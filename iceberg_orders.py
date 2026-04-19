"""
Iceberg Order Execution
Hidden order execution to minimize market impact
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class IcebergOrder:
    """Iceberg order parameters"""
    symbol: str
    side: str  # 'buy' or 'sell'
    total_quantity: float
    visible_quantity: float
    hidden_quantity: float
    price: Optional[float] = None  # None for market, specific for limit
    order_id: str = None
    creation_time: datetime = None
    status: str = 'pending'  # pending, active, partially_filled, filled, cancelled
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"iceberg_{self.symbol}_{self.side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.creation_time is None:
            self.creation_time = datetime.now()

@dataclass
class IcebergExecution:
    """Iceberg execution result"""
    order_id: str
    symbol: str
    side: str
    total_filled: float
    avg_price: float
    total_commissions: float
    execution_times: List[datetime]
    market_impact_estimate: float
    status: str

class IcebergOrderExecutor:
    """Execute iceberg orders with intelligent slicing"""
    
    def __init__(self, broker_interface, config: Dict = None):
        self.broker = broker_interface
        self.config = config or {
            'visible_ratio': 0.1,  # 10% visible
            'min_visible_size': 100,  # Minimum visible size
            'max_visible_size': 1000,  # Maximum visible size
            'replenish_threshold': 0.8,  # Replenish when 80% filled
            'replenish_delay': 2.0,  # Seconds between replenishments
            'max_slippage_bps': 5.0,  # Maximum 5 bps slippage
            'timeout_seconds': 300  # 5 minute timeout
        }
        
        self.active_orders: Dict[str, IcebergOrder] = {}
        self.execution_history: Dict[str, IcebergExecution] = {}
        self.order_tracker = {}
        
    async def execute_iceberg(self, symbol: str, side: str, 
                            total_quantity: float, 
                            price: Optional[float] = None,
                            urgency: float = 0.5) -> IcebergExecution:
        """Execute iceberg order"""
        
        logger.info(f"Executing iceberg order: {side} {total_quantity} {symbol}")
        
        # Calculate visible and hidden quantities
        visible_quantity = self._calculate_visible_quantity(total_quantity)
        hidden_quantity = total_quantity - visible_quantity
        
        # Create iceberg order
        iceberg = IcebergOrder(
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            visible_quantity=visible_quantity,
            hidden_quantity=hidden_quantity,
            price=price
        )
        
        self.active_orders[iceberg.order_id] = iceberg
        self.order_tracker[iceberg.order_id] = {
            'executions': [],
            'filled_quantity': 0,
            'remaining_hidden': hidden_quantity,
            'start_time': datetime.now()
        }
        
        # Start execution loop
        execution_task = asyncio.create_task(
            self._execute_iceberg_loop(iceberg, urgency)
        )
        
        # Wait for completion or timeout
        try:
            await asyncio.wait_for(execution_task, 
                                 timeout=self.config['timeout_seconds'])
        except asyncio.TimeoutError:
            logger.warning(f"Iceberg order {iceberg.order_id} timed out")
            await self._cancel_remaining(iceberg)
        
        # Compile execution results
        execution_result = self._compile_execution_result(iceberg)
        
        # Clean up
        if iceberg.order_id in self.active_orders:
            del self.active_orders[iceberg.order_id]
        
        return execution_result
    
    async def _execute_iceberg_loop(self, iceberg: IcebergOrder, 
                                   urgency: float):
        """Main iceberg execution loop"""
        
        tracker = self.order_tracker[iceberg.order_id]
        
        # Place initial visible order
        initial_order = await self._place_visible_order(iceberg)
        
        if not initial_order.get('success', False):
            logger.error(f"Failed to place initial order for {iceberg.order_id}")
            iceberg.status = 'cancelled'
            return
        
        tracker['current_visible_order'] = initial_order['order_id']
        iceberg.status = 'active'
        
        # Monitoring loop
        while tracker['filled_quantity'] < iceberg.total_quantity:
            # Check if we should replenish
            should_replenish = await self._should_replenish(iceberg, tracker)
            
            if should_replenish and tracker['remaining_hidden'] > 0:
                # Calculate replenishment size
                replenish_size = self._calculate_replenishment_size(iceberg, tracker)
                
                if replenish_size > 0:
                    # Place replenishment order
                    replenish_order = await self._place_replenishment_order(
                        iceberg, replenish_size
                    )
                    
                    if replenish_order.get('success', False):
                        tracker['current_visible_order'] = replenish_order['order_id']
                        tracker['remaining_hidden'] -= replenish_size
                        
                        logger.info(f"Replenished {replenish_size} for {iceberg.order_id}, "
                                  f"remaining hidden: {tracker['remaining_hidden']}")
                    
                    # Delay before next check
                    await asyncio.sleep(self.config['replenish_delay'])
            
            # Check order status
            await self._update_order_status(iceberg, tracker)
            
            # Check if we're done
            if tracker['filled_quantity'] >= iceberg.total_quantity:
                iceberg.status = 'filled'
                break
            
            # Check timeout
            elapsed = (datetime.now() - tracker['start_time']).total_seconds()
            if elapsed > self.config['timeout_seconds']:
                logger.warning(f"Iceberg order {iceberg.order_id} timeout after {elapsed}s")
                await self._cancel_remaining(iceberg)
                break
            
            # Short delay before next check
            await asyncio.sleep(1)
    
    async def _place_visible_order(self, iceberg: IcebergOrder) -> Dict:
        """Place visible portion of iceberg order"""
        
        order_params = {
            'symbol': iceberg.symbol,
            'side': iceberg.side,
            'quantity': iceberg.visible_quantity,
            'order_type': 'limit' if iceberg.price else 'market'
        }
        
        if iceberg.price:
            order_params['price'] = iceberg.price
        
        try:
            if iceberg.price:
                result = await self.broker.place_limit_order(**order_params)
            else:
                result = await self.broker.place_market_order(**order_params)
            
            if result.get('success', False):
                logger.info(f"Placed visible order: {iceberg.side} {iceberg.visible_quantity} "
                          f"{iceberg.symbol} @ {iceberg.price or 'market'}")
                
                return {
                    'success': True,
                    'order_id': result.get('order_id'),
                    'price': result.get('price'),
                    'timestamp': datetime.now()
                }
            else:
                logger.error(f"Failed to place visible order: {result.get('error', 'Unknown')}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            logger.error(f"Error placing visible order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _place_replenishment_order(self, iceberg: IcebergOrder,
                                       quantity: float) -> Dict:
        """Place replenishment order"""
        
        order_params = {
            'symbol': iceberg.symbol,
            'side': iceberg.side,
            'quantity': quantity,
            'order_type': 'limit' if iceberg.price else 'market'
        }
        
        if iceberg.price:
            order_params['price'] = iceberg.price
        
        try:
            if iceberg.price:
                result = await self.broker.place_limit_order(**order_params)
            else:
                result = await self.broker.place_market_order(**order_params)
            
            if result.get('success', False):
                return {
                    'success': True,
                    'order_id': result.get('order_id'),
                    'price': result.get('price')
                }
            else:
                logger.error(f"Failed to place replenishment order: {result.get('error')}")
                return {'success': False, 'error': result.get('error')}
                
        except Exception as e:
            logger.error(f"Error placing replenishment order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _should_replenish(self, iceberg: IcebergOrder, 
                              tracker: Dict) -> bool:
        """Check if we should replenish the visible order"""
        
        if tracker['remaining_hidden'] <= 0:
            return False
        
        # Get current visible order status
        current_order_id = tracker.get('current_visible_order')
        if not current_order_id:
            return True  # No current order, need to place one
        
        try:
            order_status = await self.broker.get_order_status(current_order_id)
            
            if not order_status.get('success', False):
                return True  # Order failed, need new one
            
            filled = order_status.get('filled', 0)
            visible_size = iceberg.visible_quantity
            
            # Replenish when mostly filled
            if filled >= visible_size * self.config['replenish_threshold']:
                return True
            
        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            return True
        
        return False
    
    def _calculate_visible_quantity(self, total_quantity: float) -> float:
        """Calculate visible portion size"""
        
        # Base on config ratio
        visible = total_quantity * self.config['visible_ratio']
        
        # Apply minimum and maximum
        visible = max(visible, self.config['min_visible_size'])
        visible = min(visible, self.config['max_visible_size'])
        
        # Round to appropriate lot size
        visible = self._round_to_lot_size(visible)
        
        return visible
    
    def _calculate_replenishment_size(self, iceberg: IcebergOrder,
                                    tracker: Dict) -> float:
        """Calculate size for replenishment order"""
        
        # Use same logic as initial visible size
        replenish_size = self._calculate_visible_quantity(
            tracker['remaining_hidden']
        )
        
        # Don't exceed remaining hidden quantity
        replenish_size = min(replenish_size, tracker['remaining_hidden'])
        
        return replenish_size
    
    async def _update_order_status(self, iceberg: IcebergOrder, 
                                 tracker: Dict):
        """Update order status and track fills"""
        
        current_order_id = tracker.get('current_visible_order')
        if not current_order_id:
            return
        
        try:
            order_status = await self.broker.get_order_status(current_order_id)
            
            if order_status.get('success', False):
                filled = order_status.get('filled', 0)
                price = order_status.get('price', 0)
                
                # Check if we have new fills
                previous_filled = tracker.get('last_checked_filled', 0)
                
                if filled > previous_filled:
                    # Record new fill
                    fill_quantity = filled - previous_filled
                    fill_price = price
                    
                    tracker['executions'].append({
                        'quantity': fill_quantity,
                        'price': fill_price,
                        'timestamp': datetime.now(),
                        'order_id': current_order_id
                    })
                    
                    tracker['filled_quantity'] += fill_quantity
                    tracker['last_checked_filled'] = filled
                    
                    logger.debug(f"New fill: {fill_quantity} @ {fill_price} for {iceberg.order_id}")
                
                # Update order status
                if order_status.get('status') == 'filled':
                    tracker['current_visible_order'] = None
        
        except Exception as e:
            logger.error(f"Error updating order status: {e}")
    
    async def _cancel_remaining(self, iceberg: IcebergOrder):
        """Cancel remaining hidden quantity"""
        
        tracker = self.order_tracker.get(iceberg.order_id)
        if not tracker:
            return
        
        current_order_id = tracker.get('current_visible_order')
        if current_order_id:
            try:
                await self.broker.cancel_order(current_order_id)
                logger.info(f"Cancelled order {current_order_id} for {iceberg.order_id}")
            except Exception as e:
                logger.error(f"Error cancelling order: {e}")
        
        iceberg.status = 'cancelled'
    
    def _round_to_lot_size(self, quantity: float, lot_size: float = 1.0) -> float:
        """Round quantity to lot size"""
        return round(quantity / lot_size) * lot_size
    
    def _compile_execution_result(self, iceberg: IcebergOrder) -> IcebergExecution:
        """Compile execution results"""
        
        tracker = self.order_tracker.get(iceberg.order_id, {})
        executions = tracker.get('executions', [])
        
        if not executions:
            return IcebergExecution(
                order_id=iceberg.order_id,
                symbol=iceberg.symbol,
                side=iceberg.side,
                total_filled=0,
                avg_price=0,
                total_commissions=0,
                execution_times=[],
                market_impact_estimate=0,
                status=iceberg.status
            )
        
        # Calculate statistics
        total_filled = sum(e['quantity'] for e in executions)
        total_value = sum(e['quantity'] * e['price'] for e in executions)
        avg_price = total_value / total_filled if total_filled > 0 else 0
        
        # Estimate market impact (simplified)
        first_price = executions[0]['price']
        last_price = executions[-1]['price']
        market_impact = abs(last_price - first_price) / first_price * 10000  # in bps
        
        execution_times = [e['timestamp'] for e in executions]
        
        # Estimate commissions (simplified)
        total_commissions = total_value * 0.001  # 0.1% estimate
        
        result = IcebergExecution(
            order_id=iceberg.order_id,
            symbol=iceberg.symbol,
            side=iceberg.side,
            total_filled=total_filled,
            avg_price=avg_price,
            total_commissions=total_commissions,
            execution_times=execution_times,
            market_impact_estimate=market_impact,
            status=iceberg.status
        )
        
        # Store in history
        self.execution_history[iceberg.order_id] = result
        
        # Clean up tracker
        if iceberg.order_id in self.order_tracker:
            del self.order_tracker[iceberg.order_id]
        
        logger.info(f"Iceberg execution complete: {result.total_filled}/{iceberg.total_quantity} "
                   f"filled @ avg {avg_price:.5f}, impact: {market_impact:.1f} bps")
        
        return result
    
    def get_performance_metrics(self) -> Dict:
        """Get iceberg execution performance metrics"""
        
        if not self.execution_history:
            return {}
        
        completed = [e for e in self.execution_history.values() 
                    if e.status == 'filled']
        
        if not completed:
            return {}
        
        # Calculate metrics
        total_filled = sum(e.total_filled for e in completed)
        avg_market_impact = np.mean([e.market_impact_estimate for e in completed])
        avg_fill_rate = np.mean([e.total_filled / e.total_filled for e in completed])  # Always 1 for filled
        avg_execution_time = np.mean([
            (e.execution_times[-1] - e.execution_times[0]).total_seconds() 
            for e in completed if len(e.execution_times) > 1
        ])
        
        return {
            'total_executions': len(self.execution_history),
            'completed_executions': len(completed),
            'total_volume': total_filled,
            'avg_market_impact_bps': avg_market_impact,
            'avg_execution_time_seconds': avg_execution_time,
            'success_rate': len(completed) / len(self.execution_history)
        }