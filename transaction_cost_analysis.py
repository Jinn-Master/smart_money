"""
Transaction Cost Analysis (TCA)
Measure and analyze execution costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class TransactionCosts:
    """Detailed transaction costs"""
    execution_cost: float  # Total execution cost
    slippage: float  # Price impact
    commission: float  # Broker commissions
    fees: float  # Exchange fees
    taxes: float  # Transaction taxes
    opportunity_cost: float  # Cost of delay
    total_cost: float  # Sum of all costs
    cost_bps: float  # Cost in basis points

@dataclass
class BenchmarkComparison:
    """Comparison against benchmarks"""
    vwap_benchmark: float  # VWAP benchmark price
    implementation_shortfall: float  # Difference from arrival price
    market_impact: float  # Temporary impact
    timing_cost: float  # Cost due to market movement
    opportunity_cost: float  # Cost of not trading

class TransactionCostAnalyzer:
    """Analyze transaction costs and execution quality"""
    
    def __init__(self):
        self.transaction_history = []
        self.benchmark_prices = {}
        
    def analyze_transaction(self, execution_data: Dict, 
                          market_data: pd.DataFrame) -> TransactionCosts:
        """Analyze transaction costs for an execution"""
        
        symbol = execution_data.get('symbol')
        side = execution_data.get('side', 'buy')
        executions = execution_data.get('executions', [])
        
        if not executions:
            return self._empty_costs()
        
        # Calculate basic costs
        total_quantity = sum(e.get('quantity', 0) for e in executions)
        total_value = sum(e.get('quantity', 0) * e.get('price', 0) for e in executions)
        avg_price = total_value / total_quantity if total_quantity > 0 else 0
        
        # Get benchmark prices
        arrival_price = self._get_arrival_price(symbol, market_data, 
                                              execution_data.get('start_time'))
        vwap_benchmark = self._calculate_vwap_benchmark(market_data, 
                                                      execution_data.get('start_time'),
                                                      execution_data.get('end_time'))
        
        # Calculate costs
        slippage = self._calculate_slippage(avg_price, arrival_price, side)
        commission = self._calculate_commission(total_value)
        fees = self._calculate_fees(total_value, symbol)
        taxes = self._calculate_taxes(total_value, symbol)
        
        # Calculate market impact
        market_impact = self._estimate_market_impact(executions, market_data, side)
        
        # Calculate opportunity cost
        opportunity_cost = self._calculate_opportunity_cost(
            executions, market_data, side
        )
        
        # Sum total costs
        execution_cost = slippage + commission + fees + taxes
        total_cost = execution_cost + market_impact + opportunity_cost
        
        # Convert to basis points
        cost_bps = (total_cost / total_value) * 10000 if total_value > 0 else 0
        
        costs = TransactionCosts(
            execution_cost=execution_cost,
            slippage=slippage,
            commission=commission,
            fees=fees,
            taxes=taxes,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost,
            cost_bps=cost_bps
        )
        
        # Store for analysis
        self.transaction_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'total_quantity': total_quantity,
            'total_value': total_value,
            'avg_price': avg_price,
            'costs': costs,
            'benchmark': {
                'arrival_price': arrival_price,
                'vwap_benchmark': vwap_benchmark
            }
        })
        
        logger.info(f"Transaction cost analysis: {cost_bps:.2f} bps for {symbol} {side}")
        
        return costs
    
    def compare_to_benchmarks(self, execution_data: Dict,
                            market_data: pd.DataFrame) -> BenchmarkComparison:
        """Compare execution to benchmarks"""
        
        symbol = execution_data.get('symbol')
        side = execution_data.get('side', 'buy')
        executions = execution_data.get('executions', [])
        
        if not executions:
            return self._empty_benchmark()
        
        total_quantity = sum(e.get('quantity', 0) for e in executions)
        total_value = sum(e.get('quantity', 0) * e.get('price', 0) for e in executions)
        avg_price = total_value / total_quantity if total_quantity > 0 else 0
        
        # Get benchmark prices
        start_time = execution_data.get('start_time')
        end_time = execution_data.get('end_time')
        
        arrival_price = self._get_arrival_price(symbol, market_data, start_time)
        vwap_benchmark = self._calculate_vwap_benchmark(market_data, start_time, end_time)
        
        # Implementation shortfall
        if side == 'buy':
            implementation_shortfall = avg_price - arrival_price
        else:  # sell
            implementation_shortfall = arrival_price - avg_price
        
        # Market impact (temporary)
        market_impact = self._estimate_temporary_impact(executions, market_data)
        
        # Timing cost (market movement during execution)
        timing_cost = self._calculate_timing_cost(executions, market_data, side)
        
        # Opportunity cost (difference from VWAP)
        if side == 'buy':
            opportunity_cost = avg_price - vwap_benchmark
        else:
            opportunity_cost = vwap_benchmark - avg_price
        
        comparison = BenchmarkComparison(
            vwap_benchmark=vwap_benchmark,
            implementation_shortfall=implementation_shortfall,
            market_impact=market_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost
        )
        
        return comparison
    
    def _get_arrival_price(self, symbol: str, market_data: pd.DataFrame,
                          start_time: datetime) -> float:
        """Get price at time of trading decision (arrival price)"""
        
        if market_data is None or len(market_data) == 0:
            return 0.0
        
        if start_time and start_time in market_data.index:
            return market_data.loc[start_time, 'Close']
        
        # Use most recent price before start
        if start_time:
            before = market_data[market_data.index <= start_time]
            if len(before) > 0:
                return before.iloc[-1]['Close']
        
        # Default to current price
        return market_data['Close'].iloc[-1]
    
    def _calculate_vwap_benchmark(self, market_data: pd.DataFrame,
                                start_time: datetime, 
                                end_time: datetime) -> float:
        """Calculate VWAP benchmark for period"""
        
        if market_data is None or len(market_data) == 0:
            return 0.0
        
        # Filter data to execution period
        if start_time and end_time:
            period_data = market_data[
                (market_data.index >= start_time) & 
                (market_data.index <= end_time)
            ]
        else:
            # Use last hour as default
            period_data = market_data.iloc[-60:]  # Last 60 periods
        
        if len(period_data) == 0:
            return market_data['Close'].iloc[-1]
        
        # Calculate VWAP
        volume = period_data['Volume'].sum()
        if volume == 0:
            return period_data['Close'].mean()
        
        vwap = (period_data['Volume'] * 
                (period_data['High'] + period_data['Low'] + period_data['Close']) / 3).sum() / volume
        
        return vwap
    
    def _calculate_slippage(self, execution_price: float,
                           arrival_price: float, side: str) -> float:
        """Calculate slippage cost"""
        
        if side == 'buy':
            slippage = execution_price - arrival_price
        else:  # sell
            slippage = arrival_price - execution_price
        
        return max(slippage, 0)  # Only positive slippage is a cost
    
    def _calculate_commission(self, total_value: float) -> float:
        """Calculate commission costs"""
        
        # Example: 0.1% commission
        commission_rate = 0.001
        return total_value * commission_rate
    
    def _calculate_fees(self, total_value: float, symbol: str) -> float:
        """Calculate exchange and regulatory fees"""
        
        # Simplified fee calculation
        # Different fees for different instruments
        if 'USD' in symbol or 'EUR' in symbol:
            # Forex: typically lower fees
            fee_rate = 0.00002  # 0.2 bps
        elif 'BTC' in symbol or 'ETH' in symbol:
            # Crypto: higher fees
            fee_rate = 0.001  # 10 bps
        else:
            # Stocks/indices
            fee_rate = 0.00005  # 0.5 bps
        
        return total_value * fee_rate
    
    def _calculate_taxes(self, total_value: float, symbol: str) -> float:
        """Calculate transaction taxes"""
        
        # Simplified tax calculation
        # This would need to be jurisdiction-specific
        tax_rate = 0.0  # Assume no transaction taxes for now
        
        return total_value * tax_rate
    
    def _estimate_market_impact(self, executions: List[Dict],
                              market_data: pd.DataFrame,
                              side: str) -> float:
        """Estimate permanent market impact"""
        
        if not executions or market_data is None:
            return 0.0
        
        total_quantity = sum(e.get('quantity', 0) for e in executions)
        
        # Get market depth information (simplified)
        # In reality, this would require order book data
        
        # Use square root impact model
        # Market impact = alpha * sqrt(order_size / market_volume)
        
        # Estimate market volume
        if len(market_data) >= 20:
            avg_daily_volume = market_data['Volume'].tail(20).mean()
        else:
            avg_daily_volume = market_data['Volume'].mean()
        
        if avg_daily_volume == 0:
            return 0.0
        
        # Impact coefficient (varies by market)
        alpha = 10.0  # 10 bps impact coefficient
        
        participation_rate = total_quantity / avg_daily_volume
        market_impact = alpha * np.sqrt(participation_rate)  # in bps
        
        # Convert to currency
        total_value = sum(e.get('quantity', 0) * e.get('price', 0) for e in executions)
        impact_cost = total_value * (market_impact / 10000)  # bps to decimal
        
        return impact_cost
    
    def _calculate_opportunity_cost(self, executions: List[Dict],
                                  market_data: pd.DataFrame,
                                  side: str) -> float:
        """Calculate opportunity cost of execution"""
        
        if not executions or len(executions) < 2:
            return 0.0
        
        # Get price at decision time and final price
        first_execution = executions[0]
        last_execution = executions[-1]
        
        start_price = first_execution.get('price', 0)
        end_price = last_execution.get('price', 0)
        
        if side == 'buy':
            # For buys, opportunity cost is if price went down during execution
            opportunity_cost = max(0, start_price - end_price)
        else:  # sell
            # For sells, opportunity cost is if price went up during execution
            opportunity_cost = max(0, end_price - start_price)
        
        total_quantity = sum(e.get('quantity', 0) for e in executions)
        
        return opportunity_cost * total_quantity
    
    def _estimate_temporary_impact(self, executions: List[Dict],
                                 market_data: pd.DataFrame) -> float:
        """Estimate temporary market impact"""
        
        # Temporary impact reverts quickly
        # Use half of estimated permanent impact
        permanent_impact = self._estimate_market_impact(executions, market_data, 'buy')
        temporary_impact = permanent_impact * 0.5
        
        return temporary_impact
    
    def _calculate_timing_cost(self, executions: List[Dict],
                             market_data: pd.DataFrame,
                             side: str) -> float:
        """Calculate timing cost due to market movement"""
        
        if not executions or len(executions) < 2:
            return 0.0
        
        # Get market movement during execution period
        start_time = executions[0].get('timestamp')
        end_time = executions[-1].get('timestamp')
        
        if not start_time or not end_time:
            return 0.0
        
        # Find market prices at start and end
        if market_data is not None and len(market_data) > 0:
            # Get closest market prices
            start_idx = market_data.index.get_indexer([start_time], method='nearest')[0]
            end_idx = market_data.index.get_indexer([end_time], method='nearest')[0]
            
            market_start = market_data.iloc[start_idx]['Close']
            market_end = market_data.iloc[end_idx]['Close']
        else:
            return 0.0
        
        # Calculate market movement
        market_movement = market_end - market_start
        
        if side == 'buy':
            # For buys, negative movement is cost (price increased)
            timing_cost = max(0, market_movement)
        else:  # sell
            # For sells, positive movement is cost (price decreased)
            timing_cost = max(0, -market_movement)
        
        total_quantity = sum(e.get('quantity', 0) for e in executions)
        
        return timing_cost * total_quantity
    
    def _empty_costs(self) -> TransactionCosts:
        """Return empty costs structure"""
        
        return TransactionCosts(
            execution_cost=0,
            slippage=0,
            commission=0,
            fees=0,
            taxes=0,
            opportunity_cost=0,
            total_cost=0,
            cost_bps=0
        )
    
    def _empty_benchmark(self) -> BenchmarkComparison:
        """Return empty benchmark comparison"""
        
        return BenchmarkComparison(
            vwap_benchmark=0,
            implementation_shortfall=0,
            market_impact=0,
            timing_cost=0,
            opportunity_cost=0
        )
    
    def generate_tca_report(self, period_days: int = 30) -> Dict:
        """Generate TCA report for period"""
        
        if not self.transaction_history:
            return {}
        
        # Filter to period
        cutoff = datetime.now() - timedelta(days=period_days)
        recent_transactions = [
            t for t in self.transaction_history 
            if t['timestamp'] >= cutoff
        ]
        
        if not recent_transactions:
            return {}
        
        # Calculate statistics
        total_cost_bps = np.mean([t['costs'].cost_bps for t in recent_transactions])
        total_volume = sum(t['total_quantity'] for t in recent_transactions)
        total_value = sum(t['total_value'] for t in recent_transactions)
        total_cost = sum(t['costs'].total_cost for t in recent_transactions)
        
        # Breakdown by cost component
        avg_slippage = np.mean([t['costs'].slippage for t in recent_transactions])
        avg_commission = np.mean([t['costs'].commission for t in recent_transactions])
        avg_fees = np.mean([t['costs'].fees for t in recent_transactions])
        
        # By side
        buys = [t for t in recent_transactions if t['side'] == 'buy']
        sells = [t for t in recent_transactions if t['side'] == 'sell']
        
        buy_cost_bps = np.mean([t['costs'].cost_bps for t in buys]) if buys else 0
        sell_cost_bps = np.mean([t['costs'].cost_bps for t in sells]) if sells else 0
        
        # By symbol
        symbols = {}
        for t in recent_transactions:
            symbol = t['symbol']
            if symbol not in symbols:
                symbols[symbol] = {
                    'count': 0,
                    'total_cost_bps': 0,
                    'total_volume': 0
                }
            
            symbols[symbol]['count'] += 1
            symbols[symbol]['total_cost_bps'] += t['costs'].cost_bps
            symbols[symbol]['total_volume'] += t['total_quantity']
        
        for symbol in symbols:
            symbols[symbol]['avg_cost_bps'] = (
                symbols[symbol]['total_cost_bps'] / symbols[symbol]['count']
            )
        
        report = {
            'period': f'last_{period_days}_days',
            'total_transactions': len(recent_transactions),
            'total_volume': total_volume,
            'total_value': total_value,
            'total_cost': total_cost,
            'avg_cost_bps': total_cost_bps,
            'cost_breakdown': {
                'slippage_bps': (avg_slippage / total_value * 10000) if total_value > 0 else 0,
                'commission_bps': (avg_commission / total_value * 10000) if total_value > 0 else 0,
                'fees_bps': (avg_fees / total_value * 10000) if total_value > 0 else 0
            },
            'by_side': {
                'buy_cost_bps': buy_cost_bps,
                'sell_cost_bps': sell_cost_bps,
                'buy_count': len(buys),
                'sell_count': len(sells)
            },
            'by_symbol': symbols,
            'recommendations': self._generate_recommendations(recent_transactions)
        }
        
        return report
    
    def _generate_recommendations(self, transactions: List) -> List[str]:
        """Generate recommendations based on TCA"""
        
        recommendations = []
        
        if not transactions:
            return recommendations
        
        avg_cost_bps = np.mean([t['costs'].cost_bps for t in transactions])
        
        # Cost-based recommendations
        if avg_cost_bps > 10:
            recommendations.append(
                "High transaction costs (>10 bps). Consider using iceberg/VWAP orders."
            )
        elif avg_cost_bps > 5:
            recommendations.append(
                "Moderate transaction costs (>5 bps). Optimize order sizes and timing."
            )
        
        # Check for patterns
        buy_costs = [t['costs'].cost_bps for t in transactions if t['side'] == 'buy']
        sell_costs = [t['costs'].cost_bps for t in transactions if t['side'] == 'sell']
        
        if buy_costs and sell_costs:
            avg_buy_cost = np.mean(buy_costs)
            avg_sell_cost = np.mean(sell_costs)
            
            if avg_buy_cost > avg_sell_cost * 1.5:
                recommendations.append(
                    "Buy costs significantly higher than sell costs. "
                    "Consider more aggressive buy execution strategies."
                )
            elif avg_sell_cost > avg_buy_cost * 1.5:
                recommendations.append(
                    "Sell costs significantly higher than buy costs. "
                    "Consider more aggressive sell execution strategies."
                )
        
        # Check time of day patterns
        morning_trades = [t for t in transactions 
                         if t['timestamp'].hour < 12]
        afternoon_trades = [t for t in transactions 
                          if t['timestamp'].hour >= 12]
        
        if morning_trades and afternoon_trades:
            morning_cost = np.mean([t['costs'].cost_bps for t in morning_trades])
            afternoon_cost = np.mean([t['costs'].cost_bps for t in afternoon_trades])
            
            if morning_cost > afternoon_cost * 1.3:
                recommendations.append(
                    "Higher costs in morning trading. Consider shifting trades to afternoon."
                )
            elif afternoon_cost > morning_cost * 1.3:
                recommendations.append(
                    "Higher costs in afternoon trading. Consider shifting trades to morning."
                )
        
        return recommendations[:3]  # Top 3 recommendations