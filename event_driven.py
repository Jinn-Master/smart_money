"""
Event-Driven Backtesting Engine
Tick-by-tick simulation with realistic execution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Trade record for backtesting"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000
    commission_bps: float = 1.0  # 1 basis point
    slippage_bps: float = 0.5    # 0.5 basis points
    position_sizing: str = "kelly"  # kelly, fixed, volatility
    max_positions: int = 10
    start_date: datetime = None
    end_date: datetime = None

class EventDrivenBacktester:
    """Tick-by-tick backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.equity_curve = []
        
        # Market data
        self.current_prices = {}
        self.order_book_snapshots = {}
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'calmar_ratio': 0
        }
    
    def run(self, strategy: Callable, data: Dict[str, pd.DataFrame], 
           tick_data: Optional[Dict] = None) -> Dict:
        """Run backtest with given strategy"""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Convert to timestamp range
        timestamps = self._generate_timestamps(data)
        
        # Event loop
        for timestamp in timestamps:
            self.current_time = timestamp
            
            # Update prices
            for symbol, df in data.items():
                if timestamp in df.index:
                    self.current_prices[symbol] = {
                        'bid': df.loc[timestamp, 'Close'] * 0.9999,
                        'ask': df.loc[timestamp, 'Close'] * 1.0001,
                        'mid': df.loc[timestamp, 'Close']
                    }
            
            # Get signals from strategy
            signals = strategy(self.current_prices, self.current_time)
            
            # Execute signals
            for signal in signals:
                if self._should_execute(signal):
                    self._execute_trade(signal)
            
            # Update portfolio value
            self._update_portfolio()
            
            # Record equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
            # Check stop losses
            self._check_stops()
        
        # Calculate final statistics
        self._calculate_statistics()
        
        return {
            'equity_curve': pd.DataFrame(self.equity_curve),
            'trade_history': self.trade_history,
            'statistics': self.stats,
            'final_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        }
    
    def _execute_trade(self, signal: Dict):
        """Execute trade with realistic costs"""
        symbol = signal['symbol']
        direction = signal['direction']
        quantity = signal.get('quantity', 0)
        
        if quantity <= 0:
            return
        
        # Get current price with bid/ask
        if symbol not in self.current_prices:
            logger.warning(f"No price data for {symbol}")
            return
        
        if direction == 'LONG':
            entry_price = self.current_prices[symbol]['ask']  # Buy at ask
        else:  # SHORT
            entry_price = self.current_prices[symbol]['bid']  # Sell at bid
        
        # Calculate costs
        trade_value = entry_price * quantity
        commission = trade_value * (self.config.commission_bps / 10000)
        slippage = trade_value * (self.config.slippage_bps / 10000)
        
        # Check if we can afford it
        total_cost = trade_value + commission + slippage
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash for trade: {total_cost} > {self.cash}")
            return
        
        # Create trade
        trade = Trade(
            entry_time=self.current_time,
            exit_time=None,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            commission=commission,
            slippage=slippage,
            metadata=signal.get('metadata', {})
        )
        
        # Update cash
        self.cash -= total_cost
        
        # Add to positions
        position_id = f"{symbol}_{direction}_{self.current_time.strftime('%Y%m%d_%H%M%S')}"
        self.positions[position_id] = trade
        
        logger.info(f"Executed {direction} {quantity} {symbol} @ {entry_price:.5f}")
    
    def _update_portfolio(self):
        """Update portfolio value"""
        positions_value = 0
        
        for position_id, trade in self.positions.items():
            if trade.symbol in self.current_prices:
                current_price = self.current_prices[trade.symbol]['mid']
                
                if trade.direction == 'LONG':
                    positions_value += current_price * trade.quantity
                else:  # SHORT
                    # For short, value is negative (we owe the shares)
                    positions_value -= current_price * trade.quantity
        
        self.portfolio_value = self.cash + positions_value
    
    def _calculate_statistics(self):
        """Calculate performance statistics"""
        if not self.trade_history:
            return
        
        # Basic stats
        winning_trades = [t for t in self.trade_history if t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl <= 0]
        
        self.stats['total_trades'] = len(self.trade_history)
        self.stats['winning_trades'] = len(winning_trades)
        self.stats['losing_trades'] = len(losing_trades)
        self.stats['win_rate'] = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        
        # P&L stats
        total_pnl = sum(t.pnl for t in self.trade_history)
        self.stats['total_pnl'] = total_pnl
        self.stats['avg_win'] = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        self.stats['avg_loss'] = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        self.stats['profit_factor'] = abs(sum(t.pnl for t in winning_trades) / 
                                        sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
        
        # Drawdown
        equity_values = [point['portfolio_value'] for point in self.equity_curve]
        running_max = pd.Series(equity_values).cummax()
        drawdown = (pd.Series(equity_values) - running_max) / running_max
        self.stats['max_drawdown'] = drawdown.min() * 100  # Percentage
        
        # Sharpe ratio (simplified)
        returns = pd.Series(equity_values).pct_change().dropna()
        if len(returns) > 1:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            self.stats['sharpe_ratio'] = sharpe