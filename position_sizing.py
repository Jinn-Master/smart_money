"""
Advanced Position Sizing Methods
Institutional position sizing with risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, optimize

logger = logging.getLogger(__name__)

class PositionSizingMethod(Enum):
    """Position sizing methods"""
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    VOLATILITY_TARGETING = "volatility_targeting"
    RISK_PARITY = "risk_parity"
    MARTINGALE = "martingale"  # Warning: High risk
    ANTI_MARTINGALE = "anti_martingale"
    FIXED_FRACTIONAL = "fixed_fractional"
    FIXED_RATIO = "fixed_ratio"

@dataclass
class PositionSizingParameters:
    """Position sizing parameters"""
    method: PositionSizingMethod
    account_balance: float
    risk_per_trade: float  # As fraction (0.01 = 1%)
    max_position_size: float  # As fraction of account
    volatility_target: Optional[float] = None  # Annualized volatility target
    win_rate: Optional[float] = None
    win_loss_ratio: Optional[float] = None
    correlation_matrix: Optional[pd.DataFrame] = None

class AdvancedPositionSizer:
    """Advanced position sizing with multiple methods"""
    
    def __init__(self, parameters: PositionSizingParameters):
        self.params = parameters
        self.trade_history = []
        self.performance_metrics = {}
        
    def calculate_position_size(self, trade_signal: Dict, 
                              market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate position size using specified method"""
        
        current_price = market_data['Close'].iloc[-1]
        stop_loss = trade_signal.get('stop_loss', current_price * 0.99)
        take_profit = trade_signal.get('take_profit', current_price * 1.01)
        
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_loss)
        if risk_per_share == 0:
            return {'position_size': 0, 'risk_amount': 0}
        
        # Calculate reward per share
        reward_per_share = abs(take_profit - current_price)
        
        # Risk-reward ratio
        risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
        
        # Select sizing method
        if self.params.method == PositionSizingMethod.KELLY:
            position_size = self._kelly_criterion(
                current_price, risk_per_share, reward_per_share
            )
        elif self.params.method == PositionSizingMethod.HALF_KELLY:
            position_size = self._half_kelly(
                current_price, risk_per_share, reward_per_share
            )
        elif self.params.method == PositionSizingMethod.VOLATILITY_TARGETING:
            position_size = self._volatility_targeting(
                current_price, risk_per_share, market_data
            )
        elif self.params.method == PositionSizingMethod.RISK_PARITY:
            position_size = self._risk_parity(
                current_price, risk_per_share, trade_signal
            )
        elif self.params.method == PositionSizingMethod.ANTI_MARTINGALE:
            position_size = self._anti_martingale(
                current_price, risk_per_share
            )
        else:  # FIXED_FRACTIONAL
            position_size = self._fixed_fractional(
                current_price, risk_per_share
            )
        
        # Apply maximum position size constraint
        max_shares = (self.params.account_balance * self.params.max_position_size) / current_price
        position_size = min(position_size, max_shares)
        
        # Calculate actual risk amount
        risk_amount = position_size * risk_per_share
        
        return {
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_percent': risk_amount / self.params.account_balance,
            'position_value': position_size * current_price,
            'risk_reward_ratio': risk_reward,
            'method': self.params.method.value
        }
    
    def _kelly_criterion(self, current_price: float, 
                        risk_per_share: float,
                        reward_per_share: float) -> float:
        """Kelly criterion position sizing"""
        
        if self.params.win_rate is None or self.params.win_loss_ratio is None:
            # Use estimated values if not provided
            win_rate = 0.55
            win_loss_ratio = 1.5
        else:
            win_rate = self.params.win_rate
            win_loss_ratio = self.params.win_loss_ratio
        
        # Kelly formula: f* = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        
        kelly_fraction = (p * b - q) / b
        
        if kelly_fraction <= 0:
            return 0.0
        
        # Convert to position size
        risk_amount = self.params.account_balance * self.params.risk_per_trade * kelly_fraction
        position_size = risk_amount / risk_per_share
        
        return position_size
    
    def _half_kelly(self, current_price: float,
                   risk_per_share: float,
                   reward_per_share: float) -> float:
        """Half-Kelly position sizing (safer)"""
        
        kelly_size = self._kelly_criterion(current_price, risk_per_share, reward_per_share)
        return kelly_size * 0.5
    
    def _volatility_targeting(self, current_price: float,
                            risk_per_share: float,
                            market_data: pd.DataFrame) -> float:
        """Volatility targeting position sizing"""
        
        if self.params.volatility_target is None:
            # Default to 15% annualized
            target_vol = 0.15
        else:
            target_vol = self.params.volatility_target
        
        # Calculate current volatility
        returns = market_data['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            # Use simple range-based estimate
            recent = market_data.iloc[-10:]
            avg_range = (recent['High'] - recent['Low']).mean() / recent['Close'].mean()
            current_vol = avg_range * np.sqrt(252)  # Annualize
        else:
            # Use standard deviation
            current_vol = returns.std() * np.sqrt(252)
        
        if current_vol == 0:
            return 0.0
        
        # Calculate position size to achieve target volatility
        # Simplified: position_size ∝ target_vol / current_vol
        
        volatility_ratio = target_vol / current_vol
        base_size = (self.params.account_balance * self.params.risk_per_trade) / risk_per_share
        
        position_size = base_size * volatility_ratio
        
        return position_size
    
    def _risk_parity(self, current_price: float,
                    risk_per_share: float,
                    trade_signal: Dict) -> float:
        """Risk parity position sizing"""
        
        if self.params.correlation_matrix is None:
            # Without correlation matrix, use simplified approach
            return self._fixed_fractional(current_price, risk_per_share)
        
        # Get symbol
        symbol = trade_signal.get('symbol', '')
        
        if symbol not in self.params.correlation_matrix.index:
            return self._fixed_fractional(current_price, risk_per_share)
        
        # Calculate risk contribution
        # In risk parity, each position contributes equally to portfolio risk
        
        # Get correlations with current positions
        # This is simplified - full implementation would consider portfolio context
        
        # For now, use inverse volatility weighting
        position_size = self._volatility_targeting(current_price, risk_per_share, 
                                                  pd.DataFrame({'Close': [current_price]}))
        
        # Adjust based on correlation
        avg_correlation = self.params.correlation_matrix.loc[symbol].mean()
        
        # Higher correlation = smaller position to maintain risk parity
        correlation_adjustment = 1.0 / (1.0 + avg_correlation)
        position_size *= correlation_adjustment
        
        return position_size
    
    def _anti_martingale(self, current_price: float,
                        risk_per_share: float) -> float:
        """Anti-martingale position sizing (increase after wins)"""
        
        if not self.trade_history:
            # First trade - use base size
            base_size = (self.params.account_balance * self.params.risk_per_trade) / risk_per_share
            return base_size
        
        # Check recent performance
        recent_trades = self.trade_history[-10:] if len(self.trade_history) >= 10 else self.trade_history
        
        if not recent_trades:
            base_size = (self.params.account_balance * self.params.risk_per_trade) / risk_per_share
            return base_size
        
        # Calculate win streak
        wins = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
        win_rate = wins / len(recent_trades)
        
        # Adjust position size based on recent performance
        base_size = (self.params.account_balance * self.params.risk_per_trade) / risk_per_share
        
        if win_rate > 0.6:
            # Winning streak - increase size
            adjustment = 1.0 + (win_rate - 0.6) * 2  # Up to 1.8x
        elif win_rate < 0.4:
            # Losing streak - decrease size
            adjustment = 0.5 + (win_rate - 0.4) * 2.5  # Down to 0.25x
        else:
            # Neutral performance
            adjustment = 1.0
        
        position_size = base_size * adjustment
        
        return position_size
    
    def _fixed_fractional(self, current_price: float,
                         risk_per_share: float) -> float:
        """Fixed fractional position sizing"""
        
        risk_amount = self.params.account_balance * self.params.risk_per_trade
        position_size = risk_amount / risk_per_share
        
        return position_size
    
    def update_trade_history(self, trade_result: Dict):
        """Update trade history for adaptive sizing"""
        
        self.trade_history.append(trade_result)
        
        # Keep history manageable
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        # Update performance metrics
        self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update performance metrics for adaptive sizing"""
        
        if len(self.trade_history) < 10:
            return
        
        # Calculate metrics
        pnls = [t.get('pnl', 0) for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        if pnls:
            self.performance_metrics = {
                'win_rate': len(wins) / len(pnls),
                'avg_win': np.mean(wins) if wins else 0,
                'avg_loss': np.mean(losses) if losses else 0,
                'win_loss_ratio': abs(np.mean(wins) / np.mean(losses)) if losses and np.mean(losses) != 0 else 0,
                'expectancy': (self.performance_metrics.get('win_rate', 0) * 
                              self.performance_metrics.get('avg_win', 0) +
                              (1 - self.performance_metrics.get('win_rate', 0)) * 
                              self.performance_metrics.get('avg_loss', 0)),
                'sharpe_ratio': self._calculate_sharpe_ratio(pnls)
            }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns"""
        
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        
        return sharpe
    
    def calculate_optimal_leverage(self, confidence_level: float = 0.95) -> float:
        """Calculate optimal leverage based on risk metrics"""
        
        if not self.performance_metrics:
            return 1.0  # No leverage
        
        win_rate = self.performance_metrics.get('win_rate', 0.5)
        win_loss_ratio = self.performance_metrics.get('win_loss_ratio', 1.0)
        
        # Kelly optimal leverage
        kelly_leverage = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply confidence adjustment
        z_score = stats.norm.ppf(confidence_level)
        
        # Estimate uncertainty in win rate
        n_trades = len(self.trade_history)
        if n_trades > 0:
            # Standard error of win rate
            se = np.sqrt(win_rate * (1 - win_rate) / n_trades)
            adjusted_win_rate = win_rate - z_score * se
        else:
            adjusted_win_rate = win_rate
        
        # Conservative leverage (half-kelly with uncertainty adjustment)
        conservative_leverage = max(0, (adjusted_win_rate - (1 - adjusted_win_rate) / win_loss_ratio) * 0.5)
        
        return min(conservative_leverage, 3.0)  # Cap at 3x leverage
    
    def calculate_value_at_risk(self, position_size: float,
                              current_price: float,
                              confidence_level: float = 0.95,
                              horizon_days: int = 1) -> Dict[str, float]:
        """Calculate Value at Risk for position"""
        
        if not self.trade_history:
            return {'var_95': 0, 'expected_shortfall': 0}
        
        # Get historical returns
        returns = []
        for trade in self.trade_history[-50:]:  # Last 50 trades
            if 'entry_price' in trade and 'exit_price' in trade:
                ret = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
                returns.append(ret)
        
        if len(returns) < 10:
            return {'var_95': 0, 'expected_shortfall': 0}
        
        returns_array = np.array(returns)
        
        # Historical VaR
        var_historical = np.percentile(returns_array, (1 - confidence_level) * 100)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        z_score = stats.norm.ppf(1 - confidence_level)
        var_parametric = mean_return + z_score * std_return
        
        # Expected Shortfall (CVaR)
        var_threshold = var_historical
        losses_beyond_var = returns_array[returns_array <= var_threshold]
        expected_shortfall = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var_threshold
        
        # Convert to dollar amounts
        position_value = position_size * current_price
        
        return {
            'var_95': abs(var_historical * position_value),
            'var_parametric': abs(var_parametric * position_value),
            'expected_shortfall': abs(expected_shortfall * position_value),
            'daily_volatility': std_return * position_value
        }