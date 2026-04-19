"""
Portfolio Construction and Optimization
Modern Portfolio Theory (MPT) and Risk Parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy import optimize, stats
import logging

logger = logging.getLogger(__name__)

@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0  # Minimum position weight
    max_weight: float = 0.3  # Maximum position weight
    max_leverage: float = 1.0  # 1.0 = no leverage
    target_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sector_limits: Optional[Dict[str, float]] = None

@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    symbols: List[str]
    weights: np.ndarray
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    risk_contributions: Dict[str, float]

class PortfolioOptimizer:
    """Portfolio optimization using Modern Portfolio Theory"""
    
    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
        self.returns_data = None
        self.covariance_matrix = None
        
    def optimize_portfolio(self, returns: pd.DataFrame,
                          method: str = 'max_sharpe') -> PortfolioAllocation:
        """Optimize portfolio allocation"""
        
        self.returns_data = returns
        self.covariance_matrix = returns.cov()
        
        symbols = returns.columns.tolist()
        n_assets = len(symbols)
        
        # Expected returns (historical mean)
        expected_returns = returns.mean()
        
        # Initial weights (equal weight)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Define constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Sum to 1
        ]
        
        # Bounds for each weight
        bounds = tuple((self.constraints.min_weight, self.constraints.max_weight) 
                      for _ in range(n_assets))
        
        # Optimize based on method
        if method == 'max_sharpe':
            result = self._maximize_sharpe_ratio(
                expected_returns, initial_weights, constraints, bounds
            )
        elif method == 'min_variance':
            result = self._minimize_variance(
                initial_weights, constraints, bounds
            )
        elif method == 'risk_parity':
            result = self._risk_parity_allocation(
                initial_weights, constraints, bounds
            )
        elif method == 'max_diversification':
            result = self._maximize_diversification(
                initial_weights, constraints, bounds
            )
        else:
            # Equal weight as fallback
            optimal_weights = initial_weights
            result = {'success': True, 'x': optimal_weights}
        
        if result['success']:
            optimal_weights = result['x']
        else:
            logger.warning("Optimization failed, using equal weights")
            optimal_weights = initial_weights
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_volatility = np.sqrt(
            optimal_weights @ self.covariance_matrix @ optimal_weights
        )
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(optimal_weights)
        
        # Calculate diversification ratio
        diversification_ratio = self._calculate_diversification_ratio(optimal_weights)
        
        return PortfolioAllocation(
            symbols=symbols,
            weights=optimal_weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio,
            risk_contributions=dict(zip(symbols, risk_contributions))
        )
    
    def _maximize_sharpe_ratio(self, expected_returns: pd.Series,
                              initial_weights: np.ndarray,
                              constraints: List, bounds: Tuple) -> Dict:
        """Maximize Sharpe ratio"""
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
            
            if portfolio_vol == 0:
                return 1e6  # Large penalty for zero volatility
            
            sharpe = portfolio_return / portfolio_vol
            
            # Penalize constraints violation
            penalty = 0
            if self.constraints.target_return is not None:
                if portfolio_return < self.constraints.target_return:
                    penalty += (self.constraints.target_return - portfolio_return) * 100
            
            return -sharpe + penalty
        
        result = optimize.minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return result
    
    def _minimize_variance(self, initial_weights: np.ndarray,
                          constraints: List, bounds: Tuple) -> Dict:
        """Minimize portfolio variance"""
        
        def portfolio_variance(weights):
            return weights @ self.covariance_matrix @ weights
        
        result = optimize.minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return result
    
    def _risk_parity_allocation(self, initial_weights: np.ndarray,
                               constraints: List, bounds: Tuple) -> Dict:
        """Risk parity allocation (equal risk contribution)"""
        
        def risk_parity_objective(weights):
            # Calculate risk contributions
            portfolio_variance = weights @ self.covariance_matrix @ weights
            marginal_contributions = self.covariance_matrix @ weights
            risk_contributions = weights * marginal_contributions / portfolio_variance
            
            # Objective: minimize squared differences from equal contribution
            target_contribution = 1.0 / len(weights)
            deviations = risk_contributions - target_contribution
            
            return np.sum(deviations ** 2)
        
        result = optimize.minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return result
    
    def _maximize_diversification(self, initial_weights: np.ndarray,
                                 constraints: List, bounds: Tuple) -> Dict:
        """Maximize diversification ratio"""
        
        def diversification_objective(weights):
            # Diversification ratio = weighted average vol / portfolio vol
            individual_vols = np.sqrt(np.diag(self.covariance_matrix))
            avg_vol = np.dot(weights, individual_vols)
            portfolio_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
            
            if portfolio_vol == 0:
                return 1e6
            
            diversification_ratio = avg_vol / portfolio_vol
            
            return -diversification_ratio  # Negative for minimization
        
        result = optimize.minimize(
            diversification_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        return result
    
    def _calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions for each asset"""
        
        portfolio_variance = weights @ self.covariance_matrix @ weights
        
        if portfolio_variance == 0:
            return np.zeros_like(weights)
        
        marginal_contributions = self.covariance_matrix @ weights
        risk_contributions = weights * marginal_contributions / portfolio_variance
        
        return risk_contributions
    
    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate diversification ratio"""
        
        individual_vols = np.sqrt(np.diag(self.covariance_matrix))
        avg_vol = np.dot(weights, individual_vols)
        portfolio_vol = np.sqrt(weights @ self.covariance_matrix @ weights)
        
        if portfolio_vol == 0:
            return 0.0
        
        return avg_vol / portfolio_vol
    
    def calculate_efficient_frontier(self, returns: pd.DataFrame,
                                   n_points: int = 50) -> pd.DataFrame:
        """Calculate efficient frontier"""
        
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        n_assets = len(expected_returns)
        
        # Find minimum variance portfolio
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        initial_weights = np.ones(n_assets) / n_assets
        
        # Minimum variance portfolio
        min_var_result = self._minimize_variance(initial_weights, constraints, bounds)
        min_var_weights = min_var_result['x']
        min_var_return = np.dot(min_var_weights, expected_returns)
        
        # Maximum return portfolio (simplified - just max return asset)
        max_return_idx = np.argmax(expected_returns)
        max_return_weights = np.zeros(n_assets)
        max_return_weights[max_return_idx] = 1.0
        max_return = expected_returns[max_return_idx]
        
        # Generate points along frontier
        target_returns = np.linspace(min_var_return, max_return, n_points)
        
        frontier_points = []
        
        for target_return in target_returns:
            # Add return constraint
            constraints_with_return = constraints + [
                {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
            ]
            
            result = optimize.minimize(
                lambda w: w @ cov_matrix @ w,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_with_return,
                options={'maxiter': 1000}
            )
            
            if result['success']:
                weights = result['x']
                volatility = np.sqrt(weights @ cov_matrix @ weights)
                sharpe = target_return / volatility if volatility > 0 else 0
                
                frontier_points.append({
                    'return': target_return,
                    'volatility': volatility,
                    'sharpe': sharpe,
                    'weights': weights
                })
        
        return pd.DataFrame(frontier_points)
    
    def calculate_black_litterman(self, market_caps: Dict[str, float],
                                 views: Dict[str, float],
                                 view_confidences: Dict[str, float],
                                 risk_aversion: float = 2.5,
                                 tau: float = 0.025) -> np.ndarray:
        """Black-Litterman model for incorporating views"""
        
        symbols = list(market_caps.keys())
        n_assets = len(symbols)
        
        # Market capitalization weights
        market_cap_values = np.array([market_caps[s] for s in symbols])
        market_weights = market_cap_values / np.sum(market_cap_values)
        
        # Implied equilibrium returns
        implied_returns = risk_aversion * self.covariance_matrix @ market_weights
        
        # Create view matrix P and view vector Q
        P = np.zeros((len(views), n_assets))
        Q = np.zeros(len(views))
        Omega = np.zeros((len(views), len(views)))  # View uncertainty
        
        for i, (symbol, view_return) in enumerate(views.items()):
            symbol_idx = symbols.index(symbol)
            P[i, symbol_idx] = 1.0
            Q[i] = view_return
            
            # View confidence (lower confidence = higher uncertainty)
            confidence = view_confidences.get(symbol, 0.5)
            Omega[i, i] = (1.0 / confidence - 1.0) * tau
        
        # Black-Litterman formula
        # Π = implied equilibrium returns
        Pi = implied_returns
        
        # τΣ
        tau_sigma = tau * self.covariance_matrix
        
        # BL expected returns: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1Π + P'Ω^-1Q]
        
        # Invert matrices
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        
        # First term
        first_term = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        
        # Second term
        second_term = tau_sigma_inv @ Pi + P.T @ omega_inv @ Q
        
        # Expected returns
        expected_returns = first_term @ second_term
        
        # Calculate posterior covariance
        # Σ_BL = Σ + [(τΣ)^-1 + P'Ω^-1P]^-1
        posterior_cov = self.covariance_matrix + first_term
        
        # Update covariance matrix
        self.covariance_matrix = posterior_cov
        
        return expected_returns
    
    def calculate_margin_requirements(self, weights: np.ndarray,
                                     prices: Dict[str, float],
                                     margin_rates: Dict[str, float]) -> float:
        """Calculate margin requirements for portfolio"""
        
        total_margin = 0.0
        
        for i, symbol in enumerate(self.returns_data.columns):
            if symbol in prices and symbol in margin_rates:
                position_value = weights[i] * prices[symbol]
                margin_requirement = position_value * margin_rates[symbol]
                total_margin += margin_requirement
        
        return total_margin
    
    def stress_test_portfolio(self, weights: np.ndarray,
                             stress_scenarios: Dict[str, float]) -> Dict[str, float]:
        """Stress test portfolio under various scenarios"""
        
        scenario_results = {}
        
        for scenario_name, shock_magnitude in stress_scenarios.items():
            # Apply shock to covariance matrix
            shocked_cov = self.covariance_matrix.copy()
            
            if scenario_name == 'volatility_spike':
                # Increase all volatilities
                shocked_cov = shocked_cov * (1 + shock_magnitude)
            elif scenario_name == 'correlation_increase':
                # Increase correlations
                for i in range(len(shocked_cov)):
                    for j in range(i+1, len(shocked_cov)):
                        if shocked_cov[i, j] > 0:  # Positive correlation
                            shocked_cov[i, j] = shocked_cov[j, i] = min(
                                0.95, shocked_cov[i, j] * (1 + shock_magnitude)
                            )
            
            # Calculate shocked portfolio volatility
            shocked_volatility = np.sqrt(weights @ shocked_cov @ weights)
            
            scenario_results[scenario_name] = {
                'portfolio_volatility': shocked_volatility,
                'increase_pct': (shocked_volatility / 
                                np.sqrt(weights @ self.covariance_matrix @ weights) - 1) * 100
            }
        
        return scenario_results