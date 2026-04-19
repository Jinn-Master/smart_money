"""
Monte Carlo Simulation for Strategy Analysis
Random path analysis and stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class MonteCarloPath:
    """Single Monte Carlo simulation path"""
    path_id: int
    returns: np.ndarray
    equity_curve: np.ndarray
    final_value: float
    max_drawdown: float
    sharpe_ratio: float

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results"""
    simulations: int
    paths: List[MonteCarloPath]
    statistics: Dict[str, float]
    percentiles: Dict[str, np.ndarray]
    risk_metrics: Dict[str, float]
    probability_of_success: float

class MonteCarloSimulator:
    """Monte Carlo simulation for strategy analysis"""
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        self.results = None
        
    def simulate_strategy(self, historical_returns: pd.Series,
                         initial_capital: float = 100000,
                         strategy_function: Optional[Callable] = None,
                         confidence_level: float = 0.95) -> MonteCarloResult:
        """Run Monte Carlo simulation for strategy"""
        
        logger.info(f"Starting Monte Carlo simulation with {self.n_simulations} paths")
        
        if len(historical_returns) < 50:
            logger.warning("Insufficient historical data for reliable simulation")
        
        # Calculate statistics from historical returns
        mu = historical_returns.mean()
        sigma = historical_returns.std()
        skew = historical_returns.skew()
        kurt = historical_returns.kurtosis()
        
        logger.info(f"Historical stats: μ={mu:.4%}, σ={sigma:.4%}, "
                   f"skew={skew:.2f}, kurt={kurt:.2f}")
        
        # Generate random paths
        paths = []
        
        for i in range(self.n_simulations):
            # Generate random returns
            if strategy_function:
                # Use strategy function if provided
                simulated_returns = strategy_function(historical_returns)
            else:
                # Use statistical simulation
                simulated_returns = self._generate_random_returns(
                    mu, sigma, skew, kurt, len(historical_returns)
                )
            
            # Calculate equity curve
            equity_curve = self._calculate_equity_curve(
                simulated_returns, initial_capital
            )
            
            # Calculate metrics
            final_value = equity_curve[-1]
            max_drawdown = self._calculate_max_drawdown(equity_curve)
            sharpe_ratio = self._calculate_sharpe_ratio(simulated_returns)
            
            path = MonteCarloPath(
                path_id=i,
                returns=simulated_returns,
                equity_curve=equity_curve,
                final_value=final_value,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio
            )
            
            paths.append(path)
        
        # Calculate statistics
        statistics = self._calculate_statistics(paths)
        percentiles = self._calculate_percentiles(paths)
        risk_metrics = self._calculate_risk_metrics(paths, confidence_level)
        probability_of_success = self._calculate_success_probability(paths, initial_capital)
        
        result = MonteCarloResult(
            simulations=self.n_simulations,
            paths=paths,
            statistics=statistics,
            percentiles=percentiles,
            risk_metrics=risk_metrics,
            probability_of_success=probability_of_success
        )
        
        self.results = result
        
        logger.info(f"Monte Carlo simulation complete. Success probability: {probability_of_success:.2%}")
        
        return result
    
    def _generate_random_returns(self, mu: float, sigma: float,
                               skew: float, kurt: float,
                               n_periods: int) -> np.ndarray:
        """Generate random returns with specified statistics"""
        
        # Use different distribution based on skew/kurtosis
        if abs(skew) < 0.5 and abs(kurt) < 1:
            # Approximately normal
            returns = np.random.normal(mu, sigma, n_periods)
        else:
            # Use Student's t-distribution for fat tails
            # Estimate degrees of freedom from kurtosis
            if kurt > 0:
                df = 6 / kurt + 4  # Approximation
                df = max(2.1, min(df, 100))  # Bound degrees of freedom
                
                # Scale to match standard deviation
                scale = sigma * np.sqrt((df - 2) / df)
                returns = stats.t.rvs(df, loc=mu, scale=scale, size=n_periods)
            else:
                # Normal if negative kurtosis (unusual)
                returns = np.random.normal(mu, sigma, n_periods)
        
        # Adjust for autocorrelation if present in historical data
        # This is a simplified approach
        
        return returns
    
    def _calculate_equity_curve(self, returns: np.ndarray,
                              initial_capital: float) -> np.ndarray:
        """Calculate equity curve from returns"""
        
        equity = initial_capital * np.cumprod(1 + returns)
        
        return equity
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        return abs(max_drawdown)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray,
                               risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_statistics(self, paths: List[MonteCarloPath]) -> Dict[str, float]:
        """Calculate statistics across all paths"""
        
        final_values = [p.final_value for p in paths]
        max_drawdowns = [p.max_drawdown for p in paths]
        sharpe_ratios = [p.sharpe_ratio for p in paths]
        
        statistics = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'median_sharpe': np.median(sharpe_ratios),
            'positive_paths': sum(1 for v in final_values if v > 0),
            'negative_paths': sum(1 for v in final_values if v <= 0)
        }
        
        # Add confidence intervals
        for metric, values in [('final_value', final_values),
                              ('max_drawdown', max_drawdowns),
                              ('sharpe', sharpe_ratios)]:
            
            statistics[f'{metric}_95ci_lower'] = np.percentile(values, 2.5)
            statistics[f'{metric}_95ci_upper'] = np.percentile(values, 97.5)
        
        return statistics
    
    def _calculate_percentiles(self, paths: List[MonteCarloPath]) -> Dict[str, np.ndarray]:
        """Calculate percentiles for key metrics"""
        
        final_values = [p.final_value for p in paths]
        max_drawdowns = [p.max_drawdown for p in paths]
        sharpe_ratios = [p.sharpe_ratio for p in paths]
        
        percentiles = np.arange(1, 100, 1)  # 1st to 99th percentile
        
        return {
            'final_value': np.percentile(final_values, percentiles),
            'max_drawdown': np.percentile(max_drawdowns, percentiles),
            'sharpe_ratio': np.percentile(sharpe_ratios, percentiles),
            'percentiles': percentiles
        }
    
    def _calculate_risk_metrics(self, paths: List[MonteCarloPath],
                              confidence_level: float) -> Dict[str, float]:
        """Calculate risk metrics"""
        
        final_values = [p.final_value for p in paths]
        returns = []
        for p in paths:
            if len(p.returns) > 0:
                returns.extend(p.returns)
        
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Value at Risk (VaR)
        var_historical = np.percentile(returns_array, (1 - confidence_level) * 100)
        
        # Expected Shortfall (CVaR)
        var_threshold = var_historical
        losses_beyond_var = returns_array[returns_array <= var_threshold]
        expected_shortfall = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var_threshold
        
        # Tail risk measures
        # Sortino ratio (only downside deviation)
        risk_free_rate = 0.02 / 252  # Daily
        downside_returns = returns_array[returns_array < risk_free_rate] - risk_free_rate
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        mean_return = np.mean(returns_array)
        sortino_ratio = (mean_return - risk_free_rate) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Omega ratio
        threshold = 0.0  # Zero return threshold
        gains = returns_array[returns_array > threshold] - threshold
        losses = threshold - returns_array[returns_array <= threshold]
        
        if len(losses) > 0 and np.sum(losses) > 0:
            omega_ratio = np.sum(gains) / np.sum(losses) if len(gains) > 0 else 0
        else:
            omega_ratio = np.inf if len(gains) > 0 else 0
        
        # Maximum drawdown statistics
        max_drawdowns = [p.max_drawdown for p in paths]
        
        return {
            'var_95': var_historical,
            'expected_shortfall_95': expected_shortfall,
            'sortino_ratio': sortino_ratio,
            'omega_ratio': omega_ratio,
            'worst_max_drawdown': np.max(max_drawdowns) if max_drawdowns else 0,
            'drawdown_95th': np.percentile(max_drawdowns, 95) if max_drawdowns else 0
        }
    
    def _calculate_success_probability(self, paths: List[MonteCarloPath],
                                     initial_capital: float) -> float:
        """Calculate probability of success (final value > initial)"""
        
        if not paths:
            return 0.0
        
        successful_paths = sum(1 for p in paths if p.final_value > initial_capital)
        probability = successful_paths / len(paths)
        
        return probability
    
    def stress_test(self, historical_returns: pd.Series,
                   stress_scenarios: Dict[str, float],
                   initial_capital: float = 100000) -> Dict[str, Dict]:
        """Stress test strategy under various scenarios"""
        
        results = {}
        
        for scenario, shock in stress_scenarios.items():
            logger.info(f"Running stress test: {scenario}")
            
            # Apply stress to historical returns
            stressed_returns = self._apply_stress_scenario(
                historical_returns, scenario, shock
            )
            
            # Run simulation with stressed returns
            result = self.simulate_strategy(
                stressed_returns,
                initial_capital=initial_capital,
                confidence_level=0.95
            )
            
            results[scenario] = {
                'statistics': result.statistics,
                'risk_metrics': result.risk_metrics,
                'success_probability': result.probability_of_success,
                'shock_magnitude': shock
            }
        
        return results
    
    def _apply_stress_scenario(self, returns: pd.Series,
                             scenario: str, shock: float) -> pd.Series:
        """Apply stress scenario to returns"""
        
        if scenario == 'volatility_shock':
            # Increase volatility
            stressed = returns * (1 + shock)
            
        elif scenario == 'mean_reversal':
            # Shift mean returns downward
            stressed = returns - shock
            
        elif scenario == 'fat_tails':
            # Increase kurtosis (more extreme events)
            # Simulate by adding occasional large moves
            stressed = returns.copy()
            n_shocks = int(len(returns) * 0.05)  # 5% of periods
            shock_indices = np.random.choice(len(returns), n_shocks, replace=False)
            stressed.iloc[shock_indices] *= (1 + shock * np.random.randn(n_shocks))
            
        elif scenario == 'correlation_breakdown':
            # This would require multiple assets
            # For single asset, increase autocorrelation
            stressed = returns.copy()
            for i in range(1, len(returns)):
                stressed.iloc[i] = returns.iloc[i] * (1 - shock) + returns.iloc[i-1] * shock
            
        elif scenario == 'liquidity_crisis':
            # Combine multiple stresses
            stressed = returns * (1 + shock) - shock/2
            
        else:
            # Default: simple shock
            stressed = returns * (1 + shock)
        
        return stressed
    
    def calculate_optimal_position_size(self, confidence_level: float = 0.95,
                                      ruin_probability: float = 0.01) -> float:
        """Calculate optimal position size based on Monte Carlo results"""
        
        if self.results is None:
            logger.warning("No Monte Carlo results available")
            return 0.0
        
        # Get worst-case drawdown
        max_drawdowns = [p.max_drawdown for p in self.results.paths]
        worst_drawdown = np.percentile(max_drawdowns, (1 - ruin_probability) * 100)
        
        if worst_drawdown == 0:
            return 1.0  # No risk, full position
        
        # Kelly-like position sizing
        # Position size = 1 / worst_drawdown * kelly_fraction
        kelly_fraction = 0.5  # Half-kelly for safety
        
        optimal_size = kelly_fraction / worst_drawdown
        
        # Cap at reasonable levels
        optimal_size = min(optimal_size, 2.0)  # Max 200% position
        optimal_size = max(optimal_size, 0.1)  # Min 10% position
        
        logger.info(f"Optimal position size: {optimal_size:.2%} "
                   f"(based on {ruin_probability:.1%} ruin probability)")
        
        return optimal_size
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive Monte Carlo report"""
        
        if self.results is None:
            return {"error": "No simulation results available"}
        
        report = {
            'simulation_summary': {
                'number_of_simulations': self.results.simulations,
                'probability_of_success': self.results.probability_of_success,
                'successful_paths': self.results.statistics.get('positive_paths', 0),
                'failed_paths': self.results.statistics.get('negative_paths', 0)
            },
            'performance_statistics': {
                'mean_final_value': self.results.statistics.get('mean_final_value', 0),
                'median_final_value': self.results.statistics.get('median_final_value', 0),
                'mean_sharpe': self.results.statistics.get('mean_sharpe', 0),
                'median_sharpe': self.results.statistics.get('median_sharpe', 0)
            },
            'risk_statistics': {
                'mean_max_drawdown': self.results.statistics.get('mean_max_drawdown', 0),
                'worst_max_drawdown': self.results.risk_metrics.get('worst_max_drawdown', 0),
                'var_95': self.results.risk_metrics.get('var_95', 0),
                'expected_shortfall_95': self.results.risk_metrics.get('expected_shortfall_95', 0),
                'sortino_ratio': self.results.risk_metrics.get('sortino_ratio', 0)
            },
            'confidence_intervals': {
                'final_value_95ci': [
                    self.results.statistics.get('final_value_95ci_lower', 0),
                    self.results.statistics.get('final_value_95ci_upper', 0)
                ],
                'max_drawdown_95ci': [
                    self.results.statistics.get('max_drawdown_95ci_lower', 0),
                    self.results.statistics.get('max_drawdown_95ci_upper', 0)
                ]
            },
            'recommendations': self._generate_mc_recommendations()
        }
        
        return report
    
    def _generate_mc_recommendations(self) -> List[str]:
        """Generate recommendations based on Monte Carlo results"""
        
        recommendations = []
        
        if self.results is None:
            return ["No simulation results available"]
        
        prob_success = self.results.probability_of_success
        mean_drawdown = self.results.statistics.get('mean_max_drawdown', 0)
        worst_drawdown = self.results.risk_metrics.get('worst_max_drawdown', 0)
        var_95 = self.results.risk_metrics.get('var_95', 0)
        
        # Success probability recommendations
        if prob_success < 0.5:
            recommendations.append(
                f"Low success probability ({prob_success:.1%}). Strategy needs improvement."
            )
        elif prob_success < 0.7:
            recommendations.append(
                f"Moderate success probability ({prob_success:.1%}). "
                "Consider risk management enhancements."
            )
        
        # Drawdown recommendations
        if mean_drawdown > 0.2:  # 20%
            recommendations.append(
                f"High average drawdown ({mean_drawdown:.1%}). "
                "Consider adding stop-losses or reducing position size."
            )
        
        if worst_drawdown > 0.5:  # 50%
            recommendations.append(
                f"Extreme worst-case drawdown ({worst_drawdown:.1%}). "
                "Implement circuit breakers or maximum loss limits."
            )
        
        # VaR recommendations
        if var_95 < -0.05:  # 5% daily loss at 95% confidence
            recommendations.append(
                f"High VaR ({var_95:.2%}). Consider reducing leverage or volatility exposure."
            )
        
        # Positive recommendations
        if prob_success > 0.8 and mean_drawdown < 0.1:
            recommendations.append(
                "Excellent risk-reward profile. Strategy appears robust."
            )
        
        # Position sizing recommendation
        optimal_size = self.calculate_optimal_position_size()
        recommendations.append(
            f"Recommended maximum position size: {optimal_size:.1%} of capital"
        )
        
        return recommendations[:3]  # Top 3 recommendations
    
    def plot_distributions(self, save_path: Optional[str] = None):
        """Plot Monte Carlo distributions"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.results is None:
                logger.warning("No results to plot")
                return
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Final value distribution
            final_values = [p.final_value for p in self.results.paths]
            
            axes[0, 0].hist(final_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(x=np.mean(final_values), color='r', linestyle='--', 
                             label=f'Mean: ${np.mean(final_values):,.0f}')
            axes[0, 0].axvline(x=np.median(final_values), color='g', linestyle='--',
                             label=f'Median: ${np.median(final_values):,.0f}')
            axes[0, 0].set_xlabel('Final Portfolio Value ($)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Final Portfolio Values')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Maximum drawdown distribution
            max_drawdowns = [p.max_drawdown for p in self.results.paths]
            
            axes[0, 1].hist(max_drawdowns, bins=50, color='red', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=np.mean(max_drawdowns), color='b', linestyle='--',
                             label=f'Mean: {np.mean(max_drawdowns):.2%}')
            axes[0, 1].set_xlabel('Maximum Drawdown')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of Maximum Drawdowns')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Sharpe ratio distribution
            sharpe_ratios = [p.sharpe_ratio for p in self.results.paths]
            
            axes[1, 0].hist(sharpe_ratios, bins=50, color='green', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=np.mean(sharpe_ratios), color='r', linestyle='--',
                             label=f'Mean: {np.mean(sharpe_ratios):.2f}')
            axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 0].set_xlabel('Sharpe Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Distribution of Sharpe Ratios')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Equity curve percentiles
            # Get all equity curves
            all_curves = [p.equity_curve for p in self.results.paths]
            max_length = max(len(curve) for curve in all_curves)
            
            # Pad curves to same length
            padded_curves = []
            for curve in all_curves:
                if len(curve) < max_length:
                    padded = np.pad(curve, (0, max_length - len(curve)), 
                                  mode='constant', constant_values=curve[-1])
                else:
                    padded = curve
                padded_curves.append(padded)
            
            curves_array = np.array(padded_curves)
            
            # Calculate percentiles
            percentiles = [5, 25, 50, 75, 95]
            percentile_curves = np.percentile(curves_array, percentiles, axis=0)
            
            time_periods = np.arange(max_length)
            
            for i, pct in enumerate(percentiles):
                axes[1, 1].plot(time_periods, percentile_curves[i], 
                              label=f'{pct}th percentile', alpha=0.8)
            
            axes[1, 1].set_xlabel('Time Period')
            axes[1, 1].set_ylabel('Portfolio Value ($)')
            axes[1, 1].set_title('Equity Curve Percentiles')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting distributions: {e}")