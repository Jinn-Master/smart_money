"""
Walk-Forward Analysis (WFA)
Robust strategy validation with rolling windows
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from itertools import product

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardWindow:
    """Walk-forward analysis window"""
    training_start: datetime
    training_end: datetime
    testing_start: datetime
    testing_end: datetime
    window_index: int

@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    window: WalkForwardWindow
    training_metrics: Dict[str, float]
    testing_metrics: Dict[str, float]
    optimized_parameters: Dict[str, Any]
    parameter_stability: float

class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy validation"""
    
    def __init__(self, initial_window_months: int = 12,
                 testing_window_months: int = 3,
                 step_months: int = 3):
        
        self.initial_window = initial_window_months
        self.testing_window = testing_window_months
        self.step = step_months
        
        self.results = []
        self.parameter_history = []
        
    def run_analysis(self, data: pd.DataFrame,
                    strategy: Callable,
                    parameter_grid: Dict[str, List[Any]],
                    metric: str = 'sharpe_ratio') -> List[WalkForwardResult]:
        """Run walk-forward analysis"""
        
        logger.info("Starting walk-forward analysis...")
        
        # Generate walk-forward windows
        windows = self._generate_windows(data.index.min(), data.index.max())
        
        results = []
        
        for i, window in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Split data
            train_data = data[
                (data.index >= window.training_start) & 
                (data.index <= window.training_end)
            ]
            
            test_data = data[
                (data.index >= window.testing_start) & 
                (data.index <= window.testing_end)
            ]
            
            if len(train_data) < 100 or len(test_data) < 20:
                logger.warning(f"Window {i+1} has insufficient data, skipping")
                continue
            
            # Optimize parameters on training data
            optimized_params = self._optimize_parameters(
                train_data, strategy, parameter_grid, metric
            )
            
            # Test on out-of-sample data
            test_result = strategy(test_data, **optimized_params)
            
            # Calculate training metrics (in-sample)
            train_result = strategy(train_data, **optimized_params)
            
            # Calculate parameter stability
            stability = self._calculate_parameter_stability(optimized_params)
            
            # Store results
            result = WalkForwardResult(
                window=window,
                training_metrics=self._extract_metrics(train_result),
                testing_metrics=self._extract_metrics(test_result),
                optimized_parameters=optimized_params,
                parameter_stability=stability
            )
            
            results.append(result)
            
            # Store parameter history for analysis
            self.parameter_history.append({
                'window': window,
                'parameters': optimized_params
            })
        
        self.results = results
        
        # Generate summary report
        self._generate_summary_report()
        
        return results
    
    def _generate_windows(self, start_date: datetime, 
                         end_date: datetime) -> List[WalkForwardWindow]:
        """Generate walk-forward windows"""
        
        windows = []
        
        # Start with initial training window
        current_train_start = start_date
        current_train_end = current_train_start + timedelta(days=30 * self.initial_window)
        current_test_end = current_train_end + timedelta(days=30 * self.testing_window)
        
        window_index = 1
        
        while current_test_end <= end_date:
            window = WalkForwardWindow(
                training_start=current_train_start,
                training_end=current_train_end,
                testing_start=current_train_end + timedelta(days=1),
                testing_end=current_test_end,
                window_index=window_index
            )
            
            windows.append(window)
            
            # Move window forward
            current_train_start += timedelta(days=30 * self.step)
            current_train_end = current_train_start + timedelta(days=30 * self.initial_window)
            current_test_end = current_train_end + timedelta(days=30 * self.testing_window)
            
            window_index += 1
        
        logger.info(f"Generated {len(windows)} walk-forward windows")
        
        return windows
    
    def _optimize_parameters(self, data: pd.DataFrame,
                           strategy: Callable,
                           parameter_grid: Dict[str, List[Any]],
                           metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        
        best_metric = -np.inf
        best_params = {}
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        # Limit number of combinations for performance
        max_combinations = 100
        if np.prod([len(v) for v in param_values]) > max_combinations:
            # Use random sampling instead of full grid
            combinations = []
            for _ in range(max_combinations):
                combo = {}
                for name, values in parameter_grid.items():
                    combo[name] = np.random.choice(values)
                combinations.append(combo)
        else:
            # Full grid search
            combinations = [
                dict(zip(param_names, combo))
                for combo in product(*param_values)
            ]
        
        # Evaluate each combination
        for params in combinations:
            try:
                result = strategy(data, **params)
                metrics = self._extract_metrics(result)
                
                current_metric = metrics.get(metric, -np.inf)
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_params = params.copy()
                    
            except Exception as e:
                logger.debug(f"Parameter combination failed: {params}, error: {e}")
                continue
        
        logger.info(f"Optimized parameters: {best_params}, metric: {best_metric:.4f}")
        
        return best_params
    
    def _extract_metrics(self, result: Any) -> Dict[str, float]:
        """Extract performance metrics from strategy result"""
        
        if isinstance(result, dict):
            # Result is already a dictionary of metrics
            return result
        
        # Try to extract from common result structures
        metrics = {}
        
        if hasattr(result, 'sharpe_ratio'):
            metrics['sharpe_ratio'] = result.sharpe_ratio
        if hasattr(result, 'total_return'):
            metrics['total_return'] = result.total_return
        if hasattr(result, 'max_drawdown'):
            metrics['max_drawdown'] = result.max_drawdown
        if hasattr(result, 'win_rate'):
            metrics['win_rate'] = result.win_rate
        
        # Default metrics if none found
        if not metrics:
            metrics = {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        return metrics
    
    def _calculate_parameter_stability(self, 
                                     current_params: Dict[str, Any]) -> float:
        """Calculate parameter stability compared to history"""
        
        if not self.parameter_history:
            return 1.0  # First window, perfect stability
        
        # Compare with average of previous parameters
        prev_params_list = [h['parameters'] for h in self.parameter_history]
        
        stability_scores = []
        
        for param_name, param_value in current_params.items():
            if param_name not in prev_params_list[0]:
                continue
            
            # Get historical values for this parameter
            historical_values = []
            for prev in prev_params_list:
                if param_name in prev:
                    historical_values.append(prev[param_name])
            
            if not historical_values:
                continue
            
            # Calculate stability (1 - normalized deviation)
            if isinstance(param_value, (int, float)):
                # Numeric parameter
                avg_historical = np.mean(historical_values)
                std_historical = np.std(historical_values)
                
                if std_historical == 0:
                    stability = 1.0
                else:
                    deviation = abs(param_value - avg_historical) / std_historical
                    stability = 1.0 / (1.0 + deviation)
            
            else:
                # Categorical parameter
                # Stability = frequency of this value in history
                freq = historical_values.count(param_value) / len(historical_values)
                stability = freq
            
            stability_scores.append(stability)
        
        if not stability_scores:
            return 1.0
        
        return np.mean(stability_scores)
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of walk-forward analysis"""
        
        if not self.results:
            return {}
        
        # Calculate summary statistics
        test_sharpe = [r.testing_metrics.get('sharpe_ratio', 0) for r in self.results]
        test_returns = [r.testing_metrics.get('total_return', 0) for r in self.results]
        test_drawdowns = [r.testing_metrics.get('max_drawdown', 0) for r in self.results]
        
        train_sharpe = [r.training_metrics.get('sharpe_ratio', 0) for r in self.results]
        train_returns = [r.training_metrics.get('total_return', 0) for r in self.results]
        
        stability_scores = [r.parameter_stability for r in self.results]
        
        # Calculate decay (difference between train and test)
        decay_sharpe = np.mean([t - s for t, s in zip(train_sharpe, test_sharpe)])
        decay_return = np.mean([t - s for t, s in zip(train_returns, test_returns)])
        
        # Consistency metrics
        positive_windows = sum(1 for s in test_sharpe if s > 0)
        consistency = positive_windows / len(test_sharpe) if test_sharpe else 0
        
        # Parameter stability analysis
        param_stability = np.mean(stability_scores)
        
        # Best and worst windows
        if test_sharpe:
            best_idx = np.argmax(test_sharpe)
            worst_idx = np.argmin(test_sharpe)
            
            best_window = self.results[best_idx]
            worst_window = self.results[worst_idx]
        else:
            best_window = worst_window = None
        
        report = {
            'summary': {
                'total_windows': len(self.results),
                'avg_test_sharpe': np.mean(test_sharpe),
                'std_test_sharpe': np.std(test_sharpe),
                'avg_test_return': np.mean(test_returns),
                'avg_test_drawdown': np.mean(test_drawdowns),
                'consistency': consistency,
                'parameter_stability': param_stability,
                'decay_sharpe': decay_sharpe,
                'decay_return': decay_return
            },
            'best_window': {
                'index': best_window.window.window_index if best_window else None,
                'sharpe': best_window.testing_metrics.get('sharpe_ratio', 0) if best_window else 0,
                'parameters': best_window.optimized_parameters if best_window else {}
            },
            'worst_window': {
                'index': worst_window.window.window_index if worst_window else None,
                'sharpe': worst_window.testing_metrics.get('sharpe_ratio', 0) if worst_window else 0,
                'parameters': worst_window.optimized_parameters if worst_window else {}
            },
            'robustness_assessment': self._assess_robustness(),
            'recommendations': self._generate_recommendations()
        }
        
        logger.info(f"Walk-forward analysis complete. Avg test Sharpe: {report['summary']['avg_test_sharpe']:.4f}")
        
        return report
    
    def _assess_robustness(self) -> Dict[str, Any]:
        """Assess strategy robustness based on WFA results"""
        
        if not self.results:
            return {'score': 0, 'assessment': 'insufficient_data'}
        
        test_sharpe = [r.testing_metrics.get('sharpe_ratio', 0) for r in self.results]
        stability_scores = [r.parameter_stability for r in self.results]
        
        # Calculate robustness score (0-100)
        score = 0
        
        # 1. Consistency (40%)
        positive_windows = sum(1 for s in test_sharpe if s > 0)
        consistency = positive_windows / len(test_sharpe)
        score += consistency * 40
        
        # 2. Sharpe ratio level (30%)
        avg_sharpe = np.mean(test_sharpe)
        if avg_sharpe > 1.5:
            score += 30
        elif avg_sharpe > 1.0:
            score += 25
        elif avg_sharpe > 0.5:
            score += 20
        elif avg_sharpe > 0:
            score += 10
        
        # 3. Parameter stability (20%)
        avg_stability = np.mean(stability_scores)
        score += avg_stability * 20
        
        # 4. Low drawdown (10%)
        test_drawdowns = [r.testing_metrics.get('max_drawdown', 0) for r in self.results]
        avg_drawdown = np.mean(test_drawdowns)
        
        if avg_drawdown < 0.1:  # Less than 10%
            score += 10
        elif avg_drawdown < 0.2:  # Less than 20%
            score += 5
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        
        # Assessment categories
        if score >= 80:
            assessment = 'Excellent'
        elif score >= 70:
            assessment = 'Good'
        elif score >= 60:
            assessment = 'Moderate'
        elif score >= 50:
            assessment = 'Marginal'
        else:
            assessment = 'Poor'
        
        return {
            'score': score,
            'assessment': assessment,
            'components': {
                'consistency': consistency,
                'avg_sharpe': avg_sharpe,
                'parameter_stability': avg_stability,
                'avg_drawdown': avg_drawdown
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on WFA results"""
        
        recommendations = []
        
        if not self.results:
            return ["Insufficient data for recommendations"]
        
        robustness = self._assess_robustness()
        
        # Strategy-specific recommendations
        if robustness['score'] < 60:
            recommendations.append(
                "Strategy shows poor robustness. Consider fundamental redesign."
            )
        elif robustness['score'] < 70:
            recommendations.append(
                "Strategy shows marginal robustness. Optimize parameters and add filters."
            )
        elif robustness['score'] < 80:
            recommendations.append(
                "Strategy shows good robustness. Consider live trading with small capital."
            )
        else:
            recommendations.append(
                "Strategy shows excellent robustness. Suitable for live trading."
            )
        
        # Parameter stability recommendations
        stability_scores = [r.parameter_stability for r in self.results]
        avg_stability = np.mean(stability_scores)
        
        if avg_stability < 0.5:
            recommendations.append(
                "High parameter instability. Consider using adaptive parameters."
            )
        
        # Decay recommendations
        test_sharpe = [r.testing_metrics.get('sharpe_ratio', 0) for r in self.results]
        train_sharpe = [r.training_metrics.get('sharpe_ratio', 0) for r in self.results]
        
        decay = np.mean([t - s for t, s in zip(train_sharpe, test_sharpe)])
        
        if decay > 0.5:
            recommendations.append(
                f"High decay ({decay:.2f}) between train and test. "
                "Strategy may be overfitting. Reduce parameter complexity."
            )
        
        # Consistency recommendations
        positive_windows = sum(1 for s in test_sharpe if s > 0)
        consistency = positive_windows / len(test_sharpe)
        
        if consistency < 0.6:
            recommendations.append(
                f"Low consistency ({consistency:.0%}). "
                "Strategy may be market-regime dependent."
            )
        
        return recommendations[:3]  # Top 3 recommendations
    
    def plot_results(self, save_path: Optional[str] = None):
        """Plot walk-forward analysis results"""
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if not self.results:
                logger.warning("No results to plot")
                return
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # 1. Sharpe ratio progression
            test_sharpe = [r.testing_metrics.get('sharpe_ratio', 0) for r in self.results]
            train_sharpe = [r.training_metrics.get('sharpe_ratio', 0) for r in self.results]
            window_indices = [r.window.window_index for r in self.results]
            
            axes[0].plot(window_indices, test_sharpe, 'b-', label='Test Sharpe', marker='o')
            axes[0].plot(window_indices, train_sharpe, 'r--', label='Train Sharpe', alpha=0.7)
            axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[0].set_xlabel('Window Index')
            axes[0].set_ylabel('Sharpe Ratio')
            axes[0].set_title('Walk-Forward Sharpe Ratio Progression')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. Parameter stability
            stability = [r.parameter_stability for r in self.results]
            
            axes[1].bar(window_indices, stability, color='green', alpha=0.7)
            axes[1].axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Stability Threshold')
            axes[1].set_xlabel('Window Index')
            axes[1].set_ylabel('Parameter Stability')
            axes[1].set_title('Parameter Stability Across Windows')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 3. Returns distribution
            test_returns = [r.testing_metrics.get('total_return', 0) for r in self.results]
            
            axes[2].hist(test_returns, bins=10, color='blue', alpha=0.7, edgecolor='black')
            axes[2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
            axes[2].set_xlabel('Window Return')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('Distribution of Window Returns')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting results: {e}")