"""
Performance Attribution Analysis
Detailed breakdown of returns by source
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class AttributionComponent:
    """Performance attribution component"""
    source: str  # 'market', 'sector', 'selection', 'timing', 'currency'
    return_contribution: float
    risk_contribution: float
    information_ratio: float

@dataclass
class PerformanceAttribution:
    """Complete performance attribution analysis"""
    total_return: float
    benchmark_return: float
    active_return: float
    components: List[AttributionComponent]
    brinson_attribution: Dict[str, float]
    risk_decomposition: Dict[str, float]

class PerformanceAttributionAnalyzer:
    """Performance attribution analysis"""
    
    def __init__(self):
        self.trade_history = []
        self.benchmark_returns = {}
        self.attribution_history = []
        
    def analyze_performance(self, portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series,
                          position_data: List[Dict],
                          market_data: pd.DataFrame) -> PerformanceAttribution:
        """Analyze performance attribution"""
        
        if len(portfolio_returns) < 20 or len(benchmark_returns) < 20:
            logger.warning("Insufficient data for attribution analysis")
            return self._empty_attribution()
        
        # Calculate returns
        total_return = (1 + portfolio_returns).prod() - 1
        benchmark_return = (1 + benchmark_returns).prod() - 1
        active_return = total_return - benchmark_return
        
        # Brinson attribution (if we have sector data)
        brinson_attribution = self._calculate_brinson_attribution(
            portfolio_returns, benchmark_returns, position_data
        )
        
        # Factor-based attribution
        factor_components = self._calculate_factor_attribution(
            portfolio_returns, benchmark_returns, market_data
        )
        
        # Risk decomposition
        risk_decomposition = self._decompose_risk(
            portfolio_returns, benchmark_returns
        )
        
        # Combine components
        components = []
        
        # Market component
        market_component = AttributionComponent(
            source='market',
            return_contribution=benchmark_return,
            risk_contribution=risk_decomposition.get('market_risk', 0),
            information_ratio=self._calculate_information_ratio(
                benchmark_return, risk_decomposition.get('market_risk', 0)
            )
        )
        components.append(market_component)
        
        # Active components from Brinson
        for source, contribution in brinson_attribution.items():
            if source != 'total':
                component = AttributionComponent(
                    source=source,
                    return_contribution=contribution,
                    risk_contribution=risk_decomposition.get(f'{source}_risk', 0),
                    information_ratio=self._calculate_information_ratio(
                        contribution, risk_decomposition.get(f'{source}_risk', 0)
                    )
                )
                components.append(component)
        
        # Factor components
        for factor in factor_components:
            components.append(factor)
        
        attribution = PerformanceAttribution(
            total_return=total_return,
            benchmark_return=benchmark_return,
            active_return=active_return,
            components=components,
            brinson_attribution=brinson_attribution,
            risk_decomposition=risk_decomposition
        )
        
        # Store for history
        self.attribution_history.append({
            'timestamp': datetime.now(),
            'attribution': attribution,
            'period': {
                'start': portfolio_returns.index[0],
                'end': portfolio_returns.index[-1]
            }
        })
        
        logger.info(f"Performance attribution: Active return: {active_return:.2%}")
        
        return attribution
    
    def _calculate_brinson_attribution(self, portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series,
                                     position_data: List[Dict]) -> Dict[str, float]:
        """Calculate Brinson-Fachler attribution"""
        
        # Simplified Brinson attribution
        # In practice, this would require detailed sector/asset class data
        
        attribution = {
            'total': 0.0,
            'allocation': 0.0,
            'selection': 0.0,
            'interaction': 0.0
        }
        
        if not position_data or len(portfolio_returns) < 30:
            return attribution
        
        # Calculate active return
        active_return = (1 + portfolio_returns).prod() - 1 - \
                       (1 + benchmark_returns).prod() + 1
        
        # Simplified decomposition
        # Allocation effect (sector weighting)
        allocation_effect = active_return * 0.4  # Assume 40% allocation
        
        # Selection effect (security selection)
        selection_effect = active_return * 0.4  # Assume 40% selection
        
        # Interaction effect
        interaction_effect = active_return * 0.2  # Assume 20% interaction
        
        attribution.update({
            'total': active_return,
            'allocation': allocation_effect,
            'selection': selection_effect,
            'interaction': interaction_effect
        })
        
        return attribution
    
    def _calculate_factor_attribution(self, portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series,
                                    market_data: pd.DataFrame) -> List[AttributionComponent]:
        """Calculate factor-based attribution"""
        
        components = []
        
        if market_data is None or len(market_data) < 30:
            return components
        
        # Common factors to analyze
        factors = {
            'market': market_data.get('Close', pd.Series([0])).pct_change().dropna(),
            'size': self._calculate_size_factor(market_data),
            'value': self._calculate_value_factor(market_data),
            'momentum': self._calculate_momentum_factor(market_data),
            'volatility': self._calculate_volatility_factor(market_data)
        }
        
        # Calculate factor exposures and contributions
        active_returns = portfolio_returns - benchmark_returns
        
        for factor_name, factor_returns in factors.items():
            if len(factor_returns) < 20:
                continue
            
            # Align dates
            common_dates = active_returns.index.intersection(factor_returns.index)
            if len(common_dates) < 10:
                continue
            
            aligned_active = active_returns.loc[common_dates]
            aligned_factor = factor_returns.loc[common_dates]
            
            # Calculate factor exposure (beta)
            try:
                # Simple regression
                beta, alpha = np.polyfit(aligned_factor, aligned_active, 1)
                
                # Calculate contribution
                factor_contribution = beta * aligned_factor.mean()
                
                # Risk contribution
                factor_risk = abs(beta) * aligned_factor.std()
                
                # Information ratio
                ir = factor_contribution / factor_risk if factor_risk > 0 else 0
                
                component = AttributionComponent(
                    source=f'factor_{factor_name}',
                    return_contribution=factor_contribution,
                    risk_contribution=factor_risk,
                    information_ratio=ir
                )
                
                components.append(component)
                
            except Exception as e:
                logger.debug(f"Factor attribution failed for {factor_name}: {e}")
                continue
        
        return components
    
    def _calculate_size_factor(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate size factor returns"""
        # Simplified - in practice would use small-cap vs large-cap returns
        return market_data.get('Close', pd.Series([0])).pct_change().rolling(5).std()
    
    def _calculate_value_factor(self, market_data: pd.DataFrame) -> pd.Series:
        """Calculate value factor returns"""
        # Simplified - would use value vs growth returns
        return pd.Series(0, index=market_data.index)