"""
Market Depth Analysis
Implements: Depth of Market (DOM) analysis, absorption, exhaustion patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class DepthLevel:
    """Market depth level with aggregated orders"""
    price: float
    total_quantity: float
    bid_quantity: float = 0.0
    ask_quantity: float = 0.0
    order_count: int = 0
    last_update: datetime = None
    level_type: str = ''  # 'bid', 'ask', 'both'
    
    @property
    def imbalance(self) -> float:
        """Calculate imbalance at this level"""
        total = self.bid_quantity + self.ask_quantity
        if total == 0:
            return 0.0
        return (self.bid_quantity - self.ask_quantity) / total
    
    @property
    def net_quantity(self) -> float:
        """Net quantity (bid - ask)"""
        return self.bid_quantity - self.ask_quantity


@dataclass
class MarketDepthSnapshot:
    """Complete market depth snapshot"""
    timestamp: datetime
    symbol: str
    levels: List[DepthLevel]
    spread: float
    best_bid: float
    best_ask: float
    total_bid_volume: float
    total_ask_volume: float
    
    def get_level_at_price(self, price: float, tolerance: float = 0.0001) -> Optional[DepthLevel]:
        """Get depth level at specific price"""
        for level in self.levels:
            if abs(level.price - price) / price <= tolerance:
                return level
        return None
    
    def get_cumulative_depth(self, side: str, levels: int = 10) -> List[Tuple[float, float]]:
        """Get cumulative depth for bid or ask side"""
        cumulative = []
        total = 0.0
        
        # Filter levels by side
        side_levels = []
        for level in self.levels:
            if side == 'bid' and level.bid_quantity > 0:
                side_levels.append((level.price, level.bid_quantity))
            elif side == 'ask' and level.ask_quantity > 0:
                side_levels.append((level.price, level.ask_quantity))
        
        # Sort by price (bids descending, asks ascending)
        if side == 'bid':
            side_levels.sort(key=lambda x: x[0], reverse=True)
        else:
            side_levels.sort(key=lambda x: x[0])
        
        # Calculate cumulative depth
        for price, quantity in side_levels[:levels]:
            total += quantity
            cumulative.append((price, total))
        
        return cumulative


@dataclass
class AbsorptionSignal:
    """Absorption pattern signal"""
    signal_type: str  # 'absorption', 'exhaustion', 'accumulation', 'distribution'
    price_level: float
    strength: float
    volume_absorbed: float
    timestamp: datetime
    context: str = ''  # 'bid', 'ask', 'both'


class MarketDepthAnalyzer:
    """Analyzes market depth and DOM patterns"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.depth_levels = self.config.get('depth_levels', 20)
        self.absorption_threshold = self.config.get('absorption_threshold', 2.0)
        self.exhaustion_threshold = self.config.get('exhaustion_threshold', 0.3)
        self.update_frequency = self.config.get('update_frequency', 100)  # ms
        
        # State
        self.depth_history: deque[MarketDepthSnapshot] = deque(maxlen=1000)
        self.absorption_signals: List[AbsorptionSignal] = []
        self.depth_imbalance_history: List[float] = []
        
        # Statistics
        self.stats = {
            'avg_spread': 0.0,
            'max_depth_imbalance': 0.0,
            'absorption_count': 0,
            'total_updates': 0
        }
        
        # Cache for performance
        self._depth_cache: Dict[str, MarketDepthSnapshot] = {}
        self._imbalance_cache: Dict[str, List[float]] = {}
        
        logger.info("MarketDepthAnalyzer initialized")
    
    def update_depth(self, 
                    bids: List[Tuple[float, float]], 
                    asks: List[Tuple[float, float]],
                    symbol: str) -> MarketDepthSnapshot:
        """
        Update market depth with new data
        
        Args:
            bids: List of (price, quantity) for bids
            asks: List of (price, quantity) for asks
            symbol: Trading symbol
        
        Returns:
            MarketDepthSnapshot
        """
        # Sort bids descending, asks ascending
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)[:self.depth_levels]
        asks_sorted = sorted(asks, key=lambda x: x[0])[:self.depth_levels]
        
        # Create depth levels
        levels_dict: Dict[float, DepthLevel] = {}
        
        # Process bids
        for price, quantity in bids_sorted:
            if price not in levels_dict:
                levels_dict[price] = DepthLevel(
                    price=price,
                    total_quantity=0,
                    bid_quantity=0,
                    ask_quantity=0,
                    last_update=datetime.now()
                )
            level = levels_dict[price]
            level.bid_quantity += quantity
            level.total_quantity += quantity
            level.order_count += 1
            level.level_type = 'bid' if level.ask_quantity == 0 else 'both'
        
        # Process asks
        for price, quantity in asks_sorted:
            if price not in levels_dict:
                levels_dict[price] = DepthLevel(
                    price=price,
                    total_quantity=0,
                    bid_quantity=0,
                    ask_quantity=0,
                    last_update=datetime.now()
                )
            level = levels_dict[price]
            level.ask_quantity += quantity
            level.total_quantity += quantity
            level.order_count += 1
            level.level_type = 'ask' if level.bid_quantity == 0 else 'both'
        
        # Convert to sorted list
        levels = sorted(levels_dict.values(), key=lambda x: x.price)
        
        # Calculate metrics
        best_bid = max([l.price for l in levels if l.bid_quantity > 0], default=0)
        best_ask = min([l.price for l in levels if l.ask_quantity > 0], default=0)
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0
        
        total_bid_volume = sum(l.bid_quantity for l in levels)
        total_ask_volume = sum(l.ask_quantity for l in levels)
        
        # Create snapshot
        snapshot = MarketDepthSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            levels=levels,
            spread=spread,
            best_bid=best_bid,
            best_ask=best_ask,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume
        )
        
        # Update history
        self.depth_history.append(snapshot)
        
        # Update statistics
        self._update_statistics(snapshot)
        
        # Analyze for patterns
        self._analyze_depth_patterns(snapshot)
        
        return snapshot
    
    def analyze_depth(self, snapshot: MarketDepthSnapshot) -> Dict[str, Any]:
        """Comprehensive market depth analysis"""
        
        analysis = {
            'depth_profile': self._analyze_depth_profile(snapshot),
            'imbalance_analysis': self._analyze_depth_imbalance(snapshot),
            'absorption_signals': self._detect_absorption(snapshot),
            'support_resistance': self._identify_depth_support_resistance(snapshot),
            'liquidity_pools': self._identify_liquidity_pools(snapshot),
            'market_conditions': self._assess_market_conditions(snapshot)
        }
        
        return analysis
    
    def _analyze_depth_profile(self, snapshot: MarketDepthSnapshot) -> Dict[str, Any]:
        """Analyze depth profile characteristics"""
        if not snapshot.levels:
            return {}
        
        # Calculate depth metrics at different distances from best bid/ask
        metrics = {
            'spread_bps': (snapshot.spread / snapshot.best_bid * 10000) if snapshot.best_bid > 0 else 0,
            'bid_ask_ratio': snapshot.total_bid_volume / snapshot.total_ask_volume if snapshot.total_ask_volume > 0 else 0,
            'depth_steepness': self._calculate_depth_steepness(snapshot),
            'depth_concentration': self._calculate_depth_concentration(snapshot),
            'market_impact': self._estimate_market_impact_from_depth(snapshot)
        }
        
        # Depth at various percentages from mid price
        mid_price = (snapshot.best_bid + snapshot.best_ask) / 2 if snapshot.best_bid > 0 and snapshot.best_ask > 0 else 0
        
        for percentage in [0.1, 0.25, 0.5, 1.0]:
            price_range = mid_price * percentage / 100
            
            # Calculate depth within range
            bid_depth = sum(
                l.bid_quantity for l in snapshot.levels 
                if snapshot.best_bid - l.price <= price_range and l.bid_quantity > 0
            )
            
            ask_depth = sum(
                l.ask_quantity for l in snapshot.levels 
                if l.price - snapshot.best_ask <= price_range and l.ask_quantity > 0
            )
            
            metrics[f'depth_{percentage}pct_bid'] = bid_depth
            metrics[f'depth_{percentage}pct_ask'] = ask_depth
            metrics[f'depth_{percentage}pct_net'] = bid_depth - ask_depth
        
        return metrics
    
    def _calculate_depth_steepness(self, snapshot: MarketDepthSnapshot) -> Dict[str, float]:
        """Calculate how steep the order book is on each side"""
        if len(snapshot.levels) < 5:
            return {'bid_steepness': 0.0, 'ask_steepness': 0.0}
        
        # Get top 5 bid levels
        bid_levels = [(l.price, l.bid_quantity) for l in snapshot.levels if l.bid_quantity > 0]
        bid_levels.sort(key=lambda x: x[0], reverse=True)
        bid_levels = bid_levels[:5]
        
        # Get top 5 ask levels
        ask_levels = [(l.price, l.ask_quantity) for l in snapshot.levels if l.ask_quantity > 0]
        ask_levels.sort(key=lambda x: x[0])
        ask_levels = ask_levels[:5]
        
        # Calculate steepness (price change per unit of volume)
        bid_steepness = 0.0
        if len(bid_levels) >= 2:
            price_changes = []
            for i in range(1, len(bid_levels)):
                price_diff = bid_levels[i-1][0] - bid_levels[i][0]
                avg_volume = (bid_levels[i-1][1] + bid_levels[i][1]) / 2
                if avg_volume > 0:
                    price_changes.append(price_diff / avg_volume)
            if price_changes:
                bid_steepness = statistics.mean(price_changes)
        
        ask_steepness = 0.0
        if len(ask_levels) >= 2:
            price_changes = []
            for i in range(1, len(ask_levels)):
                price_diff = ask_levels[i][0] - ask_levels[i-1][0]
                avg_volume = (ask_levels[i-1][1] + ask_levels[i][1]) / 2
                if avg_volume > 0:
                    price_changes.append(price_diff / avg_volume)
            if price_changes:
                ask_steepness = statistics.mean(price_changes)
        
        return {
            'bid_steepness': bid_steepness,
            'ask_steepness': ask_steepness,
            'steepness_ratio': bid_steepness / ask_steepness if ask_steepness != 0 else 0
        }
    
    def _calculate_depth_concentration(self, snapshot: MarketDepthSnapshot) -> Dict[str, float]:
        """Calculate how concentrated liquidity is"""
        if not snapshot.levels:
            return {}
        
        # Get all quantities
        bid_quantities = [l.bid_quantity for l in snapshot.levels if l.bid_quantity > 0]
        ask_quantities = [l.ask_quantity for l in snapshot.levels if l.ask_quantity > 0]
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        def calculate_hhi(quantities):
            if not quantities:
                return 0.0
            total = sum(quantities)
            if total == 0:
                return 0.0
            return sum((q / total) ** 2 for q in quantities) * 10000
        
        bid_hhi = calculate_hhi(bid_quantities)
        ask_hhi = calculate_hhi(ask_quantities)
        
        # Calculate top 3 concentration
        def top_n_concentration(quantities, n=3):
            if not quantities:
                return 0.0
            total = sum(quantities)
            if total == 0:
                return 0.0
            sorted_q = sorted(quantities, reverse=True)
            top_sum = sum(sorted_q[:min(n, len(sorted_q))])
            return top_sum / total
        
        bid_top3 = top_n_concentration(bid_quantities, 3)
        ask_top3 = top_n_concentration(ask_quantities, 3)
        
        return {
            'bid_concentration_hhi': bid_hhi,
            'ask_concentration_hhi': ask_hhi,
            'bid_top3_concentration': bid_top3,
            'ask_top3_concentration': ask_top3,
            'avg_concentration': (bid_hhi + ask_hhi) / 2
        }
    
    def _estimate_market_impact_from_depth(self, snapshot: MarketDepthSnapshot) -> Dict[str, float]:
        """Estimate market impact for different order sizes from depth"""
        if not snapshot.levels:
            return {}
        
        impact_estimates = {}
        order_sizes = [1000, 5000, 10000, 50000, 100000]
        
        for size in order_sizes:
            # Estimate buying impact (walking the ask side)
            remaining = size
            total_cost = 0
            ask_levels = [(l.price, l.ask_quantity) for l in snapshot.levels if l.ask_quantity > 0]
            ask_levels.sort(key=lambda x: x[0])  # Sort by price ascending
            
            for price, quantity in ask_levels:
                if remaining <= 0:
                    break
                executed = min(remaining, quantity)
                total_cost += executed * price
                remaining -= executed
            
            avg_buy_price = total_cost / (size - remaining) if (size - remaining) > 0 else 0
            
            # Estimate selling impact (walking the bid side)
            remaining = size
            total_proceeds = 0
            bid_levels = [(l.price, l.bid_quantity) for l in snapshot.levels if l.bid_quantity > 0]
            bid_levels.sort(key=lambda x: x[0], reverse=True)  # Sort by price descending
            
            for price, quantity in bid_levels:
                if remaining <= 0:
                    break
                executed = min(remaining, quantity)
                total_proceeds += executed * price
                remaining -= executed
            
            avg_sell_price = total_proceeds / (size - remaining) if (size - remaining) > 0 else 0
            
            # Calculate impact in basis points
            mid_price = (snapshot.best_bid + snapshot.best_ask) / 2
            if mid_price > 0:
                buy_impact = (avg_buy_price - mid_price) / mid_price * 10000
                sell_impact = (mid_price - avg_sell_price) / mid_price * 10000
            else:
                buy_impact = sell_impact = 0
            
            impact_estimates[f'impact_{size}'] = {
                'buy_impact_bps': buy_impact,
                'sell_impact_bps': sell_impact,
                'avg_buy_price': avg_buy_price,
                'avg_sell_price': avg_sell_price,
                'total_impact': (buy_impact + abs(sell_impact)) / 2
            }
        
        return impact_estimates
    
    def _analyze_depth_imbalance(self, snapshot: MarketDepthSnapshot) -> Dict[str, Any]:
        """Analyze depth imbalance at different levels"""
        if not snapshot.levels:
            return {}
        
        imbalances = {}
        
        # Calculate imbalance at different depth levels
        for depth in [1, 3, 5, 10]:
            # Get top N levels on each side
            bid_levels = [(l.price, l.bid_quantity) for l in snapshot.levels if l.bid_quantity > 0]
            bid_levels.sort(key=lambda x: x[0], reverse=True)
            bid_levels = bid_levels[:depth]
            
            ask_levels = [(l.price, l.ask_quantity) for l in snapshot.levels if l.ask_quantity > 0]
            ask_levels.sort(key=lambda x: x[0])
            ask_levels = ask_levels[:depth]
            
            bid_volume = sum(q for _, q in bid_levels)
            ask_volume = sum(q for _, q in ask_levels)
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                imbalance = (bid_volume - ask_volume) / total_volume
                imbalances[f'depth_{depth}_imbalance'] = imbalance
        
        # Calculate weighted imbalance (closer levels weighted higher)
        weighted_imbalance = 0.0
        total_weight = 0.0
        
        for i, level in enumerate(snapshot.levels):
            if level.bid_quantity > 0 or level.ask_quantity > 0:
                # Weight decreases with distance from best bid/ask
                if level.bid_quantity > 0:
                    distance = abs(snapshot.best_bid - level.price) if snapshot.best_bid > 0 else 1
                    weight = 1.0 / (1.0 + distance / snapshot.spread) if snapshot.spread > 0 else 1.0
                    weighted_imbalance += weight * level.imbalance
                    total_weight += weight
                
                if level.ask_quantity > 0:
                    distance = abs(level.price - snapshot.best_ask) if snapshot.best_ask > 0 else 1
                    weight = 1.0 / (1.0 + distance / snapshot.spread) if snapshot.spread > 0 else 1.0
                    weighted_imbalance += weight * level.imbalance
                    total_weight += weight
        
        if total_weight > 0:
            weighted_imbalance /= total_weight
        
        # Determine imbalance signal
        signal = 'neutral'
        if weighted_imbalance > self.absorption_threshold * 0.7:
            signal = 'strong_buying_pressure'
        elif weighted_imbalance < -self.absorption_threshold * 0.7:
            signal = 'strong_selling_pressure'
        elif weighted_imbalance > self.absorption_threshold * 0.3:
            signal = 'moderate_buying_pressure'
        elif weighted_imbalance < -self.absorption_threshold * 0.3:
            signal = 'moderate_selling_pressure'
        
        # Update history
        self.depth_imbalance_history.append(weighted_imbalance)
        if len(self.depth_imbalance_history) > 100:
            self.depth_imbalance_history.pop(0)
        
        # Check trend
        trend = 'stable'
        if len(self.depth_imbalance_history) >= 5:
            recent = self.depth_imbalance_history[-5:]
            if all(imb > 0 for imb in recent) and recent[-1] > recent[0]:
                trend = 'increasing_buying'
            elif all(imb < 0 for imb in recent) and recent[-1] < recent[0]:
                trend = 'increasing_selling'
            elif recent[-1] > recent[0]:
                trend = 'improving_buying'
            elif recent[-1] < recent[0]:
                trend = 'worsening_selling'
        
        return {
            'weighted_imbalance': weighted_imbalance,
            'signal': signal,
            'trend': trend,
            'depth_imbalances': imbalances,
            'history_mean': statistics.mean(self.depth_imbalance_history) if self.depth_imbalance_history else 0
        }
    
    def _detect_absorption(self, snapshot: MarketDepthSnapshot) -> List[AbsorptionSignal]:
        """Detect absorption patterns in market depth"""
        signals = []
        
        if len(self.depth_history) < 2:
            return signals
        
        current_snapshot = snapshot
        previous_snapshot = self.depth_history[-2] if len(self.depth_history) >= 2 else None
        
        if not previous_snapshot:
            return signals
        
        # Look for absorption at key levels
        for level in current_snapshot.levels:
            # Find corresponding level in previous snapshot
            prev_level = previous_snapshot.get_level_at_price(level.price)
            
            if prev_level:
                # Calculate changes
                bid_change = level.bid_quantity - prev_level.bid_quantity
                ask_change = level.ask_quantity - prev_level.ask_quantity
                
                # Check for absorption (large quantity added that wasn't immediately executed)
                if abs(bid_change) > self.absorption_threshold * 1000:  # Threshold in units
                    if bid_change > 0:
                        signal = AbsorptionSignal(
                            signal_type='absorption',
                            price_level=level.price,
                            strength=min(bid_change / 10000, 1.0),
                            volume_absorbed=bid_change,
                            timestamp=datetime.now(),
                            context='bid'
                        )
                        signals.append(signal)
                
                if abs(ask_change) > self.absorption_threshold * 1000:
                    if ask_change > 0:
                        signal = AbsorptionSignal(
                            signal_type='absorption',
                            price_level=level.price,
                            strength=min(ask_change / 10000, 1.0),
                            volume_absorbed=ask_change,
                            timestamp=datetime.now(),
                            context='ask'
                        )
                        signals.append(signal)
                
                # Check for exhaustion (large quantity removed without price movement)
                if (prev_level.bid_quantity > level.bid_quantity * 2 and 
                    bid_change < 0 and abs(bid_change) > 1000):
                    signal = AbsorptionSignal(
                        signal_type='exhaustion',
                        price_level=level.price,
                        strength=min(abs(bid_change) / 10000, 1.0),
                        volume_absorbed=abs(bid_change),
                        timestamp=datetime.now(),
                        context='bid'
                    )
                    signals.append(signal)
                
                if (prev_level.ask_quantity > level.ask_quantity * 2 and 
                    ask_change < 0 and abs(ask_change) > 1000):
                    signal = AbsorptionSignal(
                        signal_type='exhaustion',
                        price_level=level.price,
                        strength=min(abs(ask_change) / 10000, 1.0),
                        volume_absorbed=abs(ask_change),
                        timestamp=datetime.now(),
                        context='ask'
                    )
                    signals.append(signal)
        
        # Update signals history
        self.absorption_signals.extend(signals)
        self.stats['absorption_count'] += len(signals)
        
        # Keep recent signals only
        if len(self.absorption_signals) > 100:
            self.absorption_signals = self.absorption_signals[-100:]
        
        return signals
    
    def _identify_depth_support_resistance(self, snapshot: MarketDepthSnapshot) -> Dict[str, Any]:
        """Identify support and resistance levels from depth"""
        support_levels = []
        resistance_levels = []
        
        # Large bid clusters = support
        for level in snapshot.levels:
            if level.bid_quantity > self.absorption_threshold * 2000:
                support_levels.append({
                    'price': level.price,
                    'strength': min(level.bid_quantity / 10000, 5.0),
                    'quantity': level.bid_quantity,
                    'distance_to_best': (snapshot.best_bid - level.price) / snapshot.best_bid * 100 if snapshot.best_bid > 0 else 0
                })
        
        # Large ask clusters = resistance
        for level in snapshot.levels:
            if level.ask_quantity > self.absorption_threshold * 2000:
                resistance_levels.append({
                    'price': level.price,
                    'strength': min(level.ask_quantity / 10000, 5.0),
                    'quantity': level.ask_quantity,
                    'distance_to_best': (level.price - snapshot.best_ask) / snapshot.best_ask * 100 if snapshot.best_ask > 0 else 0
                })
        
        # Sort by strength
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'support_levels': support_levels[:5],
            'resistance_levels': resistance_levels[:5],
            'closest_support': support_levels[0] if support_levels else None,
            'closest_resistance': resistance_levels[0] if resistance_levels else None
        }
    
    def _identify_liquidity_pools(self, snapshot: MarketDepthSnapshot) -> List[Dict[str, Any]]:
        """Identify liquidity pools in market depth"""
        pools = []
        
        # Look for large clusters of liquidity
        for i, level in enumerate(snapshot.levels):
            # Check if this level has significantly more liquidity than neighbors
            if i > 0 and i < len(snapshot.levels) - 1:
                prev = snapshot.levels[i-1]
                next_lvl = snapshot.levels[i+1]
                
                avg_neighbor_quantity = (prev.total_quantity + next_lvl.total_quantity) / 2
                
                if level.total_quantity > avg_neighbor_quantity * 2:
                    pool_type = 'mixed'
                    if level.bid_quantity > level.ask_quantity * 2:
                        pool_type = 'bid'
                    elif level.ask_quantity > level.bid_quantity * 2:
                        pool_type = 'ask'
                    
                    pools.append({
                        'price': level.price,
                        'type': pool_type,
                        'total_quantity': level.total_quantity,
                        'bid_quantity': level.bid_quantity,
                        'ask_quantity': level.ask_quantity,
                        'imbalance': level.imbalance,
                        'is_liquidity_pool': True
                    })
        
        return pools
    
    def _assess_market_conditions(self, snapshot: MarketDepthSnapshot) -> Dict[str, str]:
        """Assess overall market conditions from depth"""
        conditions = {
            'liquidity': 'unknown',
            'volatility': 'unknown',
            'market_state': 'unknown',
            'depth_quality': 'unknown'
        }
        
        if not snapshot.levels:
            return conditions
        
        # Assess liquidity
        total_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
        if total_volume > 1000000:
            conditions['liquidity'] = 'high'
        elif total_volume > 100000:
            conditions['liquidity'] = 'medium'
        else:
            conditions['liquidity'] = 'low'
        
        # Assess volatility from spread
        spread_bps = (snapshot.spread / snapshot.best_bid * 10000) if snapshot.best_bid > 0 else 0
        if spread_bps > 50:
            conditions['volatility'] = 'high'
        elif spread_bps > 10:
            conditions['volatility'] = 'medium'
        else:
            conditions['volatility'] = 'low'
        
        # Assess market state from imbalance
        weighted_imbalance = self._analyze_depth_imbalance(snapshot).get('weighted_imbalance', 0)
        if abs(weighted_imbalance) > 0.5:
            conditions['market_state'] = 'one_sided'
        elif abs(weighted_imbalance) > 0.2:
            conditions['market_state'] = 'biased'
        else:
            conditions['market_state'] = 'balanced'
        
        # Assess depth quality
        concentration = self._calculate_depth_concentration(snapshot)
        avg_hhi = concentration.get('avg_concentration', 0)
        if avg_hhi > 2000:
            conditions['depth_quality'] = 'concentrated'
        elif avg_hhi > 1000:
            conditions['depth_quality'] = 'moderate'
        else:
            conditions['depth_quality'] = 'dispersed'
        
        return conditions
    
    def _update_statistics(self, snapshot: MarketDepthSnapshot):
        """Update running statistics"""
        # Update average spread
        if snapshot.spread > 0:
            # Keep running average
            if self.stats['avg_spread'] == 0:
                self.stats['avg_spread'] = snapshot.spread
            else:
                self.stats['avg_spread'] = (self.stats['avg_spread'] * 0.9 + snapshot.spread * 0.1)
        
        # Update max imbalance
        imbalance = self._analyze_depth_imbalance(snapshot).get('weighted_imbalance', 0)
        self.stats['max_depth_imbalance'] = max(self.stats['max_depth_imbalance'], abs(imbalance))
        
        # Update count
        self.stats['total_updates'] += 1
    
    def get_recent_signals(self, limit: int = 10) -> List[AbsorptionSignal]:
        """Get recent absorption signals"""
        return self.absorption_signals[-limit:] if self.absorption_signals else []
    
    def clear_history(self):
        """Clear depth history"""
        self.depth_history.clear()
        self.absorption_signals.clear()
        self.depth_imbalance_history.clear()
        logger.info("Market depth history cleared")