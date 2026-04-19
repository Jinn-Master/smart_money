"""
Order Book Analysis
Implements: Order book imbalance, depth analysis, market depth calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in order book"""
    price: float
    quantity: float
    order_count: int = 1
    timestamp: datetime = None
    side: str = ''  # 'bid' or 'ask'
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    symbol: str
    spread: float = 0.0
    mid_price: float = 0.0
    imbalance: float = 0.0
    weighted_mid: float = 0.0
    
    def __post_init__(self):
        # Calculate metrics
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            self.spread = best_ask - best_bid
            self.mid_price = (best_bid + best_ask) / 2
            
            # Calculate weighted mid price
            bid_volume = sum(b.quantity for b in self.bids[:5])
            ask_volume = sum(a.quantity for a in self.asks[:5])
            total_volume = bid_volume + ask_volume
            
            if total_volume > 0:
                self.weighted_mid = (
                    best_bid * (ask_volume / total_volume) +
                    best_ask * (bid_volume / total_volume)
                )
            
            # Calculate imbalance
            self.imbalance = self._calculate_imbalance()
    
    def _calculate_imbalance(self) -> float:
        """Calculate order book imbalance"""
        if not self.bids or not self.asks:
            return 0.0
        
        # Look at top 5 levels on each side
        top_bid_volume = sum(b.quantity for b in self.bids[:5])
        top_ask_volume = sum(a.quantity for a in self.asks[:5])
        total_volume = top_bid_volume + top_ask_volume
        
        if total_volume == 0:
            return 0.0
        
        return (top_bid_volume - top_ask_volume) / total_volume
    
    def get_depth_at_price(self, price: float) -> Tuple[float, float]:
        """Get total quantity at a specific price level"""
        bid_quantity = 0
        ask_quantity = 0
        
        for bid in self.bids:
            if abs(bid.price - price) / price < 0.001:  # Within 0.1%
                bid_quantity += bid.quantity
        
        for ask in self.asks:
            if abs(ask.price - price) / price < 0.001:
                ask_quantity += ask.quantity
        
        return bid_quantity, ask_quantity


class OrderBookAnalyzer:
    """Analyzes order book dynamics and microstructure"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        self.depth_levels = self.config.get('depth_levels', 10)
        self.imbalance_threshold = self.config.get('imbalance_threshold', 0.3)
        self.liquidity_threshold = self.config.get('liquidity_threshold', 1000)
        self.update_frequency = self.config.get('update_frequency', 1000)  # ms
        
        # State
        self.order_book_history: deque[OrderBookSnapshot] = deque(maxlen=1000)
        self.imbalance_history: List[float] = []
        self.spread_history: List[float] = []
        
        # Statistics
        self.stats = {
            'avg_spread': 0.0,
            'max_imbalance': 0.0,
            'liquidity_walls': [],
            'update_count': 0
        }
        
        logger.info("OrderBookAnalyzer initialized")
    
    def update_order_book(self, 
                         bids: List[Tuple[float, float]], 
                         asks: List[Tuple[float, float]],
                         symbol: str) -> OrderBookSnapshot:
        """
        Update order book with new data
        
        Args:
            bids: List of (price, quantity) for bids
            asks: List of (price, quantity) for asks
            symbol: Trading symbol
        
        Returns:
            OrderBookSnapshot
        """
        # Sort bids descending, asks ascending
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)[:self.depth_levels]
        asks_sorted = sorted(asks, key=lambda x: x[0])[:self.depth_levels]
        
        # Create OrderBookLevel objects
        bid_levels = [
            OrderBookLevel(price=price, quantity=quantity, side='bid')
            for price, quantity in bids_sorted
        ]
        
        ask_levels = [
            OrderBookLevel(price=price, quantity=quantity, side='ask')
            for price, quantity in asks_sorted
        ]
        
        # Create snapshot
        snapshot = OrderBookSnapshot(
            timestamp=datetime.now(),
            bids=bid_levels,
            asks=ask_levels,
            symbol=symbol
        )
        
        # Update history
        self.order_book_history.append(snapshot)
        self.imbalance_history.append(snapshot.imbalance)
        self.spread_history.append(snapshot.spread)
        
        # Update statistics
        self._update_statistics(snapshot)
        self.stats['update_count'] += 1
        
        return snapshot
    
    def analyze_order_book(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Comprehensive order book analysis"""
        
        analysis = {
            'basic_metrics': self._get_basic_metrics(snapshot),
            'imbalance_analysis': self._analyze_imbalance(snapshot),
            'liquidity_analysis': self._analyze_liquidity(snapshot),
            'depth_analysis': self._analyze_depth(snapshot),
            'market_impact': self._estimate_market_impact(snapshot),
            'support_resistance': self._identify_support_resistance(snapshot)
        }
        
        return analysis
    
    def _get_basic_metrics(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Get basic order book metrics"""
        if not snapshot.bids or not snapshot.asks:
            return {}
        
        best_bid = snapshot.bids[0].price
        best_ask = snapshot.asks[0].price
        
        # Calculate volume metrics
        bid_volume = sum(level.quantity for level in snapshot.bids)
        ask_volume = sum(level.quantity for level in snapshot.asks)
        total_volume = bid_volume + ask_volume
        
        # Volume-weighted average prices
        vwap_bid = sum(level.price * level.quantity for level in snapshot.bids) / bid_volume if bid_volume > 0 else 0
        vwap_ask = sum(level.price * level.quantity for level in snapshot.asks) / ask_volume if ask_volume > 0 else 0
        
        # Micro-price (more accurate mid)
        micro_price = (best_bid * ask_volume + best_ask * bid_volume) / total_volume if total_volume > 0 else snapshot.mid_price
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': snapshot.spread,
            'spread_bps': (snapshot.spread / best_bid * 10000) if best_bid > 0 else 0,
            'mid_price': snapshot.mid_price,
            'weighted_mid': snapshot.weighted_mid,
            'micro_price': micro_price,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume,
            'vwap_bid': vwap_bid,
            'vwap_ask': vwap_ask,
            'volume_ratio': bid_volume / ask_volume if ask_volume > 0 else 0
        }
    
    def _analyze_imbalance(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Analyze order book imbalance"""
        if not snapshot.bids or not snapshot.asks:
            return {}
        
        imbalance = snapshot.imbalance
        
        # Calculate imbalance at different depths
        imbalances = {}
        for depth in [1, 3, 5, 10]:
            if len(snapshot.bids) >= depth and len(snapshot.asks) >= depth:
                bid_volume = sum(b.quantity for b in snapshot.bids[:depth])
                ask_volume = sum(a.quantity for a in snapshot.asks[:depth])
                total = bid_volume + ask_volume
                
                if total > 0:
                    imbalances[f'imbalance_{depth}'] = (bid_volume - ask_volume) / total
        
        # Determine imbalance signal
        signal = 'neutral'
        if imbalance > self.imbalance_threshold:
            signal = 'strong_buying'
        elif imbalance < -self.imbalance_threshold:
            signal = 'strong_selling'
        elif imbalance > self.imbalance_threshold * 0.5:
            signal = 'moderate_buying'
        elif imbalance < -self.imbalance_threshold * 0.5:
            signal = 'moderate_selling'
        
        # Check for recent trend
        trend = 'stable'
        if len(self.imbalance_history) >= 5:
            recent_imbalances = self.imbalance_history[-5:]
            if all(i > 0 for i in recent_imbalances) and recent_imbalances[-1] > recent_imbalances[0]:
                trend = 'increasing_buying'
            elif all(i < 0 for i in recent_imbalances) and recent_imbalances[-1] < recent_imbalances[0]:
                trend = 'increasing_selling'
        
        return {
            'current_imbalance': imbalance,
            'signal': signal,
            'trend': trend,
            'depth_imbalances': imbalances,
            'threshold': self.imbalance_threshold
        }
    
    def _analyze_liquidity(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Analyze liquidity in order book"""
        liquidity_walls = {
            'bid_walls': [],
            'ask_walls': []
        }
        
        # Find large bid walls (support)
        for level in snapshot.bids:
            if level.quantity >= self.liquidity_threshold:
                liquidity_walls['bid_walls'].append({
                    'price': level.price,
                    'quantity': level.quantity,
                    'distance_bps': abs(level.price - snapshot.mid_price) / snapshot.mid_price * 10000
                })
        
        # Find large ask walls (resistance)
        for level in snapshot.asks:
            if level.quantity >= self.liquidity_threshold:
                liquidity_walls['ask_walls'].append({
                    'price': level.price,
                    'quantity': level.quantity,
                    'distance_bps': abs(level.price - snapshot.mid_price) / snapshot.mid_price * 10000
                })
        
        # Calculate liquidity metrics
        total_liquidity = sum(level.quantity for level in snapshot.bids + snapshot.asks)
        avg_bid_size = np.mean([b.quantity for b in snapshot.bids]) if snapshot.bids else 0
        avg_ask_size = np.mean([a.quantity for a in snapshot.asks]) if snapshot.asks else 0
        
        return {
            'liquidity_walls': liquidity_walls,
            'total_liquidity': total_liquidity,
            'avg_bid_size': avg_bid_size,
            'avg_ask_size': avg_ask_size,
            'bid_ask_ratio': avg_bid_size / avg_ask_size if avg_ask_size > 0 else 0,
            'liquidity_concentration': self._calculate_liquidity_concentration(snapshot)
        }
    
    def _calculate_liquidity_concentration(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate how concentrated liquidity is"""
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        
        # Calculate Gini coefficient for liquidity distribution
        all_quantities = [level.quantity for level in snapshot.bids + snapshot.asks]
        if len(all_quantities) == 0:
            return 0.0
        
        # Sort quantities
        sorted_quantities = np.sort(all_quantities)
        n = len(sorted_quantities)
        index = np.arange(1, n + 1)
        
        # Gini coefficient
        gini = (np.sum((2 * index - n - 1) * sorted_quantities)) / (n * np.sum(sorted_quantities))
        
        return gini
    
    def _analyze_depth(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Analyze order book depth profile"""
        if not snapshot.bids or not snapshot.asks:
            return {}
        
        # Calculate cumulative depth
        bid_prices = [b.price for b in snapshot.bids]
        bid_quantities = [b.quantity for b in snapshot.bids]
        ask_prices = [a.price for a in snapshot.asks]
        ask_quantities = [a.quantity for a in snapshot.asks]
        
        # Cumulative volumes
        cumulative_bid = np.cumsum(bid_quantities[::-1])[::-1]  # From inside out
        cumulative_ask = np.cumsum(ask_quantities)
        
        # Depth at percentages from mid price
        mid_price = snapshot.mid_price
        depth_levels = {}
        
        for percentage in [0.1, 0.25, 0.5, 1.0]:  # 0.1%, 0.25%, 0.5%, 1%
            price_range = mid_price * percentage / 100
            
            # Find quantity within price range on each side
            bid_quantity_in_range = sum(
                q for p, q in zip(bid_prices, bid_quantities) 
                if mid_price - p <= price_range
            )
            
            ask_quantity_in_range = sum(
                q for p, q in zip(ask_prices, ask_quantities) 
                if p - mid_price <= price_range
            )
            
            depth_levels[f'depth_{percentage}pct'] = {
                'bid_quantity': bid_quantity_in_range,
                'ask_quantity': ask_quantity_in_range,
                'net_quantity': bid_quantity_in_range - ask_quantity_in_range
            }
        
        # Calculate order book slope (steepness)
        bid_slope = self._calculate_orderbook_slope(bid_prices, cumulative_bid)
        ask_slope = self._calculate_orderbook_slope(ask_prices, cumulative_ask)
        
        return {
            'depth_profile': depth_levels,
            'bid_slope': bid_slope,
            'ask_slope': ask_slope,
            'slope_ratio': bid_slope / ask_slope if ask_slope != 0 else 0,
            'cumulative_bid': cumulative_bid.tolist(),
            'cumulative_ask': cumulative_ask.tolist()
        }
    
    def _calculate_orderbook_slope(self, prices: List[float], quantities: List[float]) -> float:
        """Calculate slope of order book (price vs cumulative quantity)"""
        if len(prices) < 2:
            return 0.0
        
        try:
            # Linear regression of price vs log(quantity)
            x = np.log(np.array(quantities) + 1)  # Add 1 to avoid log(0)
            y = np.array(prices)
            
            # Fit line
            slope, intercept = np.polyfit(x, y, 1)
            return slope
        except:
            return 0.0
    
    def _estimate_market_impact(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Estimate market impact for different order sizes"""
        if not snapshot.bids or not snapshot.asks:
            return {}
        
        impact_estimates = {}
        
        # Test order sizes (as multiples of average trade size)
        order_sizes = [1000, 5000, 10000, 50000, 100000]
        
        for size in order_sizes:
            # Estimate buying impact (walking the ask side)
            remaining = size
            total_cost = 0
            last_price = snapshot.asks[0].price
            
            for level in snapshot.asks:
                if remaining <= 0:
                    break
                
                available = level.quantity
                executed = min(remaining, available)
                total_cost += executed * level.price
                remaining -= executed
                last_price = level.price
            
            avg_buy_price = total_cost / (size - remaining) if (size - remaining) > 0 else 0
            buy_impact = (avg_buy_price - snapshot.mid_price) / snapshot.mid_price * 10000 if snapshot.mid_price > 0 else 0
            
            # Estimate selling impact (walking the bid side)
            remaining = size
            total_proceeds = 0
            
            for level in snapshot.bids:
                if remaining <= 0:
                    break
                
                available = level.quantity
                executed = min(remaining, available)
                total_proceeds += executed * level.price
                remaining -= executed
            
            avg_sell_price = total_proceeds / (size - remaining) if (size - remaining) > 0 else 0
            sell_impact = (snapshot.mid_price - avg_sell_price) / snapshot.mid_price * 10000 if snapshot.mid_price > 0 else 0
            
            impact_estimates[f'impact_{size}'] = {
                'buy_impact_bps': buy_impact,
                'sell_impact_bps': sell_impact,
                'avg_buy_price': avg_buy_price,
                'avg_sell_price': avg_sell_price,
                'execution_shortfall': (buy_impact + sell_impact) / 2  # Average impact
            }
        
        return impact_estimates
    
    def _identify_support_resistance(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Identify support and resistance levels from order book"""
        support_levels = []
        resistance_levels = []
        
        # Large bid walls = support
        for level in snapshot.bids:
            if level.quantity >= self.liquidity_threshold:
                support_levels.append({
                    'price': level.price,
                    'strength': min(level.quantity / self.liquidity_threshold, 5.0),  # 1-5 scale
                    'quantity': level.quantity,
                    'distance_to_mid': (snapshot.mid_price - level.price) / snapshot.mid_price * 100
                })
        
        # Large ask walls = resistance
        for level in snapshot.asks:
            if level.quantity >= self.liquidity_threshold:
                resistance_levels.append({
                    'price': level.price,
                    'strength': min(level.quantity / self.liquidity_threshold, 5.0),
                    'quantity': level.quantity,
                    'distance_to_mid': (level.price - snapshot.mid_price) / snapshot.mid_price * 100
                })
        
        # Sort by strength
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'support_levels': support_levels[:5],  # Top 5
            'resistance_levels': resistance_levels[:5],
            'closest_support': support_levels[0] if support_levels else None,
            'closest_resistance': resistance_levels[0] if resistance_levels else None
        }
    
    def _update_statistics(self, snapshot: OrderBookSnapshot):
        """Update running statistics"""
        # Update average spread
        if self.spread_history:
            self.stats['avg_spread'] = np.mean(self.spread_history[-100:])  # Last 100
        
        # Update max imbalance
        if self.imbalance_history:
            current_max = max(abs(i) for i in self.imbalance_history[-100:])
            self.stats['max_imbalance'] = max(self.stats['max_imbalance'], current_max)
        
        # Update liquidity walls
        liquidity_walls = self._analyze_liquidity(snapshot)['liquidity_walls']
        if liquidity_walls['bid_walls'] or liquidity_walls['ask_walls']:
            self.stats['liquidity_walls'] = liquidity_walls
    
    def detect_liquidity_pools(self, snapshot: OrderBookSnapshot) -> List[Tuple[float, float]]:
        """Detect liquidity pools (large orders) in order book"""
        pools = []
        
        # Check bid side
        for level in snapshot.bids:
            if level.quantity >= self.liquidity_threshold * 2:  # Extra large
                pools.append((level.price, level.quantity))
        
        # Check ask side
        for level in snapshot.asks:
            if level.quantity >= self.liquidity_threshold * 2:
                pools.append((level.price, level.quantity))
        
        return pools
    
    def calculate_vpoc(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate Volume Point of Control from order book"""
        if not snapshot.bids or not snapshot.asks:
            return {}
        
        # Aggregate all liquidity
        all_levels = snapshot.bids + snapshot.asks
        
        # Group by price buckets
        price_buckets = defaultdict(float)
        bucket_size = snapshot.spread * 0.1  # 10% of spread
        
        for level in all_levels:
            bucket = round(level.price / bucket_size) * bucket_size
            price_buckets[bucket] += level.quantity
        
        # Find VPOC (highest volume bucket)
        if price_buckets:
            vpoc_price = max(price_buckets, key=price_buckets.get)
            vpoc_volume = price_buckets[vpoc_price]
            
            return {
                'vpoc': vpoc_price,
                'vpoc_volume': vpoc_volume,
                'price_buckets': dict(price_buckets)
            }
        
        return {}