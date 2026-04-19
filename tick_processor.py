"""
Tick Data Processing and Aggregation
High-frequency data handling for order flow analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class Tick:
    """Individual tick data"""
    timestamp: datetime
    price: float
    volume: float
    side: Optional[str] = None  # 'buy' or 'sell'
    bid: Optional[float] = None
    ask: Optional[float] = None

@dataclass
class TickBar:
    """Aggregated tick bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    tick_count: int
    vwap: float

class TickProcessor:
    """Process and aggregate tick data"""
    
    def __init__(self, max_ticks: int = 100000):
        self.max_ticks = max_ticks
        self.ticks = deque(maxlen=max_ticks)
        self.tick_bars = {}
        
    def add_tick(self, tick: Tick):
        """Add new tick to processor"""
        self.ticks.append(tick)
    
    def create_tick_bars(self, bar_size: int = 1000) -> List[TickBar]:
        """Create bars from ticks based on tick count"""
        
        if len(self.ticks) < bar_size:
            return []
        
        bars = []
        ticks_list = list(self.ticks)
        
        for i in range(0, len(ticks_list), bar_size):
            chunk = ticks_list[i:i + bar_size]
            if len(chunk) < bar_size:
                break
            
            bar = self._aggregate_ticks_to_bar(chunk)
            bars.append(bar)
        
        return bars
    
    def create_volume_bars(self, target_volume: float) -> List[TickBar]:
        """Create bars based on volume threshold"""
        
        if not self.ticks:
            return []
        
        bars = []
        current_bar_ticks = []
        current_volume = 0
        
        for tick in self.ticks:
            current_bar_ticks.append(tick)
            current_volume += tick.volume
            
            if current_volume >= target_volume:
                bar = self._aggregate_ticks_to_bar(current_bar_ticks)
                bars.append(bar)
                
                # Reset for next bar
                current_bar_ticks = []
                current_volume = 0
        
        return bars
    
    def create_time_bars(self, interval: timedelta) -> List[TickBar]:
        """Create bars based on time intervals"""
        
        if not self.ticks:
            return []
        
        bars = []
        current_bar_ticks = []
        current_interval_start = None
        
        for tick in self.ticks:
            if current_interval_start is None:
                current_interval_start = tick.timestamp
            
            if tick.timestamp < current_interval_start + interval:
                current_bar_ticks.append(tick)
            else:
                # Create bar
                if current_bar_ticks:
                    bar = self._aggregate_ticks_to_bar(current_bar_ticks)
                    bars.append(bar)
                
                # Start new interval
                current_bar_ticks = [tick]
                current_interval_start = tick.timestamp
        
        # Process remaining ticks
        if current_bar_ticks:
            bar = self._aggregate_ticks_to_bar(current_bar_ticks)
            bars.append(bar)
        
        return bars
    
    def _aggregate_ticks_to_bar(self, ticks: List[Tick]) -> TickBar:
        """Aggregate ticks into a bar"""
        
        if not ticks:
            return None
        
        prices = [t.price for t in ticks]
        volumes = [t.volume for t in ticks]
        
        # Calculate VWAP
        vwap = np.average(prices, weights=volumes)
        
        # Separate buy/sell volumes
        buy_volume = sum(t.volume for t in ticks if t.side == 'buy')
        sell_volume = sum(t.volume for t in ticks if t.side == 'sell')
        
        bar = TickBar(
            timestamp=ticks[-1].timestamp,
            open=ticks[0].price,
            high=max(prices),
            low=min(prices),
            close=ticks[-1].price,
            volume=sum(volumes),
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            tick_count=len(ticks),
            vwap=vwap
        )
        
        return bar
    
    def calculate_delta(self, lookback_ticks: int = 1000) -> Dict[str, float]:
        """Calculate order flow delta"""
        
        if len(self.ticks) < lookback_ticks:
            return {}
        
        recent_ticks = list(self.ticks)[-lookback_ticks:]
        
        buy_volume = sum(t.volume for t in recent_ticks if t.side == 'buy')
        sell_volume = sum(t.volume for t in recent_ticks if t.side == 'sell')
        
        delta = buy_volume - sell_volume
        total_volume = buy_volume + sell_volume
        
        return {
            'delta': delta,
            'delta_percent': delta / total_volume if total_volume > 0 else 0,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_ratio': buy_volume / total_volume if total_volume > 0 else 0.5
        }
    
    def detect_large_trades(self, volume_threshold: float) -> List[Tick]:
        """Detect large trades (institutional activity)"""
        
        large_trades = []
        for tick in self.ticks:
            if tick.volume >= volume_threshold:
                large_trades.append(tick)
        
        return large_trades[-10:]  # Return last 10 large trades
    
    def calculate_microstructure_metrics(self) -> Dict[str, float]:
        """Calculate microstructure metrics"""
        
        if len(self.ticks) < 100:
            return {}
        
        recent_ticks = list(self.ticks)[-100:]
        
        # Calculate spread if bid/ask available
        spreads = []
        for tick in recent_ticks:
            if tick.bid is not None and tick.ask is not None:
                spreads.append(tick.ask - tick.bid)
        
        avg_spread = np.mean(spreads) if spreads else 0
        
        # Calculate tick frequency
        timestamps = [t.timestamp for t in recent_ticks]
        if len(timestamps) > 1:
            time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            avg_tick_interval = np.mean(time_diffs) if time_diffs else 0
            tick_frequency = 1 / avg_tick_interval if avg_tick_interval > 0 else 0
        else:
            tick_frequency = 0
        
        # Calculate price impact
        price_changes = []
        for i in range(1, len(recent_ticks)):
            if recent_ticks[i].side and recent_ticks[i-1].side:
                price_change = abs(recent_ticks[i].price - recent_ticks[i-1].price)
                volume = recent_ticks[i].volume
                if volume > 0:
                    price_impact = price_change / volume
                    price_changes.append(price_impact)
        
        avg_price_impact = np.mean(price_changes) if price_changes else 0
        
        return {
            'avg_spread': avg_spread,
            'tick_frequency': tick_frequency,
            'avg_price_impact': avg_price_impact,
            'total_ticks': len(recent_ticks)
        }