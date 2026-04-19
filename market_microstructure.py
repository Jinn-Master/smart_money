"""
Market Microstructure Analysis - Following Smart Money
Focus: Order book imbalances, liquidity, market depth
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Order book snapshot with bid/ask levels"""
    timestamp: pd.Timestamp
    bids: List[Tuple[float, float]]  # (price, volume)
    asks: List[Tuple[float, float]]
    spread: float
    mid_price: float
    
@dataclass
class OrderFlowMetrics:
    """Smart money order flow metrics"""
    bid_ask_imbalance: float  # -1 to 1
    order_book_skew: float    # Buying vs selling pressure
    liquidity_zones: List[Tuple[float, float]]  # (price, liquidity)
    absorption_levels: List[float]  # Prices with high absorption
    delta_divergence: float   # Price vs volume delta

class MarketMicrostructure:
    """Analyze order book for smart money footprints"""
    
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        self.order_book_history = deque(maxlen=1000)
        
    def calculate_bid_ask_imbalance(self, bids: List, asks: List) -> float:
        """Calculate order book imbalance (-1 to 1)"""
        if not bids or not asks:
            return 0.0
        
        total_bid_volume = sum(vol for _, vol in bids[:self.depth_levels])
        total_ask_volume = sum(vol for _, vol in asks[:self.depth_levels])
        
        if total_bid_volume + total_ask_volume == 0:
            return 0.0
        
        return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    
    def detect_liquidity_pools(self, order_book: OrderBookSnapshot) -> List[Tuple[float, float]]:
        """Identify liquidity concentrations (stop hunts)"""
        liquidity_pools = []
        
        # Look for large resting orders
        for price, volume in order_book.bids[:5]:
            if volume > np.mean([v for _, v in order_book.bids]) * 3:
                liquidity_pools.append((price, volume))
        
        for price, volume in order_book.asks[:5]:
            if volume > np.mean([v for _, v in order_book.asks]) * 3:
                liquidity_pools.append((price, volume))
        
        return liquidity_pools
    
    def calculate_vpoc(self, ticks: pd.DataFrame, period: str = '1H') -> Dict:
        """Calculate Volume Point of Control (Market Profile)"""
        if ticks.empty:
            return {}
        
        # Group by price levels
        price_bins = np.linspace(ticks['price'].min(), ticks['price'].max(), 50)
        volume_at_price = {}
        
        for price_bin in price_bins:
            mask = (ticks['price'] >= price_bin - (price_bins[1] - price_bins[0]) / 2) & \
                   (ticks['price'] < price_bin + (price_bins[1] - price_bins[0]) / 2)
            volume_at_price[price_bin] = ticks.loc[mask, 'volume'].sum()
        
        # Find POC (highest volume)
        poc_price = max(volume_at_price, key=volume_at_price.get)
        
        return {
            'poc': poc_price,
            'value_area_high': sorted(volume_at_price.keys(), 
                                     key=lambda x: volume_at_price[x], 
                                     reverse=True)[:int(len(volume_at_price) * 0.7)][-1],
            'value_area_low': sorted(volume_at_price.keys(), 
                                    key=lambda x: volume_at_price[x], 
                                    reverse=True)[:int(len(volume_at_price) * 0.7)][0],
            'volume_profile': volume_at_price
        }
    
    def detect_order_blocks(self, candles: pd.DataFrame, lookback: int = 20) -> List[Dict]:
        """Identify institutional order blocks (fair value gaps)"""
        order_blocks = []
        
        for i in range(lookback, len(candles)):
            # Look for large engulfing candles
            current = candles.iloc[i]
            prev = candles.iloc[i-1]
            
            # Bullish order block
            if (current['Close'] > prev['High'] and 
                current['Volume'] > prev['Volume'] * 1.5):
                order_blocks.append({
                    'type': 'bullish',
                    'price': prev['High'],
                    'time': candles.index[i],
                    'volume': current['Volume']
                })
            
            # Bearish order block
            elif (current['Close'] < prev['Low'] and 
                  current['Volume'] > prev['Volume'] * 1.5):
                order_blocks.append({
                    'type': 'bearish',
                    'price': prev['Low'],
                    'time': candles.index[i],
                    'volume': current['Volume']
                })
        
        return order_blocks[-5:]  # Return last 5 blocks