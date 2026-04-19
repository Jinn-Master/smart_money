"""
Order Flow Analysis - Volume Spread Analysis (VSA)
Following Wyckoff and Smart Money Concepts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VSAState:
    """Volume Spread Analysis State"""
    bar_type: str  # 'upthrust', 'spring', 'test', 'SOW', 'SOS'
    confidence: float
    volume_ratio: float
    spread_ratio: float
    close_position: str  # 'high', 'middle', 'low'

class OrderFlowAnalyzer:
    """Wyckoff Method and Volume Spread Analysis"""
    
    def __init__(self, volume_ma_period: int = 20):
        self.volume_ma_period = volume_ma_period
        self.previous_signals = []
        
    def analyze_bar(self, current: pd.Series, previous: pd.Series, 
                   avg_volume: float) -> Optional[VSAState]:
        """Analyze single bar for smart money signals"""
        
        # Calculate ratios
        volume_ratio = current['Volume'] / avg_volume
        spread = current['High'] - current['Low']
        avg_spread = (previous['High'] - previous['Low'])
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0
        
        # Determine close position
        bar_center = (current['High'] + current['Low']) / 2
        if current['Close'] > current['High'] - (spread * 0.25):
            close_position = 'high'
        elif current['Close'] < current['Low'] + (spread * 0.25):
            close_position = 'low'
        else:
            close_position = 'middle'
        
        # VSA Patterns (Wyckoff)
        # 1. Upthrust (false breakout)
        if (spread_ratio > 1.2 and volume_ratio > 1.5 and 
            close_position == 'low' and current['Close'] < previous['Close']):
            return VSAState('upthrust', 0.8, volume_ratio, spread_ratio, close_position)
        
        # 2. Spring (false breakdown)
        elif (spread_ratio > 1.2 and volume_ratio > 1.5 and 
              close_position == 'high' and current['Close'] > previous['Close']):
            return VSAState('spring', 0.8, volume_ratio, spread_ratio, close_position)
        
        # 3. Test (low volume test of support/resistance)
        elif (spread_ratio < 0.7 and volume_ratio < 0.5 and 
              abs(current['Close'] - previous['Close']) < spread * 0.1):
            return VSAState('test', 0.6, volume_ratio, spread_ratio, close_position)
        
        # 4. Sign of Weakness (SOW)
        elif (volume_ratio > 2.0 and spread_ratio > 1.5 and 
              close_position == 'low' and current['Close'] < current['Open']):
            return VSAState('SOW', 0.9, volume_ratio, spread_ratio, close_position)
        
        # 5. Sign of Strength (SOS)
        elif (volume_ratio > 2.0 and spread_ratio > 1.5 and 
              close_position == 'high' and current['Close'] > current['Open']):
            return VSAState('SOS', 0.9, volume_ratio, spread_ratio, close_position)
        
        return None
    
    def detect_wyckoff_accumulation(self, df: pd.DataFrame, window: int = 10) -> Dict:
        """Detect Wyckoff accumulation phases"""
        if len(df) < window:
            return {}
        
        # Simplified Wyckoff detection
        recent = df.iloc[-window:]
        
        # Look for spring then test pattern
        springs = 0
        tests = 0
        volume_increase = False
        
        for i in range(1, len(recent)):
            current = recent.iloc[i]
            prev = recent.iloc[i-1]
            avg_volume = recent['Volume'].rolling(5).mean().iloc[i]
            
            vsa = self.analyze_bar(current, prev, avg_volume)
            if vsa:
                if vsa.bar_type == 'spring':
                    springs += 1
                elif vsa.bar_type == 'test':
                    tests += 1
            
            # Check for volume increase on up bars
            if (current['Close'] > current['Open'] and 
                current['Volume'] > avg_volume * 1.5):
                volume_increase = True
        
        if springs >= 1 and tests >= 1 and volume_increase:
            return {
                'phase': 'accumulation',
                'confidence': min(0.9, (springs + tests) / 5),
                'last_signal': 'spring_test_complete',
                'suggested_action': 'BUY'
            }
        
        return {}
    
    def calculate_delta(self, ticks: pd.DataFrame) -> pd.Series:
        """Calculate volume delta (buying vs selling pressure)"""
        if 'side' not in ticks.columns:
            # Estimate from tick rules
            ticks['side'] = np.where(ticks['price'] > ticks['price'].shift(1), 'buy', 'sell')
        
        buy_volume = ticks.loc[ticks['side'] == 'buy', 'volume'].sum()
        sell_volume = ticks.loc[ticks['side'] == 'sell', 'volume'].sum()
        
        delta = buy_volume - sell_volume
        total = buy_volume + sell_volume
        
        return pd.Series({
            'delta': delta,
            'delta_pct': delta / total if total > 0 else 0,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume
        })