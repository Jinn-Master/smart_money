"""
Volume Profile and Market Profile Analysis
TPO (Time Price Opportunity) and Value Area calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class VolumeProfile:
    """Volume at Price (VAP) profile"""
    poc: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_at_price: Dict[float, float]
    developing_poc: Optional[float] = None
    single_prints: List[Tuple[float, float]] = None  # (price, volume)

@dataclass
class MarketProfile:
    """Market Profile (TPO)"""
    poc: float
    value_area: Tuple[float, float]
    tpo_counts: Dict[float, int]
    balance_area: Optional[Tuple[float, float]] = None
    initial_balance: Optional[Tuple[float, float]] = None

class VolumeProfileAnalyzer:
    """Volume Profile and Market Profile Analysis"""
    
    def __init__(self, price_bins: int = 50, value_area_percent: float = 0.7):
        self.price_bins = price_bins
        self.value_area_percent = value_area_percent
        self.profiles = {}
        
    def calculate_volume_profile(self, data: pd.DataFrame, 
                                volume_col: str = 'Volume') -> VolumeProfile:
        """Calculate Volume Profile from tick or candle data"""
        
        if 'price' in data.columns:
            # Tick data
            prices = data['price'].values
            volumes = data[volume_col].values
        else:
            # OHLC data - approximate
            prices = []
            volumes = []
            for _, row in data.iterrows():
                # Distribute volume across price range
                price_range = np.linspace(row['Low'], row['High'], 5)
                volume_dist = row[volume_col] / 5
                prices.extend(price_range)
                volumes.extend([volume_dist] * 5)
        
        if len(prices) == 0:
            return None
        
        # Create price bins
        min_price, max_price = np.min(prices), np.max(prices)
        bin_size = (max_price - min_price) / self.price_bins
        price_bins = np.arange(min_price, max_price + bin_size, bin_size)
        
        # Aggregate volume at price
        volume_at_price = defaultdict(float)
        for price, volume in zip(prices, volumes):
            bin_idx = int((price - min_price) / bin_size)
            bin_price = price_bins[min(bin_idx, len(price_bins) - 1)]
            volume_at_price[bin_price] += volume
        
        # Find POC (Point of Control)
        poc_price = max(volume_at_price, key=volume_at_price.get)
        total_volume = sum(volume_at_price.values())
        
        # Calculate Value Area (70% of volume)
        sorted_prices = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)
        
        cumulative_volume = 0
        value_area_prices = []
        
        for price, volume in sorted_prices:
            cumulative_volume += volume
            value_area_prices.append(price)
            
            if cumulative_volume / total_volume >= self.value_area_percent:
                break
        
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        # Find single prints (low volume areas)
        avg_volume = np.mean(list(volume_at_price.values()))
        single_prints = [(p, v) for p, v in volume_at_price.items() 
                        if v < avg_volume * 0.3]
        
        return VolumeProfile(
            poc=poc_price,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            volume_at_price=dict(volume_at_price),
            single_prints=single_prints
        )
    
    def calculate_market_profile(self, data: pd.DataFrame, 
                                period: str = '30min') -> MarketProfile:
        """Calculate Market Profile (TPO)"""
        
        # Resample to create TPO periods
        data_resampled = data.resample(period).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Create TPO counts
        tpo_counts = defaultdict(int)
        
        for idx, row in data_resampled.iterrows():
            # For each period, mark all prices between high and low
            price_range = np.arange(
                round(row['Low'], 4),
                round(row['High'], 4),
                0.0001  # Adjust based on instrument
            )
            for price in price_range:
                tpo_counts[round(price, 4)] += 1
        
        # Find POC (most TPOs)
        if not tpo_counts:
            return None
            
        poc_price = max(tpo_counts, key=tpo_counts.get)
        total_tpos = sum(tpo_counts.values())
        
        # Calculate Value Area (70% of TPOs)
        sorted_prices = sorted(tpo_counts.items(), key=lambda x: x[1], reverse=True)
        
        cumulative_tpos = 0
        value_area_prices = []
        
        for price, count in sorted_prices:
            cumulative_tpos += count
            value_area_prices.append(price)
            
            if cumulative_tpos / total_tpos >= self.value_area_percent:
                break
        
        value_area = (min(value_area_prices), max(value_area_prices))
        
        # Find Initial Balance (first 2 periods)
        initial_balance = None
        if len(data_resampled) >= 2:
            ib_high = data_resampled['High'].iloc[:2].max()
            ib_low = data_resampled['Low'].iloc[:2].min()
            initial_balance = (ib_low, ib_high)
        
        # Find Balance Area (sideways price action)
        balance_area = self._find_balance_area(data)
        
        return MarketProfile(
            poc=poc_price,
            value_area=value_area,
            tpo_counts=dict(tpo_counts),
            balance_area=balance_area,
            initial_balance=initial_balance
        )
    
    def _find_balance_area(self, data: pd.DataFrame, 
                          lookback_periods: int = 20) -> Optional[Tuple[float, float]]:
        """Identify balance/consolidation areas"""
        
        if len(data) < lookback_periods:
            return None
        
        recent = data.iloc[-lookback_periods:]
        
        # Calculate average true range
        high_low = recent['High'] - recent['Low']
        atr = high_low.rolling(14).mean().iloc[-1]
        
        # Check if price is ranging
        total_range = recent['High'].max() - recent['Low'].min()
        avg_bar_range = high_low.mean()
        
        # If total range is less than 2x ATR, likely in balance
        if total_range < atr * 2:
            return (recent['Low'].min(), recent['High'].max())
        
        return None
    
    def identify_developing_poc(self, current_profile: VolumeProfile, 
                              historical_profiles: List[VolumeProfile]) -> Optional[float]:
        """Identify developing POC (shifting value)"""
        
        if len(historical_profiles) < 5:
            return None
        
        # Check if POC is moving
        recent_pocs = [p.poc for p in historical_profiles[-5:]]
        current_poc = current_profile.poc
        
        # Calculate moving average of POCs
        poc_ma = np.mean(recent_pocs)
        poc_std = np.std(recent_pocs)
        
        # If current POC is more than 1 std from MA, it's developing
        if abs(current_poc - poc_ma) > poc_std:
            # Determine direction
            if current_poc > poc_ma:
                return current_poc  # Developing higher
            else:
                return current_poc  # Developing lower
        
        return None
    
    def calculate_profile_quality(self, profile: VolumeProfile) -> Dict[str, float]:
        """Assess quality of volume profile"""
        
        if not profile.volume_at_price:
            return {}
        
        volumes = list(profile.volume_at_price.values())
        
        # Calculate metrics
        volume_std = np.std(volumes)
        volume_mean = np.mean(volumes)
        poc_volume = profile.volume_at_price.get(profile.poc, 0)
        
        # Profile shape metrics
        skewness = self._calculate_skewness(volumes)
        kurtosis = self._calculate_kurtosis(volumes)
        
        return {
            'poc_volume_ratio': poc_volume / volume_mean if volume_mean > 0 else 0,
            'volume_std': volume_std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'value_area_width': (profile.value_area_high - profile.value_area_low) / profile.poc,
            'single_prints_count': len(profile.single_prints or [])
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of distribution"""
        if len(data) < 3:
            return 0.0
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((data_array - mean) / std) ** 3)
        return skew
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of distribution"""
        if len(data) < 4:
            return 0.0
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std == 0:
            return 0.0
        
        kurt = np.mean(((data_array - mean) / std) ** 4) - 3  # Excess kurtosis
        return kurt