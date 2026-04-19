"""
Auction Market Theory (AMT) and Market Profile Concepts
Based on Jim Dalton's work
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AuctionState(Enum):
    """Auction Market States"""
    BALANCE = "balance"           # Two-timeframe trade
    BALANCE_EXTENSION = "balance_extension"
    TREND = "trend"
    TREND_TERMINATION = "trend_termination"
    INITIATIVE = "initiative"     # One-timeframe trade

@dataclass
class AuctionMetrics:
    """Auction market metrics"""
    state: AuctionState
    confidence: float
    accepting_prices: bool        # Market accepting higher/lower prices
    rejecting_prices: bool        # Market rejecting extremes
    value_area_development: str   # 'expanding', 'contracting', 'stable'
    auction_rotation: List[str]   # Sequence of auction movements

class AuctionMarketTheory:
    """Auction Market Theory Analysis"""
    
    def __init__(self):
        self.auction_history = []
        self.value_area_history = []
        
    def analyze_auction(self, data: pd.DataFrame, 
                       profile: 'MarketProfile') -> AuctionMetrics:
        """Analyze auction market behavior"""
        
        if len(data) < 20 or profile is None:
            return AuctionMetrics(
                state=AuctionState.BALANCE,
                confidence=0.0,
                accepting_prices=False,
                rejecting_prices=False,
                value_area_development='stable',
                auction_rotation=[]
            )
        
        # Get recent price action
        recent = data.iloc[-5:]
        prev = data.iloc[-10:-5]
        
        # Calculate auction metrics
        accepting = self._is_accepting_prices(recent, profile)
        rejecting = self._is_rejecting_prices(recent, profile)
        va_development = self._assess_value_area_development()
        auction_rotation = self._track_auction_rotation(data, profile)
        
        # Determine auction state
        state = self._determine_auction_state(
            accepting, rejecting, va_development, data, profile
        )
        
        confidence = self._calculate_confidence(
            state, accepting, rejecting, data, profile
        )
        
        metrics = AuctionMetrics(
            state=state,
            confidence=confidence,
            accepting_prices=accepting,
            rejecting_prices=rejecting,
            value_area_development=va_development,
            auction_rotation=auction_rotation
        )
        
        self.auction_history.append(metrics)
        if len(self.auction_history) > 100:
            self.auction_history = self.auction_history[-100:]
        
        return metrics
    
    def _is_accepting_prices(self, recent: pd.DataFrame, 
                           profile: 'MarketProfile') -> bool:
        """Check if market is accepting higher/lower prices"""
        
        # Price is moving beyond value area with volume
        current_price = recent['Close'].iloc[-1]
        value_area_low, value_area_high = profile.value_area
        
        # Check if price is breaking out of value area
        if current_price > value_area_high:
            # Check volume on breakout
            breakout_bars = recent[recent['High'] > value_area_high]
            if not breakout_bars.empty:
                avg_volume = recent['Volume'].mean()
                breakout_volume = breakout_bars['Volume'].mean()
                return breakout_volume > avg_volume * 1.2
        
        elif current_price < value_area_low:
            # Check volume on breakdown
            breakdown_bars = recent[recent['Low'] < value_area_low]
            if not breakdown_bars.empty:
                avg_volume = recent['Volume'].mean()
                breakdown_volume = breakdown_bars['Volume'].mean()
                return breakdown_volume > avg_volume * 1.2
        
        return False
    
    def _is_rejecting_prices(self, recent: pd.DataFrame, 
                           profile: 'MarketProfile') -> bool:
        """Check if market is rejecting price extremes"""
        
        current_price = recent['Close'].iloc[-1]
        value_area_low, value_area_high = profile.value_area
        
        # Look for tails/wicks outside value area that get rejected
        for _, bar in recent.iterrows():
            # Upper tail rejection
            if bar['High'] > value_area_high and bar['Close'] < value_area_high:
                upper_tail = bar['High'] - max(bar['Open'], bar['Close'])
                if upper_tail > (bar['High'] - bar['Low']) * 0.3:  # Significant tail
                    return True
            
            # Lower tail rejection
            if bar['Low'] < value_area_low and bar['Close'] > value_area_low:
                lower_tail = min(bar['Open'], bar['Close']) - bar['Low']
                if lower_tail > (bar['High'] - bar['Low']) * 0.3:  # Significant tail
                    return True
        
        return False
    
    def _assess_value_area_development(self) -> str:
        """Assess if value area is expanding or contracting"""
        
        if len(self.value_area_history) < 3:
            return 'stable'
        
        recent_widths = [va[1] - va[0] for va in self.value_area_history[-3:]]
        
        # Calculate trend
        if len(recent_widths) >= 2:
            width_change = recent_widths[-1] - recent_widths[-2]
            width_change_pct = width_change / recent_widths[-2] if recent_widths[-2] > 0 else 0
            
            if width_change_pct > 0.05:
                return 'expanding'
            elif width_change_pct < -0.05:
                return 'contracting'
        
        return 'stable'
    
    def _track_auction_rotation(self, data: pd.DataFrame, 
                              profile: 'MarketProfile') -> List[str]:
        """Track auction rotation sequence"""
        
        rotation = []
        current_price = data['Close'].iloc[-1]
        value_area_low, value_area_high = profile.value_area
        
        # Determine current location
        if current_price > value_area_high:
            rotation.append('above_va')
        elif current_price < value_area_low:
            rotation.append('below_va')
        else:
            rotation.append('inside_va')
        
        # Add previous rotations
        if len(self.auction_history) > 0:
            prev_rotation = self.auction_history[-1].auction_rotation
            if prev_rotation:
                rotation.insert(0, prev_rotation[-1])
        
        return rotation[-2:]  # Keep last 2 rotations
    
    def _determine_auction_state(self, accepting: bool, rejecting: bool,
                               va_development: str, data: pd.DataFrame,
                               profile: 'MarketProfile') -> AuctionState:
        """Determine current auction state"""
        
        current_price = data['Close'].iloc[-1]
        value_area_low, value_area_high = profile.value_area
        
        # Check for trend
        if accepting:
            if current_price > value_area_high:
                return AuctionState.TREND
            elif current_price < value_area_low:
                return AuctionState.TREND
        elif rejecting:
            return AuctionState.BALANCE
        
        # Check for balance
        if va_development == 'stable' and not accepting:
            # Check if price is rotating within value area
            price_inside_va = (current_price >= value_area_low and 
                             current_price <= value_area_high)
            
            if price_inside_va:
                # Check for two-timeframe trade (balance)
                recent_high = data['High'].iloc[-5:].max()
                recent_low = data['Low'].iloc[-5:].min()
                
                if (recent_high <= value_area_high and 
                    recent_low >= value_area_low):
                    return AuctionState.BALANCE
        
        # Default to initiative (one-timeframe trade)
        return AuctionState.INITIATIVE
    
    def _calculate_confidence(self, state: AuctionState, accepting: bool,
                            rejecting: bool, data: pd.DataFrame,
                            profile: 'MarketProfile') -> float:
        """Calculate confidence in auction state assessment"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on evidence
        if accepting or rejecting:
            confidence += 0.2
        
        # Check for clear profile structure
        if profile.tpo_counts:
            tpos = list(profile.tpo_counts.values())
            if len(tpos) >= 10:
                # Clear POC with good distribution
                poc_volume = profile.tpo_counts.get(profile.poc, 0)
                avg_volume = np.mean(tpos)
                if poc_volume > avg_volume * 1.5:
                    confidence += 0.15
        
        # Check for recent auction consistency
        if len(self.auction_history) >= 3:
            recent_states = [h.state for h in self.auction_history[-3:]]
            if all(s == state for s in recent_states):
                confidence += 0.1
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def identify_trade_location(self, data: pd.DataFrame, 
                              profile: 'MarketProfile') -> Dict[str, any]:
        """Identify optimal trade locations based on AMT"""
        
        current_price = data['Close'].iloc[-1]
        value_area_low, value_area_high = profile.value_area
        
        trade_locations = {
            'inside_value_area': {
                'long_zone': value_area_low * 1.001,  # Just above VAL
                'short_zone': value_area_high * 0.999,  # Just below VAH
                'confidence': 0.6
            },
            'outside_value_area': {
                'long_zone': value_area_low * 0.995,  # Just below VAL for reversal
                'short_zone': value_area_high * 1.005,  # Just above VAH for reversal
                'confidence': 0.4
            }
        }
        
        # Adjust based on auction state
        auction_metrics = self.analyze_auction(data, profile)
        
        if auction_metrics.state == AuctionState.BALANCE:
            # Fade extremes in balance
            trade_locations['inside_value_area']['confidence'] += 0.2
        elif auction_metrics.state == AuctionState.TREND:
            # Go with trend outside value area
            trade_locations['outside_value_area']['confidence'] += 0.3
        
        return trade_locations