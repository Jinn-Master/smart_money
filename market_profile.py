"""
Market Profile Trading Strategies
Trading based on Value Area, POC, and Market Profile concepts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from core.volume_profile import VolumeProfileAnalyzer, VolumeProfile
from core.auction_market_theory import AuctionMarketTheory, AuctionState

logger = logging.getLogger(__name__)

class ProfileTradeLocation(Enum):
    """Market profile trade locations"""
    BELOW_VALUE_AREA = "below_value_area"
    ABOVE_VALUE_AREA = "above_value_area"
    AT_POC = "at_poc"
    VALUE_AREA_EXTREME = "value_area_extreme"
    SINGLE_PRINT = "single_print"

@dataclass
class ProfileTradeSignal:
    """Market profile based trading signal"""
    symbol: str
    location: ProfileTradeLocation
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    profile_metrics: Dict[str, float]
    auction_state: AuctionState
    trade_justification: str

class MarketProfileStrategy:
    """Market Profile based trading strategies"""
    
    def __init__(self):
        self.profile_analyzer = VolumeProfileAnalyzer()
        self.auction_theory = AuctionMarketTheory()
        self.trade_history = []
        
    def generate_profile_signals(self, symbol: str,
                               data: pd.DataFrame,
                               volume_profile: VolumeProfile,
                               auction_state: AuctionState) -> List[ProfileTradeSignal]:
        """Generate trading signals based on market profile"""
        
        signals = []
        
        if volume_profile is None or len(data) < 20:
            return signals
        
        current_price = data['Close'].iloc[-1]
        poc = volume_profile.poc
        val = volume_profile.value_area_low
        vah = volume_profile.value_area_high
        
        # 1. Trade at Value (mean reversion)
        if current_price < val:
            signal = self._create_value_area_trade(
                symbol=symbol,
                direction='LONG',
                current_price=current_price,
                val=val,
                vah=vah,
                poc=poc,
                data=data,
                volume_profile=volume_profile,
                auction_state=auction_state
            )
            if signal:
                signals.append(signal)
        
        elif current_price > vah:
            signal = self._create_value_area_trade(
                symbol=symbol,
                direction='SHORT',
                current_price=current_price,
                val=val,
                vah=vah,
                poc=poc,
                data=data,
                volume_profile=volume_profile,
                auction_state=auction_state
            )
            if signal:
                signals.append(signal)
        
        # 2. POC Trade (momentum/breakout)
        poc_signal = self._create_poc_trade(
            symbol=symbol,
            current_price=current_price,
            poc=poc,
            val=val,
            vah=vah,
            data=data,
            volume_profile=volume_profile,
            auction_state=auction_state
        )
        if poc_signal:
            signals.append(poc_signal)
        
        # 3. Single Print Trades (low volume nodes)
        single_print_signals = self._create_single_print_trades(
            symbol=symbol,
            current_price=current_price,
            volume_profile=volume_profile,
            data=data,
            auction_state=auction_state
        )
        signals.extend(single_print_signals)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals[:2]  # Return top 2 signals
    
    def _create_value_area_trade(self, symbol: str, direction: str,
                               current_price: float, val: float, vah: float,
                               poc: float, data: pd.DataFrame,
                               volume_profile: VolumeProfile,
                               auction_state: AuctionState) -> Optional[ProfileTradeSignal]:
        """Create trade at value area extremes (mean reversion)"""
        
        # Calculate distance from value area
        if direction == 'LONG':
            # Price below value area
            distance = (val - current_price) / current_price
            
            # Entry: near value area low
            entry = val * 0.999
            stop = current_price * 0.995
            target = poc * 1.005
            
            trade_location = ProfileTradeLocation.BELOW_VALUE_AREA
            justification = "Price below value area, expecting mean reversion to POC"
            
        else:  # SHORT
            # Price above value area
            distance = (current_price - vah) / current_price
            
            # Entry: near value area high
            entry = vah * 1.001
            stop = current_price * 1.005
            target = poc * 0.995
            
            trade_location = ProfileTradeLocation.ABOVE_VALUE_AREA
            justification = "Price above value area, expecting mean reversion to POC"
        
        # Need sufficient distance from value area
        if distance < 0.001:  # Less than 0.1%
            return None
        
        # Check auction state suitability
        if auction_state not in [AuctionState.BALANCE, AuctionState.BALANCE_EXTENSION]:
            # Mean reversion works best in balance
            return None
        
        # Calculate confidence
        confidence = self._calculate_value_trade_confidence(
            distance, volume_profile, data, direction
        )
        
        if confidence < 0.6:
            return None
        
        # Risk-reward check
        risk = abs(entry - stop)
        reward = abs(target - entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        if risk_reward < 1.5:
            return None
        
        # Profile metrics
        profile_metrics = self.profile_analyzer.calculate_profile_quality(volume_profile)
        
        signal = ProfileTradeSignal(
            symbol=symbol,
            location=trade_location,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=confidence,
            profile_metrics=profile_metrics,
            auction_state=auction_state,
            trade_justification=justification
        )
        
        return signal
    
    def _calculate_value_trade_confidence(self, distance: float,
                                        volume_profile: VolumeProfile,
                                        data: pd.DataFrame,
                                        direction: str) -> float:
        """Calculate confidence for value area trades"""
        
        confidence = 0.5
        
        # 1. Distance factor (optimal 0.5-1%)
        if 0.005 <= distance <= 0.01:
            confidence += 0.2
        elif distance > 0.01:
            confidence += 0.1  # Too far might indicate trend
        
        # 2. Volume profile quality
        profile_quality = self.profile_analyzer.calculate_profile_quality(volume_profile)
        
        # Higher confidence with clear POC and value area
        if profile_quality.get('poc_volume_ratio', 0) > 1.5:
            confidence += 0.15
        
        if profile_quality.get('value_area_width', 1) < 0.01:  # Tight value area
            confidence += 0.1
        
        # 3. Recent price action
        recent = data.iloc[-5:]
        
        if direction == 'LONG':
            # Look for signs of support
            lower_wicks = []
            for _, bar in recent.iterrows():
                wick = min(bar['Open'], bar['Close']) - bar['Low']
                if wick > (bar['High'] - bar['Low']) * 0.3:
                    lower_wicks.append(wick)
            
            if lower_wicks:
                confidence += 0.1
        else:  # SHORT
            # Look for signs of resistance
            upper_wicks = []
            for _, bar in recent.iterrows():
                wick = bar['High'] - max(bar['Open'], bar['Close'])
                if wick > (bar['High'] - bar['Low']) * 0.3:
                    upper_wicks.append(wick)
            
            if upper_wicks:
                confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _create_poc_trade(self, symbol: str, current_price: float,
                         poc: float, val: float, vah: float,
                         data: pd.DataFrame, volume_profile: VolumeProfile,
                         auction_state: AuctionState) -> Optional[ProfileTradeSignal]:
        """Create trade at Point of Control"""
        
        # Check if price is near POC
        distance_to_poc = abs(current_price - poc) / poc
        
        if distance_to_poc > 0.002:  # More than 0.2% from POC
            return None
        
        # Determine direction based on auction state and momentum
        recent_trend = self._analyze_recent_trend(data)
        
        if auction_state == AuctionState.TREND:
            # Go with trend at POC retest
            direction = 'LONG' if recent_trend == 'up' else 'SHORT'
            entry = poc
            stop = poc * (0.995 if direction == 'LONG' else 1.005)
            target = vah if direction == 'LONG' else val
            
            justification = "POC retest in trend, entering with trend"
            
        elif auction_state == AuctionState.BALANCE:
            # Fade POC touches in balance
            # Check if POC has been tested recently
            poc_tests = self._count_poc_tests(data, poc)
            
            if poc_tests >= 2:
                # Multiple POC tests in balance - expect break
                # Determine break direction from auction rotation
                rotation = self.auction_theory.auction_history[-1].auction_rotation if self.auction_theory.auction_history else []
                
                if 'below_va' in rotation and 'above_va' in rotation:
                    # Rotation suggests continued balance
                    return None
                elif 'below_va' in rotation:
                    direction = 'LONG'  # Rotation from below suggests upward break
                else:
                    direction = 'SHORT'
            else:
                # First POC test in balance - fade
                direction = 'SHORT' if current_price > poc else 'LONG'
            
            entry = poc
            stop = poc * (1.005 if direction == 'LONG' else 0.995)
            target = val if direction == 'LONG' else vah
            
            justification = "POC fade in balance phase"
        
        else:
            return None
        
        # Calculate confidence
        confidence = self._calculate_poc_trade_confidence(
            distance_to_poc, auction_state, recent_trend, volume_profile
        )
        
        if confidence < 0.65:
            return None
        
        profile_metrics = self.profile_analyzer.calculate_profile_quality(volume_profile)
        
        signal = ProfileTradeSignal(
            symbol=symbol,
            location=ProfileTradeLocation.AT_POC,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=confidence,
            profile_metrics=profile_metrics,
            auction_state=auction_state,
            trade_justification=justification
        )
        
        return signal
    
    def _analyze_recent_trend(self, data: pd.DataFrame) -> str:
        """Analyze recent price trend"""
        
        if len(data) < 20:
            return 'neutral'
        
        recent = data.iloc[-10:]
        
        # Simple trend detection
        price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]
        
        if price_change > 0.01:  # 1% up
            return 'up'
        elif price_change < -0.01:  # 1% down
            return 'down'
        else:
            return 'neutral'
    
    def _count_poc_tests(self, data: pd.DataFrame, poc: float) -> int:
        """Count how many times POC has been tested recently"""
        
        if len(data) < 10:
            return 0
        
        recent = data.iloc[-10:]
        
        tests = 0
        for _, bar in recent.iterrows():
            # Bar touches POC (within 0.1%)
            if bar['Low'] <= poc * 1.001 and bar['High'] >= poc * 0.999:
                tests += 1
        
        return tests
    
    def _calculate_poc_trade_confidence(self, distance_to_poc: float,
                                      auction_state: AuctionState,
                                      recent_trend: str,
                                      volume_profile: VolumeProfile) -> float:
        """Calculate confidence for POC trades"""
        
        confidence = 0.6
        
        # 1. Distance to POC (closer is better)
        if distance_to_poc < 0.001:  # Within 0.1%
            confidence += 0.15
        
        # 2. Auction state alignment
        if auction_state == AuctionState.TREND:
            confidence += 0.1
        elif auction_state == AuctionState.BALANCE:
            confidence += 0.05
        
        # 3. Volume profile structure
        profile_quality = self.profile_analyzer.calculate_profile_quality(volume_profile)
        
        # Strong POC
        if profile_quality.get('poc_volume_ratio', 0) > 2.0:
            confidence += 0.2
        
        # Symmetrical profile
        if abs(profile_quality.get('skewness', 0)) < 0.5:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _create_single_print_trades(self, symbol: str, current_price: float,
                                  volume_profile: VolumeProfile,
                                  data: pd.DataFrame,
                                  auction_state: AuctionState) -> List[ProfileTradeSignal]:
        """Create trades at single print areas (low volume nodes)"""
        
        signals = []
        
        if not volume_profile.single_prints:
            return signals
        
        # Get single prints near current price
        for single_print_price, volume in volume_profile.single_prints:
            distance_pct = abs(single_print_price - current_price) / current_price
            
            # Only consider nearby single prints (within 0.5%)
            if distance_pct > 0.005:
                continue
            
            # Single prints act as weak points - price tends to move through them
            # Trade break of single print
            
            # Determine likely break direction
            # Check which side has more volume
            prices = list(volume_profile.volume_at_price.keys())
            volumes = list(volume_profile.volume_at_price.values())
            
            # Find single print index
            try:
                sp_index = prices.index(single_print_price)
            except ValueError:
                continue
            
            # Check volume profile around single print
            left_volume = sum(volumes[max(0, sp_index-3):sp_index]) if sp_index > 0 else 0
            right_volume = sum(volumes[sp_index+1:min(len(volumes), sp_index+4)]) if sp_index < len(volumes)-1 else 0
            
            if left_volume > right_volume * 1.5:
                # More volume to the left - likely break right (up)
                direction = 'LONG'
                entry = single_print_price * 1.001
                stop = single_print_price * 0.998
                target = single_print_price * 1.01
                justification = "Single print with higher volume left, expecting upward break"
            elif right_volume > left_volume * 1.5:
                # More volume to the right - likely break left (down)
                direction = 'SHORT'
                entry = single_print_price * 0.999
                stop = single_print_price * 1.002
                target = single_print_price * 0.99
                justification = "Single print with higher volume right, expecting downward break"
            else:
                # Balanced - no clear direction
                continue
            
            # Check auction state
            if auction_state not in [AuctionState.TREND, AuctionState.INITIATIVE]:
                # Single prints work best in trending markets
                continue
            
            # Calculate confidence
            volume_ratio = max(left_volume, right_volume) / min(left_volume, right_volume) if min(left_volume, right_volume) > 0 else 1
            confidence = min(0.8, 0.5 + (volume_ratio - 1) * 0.1)
            
            profile_metrics = self.profile_analyzer.calculate_profile_quality(volume_profile)
            
            signal = ProfileTradeSignal(
                symbol=symbol,
                location=ProfileTradeLocation.SINGLE_PRINT,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                confidence=confidence,
                profile_metrics=profile_metrics,
                auction_state=auction_state,
                trade_justification=justification
            )
            
            signals.append(signal)
        
        return signals
    
    def calculate_optimal_position_size(self, signal: ProfileTradeSignal,
                                      account_balance: float,
                                      risk_per_trade: float = 0.01) -> float:
        """Calculate optimal position size for profile trade"""
        
        risk_amount = account_balance * risk_per_trade
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        
        if stop_distance == 0:
            return 0.0
        
        # Base position size
        position_size = risk_amount / stop_distance
        
        # Adjust based on profile metrics
        profile_score = self._calculate_profile_score(signal.profile_metrics)
        adjustment_factor = 0.5 + (profile_score * 0.5)  # 0.5 to 1.0
        
        final_size = position_size * adjustment_factor
        
        return final_size
    
    def _calculate_profile_score(self, profile_metrics: Dict[str, float]) -> float:
        """Calculate score for profile quality (0 to 1)"""
        
        if not profile_metrics:
            return 0.5
        
        score = 0.5
        
        # POC strength
        poc_ratio = profile_metrics.get('poc_volume_ratio', 1)
        if poc_ratio > 1.5:
            score += 0.2
        elif poc_ratio > 2.0:
            score += 0.3
        
        # Profile symmetry
        skewness = abs(profile_metrics.get('skewness', 0))
        if skewness < 0.3:
            score += 0.15
        
        # Value area tightness
        va_width = profile_metrics.get('value_area_width', 0.02)
        if va_width < 0.01:
            score += 0.1
        elif va_width > 0.03:
            score -= 0.1
        
        return min(max(score, 0.1), 1.0)