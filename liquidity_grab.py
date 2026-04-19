"""
Liquidity Grab Detection and Trading
Identifying and trading stop hunts, liquidity pools
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from core.market_microstructure import MarketMicrostructure
from core.order_flow import OrderFlowAnalyzer

logger = logging.getLogger(__name__)

class LiquidityType(Enum):
    """Types of liquidity pools"""
    STOP_HUNT = "stop_hunt"
    LIQUIDITY_POOL = "liquidity_pool"
    OPTION_BARRIER = "option_barrier"
    MARKET_MAKER_DEFENSE = "market_maker_defense"

@dataclass
class LiquiditySignal:
    """Liquidity grab trading signal"""
    symbol: str
    liquidity_type: LiquidityType
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    liquidity_pool_price: float
    estimated_pool_size: float
    trigger: str  # What triggered the signal

class LiquidityGrabStrategy:
    """Detect and trade liquidity grabs"""
    
    def __init__(self):
        self.microstructure = MarketMicrostructure()
        self.order_flow = OrderFlowAnalyzer()
        self.signals_history = []
        
        # Configuration
        self.stop_hunt_threshold = 1.5  # ATR multiplier for stop hunt detection
        self.liquidity_pool_threshold = 3.0  # Size multiplier for pool detection
        
    def detect_liquidity_grabs(self, symbol: str, 
                              data: pd.DataFrame,
                              order_book: Optional[Dict] = None,
                              market_profile: Optional[Dict] = None) -> List[LiquiditySignal]:
        """Detect potential liquidity grabs"""
        
        signals = []
        
        # Need sufficient data
        if len(data) < 50:
            return signals
        
        # 1. Detect stop hunts
        stop_hunt_signals = self._detect_stop_hunts(symbol, data, order_book)
        signals.extend(stop_hunt_signals)
        
        # 2. Detect liquidity pools
        pool_signals = self._detect_liquidity_pools(symbol, data, order_book, market_profile)
        signals.extend(pool_signals)
        
        # 3. Detect option barriers (if available)
        if 'option_data' in data.columns:
            barrier_signals = self._detect_option_barriers(symbol, data)
            signals.extend(barrier_signals)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Keep history
        self.signals_history.extend(signals)
        if len(self.signals_history) > 100:
            self.signals_history = self.signals_history[-100:]
        
        return signals[:3]  # Return top 3 signals
    
    def _detect_stop_hunts(self, symbol: str, data: pd.DataFrame,
                          order_book: Optional[Dict]) -> List[LiquiditySignal]:
        """Detect stop hunt patterns"""
        
        signals = []
        
        # Calculate ATR for volatility context
        atr = self._calculate_atr(data, period=14)
        if atr == 0:
            return signals
        
        # Look for recent wicks/tails beyond typical range
        recent = data.iloc[-10:]
        
        for i in range(1, len(recent)):
            current = recent.iloc[i]
            previous = recent.iloc[i-1]
            
            bar_range = current['High'] - current['Low']
            
            # Check for upper wick rejection (bearish stop hunt)
            upper_wick = current['High'] - max(current['Open'], current['Close'])
            if upper_wick > bar_range * 0.4:  # Significant upper wick
                # Check if wick exceeds previous highs
                prev_highs = recent['High'].iloc[:i].tail(5)
                if current['High'] > prev_highs.max() * 1.001:
                    # Possible bearish stop hunt
                    signal = self._create_stop_hunt_signal(
                        symbol=symbol,
                        direction='SHORT',
                        wick_price=current['High'],
                        current_price=current['Close'],
                        atr=atr,
                        volume=current['Volume'],
                        avg_volume=data['Volume'].rolling(20).mean().iloc[i]
                    )
                    if signal:
                        signals.append(signal)
            
            # Check for lower wick rejection (bullish stop hunt)
            lower_wick = min(current['Open'], current['Close']) - current['Low']
            if lower_wick > bar_range * 0.4:  # Significant lower wick
                # Check if wick exceeds previous lows
                prev_lows = recent['Low'].iloc[:i].tail(5)
                if current['Low'] < prev_lows.min() * 0.999:
                    # Possible bullish stop hunt
                    signal = self._create_stop_hunt_signal(
                        symbol=symbol,
                        direction='LONG',
                        wick_price=current['Low'],
                        current_price=current['Close'],
                        atr=atr,
                        volume=current['Volume'],
                        avg_volume=data['Volume'].rolling(20).mean().iloc[i]
                    )
                    if signal:
                        signals.append(signal)
        
        return signals
    
    def _create_stop_hunt_signal(self, symbol: str, direction: str,
                               wick_price: float, current_price: float,
                               atr: float, volume: float, 
                               avg_volume: float) -> Optional[LiquiditySignal]:
        """Create stop hunt trading signal"""
        
        # Check volume confirmation
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        # Need above average volume for confirmation
        if volume_ratio < 1.2:
            return None
        
        # Calculate entry levels
        if direction == 'LONG':
            # Enter above the wick (after rejection)
            entry = wick_price + (atr * 0.1)
            stop = wick_price - (atr * 0.5)
            target = entry + (atr * 2)
            
            # Check if we're already above wick
            if current_price <= wick_price:
                return None
        else:  # SHORT
            # Enter below the wick (after rejection)
            entry = wick_price - (atr * 0.1)
            stop = wick_price + (atr * 0.5)
            target = entry - (atr * 2)
            
            # Check if we're already below wick
            if current_price >= wick_price:
                return None
        
        # Calculate confidence
        confidence = min(0.9, volume_ratio * 0.3)
        
        # Risk-reward check
        risk = abs(entry - stop)
        reward = abs(target - entry)
        risk_reward = reward / risk if risk > 0 else 0
        
        if risk_reward < 1.5:
            return None
        
        return LiquiditySignal(
            symbol=symbol,
            liquidity_type=LiquidityType.STOP_HUNT,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=confidence,
            liquidity_pool_price=wick_price,
            estimated_pool_size=volume,
            trigger=f"wick_rejection_{direction.lower()}"
        )
    
    def _detect_liquidity_pools(self, symbol: str, data: pd.DataFrame,
                              order_book: Optional[Dict],
                              market_profile: Optional[Dict]) -> List[LiquiditySignal]:
        """Detect liquidity pools from order book and market profile"""
        
        signals = []
        
        if order_book is None or market_profile is None:
            return signals
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Look for liquidity concentrations in order book
        liquidity_pools = self.microstructure.detect_liquidity_pools(order_book)
        
        for price, volume in liquidity_pools:
            # Check if price is near significant levels
            distance_pct = abs(price - current_price) / current_price
            
            # Only trade if price is near liquidity pool (within 0.5%)
            if distance_pct > 0.005:
                continue
            
            # Determine direction based on position relative to current price
            if price > current_price:
                direction = 'SHORT'  # Pool above price - expect rejection
                entry = current_price
                stop = price + (abs(price - current_price) * 1.5)
                target = current_price - (abs(price - current_price) * 2)
            else:
                direction = 'LONG'  # Pool below price - expect rejection
                entry = current_price
                stop = price - (abs(current_price - price) * 1.5)
                target = current_price + (abs(current_price - price) * 2)
            
            # Calculate confidence based on pool size and distance
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            size_ratio = volume / avg_volume if avg_volume > 0 else 1
            
            confidence = min(0.8, size_ratio * 0.2 + (1 - distance_pct * 100) * 0.3)
            
            # Check market profile context
            if 'value_area_low' in market_profile and 'value_area_high' in market_profile:
                val = market_profile['value_area_low']
                vah = market_profile['value_area_high']
                
                # Higher confidence if pool is outside value area
                if price < val or price > vah:
                    confidence += 0.1
            
            signal = LiquiditySignal(
                symbol=symbol,
                liquidity_type=LiquidityType.LIQUIDITY_POOL,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                confidence=confidence,
                liquidity_pool_price=price,
                estimated_pool_size=volume,
                trigger=f"order_book_pool_{direction.lower()}"
            )
            
            signals.append(signal)
        
        return signals
    
    def _detect_option_barriers(self, symbol: str, 
                              data: pd.DataFrame) -> List[LiquiditySignal]:
        """Detect option barrier levels (simplified)"""
        
        signals = []
        
        # This would require option chain data
        # Simplified version looks for round numbers as psychological barriers
        
        current_price = data['Close'].iloc[-1]
        
        # Common psychological levels (00, 50 levels)
        round_levels = []
        base_level = round(current_price, -2)  # Round to nearest 100
        
        for offset in [-100, -50, 0, 50, 100]:
            level = base_level + offset
            round_levels.append(level)
        
        # Check if price is approaching round levels
        for level in round_levels:
            distance_pct = abs(level - current_price) / current_price
            
            # Only consider levels within 0.5%
            if distance_pct <= 0.005:
                # Determine likely direction (price tends to bounce at round numbers)
                recent_action = self._analyze_price_action_near_level(data, level)
                
                if recent_action == 'rejection':
                    direction = 'SHORT' if current_price < level else 'LONG'
                    
                    signal = LiquiditySignal(
                        symbol=symbol,
                        liquidity_type=LiquidityType.OPTION_BARRIER,
                        direction=direction,
                        entry_price=current_price,
                        stop_loss=level + (0.005 * current_price * (1 if direction == 'SHORT' else -1)),
                        take_profit=current_price + (0.01 * current_price * (1 if direction == 'LONG' else -1)),
                        confidence=0.6,
                        liquidity_pool_price=level,
                        estimated_pool_size=data['Volume'].mean() * 2,
                        trigger=f"round_number_{direction.lower()}"
                    )
                    
                    signals.append(signal)
        
        return signals
    
    def _analyze_price_action_near_level(self, data: pd.DataFrame, 
                                       level: float) -> str:
        """Analyze how price behaves near a level"""
        
        recent = data.iloc[-20:]
        
        # Check for rejections
        rejections = 0
        touches = 0
        
        for _, bar in recent.iterrows():
            # Check if bar touched the level
            if bar['Low'] <= level <= bar['High']:
                touches += 1
                
                # Check for rejection (wick/tail)
                upper_wick = bar['High'] - max(bar['Open'], bar['Close'])
                lower_wick = min(bar['Open'], bar['Close']) - bar['Low']
                
                if level >= bar['Close'] and upper_wick > (bar['High'] - bar['Low']) * 0.3:
                    # Price rejected above level
                    rejections += 1
                elif level <= bar['Close'] and lower_wick > (bar['High'] - bar['Low']) * 0.3:
                    # Price rejected below level
                    rejections += 1
        
        if touches >= 2 and rejections / touches >= 0.5:
            return 'rejection'
        elif touches >= 3:
            return 'consolidation'
        else:
            return 'breakthrough'
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        
        if len(data) < period:
            return 0.0
        
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else (high.iloc[-1] - low.iloc[-1])
    
    def calculate_optimal_entry(self, signal: LiquiditySignal,
                              data: pd.DataFrame) -> Dict[str, float]:
        """Calculate optimal entry parameters for liquidity grab"""
        
        current_price = data['Close'].iloc[-1]
        atr = self._calculate_atr(data)
        
        # Entry strategies based on liquidity type
        if signal.liquidity_type == LiquidityType.STOP_HUNT:
            # For stop hunts, wait for confirmation of rejection
            if signal.direction == 'LONG':
                optimal_entry = max(signal.liquidity_pool_price, current_price) + (atr * 0.1)
                max_slippage = atr * 0.05
            else:  # SHORT
                optimal_entry = min(signal.liquidity_pool_price, current_price) - (atr * 0.1)
                max_slippage = atr * 0.05
                
        elif signal.liquidity_type == LiquidityType.LIQUIDITY_POOL:
            # For liquidity pools, enter on test of the level
            if signal.direction == 'LONG':
                optimal_entry = signal.liquidity_pool_price + (atr * 0.05)
                max_slippage = atr * 0.03
            else:  # SHORT
                optimal_entry = signal.liquidity_pool_price - (atr * 0.05)
                max_slippage = atr * 0.03
                
        else:  # OPTION_BARRIER or MARKET_MAKER_DEFENSE
            # More conservative entries
            optimal_entry = current_price
            max_slippage = atr * 0.02
        
        return {
            'optimal_entry': optimal_entry,
            'max_slippage': max_slippage,
            'entry_zone_low': optimal_entry - max_slippage,
            'entry_zone_high': optimal_entry + max_slippage,
            'patience_score': 0.7 if signal.liquidity_type == LiquidityType.STOP_HUNT else 0.5
        }