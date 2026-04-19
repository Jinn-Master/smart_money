"""
Smart Money Strategy - Follow institutional order flow
Combines: Market Profile + VSA + Order Blocks + Liquidity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from core.market_microstructure import MarketMicrostructure, OrderFlowMetrics
from core.order_flow import OrderFlowAnalyzer, VSAState

logger = logging.getLogger(__name__)

class MarketContext(Enum):
    """Market context for regime detection"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    MARKUP = "markup"
    MARKDOWN = "markdown"
    BALANCE = "balance"

@dataclass
class SmartMoneySignal:
    """Comprehensive smart money signal"""
    symbol: str
    direction: str  # LONG, SHORT, NEUTRAL
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    context: MarketContext
    triggers: List[str]  # VSA, Order Block, Liquidity, etc.
    risk_reward: float
    position_size: float
    validity_window: int  # Bars signal is valid

class SmartMoneyStrategy:
    """Main institutional strategy combining multiple concepts"""
    
    def __init__(self):
        self.microstructure = MarketMicrostructure()
        self.order_flow = OrderFlowAnalyzer()
        self.signals_history = []
        
        # Risk parameters
        self.max_positions = 5
        self.correlation_threshold = 0.7
        
    def analyze_market(self, symbol: str, 
                      candles: pd.DataFrame,
                      order_book: Optional[Dict] = None,
                      ticks: Optional[pd.DataFrame] = None) -> SmartMoneySignal:
        """Complete market analysis for smart money signals"""
        
        signals = []
        confidence_scores = []
        
        # 1. VSA Analysis
        vsa_signals = self._analyze_vsa(candles)
        if vsa_signals:
            signals.extend(vsa_signals['triggers'])
            confidence_scores.append(vsa_signals['confidence'])
        
        # 2. Order Block Analysis
        order_blocks = self.microstructure.detect_order_blocks(candles)
        if order_blocks:
            signals.append(f"order_block_{order_blocks[-1]['type']}")
            confidence_scores.append(0.7)
        
        # 3. Market Profile (if tick data available)
        if ticks is not None:
            profile = self.microstructure.calculate_vpoc(ticks)
            current_price = candles['Close'].iloc[-1]
            
            # Trade at value (buy below value area, sell above)
            if current_price < profile.get('value_area_low', current_price):
                signals.append("below_value_area")
                confidence_scores.append(0.6)
            elif current_price > profile.get('value_area_high', current_price):
                signals.append("above_value_area")
                confidence_scores.append(0.6)
        
        # 4. Liquidity Analysis (if order book available)
        if order_book:
            liquidity_pools = self.microstructure.detect_liquidity_pools(order_book)
            current_price = candles['Close'].iloc[-1]
            
            for price, volume in liquidity_pools:
                # If price near liquidity pool, anticipate liquidity grab
                if abs(current_price - price) / current_price < 0.001:
                    direction = "SHORT" if price > current_price else "LONG"
                    signals.append(f"liquidity_grab_{direction}")
                    confidence_scores.append(0.75)
        
        # 5. Determine market context
        context = self._determine_market_context(candles)
        
        # 6. Generate final signal
        if signals and confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            
            # Determine direction based on signals
            direction = self._determine_direction(signals, context)
            
            if direction != "NEUTRAL" and avg_confidence > 0.6:
                # Calculate entry, stop, target
                current_price = candles['Close'].iloc[-1]
                atr = self._calculate_atr(candles)
                
                entry, stop, target = self._calculate_levels(
                    direction, current_price, atr, signals
                )
                
                risk_reward = abs(target - entry) / abs(entry - stop)
                
                signal = SmartMoneySignal(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    confidence=avg_confidence,
                    context=context,
                    triggers=signals,
                    risk_reward=risk_reward,
                    position_size=0.0,  # Calculated by risk manager
                    validity_window=5  # 5 bars
                )
                
                self.signals_history.append(signal)
                return signal
        
        return SmartMoneySignal(
            symbol=symbol,
            direction="NEUTRAL",
            entry_price=candles['Close'].iloc[-1],
            stop_loss=0,
            take_profit=0,
            confidence=0,
            context=context,
            triggers=[],
            risk_reward=0,
            position_size=0,
            validity_window=0
        )
    
    def _analyze_vsa(self, candles: pd.DataFrame) -> Dict:
        """Analyze Volume Spread Analysis"""
        if len(candles) < 30:
            return {}
        
        recent = candles.iloc[-5:]
        avg_volume = candles['Volume'].rolling(20).mean().iloc[-1]
        
        triggers = []
        confidence = 0
        
        for i in range(1, len(recent)):
            current = recent.iloc[i]
            previous = recent.iloc[i-1]
            
            vsa_state = self.order_flow.analyze_bar(current, previous, avg_volume)
            if vsa_state:
                triggers.append(vsa_state.bar_type)
                confidence = max(confidence, vsa_state.confidence)
        
        # Check for Wyckoff accumulation
        wyckoff = self.order_flow.detect_wyckoff_accumulation(candles)
        if wyckoff:
            triggers.append(wyckoff['phase'])
            confidence = max(confidence, wyckoff['confidence'])
        
        return {'triggers': triggers, 'confidence': confidence} if triggers else {}
    
    def _determine_market_context(self, candles: pd.DataFrame) -> MarketContext:
        """Determine Wyckoff market context"""
        if len(candles) < 50:
            return MarketContext.BALANCE
        
        # Simplified context detection
        price_50 = candles['Close'].rolling(50).mean().iloc[-1]
        price_20 = candles['Close'].rolling(20).mean().iloc[-1]
        current = candles['Close'].iloc[-1]
        
        # Check for accumulation (sideways with volume spikes)
        recent_range = candles['High'].iloc[-20:].max() - candles['Low'].iloc[-20:].min()
        avg_range = (candles['High'] - candles['Low']).rolling(50).mean().iloc[-1]
        
        if recent_range < avg_range * 0.7:
            # Check volume patterns for accumulation
            volume_std = candles['Volume'].rolling(20).std().iloc[-1]
            volume_mean = candles['Volume'].rolling(20).mean().iloc[-1]
            
            if volume_std > volume_mean * 0.5:
                return MarketContext.ACCUMULATION
            else:
                return MarketContext.BALANCE
        elif current > price_50 and current > price_20:
            return MarketContext.MARKUP
        elif current < price_50 and current < price_20:
            return MarketContext.MARKDOWN
        else:
            return MarketContext.BALANCE
    
    def _determine_direction(self, signals: List[str], context: MarketContext) -> str:
        """Determine trade direction from signals"""
        direction_score = 0
        
        for signal in signals:
            if any(word in signal for word in ['spring', 'SOS', 'bullish', 'below_value', 'LONG']):
                direction_score += 1
            elif any(word in signal for word in ['upthrust', 'SOW', 'bearish', 'above_value', 'SHORT']):
                direction_score -= 1
        
        # Context weighting
        if context == MarketContext.ACCUMULATION:
            direction_score += 1
        elif context == MarketContext.MARKUP:
            direction_score += 2
        elif context == MarketContext.MARKDOWN:
            direction_score -= 2
        elif context == MarketContext.DISTRIBUTION:
            direction_score -= 1
        
        if direction_score >= 2:
            return "LONG"
        elif direction_score <= -2:
            return "SHORT"
        else:
            return "NEUTRAL"
    
    def _calculate_atr(self, candles: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = candles['High']
        low = candles['Low']
        close = candles['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else (high.iloc[-1] - low.iloc[-1])
    
    def _calculate_levels(self, direction: str, current_price: float, 
                         atr: float, signals: List[str]) -> Tuple[float, float, float]:
        """Calculate entry, stop, and target levels"""
        
        # Institutional entries often at retracements
        if direction == "LONG":
            entry = current_price - (atr * 0.382)  # Fibonacci retracement
            stop = entry - (atr * 1.5)
            target = entry + (atr * 3)  # 2:1 risk-reward
            
            # Adjust based on signals
            if 'liquidity_grab_LONG' in signals:
                # Enter more aggressively for liquidity grabs
                entry = current_price - (atr * 0.236)
        else:  # SHORT
            entry = current_price + (atr * 0.382)
            stop = entry + (atr * 1.5)
            target = entry - (atr * 3)
            
            if 'liquidity_grab_SHORT' in signals:
                entry = current_price + (atr * 0.236)
        
        return entry, stop, target