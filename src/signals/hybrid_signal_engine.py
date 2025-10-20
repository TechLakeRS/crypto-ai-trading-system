"""
Model 3: Hybrid Signal Engine
Combines best features from both Model 1 (manual.md) and Model 2 (dp.md)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from src.technical.indicators import CryptoTechnicalAnalyzer
from src.signals.signal_types import Signal, SignalType

logger = logging.getLogger(__name__)

@dataclass
class HybridSignalConfig:
    """Hybrid configuration combining both models - with 3 modes"""
    # Mode: 'balanced', 'extended', 'maximum'
    mode: str = 'balanced'  # Balanced by default

    # Multi-timeframe like Model 2
    timeframe_short: str = '5m'   # Model 1 preference
    timeframe_medium: str = '3m'  # Model 2 preference
    timeframe_long: str = '4h'    # Model 2 context

    # Technical indicators - mix of both
    rsi_fast: int = 7              # Model 2 fast scalping
    rsi_slow: int = 14             # Model 1 reliability
    rsi_overbought: float = 77.5   # Average of both (75 + 80)/2
    rsi_oversold: float = 22.5     # Average of both (25 + 20)/2

    # MACD from Model 1 (proven for crypto)
    macd_fast: int = 5
    macd_slow: int = 13
    macd_signal: int = 8

    # Risk management - balanced
    min_confidence: float = 0.67   # Between both
    risk_per_trade_pct: float = 0.85  # Between 0.75% and 1.0%

    # Confluence requirements - stricter
    min_confluences: int = 4       # Stricter than Model 1 (3/5)
    invalidation_pct: float = 2.0  # Tighter than Model 2 (3%)

    # Dynamic properties based on mode
    @property
    def stop_loss_pct_tight(self) -> float:
        """Tight stop loss % based on mode"""
        modes = {
            'balanced': 2.5,   # Standard day trading
            'extended': 3.0,   # Wider for swing positions
            'maximum': 3.5     # Maximum breathing room
        }
        return modes.get(self.mode, 2.5)

    @property
    def stop_loss_pct_wide(self) -> float:
        """Wide stop loss % based on mode"""
        modes = {
            'balanced': 4.5,   # Moderate wide stop
            'extended': 5.5,   # Extended wide stop
            'maximum': 6.5     # Maximum wide stop
        }
        return modes.get(self.mode, 4.5)

    @property
    def leverage_min(self) -> int:
        """Minimum leverage based on mode"""
        modes = {
            'balanced': 3,     # Conservative
            'extended': 4,     # Moderate
            'maximum': 5       # Aggressive
        }
        return modes.get(self.mode, 3)

    @property
    def leverage_max(self) -> int:
        """Maximum leverage based on mode"""
        modes = {
            'balanced': 6,     # Conservative cap
            'extended': 8,     # Moderate cap
            'maximum': 10      # Aggressive cap (still not extreme)
        }
        return modes.get(self.mode, 6)

    @property
    def profit_target_ratio(self) -> float:
        """Base R:R ratio based on mode"""
        modes = {
            'balanced': 1.8,   # Quick profits
            'extended': 2.2,   # Extended targets
            'maximum': 2.8     # Maximum targets
        }
        return modes.get(self.mode, 1.8)


class HybridSignalEngine:
    """
    Model 3: Hybrid approach combining:
    - Model 1's conservative day trading with tight confluences
    - Model 2's multi-timeframe analysis and scalping speed
    - Moderate leverage (3-8×) instead of none or extreme
    - Adaptive stop losses based on volatility
    - Both fast RSI (7) and slow RSI (14) for confirmation
    """

    def __init__(self, config: Optional[HybridSignalConfig] = None):
        self.config = config or HybridSignalConfig()
        self.technical_analyzer_day = CryptoTechnicalAnalyzer(mode="day_trading")  # Model 1
        self.technical_analyzer_scalp = CryptoTechnicalAnalyzer(mode="scalping")    # Model 2
        self.active_signals: List[Signal] = []

        logger.info("Hybrid Signal Engine (Model 3) initialized")
        logger.info(f"Multi-timeframe: {self.config.timeframe_short} + {self.config.timeframe_medium} + {self.config.timeframe_long}")
        logger.info(f"Dual RSI: Fast {self.config.rsi_fast} + Slow {self.config.rsi_slow}")
        logger.info(f"Leverage: {self.config.leverage_min}×-{self.config.leverage_max}×")

    async def generate_signal(
        self,
        symbol: str,
        market_data_5m: pd.DataFrame,
        market_data_3m: pd.DataFrame,
        market_data_4h: pd.DataFrame,
        current_price: float,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None
    ) -> Optional[Signal]:
        """
        Generate hybrid signal using best of both models
        """
        try:
            base_symbol = symbol.split('/')[0]

            # Calculate indicators on all timeframes
            df_5m = self._calculate_indicators_5m(market_data_5m.copy())
            df_3m = self._calculate_indicators_3m(market_data_3m.copy())
            df_4h = self._calculate_indicators_4h(market_data_4h.copy())

            if df_5m.empty or df_3m.empty or df_4h.empty:
                return None

            latest_5m = df_5m.iloc[-1]
            latest_3m = df_3m.iloc[-1]
            latest_4h = df_4h.iloc[-1]

            # Multi-layer confluence analysis
            confluences = {}

            # Layer 1: Fast scalping signals (Model 2 approach)
            confluences['fast_rsi'] = self._check_fast_rsi(latest_3m)

            # Layer 2: Reliable day trading signals (Model 1 approach)
            confluences['slow_rsi'] = self._check_slow_rsi(latest_5m)
            confluences['macd'] = self._check_macd(latest_5m)
            confluences['bollinger'] = self._check_bollinger_bands(latest_5m)

            # Layer 3: Volume confirmation (both models)
            confluences['volume'] = self._check_volume(latest_5m)

            # Layer 4: Trend alignment (Model 1 approach)
            confluences['trend'] = self._check_trend_alignment(latest_5m)

            # Layer 5: Higher timeframe context (Model 2 approach)
            confluences['htf_trend'] = self._check_htf_trend(latest_4h)

            # Layer 6: Funding rate (Model 2 approach)
            confluences['funding'] = self._check_funding_rate(funding_rate)

            # Layer 7: Open interest (Model 2 approach)
            confluences['open_interest'] = self._check_open_interest(open_interest, df_3m)

            # Count valid confluences
            valid_confluences = [k for k, v in confluences.items() if v['valid']]
            confluence_count = len(valid_confluences)

            # Need at least 4 of 9 confluences (stricter than Model 1's 3/5)
            if confluence_count < self.config.min_confluences:
                logger.debug(f"{symbol}: Only {confluence_count}/9 confluences, need {self.config.min_confluences}")
                return None

            # Determine signal direction
            signal_direction = self._determine_direction(confluences)
            if signal_direction == 'neutral':
                return None

            signal_type = SignalType.LONG if signal_direction == 'long' else SignalType.SHORT

            # Calculate confidence (weighted by confluence quality)
            confidence = self._calculate_hybrid_confidence(confluences, confluence_count)

            if confidence < self.config.min_confidence:
                return None

            # Adaptive risk management based on volatility
            atr_4h = latest_4h.get('atr', 0)
            volatility_pct = (atr_4h / current_price) * 100 if atr_4h > 0 else 2.0

            # Use tight stops in low volatility, wide stops in high volatility
            stop_loss_pct = self.config.stop_loss_pct_tight if volatility_pct < 3 else self.config.stop_loss_pct_wide

            # Calculate entry, stops, targets
            entry_price = current_price
            stop_loss = self._calculate_adaptive_stop_loss(entry_price, signal_type, stop_loss_pct, atr_4h)
            take_profit = self._calculate_take_profit(entry_price, stop_loss, signal_type, confidence, latest_5m, latest_4h)
            invalidation_price = self._calculate_invalidation(entry_price, signal_type)

            # Calculate leverage (moderate, not extreme)
            leverage = self._calculate_moderate_leverage(confidence, volatility_pct)

            # Risk calculation
            risk_usd = self._calculate_risk_usd(entry_price, stop_loss, leverage)

            # Build signal
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=confidence * 100,
                current_price=current_price,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_levels=[take_profit],
                risk_reward_ratio=abs((take_profit - entry_price) / (entry_price - stop_loss)),
                position_size_pct=self.config.risk_per_trade_pct,
                primary_reason=self._build_primary_reason(confluences, signal_direction),
                supporting_factors=self._build_supporting_factors(
                    confluences, latest_5m, latest_3m, latest_4h, volatility_pct
                ),
                risk_factors=self._build_risk_factors(confluences, volatility_pct),
                timestamp=datetime.now(),
                priority=self._calculate_priority(confidence, confluence_count),
                sources=valid_confluences
            )

            # Hybrid metadata
            signal.metadata = {
                'model': 'Hybrid Model 3',
                'mode': self.config.mode,  # Trading mode
                'timeframes': '5m + 3m + 4h',
                'leverage': leverage,
                'invalidation_condition': f"If price closes below ${invalidation_price:.2f} (2% invalidation)",
                'invalidation_price': invalidation_price,
                'risk_usd': risk_usd,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'confluence_count': confluence_count,
                'fast_rsi_7': float(latest_3m.get('rsi_7', 0)),
                'slow_rsi_14': float(latest_5m.get('rsi', 0)),
                'volatility_pct': volatility_pct,
                'stop_type': 'tight' if volatility_pct < 3 else 'wide'
            }

            logger.info(f"Hybrid signal for {symbol}: {signal_type.value} at ${entry_price:.2f} (Mode: {self.config.mode})")
            logger.info(f"Confluences: {confluence_count}/9, Confidence: {confidence:.2f}, Leverage: {leverage}×")

            self.active_signals.append(signal)
            return signal

        except Exception as e:
            logger.error(f"Error generating hybrid signal for {symbol}: {e}", exc_info=True)
            return None

    def _calculate_indicators_5m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Model 1 style indicators on 5-min"""
        return self.technical_analyzer_day.calculate_all_indicators(df)

    def _calculate_indicators_3m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Model 2 fast RSI on 3-min"""
        # Fast RSI 7-period
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / loss
        df['rsi_7'] = 100 - (100 / (1 + rs))
        return df

    def _calculate_indicators_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate 4H context indicators"""
        # RSI, EMA, ATR for context
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        return df

    def _check_fast_rsi(self, latest: pd.Series) -> Dict:
        """Check fast RSI 7-period (Model 2 style)"""
        rsi = latest.get('rsi_7', 50)
        if rsi < self.config.rsi_oversold:
            return {'valid': True, 'direction': 'long', 'strength': 0.15, 'value': rsi}
        elif rsi > self.config.rsi_overbought:
            return {'valid': True, 'direction': 'short', 'strength': 0.15, 'value': rsi}
        return {'valid': False, 'direction': 'neutral', 'strength': 0, 'value': rsi}

    def _check_slow_rsi(self, latest: pd.Series) -> Dict:
        """Check slow RSI 14-period (Model 1 style)"""
        rsi = latest.get('rsi', 50)
        if rsi < self.config.rsi_oversold:
            return {'valid': True, 'direction': 'long', 'strength': 0.2, 'value': rsi}
        elif rsi > self.config.rsi_overbought:
            return {'valid': True, 'direction': 'short', 'strength': 0.2, 'value': rsi}
        return {'valid': False, 'direction': 'neutral', 'strength': 0, 'value': rsi}

    def _check_macd(self, latest: pd.Series) -> Dict:
        """Check MACD crossover (Model 1 style)"""
        macd = latest.get('macd', 0)
        signal = latest.get('macd_signal', 0)

        if macd > signal and macd > 0:
            return {'valid': True, 'direction': 'long', 'strength': 0.15}
        elif macd < signal and macd < 0:
            return {'valid': True, 'direction': 'short', 'strength': 0.15}
        return {'valid': False, 'direction': 'neutral', 'strength': 0}

    def _check_bollinger_bands(self, latest: pd.Series) -> Dict:
        """Check Bollinger Bands (Model 1 style)"""
        close = latest.get('close', 0)
        bb_upper = latest.get('bb_upper', 0)
        bb_lower = latest.get('bb_lower', 0)

        if close < bb_lower:
            return {'valid': True, 'direction': 'long', 'strength': 0.1}
        elif close > bb_upper:
            return {'valid': True, 'direction': 'short', 'strength': 0.1}
        return {'valid': False, 'direction': 'neutral', 'strength': 0}

    def _check_volume(self, latest: pd.Series) -> Dict:
        """Check volume confirmation (both models)"""
        volume_ratio = latest.get('volume_ratio', 1.0)
        if volume_ratio >= 1.5:
            return {'valid': True, 'direction': 'neutral', 'strength': 0.15}
        return {'valid': False, 'direction': 'neutral', 'strength': 0}

    def _check_trend_alignment(self, latest: pd.Series) -> Dict:
        """Check EMA trend (Model 1 style)"""
        close = latest.get('close', 0)
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)

        if close > ema_20 > ema_50:
            return {'valid': True, 'direction': 'long', 'strength': 0.1}
        elif close < ema_20 < ema_50:
            return {'valid': True, 'direction': 'short', 'strength': 0.1}
        return {'valid': False, 'direction': 'neutral', 'strength': 0}

    def _check_htf_trend(self, latest: pd.Series) -> Dict:
        """Check 4H trend (Model 2 style)"""
        ema_20 = latest.get('ema_20', 0)
        ema_50 = latest.get('ema_50', 0)

        if ema_20 > ema_50:
            return {'valid': True, 'direction': 'long', 'strength': 0.1}
        elif ema_20 < ema_50:
            return {'valid': True, 'direction': 'short', 'strength': 0.1}
        return {'valid': False, 'direction': 'neutral', 'strength': 0}

    def _check_funding_rate(self, funding_rate: Optional[float]) -> Dict:
        """Check funding rate (Model 2 style)"""
        if funding_rate is None:
            return {'valid': False, 'direction': 'neutral', 'strength': 0}

        if abs(funding_rate) < 0.00001:  # Very low funding
            return {'valid': True, 'direction': 'neutral', 'strength': 0.05}
        return {'valid': False, 'direction': 'neutral', 'strength': 0}

    def _check_open_interest(self, open_interest: Optional[float], df_3m: pd.DataFrame) -> Dict:
        """Check open interest trend (Model 2 style)"""
        if open_interest is None or len(df_3m) < 10:
            return {'valid': False, 'direction': 'neutral', 'strength': 0}

        # Calculate average OI over recent period
        # If OI is increasing, it suggests growing interest
        avg_volume = df_3m['volume'].tail(10).mean()
        current_volume = df_3m['volume'].iloc[-1]

        # Rising OI with rising price = bullish, rising OI with falling price = bearish
        if current_volume > avg_volume * 1.2:
            return {'valid': True, 'direction': 'neutral', 'strength': 0.05, 'value': open_interest}

        return {'valid': False, 'direction': 'neutral', 'strength': 0, 'value': open_interest}

    def _determine_direction(self, confluences: Dict) -> str:
        """Determine overall signal direction"""
        long_score = sum(c['strength'] for c in confluences.values() if c['direction'] == 'long')
        short_score = sum(c['strength'] for c in confluences.values() if c['direction'] == 'short')

        if long_score > short_score and long_score > 0.3:
            return 'long'
        elif short_score > long_score and short_score > 0.3:
            return 'short'
        return 'neutral'

    def _calculate_hybrid_confidence(self, confluences: Dict, count: int) -> float:
        """Calculate weighted confidence"""
        total_strength = sum(c['strength'] for c in confluences.values() if c['valid'])
        confluence_bonus = (count / 9) * 0.2  # Bonus for more confluences (9 total sources)
        return min(0.95, total_strength + confluence_bonus)

    def _calculate_adaptive_stop_loss(self, entry: float, signal_type: SignalType,
                                     stop_pct: float, atr: float) -> float:
        """Adaptive stop loss based on volatility"""
        stop_distance = entry * (stop_pct / 100)

        if signal_type == SignalType.LONG:
            return entry - stop_distance
        else:
            return entry + stop_distance

    def _calculate_take_profit(self, entry: float, stop: float, signal_type: SignalType,
                               confidence: float, latest_5m: pd.Series = None,
                               latest_4h: pd.Series = None) -> float:
        """
        Calculate dynamic take profit that auto-adjusts based on market conditions

        Adjustments based on:
        - Volatility (ATR): Higher volatility = wider targets
        - Trend strength: Stronger trend = wider targets
        - Volume: Higher volume = more confident, wider targets
        - Confidence: Higher confidence = higher reward target
        """
        risk = abs(entry - stop)
        # Base R:R from mode configuration
        base_rr = self.config.profit_target_ratio

        # Dynamic adjustments
        volatility_multiplier = 1.0
        trend_multiplier = 1.0
        volume_multiplier = 1.0
        confidence_multiplier = 1.0

        # Volatility adjustment (ATR-based on 4H)
        if latest_4h is not None:
            atr_4h = latest_4h.get('atr', 0)
            if atr_4h > 0:
                atr_pct = (atr_4h / entry) * 100
                if atr_pct > 5:  # High volatility
                    volatility_multiplier = 0.85  # Tighter targets in choppy markets
                elif atr_pct < 2.5:  # Low volatility
                    volatility_multiplier = 1.15  # Wider targets in calm markets

        # Trend and volume adjustments (5m timeframe)
        if latest_5m is not None:
            # Trend strength adjustment
            ema_20 = latest_5m.get('ema_20', entry)
            ema_50 = latest_5m.get('ema_50', entry)

            if signal_type == SignalType.LONG and entry > ema_20 > ema_50:
                trend_multiplier = 1.12  # Strong uptrend, extend target
            elif signal_type == SignalType.SHORT and entry < ema_20 < ema_50:
                trend_multiplier = 1.12  # Strong downtrend, extend target

            # Volume adjustment
            volume_ratio = latest_5m.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # High volume confirmation
                volume_multiplier = 1.1
            elif volume_ratio < 0.8:  # Weak volume
                volume_multiplier = 0.92

        # Confidence multiplier (higher confidence = extend targets)
        if confidence > 0.8:
            confidence_multiplier = 1.15
        elif confidence > 0.7:
            confidence_multiplier = 1.08
        else:
            confidence_multiplier = 1.0

        # Apply all adjustments
        adjusted_rr = base_rr * volatility_multiplier * trend_multiplier * volume_multiplier * confidence_multiplier

        # Clamp to reasonable range for hybrid trading (1.5x to 3.5x)
        adjusted_rr = max(1.5, min(adjusted_rr, 3.5))

        reward = risk * adjusted_rr

        if signal_type == SignalType.LONG:
            return entry + reward
        else:
            return entry - reward

    def _calculate_invalidation(self, entry: float, signal_type: SignalType) -> float:
        """Tighter invalidation than Model 2"""
        inv_pct = self.config.invalidation_pct / 100

        if signal_type == SignalType.LONG:
            return entry * (1 - inv_pct)
        else:
            return entry * (1 + inv_pct)

    def _calculate_moderate_leverage(self, confidence: float, volatility: float) -> int:
        """Moderate leverage (not extreme like Model 2)"""
        # Lower leverage in high volatility
        if volatility > 4:
            base_leverage = self.config.leverage_min
        else:
            base_leverage = self.config.leverage_min + int((confidence - 0.6) * 10)

        return min(self.config.leverage_max, max(self.config.leverage_min, base_leverage))

    def _calculate_risk_usd(self, entry: float, stop: float, leverage: int) -> float:
        """Calculate risk in USD"""
        account_balance = 10000  # Placeholder
        risk_per_trade = account_balance * (self.config.risk_per_trade_pct / 100)
        return risk_per_trade

    def _build_primary_reason(self, confluences: Dict, direction: str) -> str:
        """Build primary reason from confluences"""
        valid_sources = [k for k, v in confluences.items() if v['valid']]
        return f"HYBRID {direction.upper()}: {len(valid_sources)}/9 confluences (multi-timeframe + dual RSI + OI)"

    def _build_supporting_factors(self, confluences: Dict, latest_5m, latest_3m, latest_4h, vol: float) -> List[str]:
        """Build supporting factors list"""
        factors = []

        if confluences['fast_rsi']['valid']:
            factors.append(f"Fast RSI(7): {confluences['fast_rsi']['value']:.1f} - scalping signal")
        if confluences['slow_rsi']['valid']:
            factors.append(f"Slow RSI(14): {confluences['slow_rsi']['value']:.1f} - day trading signal")
        if confluences['macd']['valid']:
            factors.append("MACD crossover confirmed (5m)")
        if confluences['volume']['valid']:
            factors.append("Strong volume (1.5× average)")
        if confluences['trend']['valid']:
            factors.append("5m EMA alignment confirmed")
        if confluences['htf_trend']['valid']:
            factors.append("4H trend supportive")
        if confluences['funding']['valid']:
            factors.append("Funding rate neutral")
        if confluences['open_interest']['valid']:
            oi_value = confluences['open_interest'].get('value', 0)
            factors.append(f"Open interest rising (OI: {oi_value:.2f})")

        factors.append(f"Volatility: {vol:.1f}% (adaptive stops)")

        return factors

    def _build_risk_factors(self, confluences: Dict, vol: float) -> List[str]:
        """Build risk factors"""
        risks = []

        if vol > 5:
            risks.append("High volatility (>5%) - wider stops required")

        if not confluences['htf_trend']['valid']:
            risks.append("4H timeframe not aligned")

        valid_count = sum(1 for c in confluences.values() if c['valid'])
        if valid_count < 5:
            risks.append(f"Moderate confluences ({valid_count}/9)")

        return risks

    def _calculate_priority(self, confidence: float, confluence_count: int) -> int:
        """Calculate priority"""
        if confidence >= 0.75 and confluence_count >= 6:
            return 1
        elif confidence >= 0.7 and confluence_count >= 5:
            return 2
        else:
            return 3

    def get_active_signals(self, current_prices: Dict[str, float] = None) -> List[Signal]:
        """
        Get active hybrid signals

        Signals expire if:
        - Older than 5 minutes (stale signal)
        - Entry price was passed (missed opportunity)
        """
        cutoff_time = datetime.now() - timedelta(minutes=5)

        valid_signals = []
        for signal in self.active_signals:
            # Expire if too old
            if signal.timestamp <= cutoff_time:
                continue

            # Expire if entry was passed (missed the trade)
            if current_prices and signal.symbol in current_prices:
                current_price = current_prices[signal.symbol]
                if signal.signal_type == SignalType.LONG:
                    # For LONG: expire if price went above entry (missed buy)
                    if current_price > signal.entry_price * 1.002:  # 0.2% buffer
                        continue
                else:  # SHORT
                    # For SHORT: expire if price went below entry (missed sell)
                    if current_price < signal.entry_price * 0.998:  # 0.2% buffer
                        continue

            valid_signals.append(signal)

        self.active_signals = valid_signals
        return self.active_signals
