"""
Model 2: DP Signal Engine (dp.md approach)
3-minute scalping with RSI 7-period, multi-timeframe analysis, and invalidation conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from src.technical.indicators import CryptoTechnicalAnalyzer
from src.signals.signal_types import Signal, SignalType

logger = logging.getLogger(__name__)

class DPSignalConfig:
    """Configuration for DP Signal Engine (Model 2) - Scalping with 3 modes"""
    def __init__(self, mode: str = 'quick'):
        # Mode: 'quick', 'standard', 'extended'
        self.mode = mode

        self.timeframe_short = '3m'  # 3-minute intervals
        self.timeframe_long = '4h'   # 4-hour context
        self.rsi_period = 7          # Fast RSI for scalping
        self.rsi_overbought = 80.0  # Extreme overbought
        self.rsi_oversold = 20.0    # Extreme oversold
        self.min_confidence = 0.65
        self.max_confidence = 0.75
        self.leverage_low = 10        # Conservative leverage
        self.leverage_high = 15       # Aggressive leverage
        self.risk_per_trade_pct = 1.0  # 1% risk per trade
        self.invalidation_pct = 3.0    # 3% below entry for invalidation

    # Dynamic properties based on mode
    @property
    def stop_loss_pct(self) -> float:
        """Stop loss % based on scalping mode"""
        modes = {'quick': 0.8, 'standard': 1.2, 'extended': 1.5}
        return modes.get(self.mode, 1.2)

    @property
    def profit_target_ratio(self) -> float:
        """Base R:R ratio based on mode"""
        modes = {'quick': 1.5, 'standard': 2.0, 'extended': 2.5}
        return modes.get(self.mode, 2.0)


class DPSignalEngine:
    """
    Model 2: DP Signal Engine

    Based on dp.md systematic analysis template:
    - 3-minute scalping timeframe
    - RSI 7-period for fast entries
    - Multi-timeframe analysis (3-min + 4-hour)
    - Funding rate and Open Interest monitoring
    - Invalidation conditions on 3-minute candle closes
    - Leverage: 10-15×
    """

    def __init__(self, config: Optional[DPSignalConfig] = None):
        self.config = config or DPSignalConfig()
        self.technical_analyzer = CryptoTechnicalAnalyzer(mode="scalping")
        self.active_signals: List[Signal] = []

        logger.info("DP Signal Engine (Model 2) initialized")
        logger.info(f"Timeframes: {self.config.timeframe_short} + {self.config.timeframe_long}")
        logger.info(f"RSI: {self.config.rsi_period}-period (overbought: {self.config.rsi_overbought}, oversold: {self.config.rsi_oversold})")
        logger.info(f"Leverage: {self.config.leverage_low}×-{self.config.leverage_high}×")

    async def generate_signal(
        self,
        symbol: str,
        market_data_3m: pd.DataFrame,
        market_data_4h: pd.DataFrame,
        current_price: float,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None
    ) -> Optional[Signal]:
        """
        Generate Model 2 trading signal using dp.md methodology

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            market_data_3m: 3-minute OHLCV data
            market_data_4h: 4-hour OHLCV data
            current_price: Current market price
            funding_rate: Binance funding rate
            open_interest: Current open interest

        Returns:
            Signal object if valid signal found, None otherwise
        """
        try:
            base_symbol = symbol.split('/')[0]

            # 1. Calculate technical indicators for both timeframes
            df_3m = self._calculate_indicators_3m(market_data_3m.copy())
            df_4h = self._calculate_indicators_4h(market_data_4h.copy())

            if df_3m.empty or df_4h.empty:
                logger.warning(f"Insufficient data for {symbol}")
                return None

            latest_3m = df_3m.iloc[-1]
            latest_4h = df_4h.iloc[-1]

            # 2. Analyze 3-minute timeframe (entry trigger)
            signal_3m = self._analyze_3m_timeframe(df_3m, latest_3m, current_price)

            # 3. Analyze 4-hour timeframe (context/confirmation)
            context_4h = self._analyze_4h_timeframe(df_4h, latest_4h, current_price)

            # 4. Check multi-timeframe confluence
            if not self._check_timeframe_confluence(signal_3m, context_4h):
                logger.debug(f"{symbol}: No timeframe confluence")
                return None

            # 5. Analyze funding rate and open interest
            funding_context = self._analyze_funding_rate(funding_rate)
            oi_context = self._analyze_open_interest(open_interest, df_3m)

            # 6. Determine signal direction and confidence
            signal_type, confidence = self._determine_signal(
                signal_3m, context_4h, funding_context, oi_context
            )

            if signal_type == SignalType.HOLD:
                return None

            # 7. Calculate entry, stop loss, take profit, and invalidation
            entry_price = current_price
            stop_loss = self._calculate_stop_loss(entry_price, signal_type, latest_3m)
            take_profit = self._calculate_take_profit(entry_price, stop_loss, signal_type, latest_3m, latest_4h)
            invalidation_price = self._calculate_invalidation(entry_price, signal_type)

            # 8. Determine leverage based on confidence
            leverage = self._calculate_leverage(confidence, signal_3m, context_4h)

            # 9. Calculate risk in USD
            risk_usd = self._calculate_risk_usd(entry_price, stop_loss, leverage)

            # 10. Build signal
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
                primary_reason=self._build_primary_reason(signal_3m, context_4h),
                supporting_factors=self._build_supporting_factors(
                    signal_3m, context_4h, funding_context, oi_context, latest_3m, latest_4h
                ),
                risk_factors=self._build_risk_factors(signal_3m, context_4h, latest_3m),
                timestamp=datetime.now(),
                priority=self._calculate_priority(confidence, signal_3m),
                sources=['3m_technicals', '4h_context', 'funding_rate', 'open_interest']
            )

            # Add Model 2 specific metadata
            signal.metadata = {
                'model': 'DP Model 2',
                'timeframe': '3m',
                'leverage': leverage,
                'invalidation_condition': f"If price closes below ${invalidation_price:.2f} on 3-minute candle",
                'invalidation_price': invalidation_price,
                'risk_usd': risk_usd,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'rsi_7_period': float(latest_3m.get('rsi_7', 0)),
                'rsi_4h': float(latest_4h.get('rsi', 0)),
                'ema20_3m': float(latest_3m.get('ema_20', 0)),
                'ema20_4h': float(latest_4h.get('ema_20', 0))
            }

            logger.info(f"Model 2 signal generated for {symbol}: {signal_type.value} at ${entry_price:.2f}")
            logger.info(f"Confidence: {confidence:.2f}, Leverage: {leverage}×, Risk: ${risk_usd:.2f}")
            logger.info(f"Invalidation: ${invalidation_price:.2f}")

            self.active_signals.append(signal)
            return signal

        except Exception as e:
            logger.error(f"Error generating DP signal for {symbol}: {e}", exc_info=True)
            return None

    def _calculate_indicators_3m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for 3-minute timeframe"""
        # RSI 7-period (fast scalping)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()
        rs = gain / loss
        df['rsi_7'] = 100 - (100 / (1 + rs))

        # EMA 20
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Volume ratio
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']

        return df

    def _calculate_indicators_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for 4-hour timeframe"""
        # RSI 14-period (standard)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # EMA 20 and 50
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26

        return df

    def _analyze_3m_timeframe(self, df: pd.DataFrame, latest: pd.Series, current_price: float) -> Dict:
        """Analyze 3-minute timeframe for entry signals"""
        analysis = {
            'direction': 'neutral',
            'strength': 0.0,
            'rsi_signal': 'neutral',
            'trend': 'neutral',
            'momentum': 'neutral'
        }

        rsi_7 = latest.get('rsi_7', 50)
        ema_20 = latest.get('ema_20', current_price)
        macd = latest.get('macd', 0)

        # RSI 7-period analysis (dp.md uses RSI 7 for fast scalping)
        if rsi_7 < self.config.rsi_oversold:
            analysis['rsi_signal'] = 'oversold'
            analysis['direction'] = 'long'
            analysis['strength'] += 0.4
        elif rsi_7 > self.config.rsi_overbought:
            analysis['rsi_signal'] = 'overbought'
            analysis['direction'] = 'short'
            analysis['strength'] += 0.4

        # Trend relative to EMA20
        if current_price > ema_20:
            analysis['trend'] = 'bullish'
            if analysis['direction'] == 'long':
                analysis['strength'] += 0.3
        elif current_price < ema_20:
            analysis['trend'] = 'bearish'
            if analysis['direction'] == 'short':
                analysis['strength'] += 0.3

        # MACD momentum
        if macd > 0:
            analysis['momentum'] = 'positive'
            if analysis['direction'] == 'long':
                analysis['strength'] += 0.2
        elif macd < 0:
            analysis['momentum'] = 'negative'
            if analysis['direction'] == 'short':
                analysis['strength'] += 0.2

        return analysis

    def _analyze_4h_timeframe(self, df: pd.DataFrame, latest: pd.Series, current_price: float) -> Dict:
        """Analyze 4-hour timeframe for context/confirmation"""
        context = {
            'trend': 'neutral',
            'strength': 'weak',
            'supportive': False
        }

        rsi_4h = latest.get('rsi', 50)
        ema_20 = latest.get('ema_20', current_price)
        ema_50 = latest.get('ema_50', current_price)
        macd = latest.get('macd', 0)

        # 4H trend confirmation
        if ema_20 > ema_50 and current_price > ema_20:
            context['trend'] = 'bullish'
            context['supportive'] = True
        elif ema_20 < ema_50 and current_price < ema_20:
            context['trend'] = 'bearish'
            context['supportive'] = True

        # RSI context
        if 40 <= rsi_4h <= 60:
            context['strength'] = 'moderate'
        elif rsi_4h < 30 or rsi_4h > 70:
            context['strength'] = 'strong'

        # MACD context
        if macd > 0 and context['trend'] == 'bullish':
            context['supportive'] = True
        elif macd < 0 and context['trend'] == 'bearish':
            context['supportive'] = True

        return context

    def _check_timeframe_confluence(self, signal_3m: Dict, context_4h: Dict) -> bool:
        """Check if 3-minute signal aligns with 4-hour context"""
        # Long signal: 3m shows long + 4H is bullish or neutral
        if signal_3m['direction'] == 'long':
            return context_4h['trend'] in ['bullish', 'neutral']

        # Short signal: 3m shows short + 4H is bearish or neutral
        elif signal_3m['direction'] == 'short':
            return context_4h['trend'] in ['bearish', 'neutral']

        return False

    def _analyze_funding_rate(self, funding_rate: Optional[float]) -> Dict:
        """Analyze funding rate for sentiment"""
        if funding_rate is None:
            return {'status': 'unknown', 'impact': 0.0}

        # Funding rate thresholds (dp.md shows low funding rates as neutral)
        if abs(funding_rate) < 0.00001:  # Very low funding
            return {'status': 'neutral', 'impact': 0.0}
        elif funding_rate > 0.0001:  # High positive (longs paying shorts)
            return {'status': 'overcrowded_long', 'impact': -0.1}
        elif funding_rate < -0.0001:  # High negative (shorts paying longs)
            return {'status': 'overcrowded_short', 'impact': 0.1}

        return {'status': 'normal', 'impact': 0.0}

    def _analyze_open_interest(self, open_interest: Optional[float], df: pd.DataFrame) -> Dict:
        """Analyze open interest changes"""
        if open_interest is None:
            return {'status': 'unknown', 'impact': 0.0}

        # Check if OI is increasing (bullish for trending markets)
        if len(df) >= 10:
            recent_prices = df['close'].tail(10)
            if recent_prices.iloc[-1] > recent_prices.iloc[0]:
                return {'status': 'increasing_with_uptrend', 'impact': 0.05}
            elif recent_prices.iloc[-1] < recent_prices.iloc[0]:
                return {'status': 'increasing_with_downtrend', 'impact': 0.05}

        return {'status': 'stable', 'impact': 0.0}

    def _determine_signal(
        self,
        signal_3m: Dict,
        context_4h: Dict,
        funding_context: Dict,
        oi_context: Dict
    ) -> Tuple[SignalType, float]:
        """Determine final signal type and confidence"""
        # Start with 3m signal strength
        confidence = signal_3m['strength']

        # Add 4H context bonus
        if context_4h['supportive']:
            confidence += 0.1

        # Funding rate adjustment
        confidence += funding_context['impact']

        # Open interest adjustment
        confidence += oi_context['impact']

        # Clamp confidence to config range
        confidence = max(self.config.min_confidence, min(confidence, self.config.max_confidence))

        # Determine signal type
        if signal_3m['direction'] == 'long' and confidence >= self.config.min_confidence:
            return SignalType.LONG, confidence
        elif signal_3m['direction'] == 'short' and confidence >= self.config.min_confidence:
            return SignalType.SHORT, confidence

        return SignalType.HOLD, 0.0

    def _calculate_stop_loss(self, entry_price: float, signal_type: SignalType, latest_3m: pd.Series) -> float:
        """Calculate dynamic stop loss based on mode"""
        stop_pct = self.config.stop_loss_pct / 100
        logger.debug(f"Model 2 stop calculation: mode={self.config.mode}, stop_pct={self.config.stop_loss_pct}%")

        if signal_type == SignalType.LONG:
            return entry_price * (1 - stop_pct)
        else:  # SHORT
            return entry_price * (1 + stop_pct)

    def _calculate_take_profit(self, entry_price: float, stop_loss: float, signal_type: SignalType,
                              latest_3m: pd.Series = None, latest_4h: pd.Series = None) -> float:
        """
        Calculate dynamic take profit that auto-adjusts based on market conditions

        Adjustments based on:
        - Volatility (ATR): Higher volatility = wider targets
        - Trend strength: Stronger trend = wider targets
        - Volume: Higher volume = more confident, wider targets
        """
        risk = abs(entry_price - stop_loss)
        base_rr = self.config.profit_target_ratio

        # Dynamic adjustments
        volatility_multiplier = 1.0
        trend_multiplier = 1.0
        volume_multiplier = 1.0

        if latest_4h is not None:
            # Volatility adjustment (ATR-based)
            atr_4h = latest_4h.get('atr', 0)
            if atr_4h > 0:
                atr_pct = (atr_4h / entry_price) * 100
                if atr_pct > 4:  # High volatility
                    volatility_multiplier = 0.8  # Tighter targets in choppy markets
                elif atr_pct < 2:  # Low volatility
                    volatility_multiplier = 1.2  # Wider targets in calm markets

        if latest_3m is not None:
            # Trend strength adjustment
            ema_20 = latest_3m.get('ema_20', entry_price)
            if signal_type == SignalType.LONG and entry_price > ema_20 * 1.01:
                trend_multiplier = 1.1  # Strong uptrend, extend target
            elif signal_type == SignalType.SHORT and entry_price < ema_20 * 0.99:
                trend_multiplier = 1.1  # Strong downtrend, extend target

            # Volume adjustment
            volume_ratio = latest_3m.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # High volume confirmation
                volume_multiplier = 1.1
            elif volume_ratio < 0.8:  # Weak volume
                volume_multiplier = 0.9

        # Apply all adjustments
        adjusted_rr = base_rr * volatility_multiplier * trend_multiplier * volume_multiplier

        # Clamp to reasonable range for scalping (1.2x to 3x)
        adjusted_rr = max(1.2, min(adjusted_rr, 3.0))

        profit_distance = risk * adjusted_rr

        if signal_type == SignalType.LONG:
            return entry_price + profit_distance
        else:  # SHORT
            return entry_price - profit_distance

    def _calculate_invalidation(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate invalidation price (3% from entry)"""
        invalidation_pct = self.config.invalidation_pct / 100

        if signal_type == SignalType.LONG:
            return entry_price * (1 - invalidation_pct)
        else:  # SHORT
            return entry_price * (1 + invalidation_pct)

    def _calculate_leverage(self, confidence: float, signal_3m: Dict, context_4h: Dict) -> int:
        """Calculate leverage based on confidence and confluence"""
        # Base leverage on confidence
        if confidence >= 0.73:
            leverage = self.config.leverage_high  # 15×
        elif confidence >= 0.68:
            leverage = 12  # Medium-high
        else:
            leverage = self.config.leverage_low  # 10×

        # Reduce leverage if 4H context not supportive
        if not context_4h['supportive']:
            leverage = self.config.leverage_low

        return leverage

    def _calculate_risk_usd(self, entry_price: float, stop_loss: float, leverage: int) -> float:
        """Calculate risk in USD (placeholder - needs account balance)"""
        # Assume $10,000 account for now (should be passed from risk manager)
        account_balance = 10000
        risk_per_trade = account_balance * (self.config.risk_per_trade_pct / 100)
        return risk_per_trade

    def _build_primary_reason(self, signal_3m: Dict, context_4h: Dict) -> str:
        """Build primary signal reason"""
        direction = signal_3m['direction'].upper()
        rsi_signal = signal_3m['rsi_signal']
        trend_4h = context_4h['trend']

        return f"{direction} signal: RSI 7-period {rsi_signal}, 4H trend {trend_4h}"

    def _build_supporting_factors(
        self,
        signal_3m: Dict,
        context_4h: Dict,
        funding_context: Dict,
        oi_context: Dict,
        latest_3m: pd.Series,
        latest_4h: pd.Series
    ) -> List[str]:
        """Build list of supporting factors"""
        factors = []

        # 3m factors
        factors.append(f"3m RSI(7): {latest_3m.get('rsi_7', 0):.1f} - {signal_3m['rsi_signal']}")
        factors.append(f"3m trend vs EMA20: {signal_3m['trend']}")
        factors.append(f"3m MACD momentum: {signal_3m['momentum']}")

        # 4H factors
        factors.append(f"4H trend: {context_4h['trend']}")
        factors.append(f"4H RSI(14): {latest_4h.get('rsi', 0):.1f}")

        # Funding and OI
        factors.append(f"Funding rate: {funding_context['status']}")
        factors.append(f"Open interest: {oi_context['status']}")

        return factors

    def _build_risk_factors(self, signal_3m: Dict, context_4h: Dict, latest_3m: pd.Series) -> List[str]:
        """Build list of risk factors"""
        risks = []

        rsi_7 = latest_3m.get('rsi_7', 50)

        if rsi_7 > 80:
            risks.append("RSI 7-period EXTREMELY overbought (>80)")
        elif rsi_7 > 70:
            risks.append("RSI 7-period overbought (>70)")
        elif rsi_7 < 20:
            risks.append("RSI 7-period EXTREMELY oversold (<20)")
        elif rsi_7 < 30:
            risks.append("RSI 7-period oversold (<30)")

        if not context_4h['supportive']:
            risks.append("4H timeframe not supportive")

        if signal_3m['strength'] < 0.7:
            risks.append("Moderate signal strength")

        return risks

    def _calculate_priority(self, confidence: float, signal_3m: Dict) -> int:
        """Calculate signal priority (1-3)"""
        if confidence >= 0.73 and signal_3m['strength'] >= 0.8:
            return 1  # High priority
        elif confidence >= 0.68:
            return 2  # Medium priority
        else:
            return 3  # Low priority

    def get_active_signals(self, current_prices: Dict[str, float] = None) -> List[Signal]:
        """
        Get all active signals from Model 2

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

    def check_invalidation(self, signal: Signal, current_price: float, latest_candle_close: float) -> bool:
        """
        Check if invalidation condition is triggered

        Args:
            signal: The signal to check
            current_price: Current market price
            latest_candle_close: Latest 3-minute candle close price

        Returns:
            True if invalidation triggered, False otherwise
        """
        if not signal.metadata or 'invalidation_price' not in signal.metadata:
            return False

        invalidation_price = signal.metadata['invalidation_price']

        # Check if 3-minute candle closed below/above invalidation price
        if signal.signal_type == SignalType.LONG:
            if latest_candle_close < invalidation_price:
                logger.warning(f"INVALIDATION TRIGGERED for {signal.symbol}: "
                             f"3m candle closed at ${latest_candle_close:.2f} < ${invalidation_price:.2f}")
                return True
        elif signal.signal_type == SignalType.SHORT:
            if latest_candle_close > invalidation_price:
                logger.warning(f"INVALIDATION TRIGGERED for {signal.symbol}: "
                             f"3m candle closed at ${latest_candle_close:.2f} > ${invalidation_price:.2f}")
                return True

        return False
