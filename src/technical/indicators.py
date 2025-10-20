"""
Technical Analysis Engine with Crypto-Specific Indicators
Based on strategies from both markdown files
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import signal
from scipy.stats import linregress

logger = logging.getLogger(__name__)

@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    indicator: str
    timeframe: str
    signal_type: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0 to 1
    price_level: float
    confidence: float
    reasoning: str

class CryptoTechnicalAnalyzer:
    """Technical analysis engine optimized for cryptocurrency markets"""

    def __init__(self, mode: str = "scalping"):
        """
        Initialize analyzer with specific mode
        Modes: 'scalping', 'day_trading', 'swing', 'position'
        """
        self.mode = mode
        self.configure_for_mode()

    def configure_for_mode(self):
        """Configure indicators based on trading mode (from markdown strategies)"""
        if self.mode == "scalping":
            # 1-5 minute timeframe settings
            self.config = {
                'rsi': {'period': 5, 'overbought': 85, 'oversold': 15},
                'macd': {'fast': 3, 'slow': 10, 'signal': 16},
                'bb': {'period': 20, 'std': 2.5},
                'ema': [9, 20],
                'atr': {'period': 14},
                'volume_ma': 20,
                'stoch': {'period': 5, 'smooth': 3}
            }
        elif self.mode == "day_trading":
            # 5-15 minute timeframe settings
            self.config = {
                'rsi': {'period': 14, 'overbought': 75, 'oversold': 25},
                'macd': {'fast': 5, 'slow': 35, 'signal': 5},
                'bb': {'period': 20, 'std': 2.5},
                'ema': [20, 50],
                'atr': {'period': 14},
                'volume_ma': 20,
                'stoch': {'period': 14, 'smooth': 3}
            }
        elif self.mode == "swing":
            # 15min-1hr timeframe settings
            self.config = {
                'rsi': {'period': 21, 'overbought': 70, 'oversold': 30},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bb': {'period': 20, 'std': 2.0},
                'ema': [50, 200],
                'atr': {'period': 14},
                'volume_ma': 50,
                'stoch': {'period': 14, 'smooth': 5}
            }
        else:  # position
            # Multi-hour/day settings
            self.config = {
                'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bb': {'period': 20, 'std': 2.0},
                'ema': [50, 200],
                'atr': {'period': 20},
                'volume_ma': 50,
                'stoch': {'period': 21, 'smooth': 7}
            }

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators"""
        df = df.copy()

        # Basic price features
        df = self._calculate_price_features(df)

        # Trend indicators
        df = self._calculate_trend_indicators(df)

        # Momentum indicators
        df = self._calculate_momentum_indicators(df)

        # Volatility indicators
        df = self._calculate_volatility_indicators(df)

        # Volume indicators
        df = self._calculate_volume_indicators(df)

        # Custom crypto indicators
        df = self._calculate_crypto_specific_indicators(df)

        # Pattern detection
        df = self._detect_patterns(df)

        return df

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price features"""
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price ranges
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_to_high'] = (df['high'] - df['close']) / df['high']
        df['close_to_low'] = (df['close'] - df['low']) / df['low']

        # Typical price (HLC average)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Weighted close
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4

        return df

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend following indicators"""
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        for period in self.config['ema']:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # MACD - Modified for crypto volatility
        macd_config = self.config['macd']
        exp1 = df['close'].ewm(span=macd_config['fast'], adjust=False).mean()
        exp2 = df['close'].ewm(span=macd_config['slow'], adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=macd_config['signal'], adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Parabolic SAR
        psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
        df['psar'] = psar.psar()
        df['psar_up'] = psar.psar_up()
        df['psar_down'] = psar.psar_down()

        # Ichimoku Cloud (important for crypto)
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()

        # Trend strength
        df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['close']

        return df

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        # RSI with crypto-optimized levels
        rsi_config = self.config['rsi']
        df['rsi'] = ta.momentum.RSIIndicator(
            df['close'], window=rsi_config['period']
        ).rsi()

        # Stochastic Oscillator
        stoch_config = self.config['stoch']
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'],
            window=stoch_config['period'],
            smooth_window=stoch_config['smooth']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close']
        ).williams_r()

        # Rate of Change (ROC)
        df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()

        # Commodity Channel Index (CCI)
        df['cci'] = ta.trend.CCIIndicator(
            df['high'], df['low'], df['close'], window=20
        ).cci()

        # Money Flow Index (MFI)
        df['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume'], window=14
        ).money_flow_index()

        # Custom Momentum Score for Crypto
        df['momentum_score'] = (
            (df['rsi'] / 100) * 0.3 +
            ((df['stoch_k'] / 100) * 0.2) +
            ((df['mfi'] / 100) * 0.2) +
            ((df['roc'] > 0).astype(int) * 0.3)
        )

        return df

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        # Bollinger Bands with crypto-adjusted settings
        bb_config = self.config['bb']
        bollinger = ta.volatility.BollingerBands(
            df['close'],
            window=bb_config['period'],
            window_dev=bb_config['std']
        )
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / df['bb_width']

        # Average True Range (ATR)
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'],
            window=self.config['atr']['period']
        ).average_true_range()

        # ATR percentage (crypto-specific)
        df['atr_percent'] = (df['atr'] / df['close']) * 100

        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(
            df['high'], df['low'], df['close'], window=20
        )
        df['kc_upper'] = keltner.keltner_channel_hband()
        df['kc_lower'] = keltner.keltner_channel_lband()
        df['kc_middle'] = keltner.keltner_channel_mband()

        # Donchian Channels
        donchian = ta.volatility.DonchianChannel(
            df['high'], df['low'], df['close'], window=20
        )
        df['dc_upper'] = donchian.donchian_channel_hband()
        df['dc_lower'] = donchian.donchian_channel_lband()

        # Volatility ratio
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(window=50).mean()

        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators"""
        # Volume moving average
        df['volume_sma'] = df['volume'].rolling(window=self.config['volume_ma']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        # On Balance Volume (OBV)
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()

        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = (df['volume'] * df['typical_price']).cumsum() / df['volume'].cumsum()

        # Chaikin Money Flow
        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()

        # Force Index
        df['force_index'] = ta.volume.ForceIndexIndicator(
            df['close'], df['volume']
        ).force_index()

        # Volume Price Trend (VPT)
        df['vpt'] = ta.volume.VolumePriceTrendIndicator(
            df['close'], df['volume']
        ).volume_price_trend()

        # Accumulation/Distribution Line
        df['ad_line'] = ta.volume.AccDistIndexIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).acc_dist_index()

        return df

    def _calculate_crypto_specific_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cryptocurrency-specific indicators"""
        # Fibonacci retracement levels
        recent_high = df['high'].rolling(window=50).max()
        recent_low = df['low'].rolling(window=50).min()
        diff = recent_high - recent_low

        df['fib_0'] = recent_low
        df['fib_236'] = recent_low + 0.236 * diff
        df['fib_382'] = recent_low + 0.382 * diff
        df['fib_500'] = recent_low + 0.5 * diff
        df['fib_618'] = recent_low + 0.618 * diff
        df['fib_786'] = recent_low + 0.786 * diff
        df['fib_1'] = recent_high

        # Distance from Fibonacci levels
        for level in ['fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786']:
            df[f'{level}_distance'] = abs(df['close'] - df[level]) / df['close']

        # Support and Resistance levels
        df = self._calculate_support_resistance(df)

        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & \
                           (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & \
                         (df['low'].shift(1) < df['low'].shift(2))

        # Crypto Fear & Greed proxy (based on multiple indicators)
        df['technical_fear_greed'] = (
            (df['rsi'] / 100) * 0.25 +
            df['bb_percent'].clip(0, 1) * 0.25 +
            ((df['volume_ratio'] - 1).clip(-1, 1) + 1) / 2 * 0.25 +
            df['momentum_score'] * 0.25
        ) * 100

        # Whale activity indicator (large volume candles)
        volume_std = df['volume'].rolling(window=20).std()
        df['whale_candle'] = df['volume'] > (df['volume_sma'] + 2 * volume_std)

        return df

    def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate dynamic support and resistance levels"""
        # Find local maxima and minima
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()

        # Identify pivot points
        df['resistance'] = df['high'].where(df['high'] == highs)
        df['support'] = df['low'].where(df['low'] == lows)

        # Forward fill the levels
        df['resistance'] = df['resistance'].fillna(method='ffill')
        df['support'] = df['support'].fillna(method='ffill')

        # Distance from support/resistance
        df['distance_from_resistance'] = (df['resistance'] - df['close']) / df['close']
        df['distance_from_support'] = (df['close'] - df['support']) / df['close']

        return df

    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect chart patterns"""
        # Bullish/Bearish Engulfing
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        )

        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        )

        # Hammer and Shooting Star
        body = abs(df['close'] - df['open'])
        candle_range = df['high'] - df['low']

        df['hammer'] = (
            (df['low'] < df[['open', 'close']].min(axis=1) - 2 * body) &
            (df['high'] - df[['open', 'close']].max(axis=1) < body * 0.3) &
            (body > 0) & (candle_range > 0)
        )

        df['shooting_star'] = (
            (df['high'] > df[['open', 'close']].max(axis=1) + 2 * body) &
            (df[['open', 'close']].min(axis=1) - df['low'] < body * 0.3) &
            (body > 0) & (candle_range > 0)
        )

        # Doji
        df['doji'] = body / candle_range < 0.1

        # Three White Soldiers / Three Black Crows
        df['three_white_soldiers'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'].shift(2) > df['open'].shift(2)) &
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2))
        )

        df['three_black_crows'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'].shift(2) < df['open'].shift(2)) &
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2))
        )

        return df

    def generate_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate trading signals from indicators"""
        signals = []
        latest = df.iloc[-1]

        # RSI signals
        rsi_config = self.config['rsi']
        if latest['rsi'] < rsi_config['oversold']:
            signals.append(TechnicalSignal(
                indicator='RSI',
                timeframe=self.mode,
                signal_type='buy',
                strength=(rsi_config['oversold'] - latest['rsi']) / rsi_config['oversold'],
                price_level=latest['close'],
                confidence=0.7,
                reasoning=f"RSI oversold at {latest['rsi']:.2f}"
            ))
        elif latest['rsi'] > rsi_config['overbought']:
            signals.append(TechnicalSignal(
                indicator='RSI',
                timeframe=self.mode,
                signal_type='sell',
                strength=(latest['rsi'] - rsi_config['overbought']) / (100 - rsi_config['overbought']),
                price_level=latest['close'],
                confidence=0.7,
                reasoning=f"RSI overbought at {latest['rsi']:.2f}"
            ))

        # MACD signals
        if latest['macd'] > latest['macd_signal'] and df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]:
            signals.append(TechnicalSignal(
                indicator='MACD',
                timeframe=self.mode,
                signal_type='buy',
                strength=min(1, abs(latest['macd'] - latest['macd_signal']) / latest['close'] * 100),
                price_level=latest['close'],
                confidence=0.75,
                reasoning="MACD bullish crossover"
            ))
        elif latest['macd'] < latest['macd_signal'] and df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]:
            signals.append(TechnicalSignal(
                indicator='MACD',
                timeframe=self.mode,
                signal_type='sell',
                strength=min(1, abs(latest['macd'] - latest['macd_signal']) / latest['close'] * 100),
                price_level=latest['close'],
                confidence=0.75,
                reasoning="MACD bearish crossover"
            ))

        # Bollinger Bands signals
        if latest['close'] < latest['bb_lower'] and latest['volume_ratio'] > 1.5:
            signals.append(TechnicalSignal(
                indicator='Bollinger_Bands',
                timeframe=self.mode,
                signal_type='buy',
                strength=min(1, (latest['bb_lower'] - latest['close']) / latest['bb_width']),
                price_level=latest['close'],
                confidence=0.65,
                reasoning="Price below lower BB with volume"
            ))
        elif latest['close'] > latest['bb_upper'] and latest['volume_ratio'] > 1.5:
            signals.append(TechnicalSignal(
                indicator='Bollinger_Bands',
                timeframe=self.mode,
                signal_type='sell',
                strength=min(1, (latest['close'] - latest['bb_upper']) / latest['bb_width']),
                price_level=latest['close'],
                confidence=0.65,
                reasoning="Price above upper BB with volume"
            ))

        # Pattern signals
        if latest.get('bullish_engulfing', False):
            signals.append(TechnicalSignal(
                indicator='Pattern',
                timeframe=self.mode,
                signal_type='buy',
                strength=0.8,
                price_level=latest['close'],
                confidence=0.7,
                reasoning="Bullish engulfing pattern detected"
            ))

        if latest.get('bearish_engulfing', False):
            signals.append(TechnicalSignal(
                indicator='Pattern',
                timeframe=self.mode,
                signal_type='sell',
                strength=0.8,
                price_level=latest['close'],
                confidence=0.7,
                reasoning="Bearish engulfing pattern detected"
            ))

        return signals

    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict:
        """Calculate overall trend strength and direction"""
        latest = df.iloc[-1]
        lookback = min(50, len(df))
        recent_df = df.iloc[-lookback:]

        # Multiple timeframe trend
        trends = {}

        # Short-term trend (20 periods)
        if len(df) >= 20:
            short_slope, _, r_value, _, _ = linregress(range(20), recent_df['close'].iloc[-20:])
            trends['short'] = {
                'direction': 'up' if short_slope > 0 else 'down',
                'strength': abs(r_value),
                'slope': short_slope
            }

        # Medium-term trend (50 periods)
        if len(df) >= 50:
            medium_slope, _, r_value, _, _ = linregress(range(50), df['close'].iloc[-50:])
            trends['medium'] = {
                'direction': 'up' if medium_slope > 0 else 'down',
                'strength': abs(r_value),
                'slope': medium_slope
            }

        # Overall trend score
        trend_score = 0
        weights = {'short': 0.6, 'medium': 0.4}

        for timeframe, weight in weights.items():
            if timeframe in trends:
                direction_multiplier = 1 if trends[timeframe]['direction'] == 'up' else -1
                trend_score += direction_multiplier * trends[timeframe]['strength'] * weight

        return {
            'trends': trends,
            'overall_score': trend_score,
            'overall_direction': 'bullish' if trend_score > 0.2 else 'bearish' if trend_score < -0.2 else 'neutral',
            'confidence': abs(trend_score)
        }