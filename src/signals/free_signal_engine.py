"""
Free Signal Generation Engine
Uses only FREE APIs + Technical Analysis
No paid AI APIs required
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.sentiment.free_sentiment_apis import FreeSentimentAggregator
from src.signals.signal_generator import TradingSignal, SignalType, SignalSource
from src.technical.indicators import CryptoTechnicalAnalyzer
from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class FreeSignalEngine:
    """
    Generates trading signals using 100% FREE data sources:
    - Technical Analysis (RSI, MACD, Bollinger Bands, etc.)
    - Crypto Fear & Greed Index
    - CoinGecko sentiment
    - CryptoCompare news
    - Volume analysis
    - Pattern recognition
    """

    def __init__(self):
        self.sentiment_aggregator = FreeSentimentAggregator()
        self.technical_analyzer = CryptoTechnicalAnalyzer(mode='day_trading')
        self.risk_manager = RiskManager()
        self.active_signals = {}
        self.signal_history = []

    async def generate_signal(self, symbol: str, market_data: pd.DataFrame,
                             current_price: float) -> Optional[TradingSignal]:
        """
        Generate comprehensive trading signal using all free sources

        Parameters:
        - symbol: Trading pair (e.g., 'BTC/USDT')
        - market_data: OHLCV DataFrame with technical indicators
        - current_price: Current market price

        Returns:
        - TradingSignal object if signal meets criteria, None otherwise
        """

        try:
            # Extract base symbol (BTC from BTC/USDT)
            base_symbol = symbol.split('/')[0]

            # 1. Get sentiment data from free APIs
            sentiment_data = await self.sentiment_aggregator.get_market_sentiment(base_symbol)

            # 2. Calculate technical indicators
            market_data = self.technical_analyzer.calculate_all_indicators(market_data)
            tech_signals = self.technical_analyzer.generate_signals(market_data)

            # 3. Analyze all data sources (TECHNICAL ONLY - no sentiment in signals)
            sources_analysis = {
                'technical': self._analyze_technical(market_data, tech_signals),
                'volume': self._analyze_volume(market_data),
                'momentum': self._analyze_momentum(market_data),
                'pattern': self._analyze_patterns(market_data),
                'trend': self._analyze_trend(market_data)
            }

            # 4. Calculate consensus
            consensus = self._calculate_consensus(sources_analysis)

            # 5. Check if signal meets minimum requirements
            if not self._validate_signal(consensus):
                logger.info(f"Signal for {symbol} did not meet requirements")
                return None

            # 6. Determine signal type
            signal_type = self._determine_signal_type(consensus['score'])

            if signal_type == SignalType.NEUTRAL:
                return None

            # 7. Calculate entry, stop loss, and take profit
            position_type = 'long' if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else 'short'

            # Get ATR for dynamic stop loss
            atr = market_data['atr'].iloc[-1] if 'atr' in market_data else current_price * 0.02
            volatility_pct = (atr / current_price) * 100

            # Position sizing
            position_sizing = self.risk_manager.calculate_position_size(
                symbol=symbol,
                confidence=consensus['confidence'],
                volatility=volatility_pct / 100,
                signal_strength=consensus['strength']
            )

            # Stop loss
            stop_loss = self.risk_manager.calculate_stop_loss(
                entry_price=current_price,
                position_type=position_type,
                atr=atr,
                volatility_pct=volatility_pct
            )

            # Take profit levels
            take_profit_levels = self.risk_manager.calculate_take_profit(
                entry_price=current_price,
                stop_loss=stop_loss,
                position_type=position_type
            )

            # Risk/Reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit_levels[1][1] - current_price) if len(take_profit_levels) > 1 else risk * 2
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # 8. Generate reasoning
            primary_reason, supporting_factors, risk_factors = self._generate_reasoning(
                sources_analysis, consensus, sentiment_data
            )

            # 9. Calculate priority
            priority = self._calculate_priority(signal_type, consensus['confidence'], consensus['strength'])

            # 10. Create trading signal
            trading_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=consensus['confidence'],
                strength=consensus['strength'],
                sources=consensus['active_sources'],
                current_price=current_price,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_levels=take_profit_levels,
                risk_reward_ratio=risk_reward_ratio,
                position_size_pct=position_sizing['position_size_pct'],
                max_risk_amount=position_sizing['max_risk_amount'],
                technical_signals=sources_analysis['technical'],
                ai_consensus={'sentiment_data': sentiment_data},
                on_chain_signals={},
                primary_reason=primary_reason,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                timeframe='5m',
                market_condition=f"{consensus.get('trend', 'neutral')}_{consensus.get('volatility', 'medium')}",
                priority=priority
            )

            # Store signal
            self.active_signals[symbol] = trading_signal
            self.signal_history.append(trading_signal)

            logger.info(f"Generated {signal_type.value} signal for {symbol} with {len(consensus['active_sources'])} confluences")

            return trading_signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _analyze_technical(self, market_data: pd.DataFrame, signals: List[str]) -> Dict:
        """Analyze technical indicators with crypto-specific thresholds FROM MANUAL.MD"""
        latest = market_data.iloc[-1]
        score = 0
        details = []

        # RSI - Manual.md: 5-min trading uses 75/25 levels
        if 'rsi' in latest:
            rsi = latest['rsi']
            if rsi < 25:
                score += 0.3
                details.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 75:
                score -= 0.3
                details.append(f"RSI overbought ({rsi:.1f})")

        # MACD
        if 'macd' in latest and 'macd_signal' in latest:
            if latest['macd'] > latest['macd_signal']:
                score += 0.2
                details.append("MACD bullish crossover")
            else:
                score -= 0.2
                details.append("MACD bearish")

        # Bollinger Bands
        if 'bb_upper' in latest and 'bb_lower' in latest:
            if latest['close'] < latest['bb_lower']:
                score += 0.2
                details.append("Price below lower BB")
            elif latest['close'] > latest['bb_upper']:
                score -= 0.2
                details.append("Price above upper BB")

        # Moving averages
        if 'ema_20' in latest and 'ema_50' in latest:
            if latest['close'] > latest['ema_20'] > latest['ema_50']:
                score += 0.3
                details.append("Bullish MA alignment")
            elif latest['close'] < latest['ema_20'] < latest['ema_50']:
                score -= 0.3
                details.append("Bearish MA alignment")

        confidence = min(0.9, 0.5 + abs(score) * 0.5)

        return {
            'score': score,
            'confidence': confidence,
            'details': details,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_sentiment(self, sentiment_data: Dict) -> Dict:
        """Analyze free API sentiment data"""
        if not sentiment_data:
            return {'score': 0, 'confidence': 0, 'details': [], 'direction': 'neutral'}

        score = sentiment_data['overall_sentiment']
        confidence = sentiment_data['confidence']
        details = []

        # Fear & Greed
        if sentiment_data.get('fear_greed_data'):
            fg = sentiment_data['fear_greed_data']
            details.append(f"Fear & Greed: {fg['classification']} ({fg['value']})")

        # CoinGecko sentiment
        if sentiment_data.get('coingecko_data'):
            cg = sentiment_data['coingecko_data']
            details.append(f"Community votes: {cg.get('sentiment_votes_up_percentage', 50):.0f}% bullish")

        # Trending
        if sentiment_data.get('is_trending'):
            details.append("Trending on CoinGecko")
            score += 0.1

        # News sentiment
        news_items = sentiment_data.get('news_items', [])
        if news_items:
            positive = sum(1 for n in news_items if n['sentiment'] == 'positive')
            negative = sum(1 for n in news_items if n['sentiment'] == 'negative')
            details.append(f"News: {positive} positive, {negative} negative")

        return {
            'score': score,
            'confidence': confidence,
            'details': details,
            'direction': 'bullish' if score > 0.2 else 'bearish' if score < -0.2 else 'neutral'
        }

    def _analyze_volume(self, market_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns - Manual.md requires 1.5× average volume"""
        latest = market_data.iloc[-1]
        score = 0
        details = []

        if 'volume_ratio' in latest:
            vol_ratio = latest['volume_ratio']
            # Manual.md: Volume confirmation requires 1.5× average
            if vol_ratio >= 1.5:
                score += 0.3  # Stronger weight for volume confirmation
                details.append(f"Volume confirmed ({vol_ratio:.1f}x avg)")
            elif vol_ratio > 1.2:
                score += 0.15
                details.append(f"Above average volume ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.6:
                score -= 0.1
                details.append(f"Low volume warning ({vol_ratio:.1f}x avg)")

        confidence = 0.7 if vol_ratio >= 1.5 else 0.4 if details else 0

        return {
            'score': score,
            'confidence': confidence,
            'details': details,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_momentum(self, market_data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        latest = market_data.iloc[-1]
        score = 0
        details = []

        if 'returns' in latest:
            returns = latest['returns']
            if returns > 2:
                score += 0.2
                details.append(f"Strong positive momentum ({returns:.1f}%)")
            elif returns < -2:
                score -= 0.2
                details.append(f"Strong negative momentum ({returns:.1f}%)")

        confidence = 0.7 if details else 0

        return {
            'score': score,
            'confidence': confidence,
            'details': details,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_patterns(self, market_data: pd.DataFrame) -> Dict:
        """Analyze candlestick patterns"""
        latest = market_data.iloc[-1]
        score = 0
        details = []

        # Check for common patterns
        if 'bullish_engulfing' in latest and latest['bullish_engulfing']:
            score += 0.3
            details.append("Bullish engulfing pattern")

        if 'bearish_engulfing' in latest and latest['bearish_engulfing']:
            score -= 0.3
            details.append("Bearish engulfing pattern")

        if 'hammer' in latest and latest['hammer']:
            score += 0.2
            details.append("Hammer pattern (bullish reversal)")

        if 'shooting_star' in latest and latest['shooting_star']:
            score -= 0.2
            details.append("Shooting star (bearish reversal)")

        confidence = 0.7 if details else 0

        return {
            'score': score,
            'confidence': confidence,
            'details': details,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_trend(self, market_data: pd.DataFrame) -> Dict:
        """Analyze overall trend using moving averages"""
        latest = market_data.iloc[-1]
        score = 0
        details = []

        # EMA alignment
        if 'ema_20' in latest and 'ema_50' in latest:
            if latest['close'] > latest['ema_20'] > latest['ema_50']:
                score += 0.4
                details.append("Strong uptrend (price > EMA20 > EMA50)")
            elif latest['close'] < latest['ema_20'] < latest['ema_50']:
                score -= 0.4
                details.append("Strong downtrend (price < EMA20 < EMA50)")
            elif latest['close'] > latest['ema_20']:
                score += 0.2
                details.append("Price above EMA20")
            elif latest['close'] < latest['ema_20']:
                score -= 0.2
                details.append("Price below EMA20")

        # Price vs VWAP
        if 'vwap' in latest:
            if latest['close'] > latest['vwap']:
                score += 0.1
                details.append("Price above VWAP")
            else:
                score -= 0.1
                details.append("Price below VWAP")

        confidence = 0.8 if details else 0

        return {
            'score': score,
            'confidence': confidence,
            'details': details,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _calculate_consensus(self, sources: Dict) -> Dict:
        """Calculate overall consensus from all TECHNICAL sources"""
        weights = {
            'technical': 0.30,  # RSI, MACD, Bollinger Bands
            'volume': 0.20,     # Volume analysis
            'momentum': 0.20,   # Momentum indicators
            'pattern': 0.15,    # Candlestick patterns
            'trend': 0.15       # Trend analysis (EMAs, VWAP)
        }

        total_score = 0
        total_confidence = 0
        total_weight = 0
        active_sources = []

        for source, data in sources.items():
            if data['details']:  # Only count sources with actual signals
                weight = weights.get(source, 0.1)
                score = data['score'] * weight
                confidence = data['confidence'] * weight

                total_score += score
                total_confidence += confidence
                total_weight += weight

                # Map to SignalSource enum
                if source == 'technical':
                    active_sources.append(SignalSource.TECHNICAL)
                elif source == 'volume':
                    active_sources.append(SignalSource.VOLUME)
                elif source == 'momentum':
                    active_sources.append(SignalSource.MOMENTUM)
                elif source == 'pattern':
                    active_sources.append(SignalSource.PATTERN)
                elif source == 'trend':
                    active_sources.append(SignalSource.MOMENTUM)  # Trend is a form of momentum

        # Normalize
        if total_weight > 0:
            overall_score = total_score / total_weight
            overall_confidence = total_confidence / total_weight
        else:
            overall_score = 0
            overall_confidence = 0

        strength = min(1, abs(overall_score))

        return {
            'score': overall_score,
            'confidence': overall_confidence,
            'strength': strength,
            'active_sources': active_sources,
            'source_count': len(active_sources)
        }

    def _validate_signal(self, consensus: Dict) -> bool:
        """Check if signal meets minimum requirements (crypto-specific from strategy doc)"""
        # Per strategy doc: minimum 3 of 5 confluences required
        # But we'll be slightly flexible: 2+ sources with good strength OR 3+ sources
        return (
            (consensus['source_count'] >= 3) or  # Ideal: 3+ technical sources agree
            (consensus['source_count'] >= 2 and abs(consensus['score']) >= 0.4)  # Or 2 sources with strong signal
        )

    def _determine_signal_type(self, score: float) -> SignalType:
        """Determine signal type based on score"""
        if score >= 0.7:
            return SignalType.STRONG_BUY
        elif score >= 0.3:
            return SignalType.BUY
        elif score <= -0.7:
            return SignalType.STRONG_SELL
        elif score <= -0.3:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    def _generate_reasoning(self, sources: Dict, consensus: Dict, sentiment_data: Dict) -> tuple:
        """Generate human-readable reasoning"""
        # Primary reason
        strongest_source = max(sources.items(), key=lambda x: abs(x[1]['score']))
        primary_reason = strongest_source[1]['details'][0] if strongest_source[1]['details'] else "Multiple indicators aligned"

        # Supporting factors
        supporting_factors = []
        for source_name, data in sources.items():
            for detail in data['details'][:2]:
                if detail not in supporting_factors:
                    supporting_factors.append(detail)

        # Risk factors
        risk_factors = []
        if consensus['confidence'] < 0.7:
            risk_factors.append(f"Moderate confidence ({consensus['confidence']:.0%})")
        if consensus['source_count'] < 3:
            risk_factors.append(f"Limited sources ({consensus['source_count']})")

        # Add sentiment-based risk factors
        if sentiment_data and sentiment_data.get('fear_greed_data'):
            fg_value = sentiment_data['fear_greed_data']['value']
            if fg_value > 75:
                risk_factors.append("Extreme greed in market")
            elif fg_value < 25:
                risk_factors.append("Extreme fear in market")

        return primary_reason, supporting_factors[:5], risk_factors

    def _calculate_priority(self, signal_type: SignalType, confidence: float, strength: float) -> int:
        """Calculate signal priority (1-10)"""
        base = 5

        if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            base += 2
        elif signal_type in [SignalType.BUY, SignalType.SELL]:
            base += 1

        if confidence > 0.8:
            base += 2
        elif confidence > 0.7:
            base += 1

        if strength > 0.8:
            base += 1

        return min(10, max(1, base))

    def get_active_signals(self, current_prices: Dict[str, float] = None) -> List[TradingSignal]:
        """
        Get all active signals from Model 1

        Signals expire if:
        - Older than 5 minutes (stale signal)
        - Entry price was passed (missed opportunity)
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(minutes=5)

        # Filter out expired signals
        symbols_to_remove = []
        for symbol, signal in self.active_signals.items():
            # Expire if too old
            if signal.timestamp <= cutoff_time:
                symbols_to_remove.append(symbol)
                continue

            # Expire if entry was passed (missed the trade)
            if current_prices and symbol in current_prices:
                current_price = current_prices[symbol]
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    # For BUY: expire if price went above entry (missed buy)
                    if current_price > signal.entry_price * 1.002:  # 0.2% buffer
                        symbols_to_remove.append(symbol)
                elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    # For SELL: expire if price went below entry (missed sell)
                    if current_price < signal.entry_price * 0.998:  # 0.2% buffer
                        symbols_to_remove.append(symbol)

        # Remove expired signals
        for symbol in symbols_to_remove:
            del self.active_signals[symbol]

        return list(self.active_signals.values())

    def clear_signal(self, symbol: str):
        """Remove a signal for a symbol"""
        if symbol in self.active_signals:
            del self.active_signals[symbol]
