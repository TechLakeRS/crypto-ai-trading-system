"""
Signal Generation and Consensus System
Aggregates signals from all sources and generates trading recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class SignalSource(Enum):
    """Sources of trading signals"""
    TECHNICAL = "technical"
    AI_SENTIMENT = "ai_sentiment"
    ON_CHAIN = "on_chain"
    SOCIAL = "social"
    PATTERN = "pattern"
    VOLUME = "volume"
    MOMENTUM = "momentum"

@dataclass
class TradingSignal:
    """Complete trading signal with all necessary information"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    confidence: float  # 0 to 1
    strength: float  # Signal strength 0 to 1
    sources: List[SignalSource]

    # Price levels
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit_levels: List[Tuple[str, float]]

    # Risk metrics
    risk_reward_ratio: float
    position_size_pct: float
    max_risk_amount: float

    # Analysis details
    technical_signals: Dict
    ai_consensus: Dict
    on_chain_signals: Dict

    # Reasoning
    primary_reason: str
    supporting_factors: List[str]
    risk_factors: List[str]

    # Meta information
    timeframe: str
    market_condition: str
    correlation_warning: Optional[str] = None
    priority: int = 5  # 1-10, higher is more urgent

@dataclass
class MarketCondition:
    """Current market conditions"""
    trend: str  # 'bullish', 'bearish', 'neutral'
    volatility: str  # 'low', 'medium', 'high', 'extreme'
    volume_profile: str  # 'increasing', 'decreasing', 'stable'
    sentiment: str  # 'fear', 'neutral', 'greed', 'extreme_greed'
    liquidity: str  # 'high', 'medium', 'low'
    correlation_level: str  # 'high', 'medium', 'low'

class SignalGenerator:
    """Generates and validates trading signals from multiple sources"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.signal_history = []
        self.active_signals = {}
        self.signal_performance = {}

    def _default_config(self) -> Dict:
        """Default configuration for signal generation"""
        return {
            # Signal thresholds
            'min_confidence': 0.6,  # Minimum confidence for signal generation
            'min_sources': 2,  # Minimum number of confirming sources
            'min_ai_agreement': 0.6,  # Minimum AI consensus

            # Weight configuration for different sources
            'source_weights': {
                SignalSource.TECHNICAL: 0.3,
                SignalSource.AI_SENTIMENT: 0.25,
                SignalSource.ON_CHAIN: 0.2,
                SignalSource.SOCIAL: 0.1,
                SignalSource.PATTERN: 0.1,
                SignalSource.VOLUME: 0.05,
                SignalSource.MOMENTUM: 0.05
            },

            # Market condition adjustments
            'volatility_adjustments': {
                'low': 1.0,
                'medium': 0.9,
                'high': 0.7,
                'extreme': 0.5
            },

            # Signal strength thresholds
            'signal_thresholds': {
                'strong_buy': 0.8,
                'buy': 0.6,
                'neutral': 0.4,
                'sell': -0.6,
                'strong_sell': -0.8
            },

            # Timeframe priorities
            'timeframe_weights': {
                'scalping': 0.6,
                'day_trading': 0.8,
                'swing': 1.0,
                'position': 0.9
            }
        }

    async def generate_signal(self,
                             symbol: str,
                             technical_data: Dict,
                             ai_consensus: Dict,
                             on_chain_data: Dict,
                             social_data: Dict,
                             market_data: pd.DataFrame,
                             risk_manager: Any) -> Optional[TradingSignal]:
        """
        Generate a comprehensive trading signal from all data sources
        """
        try:
            # Analyze market conditions
            market_condition = self._analyze_market_conditions(market_data)

            # Collect signals from all sources
            signals = {
                SignalSource.TECHNICAL: self._analyze_technical_signals(technical_data),
                SignalSource.AI_SENTIMENT: self._analyze_ai_consensus(ai_consensus),
                SignalSource.ON_CHAIN: self._analyze_onchain_signals(on_chain_data),
                SignalSource.SOCIAL: self._analyze_social_signals(social_data),
                SignalSource.PATTERN: self._detect_patterns(market_data),
                SignalSource.VOLUME: self._analyze_volume(market_data),
                SignalSource.MOMENTUM: self._analyze_momentum(market_data)
            }

            # Calculate consensus
            consensus = self._calculate_consensus(signals, market_condition)

            # Check if signal meets minimum requirements
            if not self._validate_signal_requirements(consensus, signals):
                logger.info(f"Signal for {symbol} did not meet minimum requirements")
                return None

            # Generate signal type
            signal_type = self._determine_signal_type(consensus['overall_score'])

            if signal_type == SignalType.NEUTRAL:
                logger.info(f"Neutral signal for {symbol} - no action recommended")
                return None

            # Calculate entry and exit levels
            current_price = market_data['close'].iloc[-1]
            atr = market_data['atr'].iloc[-1] if 'atr' in market_data else current_price * 0.02

            # Get risk management recommendations
            position_sizing = risk_manager.calculate_position_size(
                symbol=symbol,
                confidence=consensus['confidence'],
                volatility=market_data['atr_percent'].iloc[-1] if 'atr_percent' in market_data else 0.03,
                signal_strength=consensus['strength']
            )

            # Calculate stop loss and take profit
            position_type = 'long' if signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else 'short'

            stop_loss = risk_manager.calculate_stop_loss(
                entry_price=current_price,
                position_type=position_type,
                atr=atr,
                support_level=market_data.get('support', current_price * 0.98),
                volatility_pct=market_data['atr_percent'].iloc[-1] if 'atr_percent' in market_data else 3
            )

            take_profit_levels = risk_manager.calculate_take_profit(
                entry_price=current_price,
                stop_loss=stop_loss,
                position_type=position_type,
                resistance_level=market_data.get('resistance', current_price * 1.02)
            )

            # Calculate risk-reward ratio
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit_levels[1][1] - current_price) if len(take_profit_levels) > 1 else risk * 2
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Check correlation risk
            correlation_check = risk_manager.check_correlation_risk(symbol, pd.DataFrame())
            correlation_warning = correlation_check.get('warning')

            # Generate reasoning
            primary_reason, supporting_factors, risk_factors = self._generate_reasoning(
                signals, consensus, market_condition
            )

            # Calculate priority
            priority = self._calculate_signal_priority(
                signal_type, consensus['confidence'], consensus['strength'], market_condition
            )

            # Create the trading signal
            trading_signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=consensus['confidence'],
                strength=consensus['strength'],
                sources=consensus['active_sources'],
                current_price=current_price,
                entry_price=current_price,  # Could be adjusted for limit orders
                stop_loss=stop_loss,
                take_profit_levels=take_profit_levels,
                risk_reward_ratio=risk_reward_ratio,
                position_size_pct=position_sizing['position_size_pct'],
                max_risk_amount=position_sizing['max_risk_amount'],
                technical_signals=signals[SignalSource.TECHNICAL],
                ai_consensus=signals[SignalSource.AI_SENTIMENT],
                on_chain_signals=signals[SignalSource.ON_CHAIN],
                primary_reason=primary_reason,
                supporting_factors=supporting_factors,
                risk_factors=risk_factors,
                timeframe=self._determine_timeframe(technical_data),
                market_condition=f"{market_condition.trend}_{market_condition.volatility}",
                correlation_warning=correlation_warning,
                priority=priority
            )

            # Store signal in history
            self.signal_history.append(trading_signal)
            self.active_signals[symbol] = trading_signal

            return trading_signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> MarketCondition:
        """Analyze overall market conditions"""
        latest = market_data.iloc[-1]

        # Trend analysis
        if 'ema_20' in market_data and 'ema_50' in market_data:
            if latest['close'] > latest['ema_20'] > latest['ema_50']:
                trend = 'bullish'
            elif latest['close'] < latest['ema_20'] < latest['ema_50']:
                trend = 'bearish'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'

        # Volatility analysis
        if 'atr_percent' in market_data:
            atr_pct = latest['atr_percent']
            if atr_pct < 2:
                volatility = 'low'
            elif atr_pct < 4:
                volatility = 'medium'
            elif atr_pct < 6:
                volatility = 'high'
            else:
                volatility = 'extreme'
        else:
            volatility = 'medium'

        # Volume analysis
        if 'volume_ratio' in market_data:
            vol_ratio = latest['volume_ratio']
            volume_profile = 'increasing' if vol_ratio > 1.2 else 'decreasing' if vol_ratio < 0.8 else 'stable'
        else:
            volume_profile = 'stable'

        # Sentiment (from technical fear & greed if available)
        if 'technical_fear_greed' in market_data:
            fg = latest['technical_fear_greed']
            if fg < 20:
                sentiment = 'fear'
            elif fg < 40:
                sentiment = 'neutral'
            elif fg < 70:
                sentiment = 'greed'
            else:
                sentiment = 'extreme_greed'
        else:
            sentiment = 'neutral'

        return MarketCondition(
            trend=trend,
            volatility=volatility,
            volume_profile=volume_profile,
            sentiment=sentiment,
            liquidity='medium',  # Would need order book data
            correlation_level='medium'  # Would need correlation matrix
        )

    def _analyze_technical_signals(self, technical_data: Dict) -> Dict:
        """Analyze technical indicators for signals"""
        signals = []
        score = 0
        confidence = 0

        # RSI analysis
        if 'rsi' in technical_data:
            rsi = technical_data['rsi']
            if rsi < 30:
                signals.append('RSI oversold')
                score += 0.2
            elif rsi > 70:
                signals.append('RSI overbought')
                score -= 0.2

        # MACD analysis
        if 'macd' in technical_data and 'macd_signal' in technical_data:
            if technical_data['macd'] > technical_data['macd_signal']:
                signals.append('MACD bullish')
                score += 0.15
            else:
                signals.append('MACD bearish')
                score -= 0.15

        # Bollinger Bands
        if 'bb_percent' in technical_data:
            bb = technical_data['bb_percent']
            if bb < 0:
                signals.append('Below lower BB')
                score += 0.15
            elif bb > 1:
                signals.append('Above upper BB')
                score -= 0.15

        # Moving averages
        if 'ema_20' in technical_data and 'ema_50' in technical_data:
            if technical_data['close'] > technical_data['ema_20'] > technical_data['ema_50']:
                signals.append('Bullish MA alignment')
                score += 0.2
            elif technical_data['close'] < technical_data['ema_20'] < technical_data['ema_50']:
                signals.append('Bearish MA alignment')
                score -= 0.2

        confidence = min(0.9, 0.5 + abs(score))

        return {
            'signals': signals,
            'score': score,
            'confidence': confidence,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_ai_consensus(self, ai_consensus: Dict) -> Dict:
        """Analyze AI consensus data"""
        if not ai_consensus:
            return {'signals': [], 'score': 0, 'confidence': 0, 'direction': 'neutral'}

        score = ai_consensus.get('overall_sentiment', 0)
        confidence = ai_consensus.get('confidence', 0)
        agreement = ai_consensus.get('agreement_score', 0)

        signals = []
        if score > 0.5:
            signals.append('AI consensus bullish')
        elif score < -0.5:
            signals.append('AI consensus bearish')
        else:
            signals.append('AI consensus neutral')

        if agreement > 0.8:
            signals.append('High AI agreement')
            confidence *= 1.2
        elif agreement < 0.5:
            signals.append('AI divergence detected')
            confidence *= 0.7

        return {
            'signals': signals,
            'score': score,
            'confidence': min(1, confidence),
            'direction': 'bullish' if score > 0.2 else 'bearish' if score < -0.2 else 'neutral',
            'key_insights': ai_consensus.get('key_insights', [])
        }

    def _analyze_onchain_signals(self, on_chain_data: Dict) -> Dict:
        """Analyze on-chain metrics"""
        if not on_chain_data:
            return {'signals': [], 'score': 0, 'confidence': 0, 'direction': 'neutral'}

        signals = []
        score = 0

        # Exchange flows
        if 'exchange_flows' in on_chain_data:
            flows = on_chain_data['exchange_flows']
            if flows < -1000:  # Large outflows
                signals.append('Exchange outflows (bullish)')
                score += 0.3
            elif flows > 1000:  # Large inflows
                signals.append('Exchange inflows (bearish)')
                score -= 0.3

        # Whale activity
        if 'whale_movements' in on_chain_data:
            whale_count = len(on_chain_data['whale_movements'])
            if whale_count > 5:
                signals.append(f'{whale_count} whale movements detected')
                score += 0.1 if on_chain_data.get('exchange_flows', 0) < 0 else -0.1

        # Network activity
        if 'active_addresses' in on_chain_data:
            active = on_chain_data['active_addresses']
            baseline = 1000000  # Example baseline
            if active > baseline * 1.2:
                signals.append('High network activity')
                score += 0.2
            elif active < baseline * 0.8:
                signals.append('Low network activity')
                score -= 0.2

        confidence = min(0.8, 0.4 + len(signals) * 0.1)

        return {
            'signals': signals,
            'score': score,
            'confidence': confidence,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_social_signals(self, social_data: Dict) -> Dict:
        """Analyze social media sentiment"""
        if not social_data:
            return {'signals': [], 'score': 0, 'confidence': 0, 'direction': 'neutral'}

        signals = []
        score = 0

        # Twitter sentiment
        if 'twitter' in social_data:
            twitter_sentiment = social_data['twitter'].get('sentiment', 0)
            if twitter_sentiment > 0.5:
                signals.append('Twitter bullish')
                score += 0.15
            elif twitter_sentiment < -0.5:
                signals.append('Twitter bearish')
                score -= 0.15

        # Reddit sentiment
        if 'reddit' in social_data:
            reddit_sentiment = social_data['reddit'].get('sentiment', 0)
            if reddit_sentiment > 0.5:
                signals.append('Reddit bullish')
                score += 0.1
            elif reddit_sentiment < -0.5:
                signals.append('Reddit bearish')
                score -= 0.1

        # Social momentum
        if 'momentum' in social_data:
            if social_data['momentum'] > 0.2:
                signals.append('Social momentum increasing')
                score += 0.1

        confidence = min(0.7, 0.3 + abs(score))

        return {
            'signals': signals,
            'score': score,
            'confidence': confidence,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _detect_patterns(self, market_data: pd.DataFrame) -> Dict:
        """Detect chart patterns"""
        signals = []
        score = 0
        latest = market_data.iloc[-1] if not market_data.empty else {}

        # Check for pattern signals
        if 'bullish_engulfing' in latest and latest['bullish_engulfing']:
            signals.append('Bullish engulfing pattern')
            score += 0.3

        if 'bearish_engulfing' in latest and latest['bearish_engulfing']:
            signals.append('Bearish engulfing pattern')
            score -= 0.3

        if 'hammer' in latest and latest['hammer']:
            signals.append('Hammer pattern')
            score += 0.2

        if 'shooting_star' in latest and latest['shooting_star']:
            signals.append('Shooting star pattern')
            score -= 0.2

        confidence = 0.7 if signals else 0

        return {
            'signals': signals,
            'score': score,
            'confidence': confidence,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_volume(self, market_data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        latest = market_data.iloc[-1] if not market_data.empty else {}
        signals = []
        score = 0

        if 'volume_ratio' in latest:
            vol_ratio = latest['volume_ratio']
            if vol_ratio > 2:
                signals.append('Volume spike')
                # Check if price is moving up or down
                if 'returns' in latest and latest['returns'] > 0:
                    score += 0.2
                else:
                    score -= 0.2
            elif vol_ratio < 0.5:
                signals.append('Low volume')
                score -= 0.1

        confidence = 0.6 if signals else 0

        return {
            'signals': signals,
            'score': score,
            'confidence': confidence,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _analyze_momentum(self, market_data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        latest = market_data.iloc[-1] if not market_data.empty else {}
        signals = []
        score = 0

        if 'momentum_score' in latest:
            mom = latest['momentum_score']
            if mom > 0.7:
                signals.append('Strong momentum')
                score += 0.25
            elif mom < 0.3:
                signals.append('Weak momentum')
                score -= 0.25

        if 'roc' in latest:
            roc = latest['roc']
            if roc > 5:
                signals.append('Positive rate of change')
                score += 0.15
            elif roc < -5:
                signals.append('Negative rate of change')
                score -= 0.15

        confidence = min(0.8, 0.4 + abs(score))

        return {
            'signals': signals,
            'score': score,
            'confidence': confidence,
            'direction': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
        }

    def _calculate_consensus(self, signals: Dict, market_condition: MarketCondition) -> Dict:
        """Calculate overall consensus from all signal sources"""
        weights = self.config['source_weights']
        volatility_adj = self.config['volatility_adjustments'].get(market_condition.volatility, 1.0)

        total_score = 0
        total_confidence = 0
        total_weight = 0
        active_sources = []

        for source, signal_data in signals.items():
            if signal_data['signals']:  # Only count sources with actual signals
                weight = weights.get(source, 0.1)
                score = signal_data['score'] * weight * volatility_adj
                confidence = signal_data['confidence'] * weight

                total_score += score
                total_confidence += confidence
                total_weight += weight
                active_sources.append(source)

        # Normalize
        if total_weight > 0:
            overall_score = total_score / total_weight
            overall_confidence = total_confidence / total_weight
        else:
            overall_score = 0
            overall_confidence = 0

        # Calculate signal strength (0 to 1)
        strength = min(1, abs(overall_score))

        return {
            'overall_score': overall_score,
            'confidence': overall_confidence,
            'strength': strength,
            'active_sources': active_sources,
            'source_count': len(active_sources)
        }

    def _validate_signal_requirements(self, consensus: Dict, signals: Dict) -> bool:
        """Check if signal meets minimum requirements"""
        # Check confidence threshold
        if consensus['confidence'] < self.config['min_confidence']:
            return False

        # Check minimum sources
        if consensus['source_count'] < self.config['min_sources']:
            return False

        # Check AI agreement if available
        ai_signal = signals.get(SignalSource.AI_SENTIMENT, {})
        if ai_signal and 'confidence' in ai_signal:
            if ai_signal['confidence'] < self.config['min_ai_agreement']:
                return False

        return True

    def _determine_signal_type(self, score: float) -> SignalType:
        """Determine signal type based on score"""
        thresholds = self.config['signal_thresholds']

        if score >= thresholds['strong_buy']:
            return SignalType.STRONG_BUY
        elif score >= thresholds['buy']:
            return SignalType.BUY
        elif score <= thresholds['strong_sell']:
            return SignalType.STRONG_SELL
        elif score <= thresholds['sell']:
            return SignalType.SELL
        else:
            return SignalType.NEUTRAL

    def _determine_timeframe(self, technical_data: Dict) -> str:
        """Determine appropriate timeframe for the signal"""
        # This would be determined by the technical analysis configuration
        # For now, returning a default
        return 'day_trading'

    def _generate_reasoning(self, signals: Dict, consensus: Dict,
                           market_condition: MarketCondition) -> Tuple[str, List[str], List[str]]:
        """Generate human-readable reasoning for the signal"""
        # Primary reason - strongest signal source
        primary_reasons = []
        for source, data in signals.items():
            if data['signals'] and abs(data['score']) > 0.2:
                primary_reasons.append((source, data['signals'][0], abs(data['score'])))

        primary_reasons.sort(key=lambda x: x[2], reverse=True)
        primary_reason = primary_reasons[0][1] if primary_reasons else "Multiple indicators aligned"

        # Supporting factors
        supporting_factors = []
        for source, data in signals.items():
            for signal in data['signals'][:2]:  # Top 2 from each source
                if signal not in supporting_factors:
                    supporting_factors.append(signal)

        # Risk factors
        risk_factors = []
        if market_condition.volatility in ['high', 'extreme']:
            risk_factors.append(f"High market volatility ({market_condition.volatility})")

        if market_condition.sentiment == 'extreme_greed':
            risk_factors.append("Market showing extreme greed")

        if consensus['confidence'] < 0.7:
            risk_factors.append("Moderate confidence level")

        if consensus['source_count'] < 3:
            risk_factors.append("Limited confirming sources")

        return primary_reason, supporting_factors[:5], risk_factors

    def _calculate_signal_priority(self, signal_type: SignalType, confidence: float,
                                  strength: float, market_condition: MarketCondition) -> int:
        """Calculate signal priority (1-10)"""
        base_priority = 5

        # Adjust for signal type
        if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
            base_priority += 2
        elif signal_type in [SignalType.BUY, SignalType.SELL]:
            base_priority += 1

        # Adjust for confidence
        if confidence > 0.8:
            base_priority += 2
        elif confidence > 0.7:
            base_priority += 1

        # Adjust for strength
        if strength > 0.8:
            base_priority += 1

        # Adjust for market conditions
        if market_condition.volatility == 'low' and market_condition.trend != 'neutral':
            base_priority += 1

        return min(10, max(1, base_priority))

    def get_active_signals(self) -> List[TradingSignal]:
        """Get all active trading signals"""
        return list(self.active_signals.values())

    def invalidate_signal(self, symbol: str, reason: str):
        """Invalidate an active signal"""
        if symbol in self.active_signals:
            signal = self.active_signals.pop(symbol)
            logger.info(f"Signal for {symbol} invalidated: {reason}")

            # Track performance for learning
            self.signal_performance[symbol] = {
                'signal': signal,
                'invalidation_reason': reason,
                'timestamp': datetime.now()
            }

    def get_signal_summary(self, signal: TradingSignal) -> str:
        """Get human-readable summary of a signal"""
        summary = f"""
📊 Trading Signal: {signal.symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Signal: {signal.signal_type.value.upper()}
💪 Strength: {signal.strength:.1%}
🔍 Confidence: {signal.confidence:.1%}
⏰ Time: {signal.timestamp.strftime('%Y-%m-%d %H:%M')}

💵 Price Levels:
• Current: ${signal.current_price:,.2f}
• Entry: ${signal.entry_price:,.2f}
• Stop Loss: ${signal.stop_loss:,.2f}
• Take Profit: {', '.join([f'{tp[0]}: ${tp[1]:,.2f}' for tp in signal.take_profit_levels[:2]])}

📈 Risk/Reward: 1:{signal.risk_reward_ratio:.1f}
💰 Position Size: {signal.position_size_pct:.1f}%
⚠️ Max Risk: ${signal.max_risk_amount:,.2f}

📝 Primary Reason: {signal.primary_reason}

✅ Supporting Factors:
{chr(10).join([f'• {factor}' for factor in signal.supporting_factors[:3]])}

⚠️ Risk Factors:
{chr(10).join([f'• {risk}' for risk in signal.risk_factors[:3]])}

📊 Market: {signal.market_condition}
⏳ Timeframe: {signal.timeframe}
🚨 Priority: {signal.priority}/10
"""
        return summary