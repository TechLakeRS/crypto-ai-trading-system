"""
Multi-AI Sentiment Analysis Module
Integrates Grok (Twitter/X), Claude, Deepseek, and other models for consensus-based sentiment
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import re
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class AIModel(Enum):
    """Available AI models"""
    GROK = "grok"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    GPT4 = "gpt4"
    GEMINI = "gemini"
    LOCAL_LLM = "local_llm"

@dataclass
class SentimentResult:
    """Individual sentiment analysis result"""
    source: str
    model: AIModel
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    reasoning: str
    key_factors: List[str]
    timestamp: datetime
    raw_data: Optional[Dict] = None

@dataclass
class ConsensusSentiment:
    """Consensus sentiment from multiple AI models"""
    overall_sentiment: float  # -1 to 1
    confidence: float  # 0 to 1
    agreement_score: float  # 0 to 1 (how much models agree)
    individual_results: List[SentimentResult]
    recommendation: str  # 'strong_buy', 'buy', 'neutral', 'sell', 'strong_sell'
    key_insights: List[str]
    timestamp: datetime

class AIModelInterface:
    """Base interface for AI models"""

    def __init__(self, api_key: str = "", endpoint: str = ""):
        self.api_key = api_key
        self.endpoint = endpoint
        self.rate_limit = 60  # requests per minute
        self.last_request_time = 0

    async def analyze_sentiment(self, data: Dict) -> SentimentResult:
        """Analyze sentiment using the AI model"""
        raise NotImplementedError

    async def _rate_limit_check(self):
        """Check rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (60 / self.rate_limit):
            await asyncio.sleep((60 / self.rate_limit) - time_since_last)
        self.last_request_time = time.time()

class GrokAnalyzer(AIModelInterface):
    """Grok AI for Twitter/X sentiment analysis"""

    async def analyze_sentiment(self, data: Dict) -> SentimentResult:
        """Analyze Twitter/X sentiment using Grok"""
        await self._rate_limit_check()

        try:
            # Prepare data for Grok
            tweets = data.get('tweets', [])
            trending_topics = data.get('trending', [])

            # Format prompt for Grok
            prompt = self._create_prompt(tweets, trending_topics, data.get('symbol', 'BTC'))

            # In production, this would call actual Grok API
            # For demonstration, using mock analysis
            sentiment_score, confidence, reasoning, key_factors = await self._mock_grok_analysis(tweets)

            return SentimentResult(
                source="Twitter/X",
                model=AIModel.GROK,
                sentiment_score=sentiment_score,
                confidence=confidence,
                reasoning=reasoning,
                key_factors=key_factors,
                timestamp=datetime.now(),
                raw_data={'tweet_count': len(tweets), 'trending': trending_topics}
            )

        except Exception as e:
            logger.error(f"Grok analysis failed: {e}")
            return None

    def _create_prompt(self, tweets: List[Dict], trending: List[str], symbol: str) -> str:
        """Create analysis prompt for Grok"""
        prompt = f"""Analyze the sentiment for {symbol} cryptocurrency based on the following Twitter/X data:

Recent tweets (last 100):
{json.dumps(tweets[:10], indent=2)}  # Sample of tweets

Trending topics: {', '.join(trending)}

Please provide:
1. Overall sentiment score (-1 to 1)
2. Confidence level (0 to 1)
3. Key factors driving sentiment
4. Notable influencer opinions
5. Potential market impact

Focus on:
- Distinguishing between noise and signal
- Identifying coordinated campaigns or manipulation
- Weighing verified accounts more heavily
- Considering engagement metrics"""

        return prompt

    async def _mock_grok_analysis(self, tweets: List[Dict]) -> Tuple[float, float, str, List[str]]:
        """Mock Grok analysis for demonstration"""
        # In production, would use actual Grok API

        # Simple sentiment analysis as placeholder
        if not tweets:
            return 0, 0.5, "No data available", []

        # Analyze tweet sentiments
        vader = SentimentIntensityAnalyzer()
        sentiments = []
        for tweet in tweets[:100]:
            text = tweet.get('text', '')
            scores = vader.polarity_scores(text)
            sentiments.append(scores['compound'])

        avg_sentiment = np.mean(sentiments) if sentiments else 0
        confidence = min(0.9, 0.5 + abs(avg_sentiment))

        # Determine key factors
        key_factors = []
        if avg_sentiment > 0.3:
            key_factors = ["Bullish momentum", "Positive news coverage", "Influencer support"]
        elif avg_sentiment < -0.3:
            key_factors = ["Bearish sentiment", "Negative news", "FUD spreading"]
        else:
            key_factors = ["Mixed signals", "Market uncertainty", "Consolidation phase"]

        reasoning = f"Analysis of {len(tweets)} tweets shows {'positive' if avg_sentiment > 0 else 'negative'} sentiment"

        return avg_sentiment, confidence, reasoning, key_factors

class ClaudeAnalyzer(AIModelInterface):
    """Claude AI for technical and fundamental analysis verification"""

    async def analyze_sentiment(self, data: Dict) -> SentimentResult:
        """Analyze using Claude for verification and technical insight"""
        await self._rate_limit_check()

        try:
            # Prepare comprehensive data for Claude
            technical_data = data.get('technical', {})
            news_data = data.get('news', [])
            onchain_data = data.get('onchain', {})

            # Create analysis prompt
            prompt = self._create_analysis_prompt(technical_data, news_data, onchain_data)

            # Mock Claude analysis (would use actual API in production)
            sentiment_score, confidence, reasoning, key_factors = await self._mock_claude_analysis(
                technical_data, news_data, onchain_data
            )

            return SentimentResult(
                source="Technical+Fundamental",
                model=AIModel.CLAUDE,
                sentiment_score=sentiment_score,
                confidence=confidence,
                reasoning=reasoning,
                key_factors=key_factors,
                timestamp=datetime.now(),
                raw_data={'technical': technical_data, 'news_count': len(news_data)}
            )

        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return None

    def _create_analysis_prompt(self, technical: Dict, news: List, onchain: Dict) -> str:
        """Create comprehensive analysis prompt"""
        return f"""Analyze cryptocurrency market sentiment based on:

Technical Indicators:
- RSI: {technical.get('rsi', 'N/A')}
- MACD: {technical.get('macd', 'N/A')}
- Volume: {technical.get('volume_ratio', 'N/A')}
- Price action: {technical.get('price_action', 'N/A')}

Recent News Headlines:
{json.dumps(news[:5], indent=2)}

On-chain Metrics:
- Exchange flows: {onchain.get('exchange_flows', 'N/A')}
- Active addresses: {onchain.get('active_addresses', 'N/A')}
- Whale movements: {onchain.get('whale_movements', 'N/A')}

Provide:
1. Sentiment score with reasoning
2. Verification of other AI signals
3. Risk assessment
4. Key market drivers"""

    async def _mock_claude_analysis(self, technical: Dict, news: List,
                                   onchain: Dict) -> Tuple[float, float, str, List[str]]:
        """Mock Claude analysis"""
        # Combine multiple factors for sentiment
        sentiment_components = []

        # Technical sentiment
        rsi = technical.get('rsi', 50)
        if rsi > 70:
            sentiment_components.append(-0.3)  # Overbought
        elif rsi < 30:
            sentiment_components.append(0.3)  # Oversold
        else:
            sentiment_components.append(0)

        # News sentiment (simplified)
        if news:
            news_sentiment = np.random.uniform(-0.5, 0.5)
            sentiment_components.append(news_sentiment)

        # On-chain sentiment
        if onchain.get('exchange_flows', 0) < 0:
            sentiment_components.append(0.2)  # Outflows = bullish
        else:
            sentiment_components.append(-0.2)

        sentiment_score = np.mean(sentiment_components) if sentiment_components else 0
        confidence = 0.7

        key_factors = [
            f"RSI at {rsi}",
            "Technical indicators mixed",
            "On-chain metrics neutral"
        ]

        reasoning = "Comprehensive analysis shows balanced market conditions with slight directional bias"

        return sentiment_score, confidence, reasoning, key_factors

class DeepseekAnalyzer(AIModelInterface):
    """Deepseek AI for DeFi and on-chain analysis"""

    async def analyze_sentiment(self, data: Dict) -> SentimentResult:
        """Analyze DeFi and on-chain sentiment using Deepseek"""
        await self._rate_limit_check()

        try:
            defi_data = data.get('defi', {})
            onchain_data = data.get('onchain', {})
            whale_data = data.get('whale_movements', [])

            # Mock Deepseek analysis
            sentiment_score, confidence, reasoning, key_factors = await self._mock_deepseek_analysis(
                defi_data, onchain_data, whale_data
            )

            return SentimentResult(
                source="DeFi+OnChain",
                model=AIModel.DEEPSEEK,
                sentiment_score=sentiment_score,
                confidence=confidence,
                reasoning=reasoning,
                key_factors=key_factors,
                timestamp=datetime.now(),
                raw_data={'defi': defi_data, 'whale_count': len(whale_data)}
            )

        except Exception as e:
            logger.error(f"Deepseek analysis failed: {e}")
            return None

    async def _mock_deepseek_analysis(self, defi: Dict, onchain: Dict,
                                     whales: List) -> Tuple[float, float, str, List[str]]:
        """Mock Deepseek analysis for DeFi focus"""
        sentiment_score = 0

        # DeFi metrics impact
        tvl_change = defi.get('tvl_change', 0)
        if tvl_change > 0:
            sentiment_score += 0.3
        elif tvl_change < 0:
            sentiment_score -= 0.3

        # Whale activity
        if len(whales) > 5:
            sentiment_score += 0.2 if onchain.get('exchange_flows', 0) < 0 else -0.2

        # Liquidations
        if defi.get('liquidations_24h', 0) > 100_000_000:
            sentiment_score -= 0.4

        sentiment_score = np.clip(sentiment_score, -1, 1)
        confidence = 0.65

        key_factors = [
            f"TVL change: {tvl_change}%",
            f"Whale movements: {len(whales)}",
            "DeFi activity moderate"
        ]

        reasoning = "DeFi metrics and on-chain analysis indicate market positioning"

        return sentiment_score, confidence, reasoning, key_factors

class MultiAISentimentAnalyzer:
    """Orchestrates multiple AI models for consensus sentiment"""

    def __init__(self):
        self.analyzers = {
            AIModel.GROK: GrokAnalyzer(),
            AIModel.CLAUDE: ClaudeAnalyzer(),
            AIModel.DEEPSEEK: DeepseekAnalyzer()
        }

        # Traditional sentiment tools
        self.vader = SentimentIntensityAnalyzer()

        # Results cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

    async def analyze_comprehensive_sentiment(self, market_data: Dict) -> ConsensusSentiment:
        """Get consensus sentiment from all AI models"""
        cache_key = f"sentiment_{market_data.get('symbol', 'BTC')}_{int(time.time() // 300)}"

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Prepare data for each AI model
        grok_data = {
            'tweets': market_data.get('social', {}).get('twitter', []),
            'trending': market_data.get('social', {}).get('trending', []),
            'symbol': market_data.get('symbol', 'BTC')
        }

        claude_data = {
            'technical': market_data.get('technical', {}),
            'news': market_data.get('news', []),
            'onchain': market_data.get('onchain', {})
        }

        deepseek_data = {
            'defi': market_data.get('defi', {}),
            'onchain': market_data.get('onchain', {}),
            'whale_movements': market_data.get('whale_movements', [])
        }

        # Run all analyses in parallel
        tasks = [
            self.analyzers[AIModel.GROK].analyze_sentiment(grok_data),
            self.analyzers[AIModel.CLAUDE].analyze_sentiment(claude_data),
            self.analyzers[AIModel.DEEPSEEK].analyze_sentiment(deepseek_data)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed analyses
        valid_results = [r for r in results if isinstance(r, SentimentResult)]

        if not valid_results:
            logger.error("All AI analyses failed")
            return None

        # Calculate consensus
        consensus = self._calculate_consensus(valid_results)

        # Cache the result
        self.cache[cache_key] = consensus

        return consensus

    def _calculate_consensus(self, results: List[SentimentResult]) -> ConsensusSentiment:
        """Calculate consensus from multiple AI results"""
        if not results:
            return None

        # Weight models based on their specialization and confidence
        weights = {
            AIModel.GROK: 0.3,  # Social sentiment specialist
            AIModel.CLAUDE: 0.4,  # Technical verification
            AIModel.DEEPSEEK: 0.3  # On-chain specialist
        }

        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0

        for result in results:
            model_weight = weights.get(result.model, 0.33)
            confidence_adjusted_weight = model_weight * result.confidence
            weighted_sentiment += result.sentiment_score * confidence_adjusted_weight
            total_weight += confidence_adjusted_weight

        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0

        # Calculate agreement score (how much models agree)
        sentiments = [r.sentiment_score for r in results]
        agreement_score = 1 - np.std(sentiments) if len(sentiments) > 1 else 1

        # Overall confidence (based on agreement and individual confidences)
        avg_confidence = np.mean([r.confidence for r in results])
        overall_confidence = avg_confidence * (0.5 + 0.5 * agreement_score)

        # Determine recommendation
        if overall_sentiment > 0.5 and overall_confidence > 0.7:
            recommendation = "strong_buy"
        elif overall_sentiment > 0.2:
            recommendation = "buy"
        elif overall_sentiment < -0.5 and overall_confidence > 0.7:
            recommendation = "strong_sell"
        elif overall_sentiment < -0.2:
            recommendation = "sell"
        else:
            recommendation = "neutral"

        # Aggregate key insights
        all_factors = []
        for result in results:
            all_factors.extend(result.key_factors[:2])  # Top 2 from each

        # Remove duplicates while preserving order
        key_insights = list(dict.fromkeys(all_factors))[:5]

        return ConsensusSentiment(
            overall_sentiment=overall_sentiment,
            confidence=overall_confidence,
            agreement_score=agreement_score,
            individual_results=results,
            recommendation=recommendation,
            key_insights=key_insights,
            timestamp=datetime.now()
        )

    async def verify_sentiment_divergence(self, consensus: ConsensusSentiment) -> Dict:
        """Identify and analyze divergences between AI models"""
        if not consensus or len(consensus.individual_results) < 2:
            return None

        divergences = []

        # Check for significant disagreements
        for i, result1 in enumerate(consensus.individual_results):
            for result2 in consensus.individual_results[i+1:]:
                diff = abs(result1.sentiment_score - result2.sentiment_score)
                if diff > 0.5:  # Significant divergence
                    divergences.append({
                        'models': (result1.model.value, result2.model.value),
                        'difference': diff,
                        'sentiment_1': result1.sentiment_score,
                        'sentiment_2': result2.sentiment_score,
                        'reasons': {
                            result1.model.value: result1.reasoning,
                            result2.model.value: result2.reasoning
                        }
                    })

        # Analyze divergence patterns
        if divergences:
            return {
                'has_divergence': True,
                'divergence_count': len(divergences),
                'max_divergence': max(d['difference'] for d in divergences),
                'divergences': divergences,
                'recommendation': 'caution',
                'suggested_action': 'Wait for clearer consensus or reduce position size'
            }

        return {
            'has_divergence': False,
            'recommendation': 'proceed',
            'confidence_boost': 1.1  # Boost confidence when models agree
        }

    def get_historical_accuracy(self, model: AIModel, lookback_days: int = 30) -> Dict:
        """Track historical accuracy of each AI model"""
        # In production, this would query a database of past predictions vs outcomes
        # For demonstration, returning mock data

        mock_accuracy = {
            AIModel.GROK: {
                'accuracy': 0.65,
                'precision': 0.68,
                'recall': 0.62,
                'best_for': 'Short-term momentum',
                'weak_for': 'Long-term trends'
            },
            AIModel.CLAUDE: {
                'accuracy': 0.72,
                'precision': 0.75,
                'recall': 0.70,
                'best_for': 'Technical verification',
                'weak_for': 'Social sentiment spikes'
            },
            AIModel.DEEPSEEK: {
                'accuracy': 0.68,
                'precision': 0.70,
                'recall': 0.66,
                'best_for': 'On-chain analysis',
                'weak_for': 'News-driven events'
            }
        }

        return mock_accuracy.get(model, {})