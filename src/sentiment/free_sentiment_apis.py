"""
FREE Sentiment Analysis APIs
No API keys required for basic functionality
"""

import aiohttp
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FearGreedIndexAPI:
    """
    Alternative.me Crypto Fear & Greed Index
    100% FREE - No API key needed
    """

    def __init__(self):
        self.base_url = "https://api.alternative.me/fng/"

    async def get_current_index(self) -> Dict:
        """Get current Fear & Greed Index (0-100)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params={"limit": 1}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data'):
                            index_data = data['data'][0]
                            return {
                                'value': int(index_data['value']),
                                'classification': index_data['value_classification'],
                                'timestamp': datetime.fromtimestamp(int(index_data['timestamp'])),
                                'sentiment_score': self._normalize_sentiment(int(index_data['value']))
                            }
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return None

    def _normalize_sentiment(self, value: int) -> float:
        """
        Normalize 0-100 to -1 to +1 scale
        0-25: Extreme Fear (-1 to -0.5)
        25-45: Fear (-0.5 to -0.1)
        45-55: Neutral (-0.1 to +0.1)
        55-75: Greed (+0.1 to +0.5)
        75-100: Extreme Greed (+0.5 to +1)
        """
        return (value - 50) / 50


class CoinGeckoAPI:
    """
    CoinGecko Free API
    50 calls/minute - No API key needed
    """

    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        self.last_call = 0
        self.min_interval = 1.2  # seconds between calls (to stay under rate limit)

    async def _rate_limit(self):
        """Simple rate limiting"""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_call
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_call = asyncio.get_event_loop().time()

    async def get_coin_sentiment(self, coin_id: str) -> Dict:
        """
        Get sentiment data for a specific coin
        coin_id: bitcoin, ethereum, solana, binancecoin, etc.
        """
        await self._rate_limit()

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/coins/{coin_id}"
                params = {
                    'localization': 'false',
                    'tickers': 'false',
                    'market_data': 'true',
                    'community_data': 'true',
                    'developer_data': 'false',
                    'sparkline': 'false'
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        market_data = data.get('market_data', {})
                        community = data.get('community_data', {})

                        return {
                            'coin': coin_id,
                            'sentiment_votes_up_percentage': data.get('sentiment_votes_up_percentage', 50),
                            'sentiment_votes_down_percentage': data.get('sentiment_votes_down_percentage', 50),
                            'market_cap_rank': market_data.get('market_cap_rank'),
                            'price_change_24h_pct': market_data.get('price_change_percentage_24h', 0),
                            'price_change_7d_pct': market_data.get('price_change_percentage_7d', 0),
                            'trading_volume_rank': data.get('market_cap_rank'),
                            'twitter_followers': community.get('twitter_followers'),
                            'reddit_subscribers': community.get('reddit_subscribers'),
                            'sentiment_score': self._calculate_sentiment(data, market_data)
                        }
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data for {coin_id}: {e}")
            return None

    def _calculate_sentiment(self, data: Dict, market_data: Dict) -> float:
        """
        Calculate overall sentiment score (-1 to +1)
        Based on: price action, community votes, volume
        """
        sentiment_up = data.get('sentiment_votes_up_percentage', 50)
        price_change_24h = market_data.get('price_change_percentage_24h', 0)

        # Weighted average
        vote_sentiment = (sentiment_up - 50) / 50  # -1 to +1
        price_sentiment = max(-1, min(1, price_change_24h / 10))  # Normalize price change

        # 60% votes, 40% price action
        overall = (vote_sentiment * 0.6) + (price_sentiment * 0.4)

        return round(overall, 3)

    async def get_trending_coins(self) -> List[Dict]:
        """Get trending coins (indicates market interest)"""
        await self._rate_limit()

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/search/trending"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        trending = []

                        for item in data.get('coins', []):
                            coin = item.get('item', {})
                            trending.append({
                                'id': coin.get('id'),
                                'symbol': coin.get('symbol'),
                                'name': coin.get('name'),
                                'market_cap_rank': coin.get('market_cap_rank'),
                                'score': coin.get('score', 0)
                            })

                        return trending
        except Exception as e:
            logger.error(f"Error fetching trending coins: {e}")
            return []


class CryptoCompareAPI:
    """
    CryptoCompare Free API
    News and social data - No API key needed for basic tier
    """

    def __init__(self):
        self.base_url = "https://min-api.cryptocompare.com/data/v2"

    async def get_latest_news(self, categories: str = "BTC,ETH,Trading") -> List[Dict]:
        """
        Get latest crypto news
        Returns list of news articles with sentiment indicators
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/news/"
                params = {
                    'categories': categories,
                    'excludeCategories': 'Sponsored'
                }

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        news_items = []

                        for article in data.get('Data', [])[:10]:  # Top 10 news
                            news_items.append({
                                'title': article.get('title'),
                                'body': article.get('body', '')[:200],  # First 200 chars
                                'source': article.get('source'),
                                'published_on': datetime.fromtimestamp(article.get('published_on', 0)),
                                'categories': article.get('categories', ''),
                                'url': article.get('url'),
                                'sentiment': self._analyze_title_sentiment(article.get('title', ''))
                            })

                        return news_items
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def _analyze_title_sentiment(self, title: str) -> str:
        """
        Simple keyword-based sentiment analysis of news titles
        """
        title_lower = title.lower()

        positive_words = ['surge', 'rally', 'boom', 'bullish', 'gains', 'rises',
                         'up', 'breakthrough', 'adoption', 'success', 'wins']
        negative_words = ['crash', 'dump', 'bearish', 'losses', 'falls', 'down',
                         'hack', 'scam', 'fails', 'warning', 'concerns']

        positive_count = sum(1 for word in positive_words if word in title_lower)
        negative_count = sum(1 for word in negative_words if word in title_lower)

        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'

    async def get_social_stats(self, symbol: str) -> Dict:
        """Get social media stats for a coin"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/social/coin/latest"
                params = {'coinId': symbol.upper()}

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        social_data = data.get('Data', {})

                        return {
                            'symbol': symbol,
                            'twitter_followers': social_data.get('Twitter', {}).get('followers'),
                            'reddit_subscribers': social_data.get('Reddit', {}).get('subscribers'),
                            'reddit_active_users': social_data.get('Reddit', {}).get('active_users'),
                            'points': social_data.get('Points', 0)
                        }
        except Exception as e:
            logger.error(f"Error fetching social stats for {symbol}: {e}")
            return None


class FreeSentimentAggregator:
    """
    Aggregates all free sentiment sources into a unified signal
    """

    def __init__(self):
        self.fear_greed = FearGreedIndexAPI()
        self.coingecko = CoinGeckoAPI()
        self.crypto_compare = CryptoCompareAPI()
        self._cache = {}  # Cache sentiment data to reduce API calls
        self._cache_duration = 300  # 5 minutes cache

    async def get_market_sentiment(self, symbol: str) -> Dict:
        """
        Get aggregated sentiment for a symbol
        Returns sentiment score, confidence, and supporting data
        """
        # Check cache first
        now = datetime.now()
        if symbol in self._cache:
            cache_entry = self._cache[symbol]
            age = (now - cache_entry['timestamp']).total_seconds()
            if age < self._cache_duration:
                logger.info(f"Using cached sentiment for {symbol} (age: {age:.0f}s)")
                return cache_entry['data']

        # Map trading symbols to CoinGecko IDs
        coin_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'BNB': 'binancecoin',
            'USDT': 'tether'
        }

        coin_id = coin_map.get(symbol, symbol.lower())

        # Fetch all data concurrently
        fear_greed_task = self.fear_greed.get_current_index()
        coingecko_task = self.coingecko.get_coin_sentiment(coin_id)
        news_task = self.crypto_compare.get_latest_news()
        trending_task = self.coingecko.get_trending_coins()

        fear_greed, coingecko, news, trending = await asyncio.gather(
            fear_greed_task,
            coingecko_task,
            news_task,
            trending_task,
            return_exceptions=True
        )

        # Aggregate sentiment
        sentiment_scores = []
        sources = []

        # Fear & Greed Index
        if fear_greed and not isinstance(fear_greed, Exception):
            sentiment_scores.append(fear_greed['sentiment_score'])
            sources.append('fear_greed')

        # CoinGecko sentiment
        if coingecko and not isinstance(coingecko, Exception):
            sentiment_scores.append(coingecko['sentiment_score'])
            sources.append('coingecko')

        # News sentiment
        if news and not isinstance(news, Exception):
            news_sentiment = self._analyze_news_sentiment(news)
            sentiment_scores.append(news_sentiment)
            sources.append('news')

        # Trending status (if coin is trending, slight positive bias)
        if trending and not isinstance(trending, Exception):
            is_trending = any(t['symbol'].upper() == symbol for t in trending)
            if is_trending:
                sentiment_scores.append(0.2)  # Small positive boost
                sources.append('trending')

        # Calculate overall sentiment
        if sentiment_scores:
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = len(sources) / 4  # Max 4 sources
        else:
            overall_sentiment = 0
            confidence = 0

        result = {
            'symbol': symbol,
            'overall_sentiment': round(overall_sentiment, 3),
            'confidence': round(confidence, 2),
            'sources_count': len(sources),
            'sources': sources,
            'fear_greed_data': fear_greed if not isinstance(fear_greed, Exception) else None,
            'coingecko_data': coingecko if not isinstance(coingecko, Exception) else None,
            'news_items': news if not isinstance(news, Exception) else [],
            'is_trending': symbol in [t.get('symbol', '').upper() for t in (trending if not isinstance(trending, Exception) else [])],
            'timestamp': datetime.now()
        }

        # Cache the result
        self._cache[symbol] = {
            'data': result,
            'timestamp': now
        }
        logger.info(f"Cached fresh sentiment data for {symbol}")

        return result

    def _analyze_news_sentiment(self, news_items: List[Dict]) -> float:
        """Analyze overall news sentiment"""
        if not news_items:
            return 0

        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        sentiments = [sentiment_map.get(item['sentiment'], 0) for item in news_items]

        return sum(sentiments) / len(sentiments) if sentiments else 0
