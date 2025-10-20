"""
Social Media Data Collector
Collects data from Twitter/X, Reddit, Telegram, Discord for sentiment analysis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
import re
from dataclasses import dataclass
from collections import defaultdict
import time
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    """Social media post data structure"""
    platform: str
    post_id: str
    author: str
    content: str
    timestamp: datetime
    engagement: Dict  # likes, retweets, comments, etc.
    sentiment: Optional[float] = None
    influence_score: float = 0
    is_verified: bool = False
    hashtags: List[str] = None
    mentions: List[str] = None

@dataclass
class InfluencerMetrics:
    """Metrics for crypto influencers"""
    username: str
    platform: str
    followers: int
    engagement_rate: float
    accuracy_score: float  # Historical accuracy of calls
    sentiment_bias: float  # Tends bullish/bearish
    last_post: datetime

class TwitterCollector:
    """Collect and analyze Twitter/X data"""

    def __init__(self, api_key: str = "", bearer_token: str = ""):
        self.api_key = api_key
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"

        # Known crypto influencers (would be more comprehensive in production)
        self.influencers = {
            'elonmusk': 150_000_000,
            'michael_saylor': 3_000_000,
            'APompliano': 1_700_000,
            'CathieDWood': 1_500_000,
            'VitalikButerin': 5_000_000,
            'cz_binance': 8_000_000,
            'brian_armstrong': 1_200_000
        }

        # Cache
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

    async def collect_tweets(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Collect tweets based on query"""
        cache_key = f"tweets_{hashlib.md5(query.encode()).hexdigest()}_{limit}"

        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data

        try:
            # In production, would use actual Twitter API
            # For demonstration, generating mock data
            tweets = await self._mock_tweet_collection(query, limit)

            # Cache results
            self.cache[cache_key] = (tweets, time.time())

            return tweets

        except Exception as e:
            logger.error(f"Error collecting tweets: {e}")
            return []

    async def _mock_tweet_collection(self, query: str, limit: int) -> List[SocialPost]:
        """Mock tweet collection for demonstration"""
        tweets = []

        for i in range(min(limit, 100)):
            # Generate mock tweet data
            is_influencer = np.random.random() < 0.1
            author = np.random.choice(list(self.influencers.keys())) if is_influencer else f"user_{i}"

            # Generate engagement based on whether it's an influencer
            if is_influencer:
                likes = np.random.randint(1000, 50000)
                retweets = np.random.randint(500, 20000)
            else:
                likes = np.random.randint(0, 1000)
                retweets = np.random.randint(0, 500)

            # Generate content with crypto terms
            crypto_terms = ['BTC', 'ETH', 'bullish', 'bearish', 'moon', 'dump', 'pump',
                          'HODL', 'buy', 'sell', 'support', 'resistance', 'breakout']
            content = f"Mock tweet about {query}: " + ' '.join(
                np.random.choice(crypto_terms, size=np.random.randint(2, 5))
            )

            # Extract hashtags and mentions
            hashtags = [f"#{term}" for term in np.random.choice(crypto_terms, size=2)]
            mentions = [f"@{np.random.choice(list(self.influencers.keys()))}"
                       if np.random.random() < 0.3 else None]
            mentions = [m for m in mentions if m]

            tweet = SocialPost(
                platform="twitter",
                post_id=f"tweet_{i}",
                author=author,
                content=content,
                timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
                engagement={
                    'likes': likes,
                    'retweets': retweets,
                    'replies': np.random.randint(0, 100),
                    'views': likes * np.random.randint(10, 100)
                },
                is_verified=is_influencer,
                hashtags=hashtags,
                mentions=mentions,
                influence_score=self._calculate_influence_score(author, likes, retweets)
            )

            tweets.append(tweet)

        return sorted(tweets, key=lambda x: x.timestamp, reverse=True)

    def _calculate_influence_score(self, author: str, likes: int, retweets: int) -> float:
        """Calculate influence score for a tweet"""
        base_score = 0

        # Author influence
        if author in self.influencers:
            followers = self.influencers[author]
            base_score += np.log10(followers) / 10  # Normalize to 0-1 range

        # Engagement influence
        engagement_score = (likes + retweets * 2) / 100000  # Normalize
        engagement_score = min(1, engagement_score)

        return min(1, base_score + engagement_score)

    async def get_trending_topics(self, location: str = "global") -> List[Dict]:
        """Get trending crypto-related topics"""
        try:
            # Mock trending topics
            topics = [
                {'name': '#Bitcoin', 'tweet_volume': np.random.randint(10000, 100000)},
                {'name': '#Ethereum', 'tweet_volume': np.random.randint(5000, 50000)},
                {'name': '#DeFi', 'tweet_volume': np.random.randint(1000, 20000)},
                {'name': '#NFTs', 'tweet_volume': np.random.randint(1000, 15000)},
                {'name': f'#{np.random.choice(["Bullish", "Bearish", "HODL", "Moon"])}',
                 'tweet_volume': np.random.randint(1000, 10000)}
            ]

            return sorted(topics, key=lambda x: x['tweet_volume'], reverse=True)

        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []

    async def analyze_influencer_sentiment(self, symbol: str = "BTC") -> Dict:
        """Analyze sentiment from key influencers"""
        influencer_sentiments = {}

        for influencer, followers in self.influencers.items():
            # Mock sentiment analysis for each influencer
            sentiment = np.random.uniform(-1, 1)
            confidence = min(0.9, 0.5 + abs(sentiment) * 0.5)

            influencer_sentiments[influencer] = {
                'sentiment': sentiment,
                'confidence': confidence,
                'followers': followers,
                'recent_posts': np.random.randint(0, 10),
                'influence_weight': np.log10(followers) / 10
            }

        # Calculate weighted sentiment
        total_weight = sum(i['influence_weight'] for i in influencer_sentiments.values())
        weighted_sentiment = sum(
            i['sentiment'] * i['influence_weight'] for i in influencer_sentiments.values()
        ) / total_weight if total_weight > 0 else 0

        return {
            'individual_sentiments': influencer_sentiments,
            'weighted_sentiment': weighted_sentiment,
            'top_bullish': sorted(
                [(k, v['sentiment']) for k, v in influencer_sentiments.items()
                 if v['sentiment'] > 0],
                key=lambda x: x[1], reverse=True
            )[:3],
            'top_bearish': sorted(
                [(k, v['sentiment']) for k, v in influencer_sentiments.items()
                 if v['sentiment'] < 0],
                key=lambda x: x[1]
            )[:3]
        }

class RedditCollector:
    """Collect and analyze Reddit data"""

    def __init__(self, client_id: str = "", client_secret: str = ""):
        self.client_id = client_id
        self.client_secret = client_secret

        # Key crypto subreddits
        self.subreddits = [
            'cryptocurrency',
            'bitcoin',
            'ethereum',
            'defi',
            'altcoin',
            'cryptomarkets',
            'bitcoinmarkets',
            'ethtrader',
            'cryptomoonshots',
            'satoshistreetbets'
        ]

        self.cache = {}
        self.cache_duration = 600  # 10 minutes

    async def collect_posts(self, subreddit: str = "cryptocurrency",
                           sort: str = "hot", limit: int = 50) -> List[SocialPost]:
        """Collect Reddit posts from specified subreddit"""
        cache_key = f"reddit_{subreddit}_{sort}_{limit}"

        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data

        try:
            # Mock Reddit data collection
            posts = await self._mock_reddit_collection(subreddit, sort, limit)

            self.cache[cache_key] = (posts, time.time())
            return posts

        except Exception as e:
            logger.error(f"Error collecting Reddit posts: {e}")
            return []

    async def _mock_reddit_collection(self, subreddit: str, sort: str,
                                     limit: int) -> List[SocialPost]:
        """Mock Reddit post collection"""
        posts = []

        for i in range(min(limit, 50)):
            # Generate mock post data
            upvotes = np.random.randint(1, 10000) if sort == "hot" else np.random.randint(0, 1000)
            downvotes = np.random.randint(0, upvotes // 2)

            post = SocialPost(
                platform="reddit",
                post_id=f"reddit_{subreddit}_{i}",
                author=f"redditor_{np.random.randint(1, 1000)}",
                content=f"Mock Reddit post about crypto in r/{subreddit}",
                timestamp=datetime.now() - timedelta(hours=np.random.randint(0, 72)),
                engagement={
                    'upvotes': upvotes,
                    'downvotes': downvotes,
                    'comments': np.random.randint(0, 500),
                    'awards': np.random.randint(0, 10),
                    'upvote_ratio': upvotes / (upvotes + downvotes) if (upvotes + downvotes) > 0 else 0
                },
                influence_score=(upvotes - downvotes) / 10000  # Simple influence metric
            )

            posts.append(post)

        return sorted(posts, key=lambda x: x.engagement['upvotes'], reverse=True)

    async def get_sentiment_distribution(self, subreddits: List[str] = None) -> Dict:
        """Get sentiment distribution across subreddits"""
        if not subreddits:
            subreddits = self.subreddits[:5]  # Top 5 subreddits

        distributions = {}

        for subreddit in subreddits:
            # Mock sentiment distribution
            distributions[subreddit] = {
                'bullish': np.random.uniform(0.2, 0.6),
                'neutral': np.random.uniform(0.2, 0.4),
                'bearish': np.random.uniform(0.1, 0.4)
            }

            # Normalize to sum to 1
            total = sum(distributions[subreddit].values())
            for key in distributions[subreddit]:
                distributions[subreddit][key] /= total

        # Calculate overall distribution
        overall = {
            'bullish': np.mean([d['bullish'] for d in distributions.values()]),
            'neutral': np.mean([d['neutral'] for d in distributions.values()]),
            'bearish': np.mean([d['bearish'] for d in distributions.values()])
        }

        return {
            'by_subreddit': distributions,
            'overall': overall,
            'dominant_sentiment': max(overall, key=overall.get)
        }

class SocialMediaAggregator:
    """Aggregate social media data from multiple platforms"""

    def __init__(self):
        self.twitter = TwitterCollector()
        self.reddit = RedditCollector()

        # Platform weights for overall sentiment
        self.platform_weights = {
            'twitter': 0.4,
            'reddit': 0.3,
            'telegram': 0.15,
            'discord': 0.15
        }

    async def collect_all_social_data(self, symbol: str = "BTC",
                                     lookback_hours: int = 24) -> Dict:
        """Collect data from all social platforms"""
        # Prepare queries
        twitter_query = f"${symbol} OR #{symbol} OR {self._get_full_name(symbol)}"

        # Collect from all platforms in parallel
        tasks = [
            self.twitter.collect_tweets(twitter_query, limit=200),
            self.twitter.get_trending_topics(),
            self.twitter.analyze_influencer_sentiment(symbol),
            self.reddit.collect_posts("cryptocurrency", limit=100),
            self.reddit.collect_posts(symbol.lower(), limit=50),
            self.reddit.get_sentiment_distribution()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        twitter_posts = results[0] if not isinstance(results[0], Exception) else []
        trending = results[1] if not isinstance(results[1], Exception) else []
        influencer_sentiment = results[2] if not isinstance(results[2], Exception) else {}
        reddit_crypto = results[3] if not isinstance(results[3], Exception) else []
        reddit_symbol = results[4] if not isinstance(results[4], Exception) else []
        reddit_sentiment = results[5] if not isinstance(results[5], Exception) else {}

        return {
            'twitter': {
                'posts': twitter_posts,
                'trending': trending,
                'influencer_sentiment': influencer_sentiment,
                'post_count': len(twitter_posts)
            },
            'reddit': {
                'posts': reddit_crypto + reddit_symbol,
                'sentiment_distribution': reddit_sentiment,
                'post_count': len(reddit_crypto) + len(reddit_symbol)
            },
            'timestamp': datetime.now(),
            'symbol': symbol
        }

    def _get_full_name(self, symbol: str) -> str:
        """Get full name from symbol"""
        names = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'SOL': 'Solana',
            'BNB': 'Binance',
            'ADA': 'Cardano',
            'DOGE': 'Dogecoin'
        }
        return names.get(symbol, symbol)

    def calculate_social_momentum(self, current_data: Dict, historical_data: List[Dict]) -> Dict:
        """Calculate social momentum indicators"""
        if not historical_data or len(historical_data) < 2:
            return {'momentum': 0, 'acceleration': 0, 'trend': 'neutral'}

        # Calculate post volume momentum
        current_volume = (
            current_data.get('twitter', {}).get('post_count', 0) +
            current_data.get('reddit', {}).get('post_count', 0)
        )

        historical_volumes = [
            d.get('twitter', {}).get('post_count', 0) +
            d.get('reddit', {}).get('post_count', 0)
            for d in historical_data
        ]

        avg_volume = np.mean(historical_volumes) if historical_volumes else current_volume
        momentum = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0

        # Calculate acceleration (change in momentum)
        if len(historical_data) >= 2:
            prev_momentum = (historical_volumes[-1] - np.mean(historical_volumes[:-1])) / \
                          np.mean(historical_volumes[:-1]) if np.mean(historical_volumes[:-1]) > 0 else 0
            acceleration = momentum - prev_momentum
        else:
            acceleration = 0

        # Determine trend
        if momentum > 0.2 and acceleration > 0:
            trend = 'strongly_bullish'
        elif momentum > 0.1:
            trend = 'bullish'
        elif momentum < -0.2 and acceleration < 0:
            trend = 'strongly_bearish'
        elif momentum < -0.1:
            trend = 'bearish'
        else:
            trend = 'neutral'

        return {
            'momentum': momentum,
            'acceleration': acceleration,
            'trend': trend,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_spike': current_volume > avg_volume * 1.5
        }

    def detect_social_anomalies(self, data: Dict) -> List[Dict]:
        """Detect unusual social media activity"""
        anomalies = []

        # Check for coordinated campaigns
        twitter_posts = data.get('twitter', {}).get('posts', [])
        if twitter_posts:
            # Look for similar content posted within short timeframe
            recent_posts = [p for p in twitter_posts
                          if (datetime.now() - p.timestamp).seconds < 3600]

            if len(recent_posts) > 50:
                anomalies.append({
                    'type': 'volume_spike',
                    'platform': 'twitter',
                    'severity': 'high',
                    'description': f"Unusual volume: {len(recent_posts)} posts in last hour"
                })

        # Check for influencer coordination
        influencer_sentiment = data.get('twitter', {}).get('influencer_sentiment', {})
        if influencer_sentiment:
            sentiments = [s['sentiment'] for s in
                         influencer_sentiment.get('individual_sentiments', {}).values()]
            if len(sentiments) > 3 and np.std(sentiments) < 0.1:
                anomalies.append({
                    'type': 'influencer_coordination',
                    'platform': 'twitter',
                    'severity': 'medium',
                    'description': "Multiple influencers posting similar sentiment"
                })

        return anomalies