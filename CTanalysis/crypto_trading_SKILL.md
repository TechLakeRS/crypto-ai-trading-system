# Cryptocurrency Trading and Data Analysis Skill

## Overview
This skill provides comprehensive guidance for cryptocurrency data analysis, trading strategy development, and market sentiment analysis. It includes best practices developed through extensive testing for handling crypto market data, implementing technical indicators, performing sentiment analysis, and creating effective visualizations.

## Core Dependencies
```python
# Essential libraries
import pandas as pd
import numpy as np
import ccxt  # Unified crypto exchange API
import ta  # Technical Analysis library
import yfinance as yf  # For additional market data
from datetime import datetime, timedelta
import plotly.graph_objects as go  # Interactive charts
import requests
import json
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp  # For async API calls

# Sentiment Analysis
import tweepy  # Twitter/X API
import praw  # Reddit API
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Machine Learning (optional)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf  # For advanced predictions
```

## 1. Data Collection Framework

### 1.1 Exchange API Integration
```python
class CryptoDataCollector:
    def __init__(self, exchange_id='binance'):
        """Initialize exchange connection"""
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,  # CRITICAL: Always enable rate limiting
            'options': {
                'defaultType': 'spot',  # or 'future' for derivatives
            }
        })
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data with error handling
        Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Data quality checks
            df = self._validate_ohlcv(df)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._handle_api_error(e)
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data"""
        # Remove rows with any zero values (except volume)
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        
        # Check for anomalies
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df[['open', 'close']].max(axis=1)]
        df = df[df['low'] <= df[['open', 'close']].min(axis=1)]
        
        # Handle missing data
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    async def fetch_multiple_pairs(self, pairs: List[str], timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple trading pairs asynchronously"""
        async with aiohttp.ClientSession() as session:
            tasks = [self._async_fetch(pair, timeframe) for pair in pairs]
            results = await asyncio.gather(*tasks)
            return dict(zip(pairs, results))
```

### 1.2 Alternative Data Sources
```python
class AlternativeDataCollector:
    def fetch_fear_greed_index(self) -> Dict:
        """Fetch Crypto Fear & Greed Index"""
        url = "https://api.alternative.me/fng/"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification'],
                'timestamp': data['data'][0]['timestamp']
            }
        return None
    
    def fetch_blockchain_metrics(self, coin: str = 'bitcoin') -> Dict:
        """Fetch on-chain metrics from Glassnode or similar"""
        # Implementation depends on API keys and service
        metrics = {
            'hash_rate': None,
            'active_addresses': None,
            'transaction_volume': None,
            'exchange_inflows': None,
            'exchange_outflows': None
        }
        # Add actual API calls here
        return metrics
```

## 2. Technical Analysis Components

### 2.1 Indicator Calculations
```python
class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators"""
        df = df.copy()
        
        # Trend Indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Volume Indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Custom Indicators
        df = TechnicalIndicators._calculate_custom_indicators(df)
        
        return df
    
    @staticmethod
    def _calculate_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate proprietary indicators"""
        # Volume-Weighted Average Price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Momentum Score (composite indicator)
        df['momentum_score'] = (
            (df['rsi'] / 100) * 0.3 +
            (df['bb_percent'].clip(0, 1)) * 0.3 +
            ((df['macd_histogram'] > 0).astype(int)) * 0.4
        )
        
        return df
```

### 2.2 Pattern Recognition
```python
class PatternRecognition:
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        highs = df['high'].rolling(window=window).max()
        lows = df['low'].rolling(window=window).min()
        
        # Find pivot points
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            # Resistance: local maximum
            if df['high'].iloc[i] == highs.iloc[i]:
                if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                    resistance_levels.append(df['high'].iloc[i])
            
            # Support: local minimum
            if df['low'].iloc[i] == lows.iloc[i]:
                if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                    support_levels.append(df['low'].iloc[i])
        
        # Cluster nearby levels
        resistance_levels = PatternRecognition._cluster_levels(resistance_levels)
        support_levels = PatternRecognition._cluster_levels(support_levels)
        
        return {
            'resistance': resistance_levels[:5],  # Top 5 levels
            'support': support_levels[:5]
        }
    
    @staticmethod
    def _cluster_levels(levels: List[float], threshold: float = 0.01) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(np.mean(current_cluster))
        
        return sorted(clustered, reverse=True)
    
    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame) -> List[Dict]:
        """Detect common chart patterns"""
        patterns = []
        
        # Head and Shoulders
        if PatternRecognition._detect_head_shoulders(df):
            patterns.append({'pattern': 'head_and_shoulders', 'direction': 'bearish'})
        
        # Double Top/Bottom
        double_pattern = PatternRecognition._detect_double_pattern(df)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Triangle patterns
        triangle = PatternRecognition._detect_triangle(df)
        if triangle:
            patterns.append(triangle)
        
        return patterns
```

## 3. Sentiment Analysis Framework

### 3.1 Social Media Sentiment
```python
class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.twitter_api = None  # Initialize with API keys
        self.reddit_api = None   # Initialize with API keys
    
    def analyze_twitter_sentiment(self, keyword: str, limit: int = 100) -> Dict:
        """Analyze Twitter/X sentiment for cryptocurrency"""
        tweets = self._fetch_tweets(keyword, limit)
        
        sentiments = []
        for tweet in tweets:
            # Clean tweet text
            clean_text = self._clean_text(tweet['text'])
            
            # VADER sentiment
            vader_scores = self.vader.polarity_scores(clean_text)
            
            # TextBlob sentiment
            blob = TextBlob(clean_text)
            
            sentiments.append({
                'vader_compound': vader_scores['compound'],
                'textblob_polarity': blob.sentiment.polarity,
                'engagement': tweet['engagement_score']
            })
        
        # Weight by engagement
        df_sent = pd.DataFrame(sentiments)
        weighted_sentiment = np.average(
            df_sent['vader_compound'], 
            weights=df_sent['engagement']
        )
        
        return {
            'overall_sentiment': weighted_sentiment,
            'sentiment_distribution': {
                'positive': (df_sent['vader_compound'] > 0.05).mean(),
                'neutral': ((df_sent['vader_compound'] >= -0.05) & (df_sent['vader_compound'] <= 0.05)).mean(),
                'negative': (df_sent['vader_compound'] < -0.05).mean()
            },
            'sample_size': len(sentiments)
        }
    
    def analyze_reddit_sentiment(self, subreddits: List[str] = ['cryptocurrency', 'bitcoin']) -> Dict:
        """Analyze Reddit sentiment from crypto subreddits"""
        posts = []
        for subreddit in subreddits:
            posts.extend(self._fetch_reddit_posts(subreddit))
        
        sentiments = []
        for post in posts:
            text = f"{post['title']} {post['body']}"
            clean_text = self._clean_text(text)
            
            scores = self.vader.polarity_scores(clean_text)
            sentiments.append({
                'compound': scores['compound'],
                'upvote_ratio': post['upvote_ratio'],
                'num_comments': post['num_comments']
            })
        
        df_sent = pd.DataFrame(sentiments)
        
        # Calculate weighted sentiment
        df_sent['weight'] = df_sent['upvote_ratio'] * np.log1p(df_sent['num_comments'])
        weighted_sentiment = np.average(df_sent['compound'], weights=df_sent['weight'])
        
        return {
            'overall_sentiment': weighted_sentiment,
            'trending_topics': self._extract_trending_topics(posts),
            'sample_size': len(posts)
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def aggregate_sentiment_signals(self, sources: Dict) -> float:
        """Combine sentiment from multiple sources into single score"""
        weights = {
            'twitter': 0.3,
            'reddit': 0.25,
            'news': 0.25,
            'fear_greed': 0.2
        }
        
        total_score = 0
        total_weight = 0
        
        for source, weight in weights.items():
            if source in sources and sources[source] is not None:
                # Normalize to -1 to 1 range
                normalized_score = np.tanh(sources[source])
                total_score += normalized_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
```

### 3.2 News Sentiment Analysis
```python
class NewsAnalyzer:
    def analyze_crypto_news(self, coin: str = 'bitcoin') -> Dict:
        """Analyze news sentiment from multiple sources"""
        news_sources = [
            self._fetch_coindesk_news,
            self._fetch_cointelegraph_news,
            self._fetch_cryptocompare_news
        ]
        
        all_articles = []
        for source_func in news_sources:
            articles = source_func(coin)
            all_articles.extend(articles)
        
        # Analyze sentiment
        sentiments = []
        for article in all_articles:
            sentiment = self._analyze_article_sentiment(article)
            sentiments.append({
                'sentiment': sentiment,
                'timestamp': article['timestamp'],
                'source_reliability': article.get('reliability', 0.5)
            })
        
        df_news = pd.DataFrame(sentiments)
        
        # Time-decay weighted sentiment
        current_time = datetime.now()
        df_news['age_hours'] = (current_time - df_news['timestamp']).dt.total_seconds() / 3600
        df_news['time_weight'] = np.exp(-df_news['age_hours'] / 24)  # 24-hour half-life
        df_news['final_weight'] = df_news['time_weight'] * df_news['source_reliability']
        
        weighted_sentiment = np.average(df_news['sentiment'], weights=df_news['final_weight'])
        
        return {
            'overall_sentiment': weighted_sentiment,
            'recent_sentiment': df_news[df_news['age_hours'] < 6]['sentiment'].mean(),
            'sentiment_trend': self._calculate_sentiment_trend(df_news),
            'article_count': len(df_news)
        }
    
    def _calculate_sentiment_trend(self, df: pd.DataFrame) -> str:
        """Determine if sentiment is improving or declining"""
        if len(df) < 10:
            return 'insufficient_data'
        
        df_sorted = df.sort_values('timestamp')
        recent = df_sorted.tail(len(df) // 2)['sentiment'].mean()
        older = df_sorted.head(len(df) // 2)['sentiment'].mean()
        
        if recent > older + 0.1:
            return 'improving'
        elif recent < older - 0.1:
            return 'declining'
        else:
            return 'stable'
```

## 4. Trading Strategy Framework

### 4.1 Strategy Implementation
```python
class TradingStrategy:
    def __init__(self, initial_capital: float = 10000):
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        
    def mean_reversion_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands mean reversion strategy"""
        df = df.copy()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['close'] < df['bb_lower'], 'signal'] = 1  # Buy signal
        df.loc[df['close'] > df['bb_upper'], 'signal'] = -1  # Sell signal
        
        # Position sizing based on BB width (volatility)
        df['position_size'] = 1 / (1 + df['bb_width'] / df['close'])
        
        return df
    
    def momentum_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-indicator momentum strategy"""
        df = df.copy()
        
        # Combine multiple momentum indicators
        conditions_buy = (
            (df['rsi'] > 50) & (df['rsi'] < 70) &  # RSI in bullish zone
            (df['macd'] > df['macd_signal']) &      # MACD bullish
            (df['close'] > df['sma_20']) &          # Price above short MA
            (df['volume'] > df['volume_sma'])       # Volume confirmation
        )
        
        conditions_sell = (
            (df['rsi'] < 50) |                      # RSI bearish
            (df['macd'] < df['macd_signal']) |      # MACD bearish
            (df['close'] < df['sma_20'])            # Price below MA
        )
        
        df['signal'] = 0
        df.loc[conditions_buy, 'signal'] = 1
        df.loc[conditions_sell, 'signal'] = -1
        
        return df
    
    def sentiment_enhanced_strategy(self, df: pd.DataFrame, sentiment: float) -> pd.DataFrame:
        """Combine technical and sentiment analysis"""
        df = df.copy()
        
        # Base technical signal
        df = self.momentum_strategy(df)
        
        # Adjust signals based on sentiment
        sentiment_multiplier = 1 + np.tanh(sentiment)  # Range: 0 to 2
        
        # Enhance buy signals with positive sentiment
        df.loc[df['signal'] == 1, 'position_size'] = df.loc[df['signal'] == 1, 'position_size'] * sentiment_multiplier
        
        # Reduce or reverse positions with negative sentiment
        if sentiment < -0.5:
            df.loc[df['signal'] == 1, 'signal'] = 0  # Cancel buy signals
            df.loc[df['signal'] == 0, 'signal'] = -1  # Add sell bias
        
        return df
```

### 4.2 Risk Management
```python
class RiskManager:
    def __init__(self, max_position_size: float = 0.1, stop_loss: float = 0.05):
        self.max_position_size = max_position_size  # Max 10% per trade
        self.stop_loss = stop_loss  # 5% stop loss
        self.max_drawdown = 0.2  # 20% maximum drawdown
        
    def calculate_position_size(self, 
                               capital: float, 
                               confidence: float,
                               volatility: float) -> float:
        """Kelly Criterion-based position sizing"""
        # Simplified Kelly formula
        win_probability = 0.5 + (confidence * 0.3)  # Confidence affects win probability
        win_loss_ratio = 1.5  # Assume 1.5:1 reward/risk
        
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Adjust for volatility
        volatility_adjustment = 1 / (1 + volatility)
        
        # Apply maximum position size constraint
        position_size = min(
            kelly_fraction * volatility_adjustment,
            self.max_position_size
        )
        
        return max(0, position_size) * capital
    
    def apply_stop_loss(self, entry_price: float, current_price: float, position: str) -> bool:
        """Check if stop loss should be triggered"""
        if position == 'long':
            return (entry_price - current_price) / entry_price > self.stop_loss
        elif position == 'short':
            return (current_price - entry_price) / entry_price > self.stop_loss
        return False
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive risk metrics"""
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annual_return': returns.mean() * 365,
            'volatility': returns.std() * np.sqrt(365),
            'sharpe_ratio': self._calculate_sharpe(returns),
            'sortino_ratio': self._calculate_sortino(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
        }
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 365
        if returns.std() == 0:
            return 0
        return np.sqrt(365) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside risk)"""
        excess_returns = returns - risk_free_rate / 365
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return np.sqrt(365) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
```

## 5. Backtesting Framework

### 5.1 Backtesting Engine
```python
class Backtester:
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, df: pd.DataFrame, strategy_func) -> Dict:
        """Run backtest on historical data"""
        df = strategy_func(df)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Check for signal
            if row['signal'] == 1 and position <= 0:  # Buy signal
                # Close short position if exists
                if position < 0:
                    capital = self._close_position(capital, position, row['close'], trades)
                    position = 0
                
                # Open long position
                position_size = row.get('position_size', 1.0)
                position = (capital * position_size) / row['close']
                capital -= position * row['close'] * (1 + self.commission)
                
                trades.append({
                    'timestamp': row.name,
                    'type': 'buy',
                    'price': row['close'],
                    'size': position
                })
                
            elif row['signal'] == -1 and position >= 0:  # Sell signal
                # Close long position if exists
                if position > 0:
                    capital = self._close_position(capital, position, row['close'], trades)
                    position = 0
                
                # Open short position (if allowed)
                # position = -(capital * position_size) / row['close']
                # capital += abs(position) * row['close'] * (1 - self.commission)
            
            # Track equity
            equity = capital + position * row['close'] if position != 0 else capital
            equity_curve.append({
                'timestamp': row.name,
                'equity': equity,
                'position': position
            })
        
        # Close final position
        if position != 0:
            capital = self._close_position(capital, position, df.iloc[-1]['close'], trades)
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        risk_manager = RiskManager()
        metrics = risk_manager.calculate_risk_metrics(equity_df['returns'].dropna())
        metrics['total_trades'] = len(trades)
        metrics['win_rate'] = self._calculate_win_rate(trades)
        metrics['profit_factor'] = self._calculate_profit_factor(trades)
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_df
        }
    
    def _close_position(self, capital: float, position: float, price: float, trades: List) -> float:
        """Close position and return updated capital"""
        if position > 0:  # Close long
            capital += position * price * (1 - self.commission)
            trades.append({
                'timestamp': datetime.now(),
                'type': 'sell',
                'price': price,
                'size': position
            })
        elif position < 0:  # Close short
            capital -= abs(position) * price * (1 + self.commission)
            trades.append({
                'timestamp': datetime.now(),
                'type': 'buy_to_cover',
                'price': price,
                'size': abs(position)
            })
        return capital
```

## 6. Visualization Standards

### 6.1 Interactive Charts
```python
class CryptoVisualizer:
    @staticmethod
    def create_candlestick_chart(df: pd.DataFrame, indicators: List[str] = None) -> go.Figure:
        """Create interactive candlestick chart with indicators"""
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Add indicators
        if indicators:
            for indicator in indicators:
                if indicator in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        mode='lines',
                        name=indicator.upper(),
                        line=dict(width=1)
                    ))
        
        # Add volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.4)',
            yaxis='y2'
        ))
        
        # Layout
        fig.update_layout(
            title='Cryptocurrency Price Analysis',
            yaxis_title='Price',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            xaxis_rangeslider_visible=False,
            template='plotly_dark',
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_sentiment_dashboard(sentiment_data: Dict) -> go.Figure:
        """Create sentiment analysis dashboard"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Sentiment', 'Source Distribution', 
                          'Sentiment Trend', 'Fear & Greed Index'),
            specs=[[{'type': 'indicator'}, {'type': 'pie'}],
                   [{'type': 'scatter'}, {'type': 'gauge'}]]
        )
        
        # Overall sentiment gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=sentiment_data['overall'],
            title={'text': "Sentiment Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [-1, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [-1, -0.5], 'color': "red"},
                       {'range': [-0.5, 0], 'color': "orange"},
                       {'range': [0, 0.5], 'color': "lightgreen"},
                       {'range': [0.5, 1], 'color': "green"}]}
        ), row=1, col=1)
        
        # Source distribution
        fig.add_trace(go.Pie(
            labels=['Twitter', 'Reddit', 'News'],
            values=[sentiment_data.get('twitter', 0),
                   sentiment_data.get('reddit', 0),
                   sentiment_data.get('news', 0)],
            hole=.3
        ), row=1, col=2)
        
        # Update layout
        fig.update_layout(height=600, showlegend=False, template='plotly_dark')
        
        return fig
    
    @staticmethod
    def create_portfolio_dashboard(portfolio_data: Dict) -> go.Figure:
        """Create portfolio performance dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Asset Allocation', 
                          'Daily Returns', 'Risk Metrics',
                          'Win/Loss Distribution', 'Drawdown'),
            specs=[[{'type': 'scatter'}, {'type': 'pie'}],
                   [{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Portfolio value over time
        equity = portfolio_data['equity_curve']
        fig.add_trace(go.Scatter(
            x=equity['timestamp'],
            y=equity['equity'],
            mode='lines',
            name='Portfolio Value',
            fill='tozeroy'
        ), row=1, col=1)
        
        # Add more dashboard components...
        
        fig.update_layout(height=900, showlegend=False, template='plotly_dark')
        return fig
```

## 7. Best Practices and Common Pitfalls

### 7.1 Data Quality
- **Always validate OHLC relationships**: High ≥ Low, High ≥ Max(Open, Close)
- **Handle missing data carefully**: Forward fill for prices, zero fill for volume
- **Check for outliers**: Use IQR method or z-score to detect anomalies
- **Timezone management**: Always work in UTC, convert for display only

### 7.2 API Usage
- **Rate limiting is critical**: Never disable rate limiting
- **Use exponential backoff**: For failed requests, wait 2^n seconds before retry
- **Cache frequently used data**: Reduce API calls for historical data
- **Handle API errors gracefully**: Always have fallback data sources

### 7.3 Backtesting Pitfalls
- **Look-ahead bias**: Never use future data in calculations
- **Survivorship bias**: Include delisted coins in historical analysis
- **Overfitting**: Use walk-forward analysis and out-of-sample testing
- **Transaction costs**: Always include fees and slippage

### 7.4 Risk Management
- **Position sizing**: Never risk more than 2% per trade
- **Correlation risk**: Monitor correlation between crypto assets
- **Black swan events**: Always have stop losses and maximum drawdown limits
- **Liquidity considerations**: Adjust position size based on volume

## 8. Advanced Techniques

### 8.1 Machine Learning Integration
```python
class MLPredictor:
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features = df.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        features['price_range'] = (features['high'] - features['low']) / features['close']
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'return_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'return_std_{window}'] = features['returns'].rolling(window).std()
            features[f'volume_mean_{window}'] = features['volume'].rolling(window).mean()
        
        # Technical indicator features (already calculated)
        feature_columns = ['rsi', 'macd', 'bb_percent', 'atr', 'obv', 'momentum_score']
        
        # Lag features
        for col in feature_columns:
            for lag in [1, 3, 5]:
                features[f'{col}_lag_{lag}'] = features[col].shift(lag)
        
        return features.dropna()
    
    def train_ensemble_model(self, X_train, y_train):
        """Train ensemble model for price prediction"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import VotingRegressor
        
        # Individual models
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        nn = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        
        # Ensemble
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb),
            ('nn', nn)
        ])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Train
        ensemble.fit(X_scaled, y_train)
        
        return ensemble, scaler
```

### 8.2 Portfolio Optimization
```python
class PortfolioOptimizer:
    def optimize_weights(self, returns_df: pd.DataFrame, method: str = 'sharpe') -> Dict:
        """Optimize portfolio weights using various methods"""
        from scipy.optimize import minimize
        
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        num_assets = len(mean_returns)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        if method == 'sharpe':
            # Maximize Sharpe ratio
            def neg_sharpe(weights):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -portfolio_return / portfolio_std
            
            result = minimize(neg_sharpe, 
                            num_assets * [1./num_assets], 
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
            
        elif method == 'min_variance':
            # Minimize variance
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            result = minimize(portfolio_variance,
                            num_assets * [1./num_assets],
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints)
        
        return dict(zip(returns_df.columns, result.x))
```

## 9. Production Deployment Guidelines

### 9.1 System Architecture
```python
class TradingSystem:
    def __init__(self, config: Dict):
        self.data_collector = CryptoDataCollector()
        self.analyzer = TechnicalIndicators()
        self.sentiment = SentimentAnalyzer()
        self.risk_manager = RiskManager()
        self.strategies = {}
        self.active_positions = {}
        
    async def run_trading_loop(self):
        """Main trading loop"""
        while True:
            try:
                # 1. Collect data
                market_data = await self._collect_all_data()
                
                # 2. Analyze
                signals = self._generate_signals(market_data)
                
                # 3. Risk check
                if self._risk_check_passed(signals):
                    # 4. Execute trades
                    await self._execute_trades(signals)
                
                # 5. Monitor positions
                await self._monitor_positions()
                
                # 6. Log and report
                self._log_performance()
                
                # Sleep until next interval
                await asyncio.sleep(60)  # 1 minute intervals
                
            except Exception as e:
                self._handle_error(e)
```

### 9.2 Error Handling
- Implement circuit breakers for API failures
- Use dead letter queues for failed transactions
- Set up monitoring and alerting (Prometheus/Grafana)
- Maintain audit logs for all trades

### 9.3 Security Considerations
- Never store API keys in code
- Use environment variables or secret management systems
- Implement IP whitelisting for exchange APIs
- Use read-only API keys for data collection
- Separate trading keys with minimal permissions

## 10. Testing Framework

### 10.1 Unit Tests
```python
import pytest

class TestIndicators:
    def test_rsi_calculation(self):
        """Test RSI calculation accuracy"""
        df = pd.DataFrame({
            'close': [100, 102, 101, 103, 102, 104, 103, 105]
        })
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        assert df['rsi'].iloc[-1] > 50  # Upward trend should have RSI > 50
    
    def test_position_sizing(self):
        """Test position sizing limits"""
        rm = RiskManager(max_position_size=0.1)
        size = rm.calculate_position_size(10000, confidence=0.8, volatility=0.5)
        assert size <= 1000  # Max 10% of capital
```

## Common Issues and Solutions

### Issue: Rate Limiting
**Solution**: Implement exponential backoff and request queuing
```python
async def rate_limited_request(func, *args, max_retries=3):
    for i in range(max_retries):
        try:
            return await func(*args)
        except RateLimitError:
            wait_time = 2 ** i
            await asyncio.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

### Issue: Data Gaps
**Solution**: Implement multiple data source fallbacks
```python
def get_price_data(symbol: str) -> pd.DataFrame:
    sources = [fetch_binance, fetch_coinbase, fetch_kraken]
    for source in sources:
        try:
            return source(symbol)
        except:
            continue
    raise Exception("All data sources failed")
```

### Issue: Slippage in Backtesting
**Solution**: Model realistic execution
```python
def apply_slippage(price: float, volume: float, order_size: float) -> float:
    """Calculate execution price with slippage"""
    market_impact = order_size / volume * 0.1  # 10% impact factor
    slippage = price * market_impact
    return price + slippage
```

## Performance Optimization

### 1. Use Vectorized Operations
```python
# Bad: Loop through DataFrame
for i in range(len(df)):
    df.loc[i, 'sma'] = df.loc[max(0, i-20):i, 'close'].mean()

# Good: Vectorized operation
df['sma'] = df['close'].rolling(window=20).mean()
```

### 2. Async/Await for API Calls
```python
# Fetch multiple pairs concurrently
async def fetch_all_pairs(pairs: List[str]):
    tasks = [fetch_pair(pair) for pair in pairs]
    return await asyncio.gather(*tasks)
```

### 3. Cache Frequently Used Data
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_expensive_indicator(data_hash: str) -> float:
    # Expensive calculation
    return result
```

## Final Notes

This skill represents best practices learned from extensive testing in cryptocurrency trading and analysis. Key takeaways:

1. **Data quality is paramount** - Always validate and clean data
2. **Risk management saves portfolios** - Never trade without stops
3. **Sentiment matters** - Crypto is highly sentiment-driven
4. **Backtesting lies** - Real trading has slippage, fees, and delays
5. **Diversification is essential** - Never put all capital in one position
6. **API reliability varies** - Always have fallbacks
7. **Market conditions change** - Strategies need regular revalidation

Remember: Past performance never guarantees future results. Always paper trade new strategies before risking real capital.