"""
Exchange Data Connector with multi-exchange support
Based on crypto_trading_SKILL.md recommendations
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
import json
from collections import deque
import time

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float = None
    ask: float = None
    bid_volume: float = None
    ask_volume: float = None

@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, volume), ...]
    asks: List[Tuple[float, float]]
    spread: float
    mid_price: float
    imbalance: float  # (bid_volume - ask_volume) / (bid_volume + ask_volume)

class ExchangeConnector:
    """Unified exchange connector with rate limiting and error handling"""

    def __init__(self, exchange_id: str = 'binance', testnet: bool = True):
        self.exchange_id = exchange_id
        self.testnet = testnet

        # Initialize exchange
        self.exchange = self._initialize_exchange()

        # Rate limiting
        self.request_timestamps = deque(maxlen=1200)  # Track last 1200 requests
        self.rate_limit_window = 60  # seconds

        # Data cache
        self.cache = {}
        self.cache_duration = 60  # seconds

        # Error handling
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection"""
        exchange_class = getattr(ccxt, self.exchange_id)

        config = {
            'enableRateLimit': True,
            'rateLimit': 50,  # Conservative rate limit
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        }

        if self.testnet:
            if self.exchange_id == 'binance':
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api'
                    }
                }

        exchange = exchange_class(config)

        # Load markets
        try:
            exchange.load_markets()
            logger.info(f"Connected to {self.exchange_id} {'testnet' if self.testnet else 'mainnet'}")
        except Exception as e:
            logger.error(f"Failed to connect to exchange: {e}")
            raise

        return exchange

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '5m',
                         limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data with validation
        Timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        cache_key = f"ohlcv_{symbol}_{timeframe}"

        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data

        for retry in range(self.max_retries):
            try:
                # Rate limiting check
                await self._check_rate_limit()

                # Fetch data
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Validate data
                df = self._validate_ohlcv(df)

                # Cache the data
                self.cache[cache_key] = (df, time.time())

                return df

            except ccxt.RateLimitExceeded:
                wait_time = self.retry_delay * (2 ** retry)
                logger.warning(f"Rate limit exceeded, waiting {wait_time}s")
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error fetching OHLCV data: {e}")
                if retry == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

        raise Exception("Failed to fetch OHLCV data after max retries")

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data according to best practices"""
        initial_len = len(df)

        # Remove zero prices (except volume)
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]

        # Check OHLC relationships
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df[['open', 'close']].max(axis=1)]
        df = df[df['low'] <= df[['open', 'close']].min(axis=1)]

        # Remove outliers using IQR method
        for col in ['open', 'high', 'low', 'close']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')

        if len(df) < initial_len * 0.95:
            logger.warning(f"Removed {initial_len - len(df)} invalid rows from OHLCV data")

        return df

    async def fetch_order_book(self, symbol: str, limit: int = 100) -> OrderBookSnapshot:
        """Fetch order book snapshot"""
        try:
            await self._check_rate_limit()

            order_book = self.exchange.fetch_order_book(symbol, limit)

            # Calculate metrics
            bids = order_book['bids'][:limit]
            asks = order_book['asks'][:limit]

            if not bids or not asks:
                raise ValueError("Empty order book")

            spread = asks[0][0] - bids[0][0]
            mid_price = (asks[0][0] + bids[0][0]) / 2

            # Calculate volume imbalance
            bid_volume = sum(bid[1] for bid in bids[:10])
            ask_volume = sum(ask[1] for ask in asks[:10])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

            return OrderBookSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                bids=bids,
                asks=asks,
                spread=spread,
                mid_price=mid_price,
                imbalance=imbalance
            )

        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            raise

    async def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data"""
        try:
            await self._check_rate_limit()
            ticker = self.exchange.fetch_ticker(symbol)

            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low']
            }

        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            raise

    async def fetch_recent_trades(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch recent trades for tape reading"""
        try:
            await self._check_rate_limit()
            trades = self.exchange.fetch_trades(symbol, limit=limit)

            df = pd.DataFrame([{
                'timestamp': trade['timestamp'],
                'price': trade['price'],
                'amount': trade['amount'],
                'side': trade['side'],
                'cost': trade['cost']
            } for trade in trades])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching recent trades: {e}")
            raise

    async def fetch_funding_rate(self, symbol: str) -> Dict:
        """Fetch funding rate for perpetual contracts"""
        if not symbol.endswith('PERP') and not symbol.endswith('/USDT:USDT'):
            symbol = symbol.replace('/USDT', '/USDT:USDT')

        try:
            await self._check_rate_limit()

            # Switch to futures market
            self.exchange.options['defaultType'] = 'future'
            funding_rate = self.exchange.fetch_funding_rate(symbol)
            self.exchange.options['defaultType'] = 'spot'

            return {
                'symbol': symbol,
                'funding_rate': funding_rate['fundingRate'],
                'timestamp': funding_rate['timestamp'],
                'next_funding_time': funding_rate.get('fundingDatetime')
            }

        except Exception as e:
            logger.error(f"Error fetching funding rate: {e}")
            return None

    async def fetch_multiple_pairs(self, pairs: List[str],
                                 timeframe: str = '5m') -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple pairs concurrently"""
        tasks = []
        for pair in pairs:
            tasks.append(self.fetch_ohlcv(pair, timeframe))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data = {}
        for pair, result in zip(pairs, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {pair}: {result}")
            else:
                data[pair] = result

        return data

    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()

        # Remove old timestamps
        while self.request_timestamps and \
              now - self.request_timestamps[0] > self.rate_limit_window:
            self.request_timestamps.popleft()

        # Check if we're at the limit
        if len(self.request_timestamps) >= 1180:  # Leave some buffer
            sleep_time = self.rate_limit_window - (now - self.request_timestamps[0]) + 1
            logger.warning(f"Rate limit approaching, sleeping for {sleep_time}s")
            await asyncio.sleep(sleep_time)

        # Record this request
        self.request_timestamps.append(now)

    async def get_account_balance(self) -> Dict:
        """Get account balance (requires API keys)"""
        try:
            await self._check_rate_limit()
            balance = self.exchange.fetch_balance()

            return {
                'total': balance['total'],
                'free': balance['free'],
                'used': balance['used'],
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    def calculate_fees(self, trade_value: float, is_maker: bool = False) -> float:
        """Calculate trading fees"""
        fee_rate = 0.001 if is_maker else 0.001  # Default rates

        if self.exchange_id == 'binance':
            fee_rate = 0.00075 if is_maker else 0.00075
        elif self.exchange_id == 'coinbase':
            fee_rate = 0.004 if is_maker else 0.006
        elif self.exchange_id == 'kraken':
            fee_rate = 0.0016 if is_maker else 0.0026

        return trade_value * fee_rate


class MultiExchangeAggregator:
    """Aggregate data from multiple exchanges for best execution"""

    def __init__(self, exchanges: List[str] = ['binance', 'coinbase', 'kraken']):
        self.connectors = {}
        for exchange_id in exchanges:
            try:
                self.connectors[exchange_id] = ExchangeConnector(exchange_id)
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_id}: {e}")

    async def get_best_prices(self, symbol: str) -> Dict:
        """Get best bid/ask across all exchanges"""
        prices = {}

        tasks = []
        for exchange_id, connector in self.connectors.items():
            tasks.append(connector.fetch_ticker(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        best_bid = 0
        best_ask = float('inf')
        best_bid_exchange = None
        best_ask_exchange = None

        for exchange_id, result in zip(self.connectors.keys(), results):
            if not isinstance(result, Exception):
                prices[exchange_id] = result

                if result['bid'] and result['bid'] > best_bid:
                    best_bid = result['bid']
                    best_bid_exchange = exchange_id

                if result['ask'] and result['ask'] < best_ask:
                    best_ask = result['ask']
                    best_ask_exchange = exchange_id

        return {
            'best_bid': best_bid,
            'best_bid_exchange': best_bid_exchange,
            'best_ask': best_ask,
            'best_ask_exchange': best_ask_exchange,
            'spread': best_ask - best_bid if best_ask < float('inf') else None,
            'all_prices': prices
        }

    async def aggregate_order_books(self, symbol: str) -> OrderBookSnapshot:
        """Aggregate order books from all exchanges"""
        all_bids = []
        all_asks = []

        for exchange_id, connector in self.connectors.items():
            try:
                order_book = await connector.fetch_order_book(symbol)

                # Add exchange identifier to each order
                for bid in order_book.bids:
                    all_bids.append((bid[0], bid[1], exchange_id))
                for ask in order_book.asks:
                    all_asks.append((ask[0], ask[1], exchange_id))

            except Exception as e:
                logger.error(f"Failed to get order book from {exchange_id}: {e}")

        # Sort bids descending, asks ascending
        all_bids.sort(key=lambda x: x[0], reverse=True)
        all_asks.sort(key=lambda x: x[0])

        # Calculate aggregated metrics
        if all_bids and all_asks:
            spread = all_asks[0][0] - all_bids[0][0]
            mid_price = (all_asks[0][0] + all_bids[0][0]) / 2

            bid_volume = sum(bid[1] for bid in all_bids[:30])
            ask_volume = sum(ask[1] for ask in all_asks[:30])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

            return OrderBookSnapshot(
                timestamp=datetime.now(),
                symbol=symbol,
                bids=[(b[0], b[1]) for b in all_bids[:100]],
                asks=[(a[0], a[1]) for a in all_asks[:100]],
                spread=spread,
                mid_price=mid_price,
                imbalance=imbalance
            )

        return None