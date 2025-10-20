"""
On-chain Data Collector for cryptocurrency analysis
Integrates with blockchain explorers and on-chain analytics APIs
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
import json
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class OnChainMetrics:
    """On-chain metrics data structure"""
    timestamp: datetime
    symbol: str
    active_addresses: int
    transaction_count: int
    transaction_volume: float
    exchange_inflows: float
    exchange_outflows: float
    net_exchange_flow: float
    large_transactions: int  # Whale movements
    hash_rate: float  # For PoW chains
    staking_ratio: float  # For PoS chains
    nvt_ratio: float  # Network Value to Transactions
    realized_cap: float
    mvrv_ratio: float  # Market Value to Realized Value

@dataclass
class WhaleAlert:
    """Large transaction alert"""
    timestamp: datetime
    from_address: str
    to_address: str
    amount: float
    symbol: str
    transaction_type: str  # 'exchange_inflow', 'exchange_outflow', 'wallet_transfer'
    exchange_name: Optional[str] = None
    usd_value: float = 0

class OnChainDataCollector:
    """Collect and analyze on-chain blockchain data"""

    def __init__(self):
        # API endpoints (would need actual API keys in production)
        self.glassnode_api = "https://api.glassnode.com/v1/metrics"
        self.santiment_api = "https://api.santiment.net/graphql"
        self.cryptoquant_api = "https://api.cryptoquant.com/v1"

        # Alternative free APIs
        self.blockchain_info_api = "https://blockchain.info"
        self.etherscan_api = "https://api.etherscan.io/api"
        self.blockchair_api = "https://api.blockchair.com"

        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

        # Known exchange addresses (simplified - would need comprehensive list)
        self.exchange_addresses = {
            'binance': [
                '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo',  # Bitcoin
                '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8'  # Ethereum
            ],
            'coinbase': [
                'bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h',
                '0x71660c4005BA85c37ccec55d0C4493E66Fe775d3'
            ]
        }

        # Whale threshold (in USD)
        self.whale_threshold = {
            'BTC': 1_000_000,  # $1M
            'ETH': 500_000,
            'default': 100_000
        }

    async def fetch_fear_greed_index(self) -> Dict:
        """Fetch Crypto Fear & Greed Index"""
        cache_key = "fear_greed_index"

        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data

        try:
            url = "https://api.alternative.me/fng/?limit=2"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = {
                            'value': int(data['data'][0]['value']),
                            'classification': data['data'][0]['value_classification'],
                            'timestamp': datetime.fromtimestamp(int(data['data'][0]['timestamp'])),
                            'previous_value': int(data['data'][1]['value']),
                            'previous_classification': data['data'][1]['value_classification']
                        }

                        # Cache the result
                        self.cache[cache_key] = (result, time.time())
                        return result

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            return None

    async def fetch_blockchain_metrics(self, symbol: str = 'BTC') -> OnChainMetrics:
        """Fetch comprehensive blockchain metrics"""
        try:
            # This would normally use authenticated APIs
            # For demonstration, using mock data structure

            metrics = OnChainMetrics(
                timestamp=datetime.now(),
                symbol=symbol,
                active_addresses=await self._get_active_addresses(symbol),
                transaction_count=await self._get_transaction_count(symbol),
                transaction_volume=await self._get_transaction_volume(symbol),
                exchange_inflows=await self._get_exchange_flows(symbol, 'inflow'),
                exchange_outflows=await self._get_exchange_flows(symbol, 'outflow'),
                net_exchange_flow=0,  # Will calculate
                large_transactions=await self._get_whale_transactions(symbol),
                hash_rate=await self._get_hash_rate(symbol) if symbol == 'BTC' else 0,
                staking_ratio=await self._get_staking_ratio(symbol) if symbol == 'ETH' else 0,
                nvt_ratio=0,  # Will calculate
                realized_cap=await self._get_realized_cap(symbol),
                mvrv_ratio=0  # Will calculate
            )

            # Calculate derived metrics
            metrics.net_exchange_flow = metrics.exchange_outflows - metrics.exchange_inflows
            if metrics.transaction_volume > 0:
                # Simplified NVT calculation (would need market cap)
                metrics.nvt_ratio = 1000000000 / metrics.transaction_volume  # Placeholder

            return metrics

        except Exception as e:
            logger.error(f"Error fetching blockchain metrics: {e}")
            return None

    async def _get_active_addresses(self, symbol: str) -> int:
        """Get number of active addresses"""
        # Placeholder - would use actual API
        if symbol == 'BTC':
            return np.random.randint(800_000, 1_200_000)
        elif symbol == 'ETH':
            return np.random.randint(400_000, 600_000)
        return np.random.randint(10_000, 100_000)

    async def _get_transaction_count(self, symbol: str) -> int:
        """Get daily transaction count"""
        if symbol == 'BTC':
            return np.random.randint(250_000, 350_000)
        elif symbol == 'ETH':
            return np.random.randint(1_000_000, 1_500_000)
        return np.random.randint(10_000, 100_000)

    async def _get_transaction_volume(self, symbol: str) -> float:
        """Get daily transaction volume in native currency"""
        if symbol == 'BTC':
            return np.random.uniform(10_000, 50_000)  # BTC
        elif symbol == 'ETH':
            return np.random.uniform(100_000, 500_000)  # ETH
        return np.random.uniform(1_000_000, 10_000_000)

    async def _get_exchange_flows(self, symbol: str, direction: str) -> float:
        """Get exchange inflows or outflows"""
        base_amount = 1000 if symbol == 'BTC' else 10000 if symbol == 'ETH' else 100000

        if direction == 'inflow':
            return np.random.uniform(base_amount * 0.8, base_amount * 1.2)
        else:
            return np.random.uniform(base_amount * 0.7, base_amount * 1.1)

    async def _get_whale_transactions(self, symbol: str) -> int:
        """Get count of large transactions"""
        return np.random.randint(10, 100)

    async def _get_hash_rate(self, symbol: str) -> float:
        """Get network hash rate for PoW chains"""
        if symbol == 'BTC':
            return np.random.uniform(400, 500)  # EH/s
        return 0

    async def _get_staking_ratio(self, symbol: str) -> float:
        """Get staking ratio for PoS chains"""
        if symbol == 'ETH':
            return np.random.uniform(0.25, 0.30)  # 25-30% staked
        return 0

    async def _get_realized_cap(self, symbol: str) -> float:
        """Get realized capitalization"""
        # Placeholder - would need UTXO analysis
        if symbol == 'BTC':
            return np.random.uniform(400_000_000_000, 500_000_000_000)
        return 0

    async def detect_whale_movements(self, symbol: str,
                                    lookback_hours: int = 24) -> List[WhaleAlert]:
        """Detect and analyze whale movements"""
        alerts = []

        # In production, this would monitor actual blockchain
        # For demonstration, generating sample alerts

        current_price = 95000 if symbol == 'BTC' else 3500 if symbol == 'ETH' else 100

        for i in range(np.random.randint(0, 5)):
            amount = np.random.uniform(100, 1000) if symbol == 'BTC' else \
                    np.random.uniform(1000, 10000)

            alert = WhaleAlert(
                timestamp=datetime.now() - timedelta(hours=np.random.randint(0, lookback_hours)),
                from_address=f"whale_{i}",
                to_address=np.random.choice(['exchange', 'cold_wallet', 'other_whale']),
                amount=amount,
                symbol=symbol,
                transaction_type=np.random.choice(['exchange_inflow', 'exchange_outflow',
                                                 'wallet_transfer']),
                usd_value=amount * current_price
            )

            if alert.usd_value > self.whale_threshold.get(symbol, self.whale_threshold['default']):
                alerts.append(alert)

        return alerts

    async def analyze_stablecoin_flows(self) -> Dict:
        """Analyze stablecoin movements as buying power indicator"""
        try:
            stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI']
            flows = {}

            for stable in stablecoins:
                # In production, would track actual flows
                flows[stable] = {
                    'total_supply': np.random.uniform(50_000_000_000, 100_000_000_000),
                    'exchange_balance': np.random.uniform(10_000_000_000, 30_000_000_000),
                    'recent_mints': np.random.uniform(0, 1_000_000_000),
                    'recent_burns': np.random.uniform(0, 500_000_000),
                    'net_flow_24h': np.random.uniform(-500_000_000, 1_000_000_000)
                }

            # Calculate aggregate metrics
            total_exchange_stables = sum(f['exchange_balance'] for f in flows.values())
            net_flow_24h = sum(f['net_flow_24h'] for f in flows.values())

            return {
                'individual_flows': flows,
                'total_exchange_balance': total_exchange_stables,
                'net_flow_24h': net_flow_24h,
                'buying_power_index': min(100, (total_exchange_stables / 1_000_000_000)),
                'flow_direction': 'bullish' if net_flow_24h > 0 else 'bearish'
            }

        except Exception as e:
            logger.error(f"Error analyzing stablecoin flows: {e}")
            return None

    async def get_defi_metrics(self) -> Dict:
        """Get DeFi ecosystem metrics"""
        try:
            metrics = {
                'total_value_locked': np.random.uniform(50_000_000_000, 100_000_000_000),
                'defi_dominance': np.random.uniform(0.05, 0.15),  # 5-15% of total crypto market
                'lending_rates': {
                    'USDT': np.random.uniform(0.05, 0.15),
                    'USDC': np.random.uniform(0.05, 0.15),
                    'ETH': np.random.uniform(0.01, 0.05),
                    'BTC': np.random.uniform(0.01, 0.03)
                },
                'dex_volume_24h': np.random.uniform(1_000_000_000, 5_000_000_000),
                'liquidations_24h': np.random.uniform(0, 100_000_000)
            }

            return metrics

        except Exception as e:
            logger.error(f"Error fetching DeFi metrics: {e}")
            return None

    async def get_futures_metrics(self, symbol: str) -> Dict:
        """Get futures and derivatives metrics"""
        try:
            metrics = {
                'open_interest': np.random.uniform(5_000_000_000, 15_000_000_000),
                'funding_rate': np.random.uniform(-0.01, 0.01),  # -1% to 1%
                'long_short_ratio': np.random.uniform(0.8, 1.2),
                'liquidations_24h': {
                    'long': np.random.uniform(0, 500_000_000),
                    'short': np.random.uniform(0, 500_000_000)
                },
                'basis': np.random.uniform(-0.005, 0.01),  # Spot vs futures spread
                'perpetual_volume_24h': np.random.uniform(10_000_000_000, 50_000_000_000)
            }

            # Interpret metrics
            if metrics['funding_rate'] > 0.005:
                metrics['sentiment'] = 'overleveraged_long'
            elif metrics['funding_rate'] < -0.005:
                metrics['sentiment'] = 'overleveraged_short'
            else:
                metrics['sentiment'] = 'neutral'

            return metrics

        except Exception as e:
            logger.error(f"Error fetching futures metrics: {e}")
            return None

    def calculate_on_chain_signals(self, metrics: OnChainMetrics) -> Dict:
        """Calculate trading signals from on-chain metrics"""
        signals = {
            'exchange_flow_signal': 0,
            'whale_signal': 0,
            'network_activity_signal': 0,
            'overall_signal': 0
        }

        # Exchange flow signal (negative flow = bullish)
        if metrics.net_exchange_flow < -metrics.exchange_inflows * 0.1:
            signals['exchange_flow_signal'] = 1  # Bullish
        elif metrics.net_exchange_flow > metrics.exchange_inflows * 0.1:
            signals['exchange_flow_signal'] = -1  # Bearish

        # Whale activity signal
        if metrics.large_transactions > 50:
            signals['whale_signal'] = 1 if metrics.net_exchange_flow < 0 else -1

        # Network activity signal
        baseline_active = 1_000_000 if metrics.symbol == 'BTC' else 500_000
        if metrics.active_addresses > baseline_active * 1.1:
            signals['network_activity_signal'] = 1
        elif metrics.active_addresses < baseline_active * 0.9:
            signals['network_activity_signal'] = -1

        # Overall signal (weighted average)
        signals['overall_signal'] = (
            signals['exchange_flow_signal'] * 0.4 +
            signals['whale_signal'] * 0.3 +
            signals['network_activity_signal'] * 0.3
        )

        return signals


class OnChainAggregator:
    """Aggregate on-chain data from multiple sources"""

    def __init__(self):
        self.collector = OnChainDataCollector()
        self.metrics_history = defaultdict(list)
        self.max_history = 100  # Keep last 100 data points

    async def get_comprehensive_metrics(self, symbols: List[str]) -> Dict:
        """Get comprehensive on-chain metrics for multiple assets"""
        results = {}

        for symbol in symbols:
            try:
                # Fetch all metrics
                metrics = await self.collector.fetch_blockchain_metrics(symbol)
                whale_alerts = await self.collector.detect_whale_movements(symbol)
                futures = await self.collector.get_futures_metrics(symbol)

                # Calculate signals
                if metrics:
                    signals = self.collector.calculate_on_chain_signals(metrics)
                else:
                    signals = None

                results[symbol] = {
                    'metrics': metrics,
                    'whale_alerts': whale_alerts,
                    'futures': futures,
                    'signals': signals
                }

                # Store history
                if metrics:
                    self.metrics_history[symbol].append(metrics)
                    if len(self.metrics_history[symbol]) > self.max_history:
                        self.metrics_history[symbol].pop(0)

            except Exception as e:
                logger.error(f"Error getting metrics for {symbol}: {e}")
                results[symbol] = None

        # Add market-wide metrics
        results['fear_greed'] = await self.collector.fetch_fear_greed_index()
        results['stablecoin_flows'] = await self.collector.analyze_stablecoin_flows()
        results['defi_metrics'] = await self.collector.get_defi_metrics()

        return results

    def get_trend_analysis(self, symbol: str, lookback: int = 24) -> Dict:
        """Analyze trends in on-chain metrics"""
        if symbol not in self.metrics_history or \
           len(self.metrics_history[symbol]) < lookback:
            return None

        recent_metrics = self.metrics_history[symbol][-lookback:]

        # Calculate trends
        active_addresses = [m.active_addresses for m in recent_metrics]
        exchange_flows = [m.net_exchange_flow for m in recent_metrics]
        whale_activity = [m.large_transactions for m in recent_metrics]

        return {
            'active_addresses_trend': np.polyfit(range(len(active_addresses)),
                                                active_addresses, 1)[0],
            'exchange_flow_trend': np.polyfit(range(len(exchange_flows)),
                                             exchange_flows, 1)[0],
            'whale_activity_trend': np.polyfit(range(len(whale_activity)),
                                              whale_activity, 1)[0],
            'current_vs_average': {
                'active_addresses': active_addresses[-1] / np.mean(active_addresses),
                'exchange_flow': exchange_flows[-1] / np.mean(exchange_flows) if np.mean(exchange_flows) != 0 else 0,
                'whale_activity': whale_activity[-1] / np.mean(whale_activity)
            }
        }