"""
System Configuration for Multi-AI Crypto Trading Platform
Combines strategies from both markdown files with AI-powered sentiment analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class AIModel(Enum):
    """Available AI models for analysis"""
    CLAUDE = "claude"
    GROK = "grok"
    DEEPSEEK = "deepseek"
    GPT4 = "gpt4"
    LOCAL_LLM = "local_llm"

class TradingMode(Enum):
    """Trading modes based on timeframe"""
    SCALPING = "scalping"        # 1-5 minute trades
    DAY_TRADING = "day_trading"   # 5-15 minute trades
    SWING_TRADING = "swing"       # 15min - 1hr trades
    POSITION = "position"         # Multi-hour/day trades

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    api_key: str = ""  # Will be loaded from environment
    api_secret: str = ""
    testnet: bool = True
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    rate_limit: int = 1200  # requests per minute

@dataclass
class AIConfig:
    """AI model configuration"""
    model: AIModel
    api_key: str = ""
    endpoint: str = ""
    max_tokens: int = 4000
    temperature: float = 0.7
    rate_limit: int = 60  # requests per minute
    specialized_for: List[str] = None  # ["sentiment", "technical", "news"]

@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size_pct: float = 10.0  # Max 10% per position
    max_total_exposure_pct: float = 60.0  # Max 60% total exposure
    stop_loss_pct: float = 2.0  # Default 2% stop loss
    take_profit_ratio: float = 2.0  # 1:2 risk reward ratio
    max_daily_loss_pct: float = 5.0  # Max 5% daily loss
    max_correlation_exposure: float = 0.7  # Max correlation between positions

@dataclass
class TechnicalConfig:
    """Technical analysis configuration adapted from strategies"""
    # From crypto_trading_strategy_october_2025.md
    scalping_indicators: Dict = None
    day_trading_indicators: Dict = None
    swing_indicators: Dict = None

    # Custom crypto-specific settings
    rsi_overbought: int = 80  # Adjusted for crypto volatility
    rsi_oversold: int = 20
    bb_std: float = 2.5  # Wider bands for crypto
    volume_confirmation: float = 1.5  # 1.5x average volume

    # Harmonic patterns
    enable_harmonics: bool = True
    prz_tolerance_pct: float = 1.0  # 1% tolerance for patterns

@dataclass
class SentimentConfig:
    """Sentiment analysis configuration"""
    # Social media sources
    twitter_enabled: bool = True
    reddit_enabled: bool = True
    telegram_enabled: bool = False
    discord_enabled: bool = False

    # News sources
    news_sources: List[str] = None

    # Weighting for aggregate sentiment
    twitter_weight: float = 0.3
    reddit_weight: float = 0.25
    news_weight: float = 0.25
    onchain_weight: float = 0.2

    # Thresholds
    extreme_fear: float = 20
    extreme_greed: float = 80

class SystemConfig:
    """Main system configuration"""

    def __init__(self):
        # Exchange configurations
        self.exchanges = {
            "binance": ExchangeConfig(
                name="binance",
                maker_fee=0.00075,
                taker_fee=0.00075,
                rate_limit=1200
            ),
            "coinbase": ExchangeConfig(
                name="coinbase",
                maker_fee=0.004,
                taker_fee=0.006,
                rate_limit=600
            ),
            "kraken": ExchangeConfig(
                name="kraken",
                maker_fee=0.0016,
                taker_fee=0.0026,
                rate_limit=600
            )
        }

        # AI model configurations
        self.ai_models = {
            AIModel.GROK: AIConfig(
                model=AIModel.GROK,
                specialized_for=["sentiment", "twitter", "social"],
                temperature=0.8
            ),
            AIModel.CLAUDE: AIConfig(
                model=AIModel.CLAUDE,
                specialized_for=["technical", "analysis", "verification"],
                temperature=0.5
            ),
            AIModel.DEEPSEEK: AIConfig(
                model=AIModel.DEEPSEEK,
                specialized_for=["onchain", "defi", "technical"],
                temperature=0.6
            )
        }

        # Risk configuration
        self.risk = RiskConfig()

        # Technical configurations for different trading modes
        self.technical_configs = {
            TradingMode.SCALPING: TechnicalConfig(
                scalping_indicators={
                    "rsi": {"period": 5, "overbought": 85, "oversold": 15},
                    "macd": {"fast": 3, "slow": 10, "signal": 16},
                    "bb": {"period": 20, "std": 2.5},
                    "ema": [9, 20],
                    "volume_ma": 20
                },
                rsi_overbought=85,
                rsi_oversold=15,
                bb_std=2.5,
                volume_confirmation=2.0
            ),
            TradingMode.DAY_TRADING: TechnicalConfig(
                day_trading_indicators={
                    "rsi": {"period": 14, "overbought": 75, "oversold": 25},
                    "macd": {"fast": 5, "slow": 35, "signal": 5},
                    "bb": {"period": 20, "std": 2.5},
                    "ema": [20, 50],
                    "volume_ma": 20
                },
                rsi_overbought=75,
                rsi_oversold=25,
                bb_std=2.5,
                volume_confirmation=1.5
            ),
            TradingMode.SWING_TRADING: TechnicalConfig(
                swing_indicators={
                    "rsi": {"period": 21, "overbought": 70, "oversold": 30},
                    "macd": {"fast": 12, "slow": 26, "signal": 9},
                    "bb": {"period": 20, "std": 2.0},
                    "ema": [50, 200],
                    "volume_ma": 50
                },
                rsi_overbought=70,
                rsi_oversold=30,
                bb_std=2.0,
                volume_confirmation=1.3
            )
        }

        # Sentiment configuration
        self.sentiment = SentimentConfig(
            news_sources=[
                "coindesk",
                "cointelegraph",
                "decrypt",
                "theblock",
                "bitcoinmagazine"
            ]
        )

        # Trading pairs configuration
        self.trading_pairs = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT"
        ]

        # System settings
        self.data_update_interval = 60  # seconds
        self.signal_generation_interval = 300  # 5 minutes
        self.max_concurrent_positions = 5
        self.min_volume_24h = 1_000_000  # $1M minimum daily volume

        # AI consensus requirements
        self.min_ai_agreement = 2  # At least 2 AI models must agree
        self.ai_confidence_threshold = 0.7  # 70% confidence required

        # Backtesting settings
        self.backtest_start_date = "2024-01-01"
        self.backtest_initial_capital = 10_000
        self.backtest_commission = 0.001

        # Alert settings
        self.alert_channels = ["telegram", "email", "discord"]
        self.urgent_alert_threshold = 0.9  # Confidence for urgent alerts

    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """Get configuration for specific exchange"""
        return self.exchanges.get(exchange_name)

    def get_ai_config(self, model: AIModel) -> AIConfig:
        """Get configuration for specific AI model"""
        return self.ai_models.get(model)

    def get_technical_config(self, mode: TradingMode) -> TechnicalConfig:
        """Get technical configuration for trading mode"""
        return self.technical_configs.get(mode)

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        # Check risk parameters
        if self.risk.max_position_size_pct > 20:
            raise ValueError("Position size too large for crypto trading")

        if self.risk.stop_loss_pct < 0.5:
            raise ValueError("Stop loss too tight for crypto volatility")

        # Check AI models
        if len(self.ai_models) < 2:
            raise ValueError("At least 2 AI models required for verification")

        return True