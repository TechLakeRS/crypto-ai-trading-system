"""
Environment variable loader
Loads configuration from .env file
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class EnvConfig:
    """Environment configuration loader"""

    # Exchange Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'True').lower() == 'true'

    # AI Model Configuration
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')

    GROK_API_KEY = os.getenv('GROK_API_KEY', '')
    GROK_API_URL = os.getenv('GROK_API_URL', 'https://api.x.ai/v1')

    PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY', '')
    PERPLEXITY_API_URL = os.getenv('PERPLEXITY_API_URL', 'https://api.perplexity.ai')

    CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY', '')
    GPT4_API_KEY = os.getenv('GPT4_API_KEY', '')

    # Social Media APIs
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')

    # Dashboard Configuration
    DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '127.0.0.1')
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '8000'))
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

    # Trading Configuration
    DEFAULT_TRADING_PAIRS = os.getenv('DEFAULT_TRADING_PAIRS', 'BTC/USDT,ETH/USDT,SOL/USDT').split(',')
    UPDATE_INTERVAL_SECONDS = int(os.getenv('UPDATE_INTERVAL_SECONDS', '60'))

    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///crypto_trading.db')

    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'change-this-secret-key-in-production')

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        missing = []

        if not cls.BINANCE_API_KEY:
            missing.append('BINANCE_API_KEY')
        if not cls.BINANCE_SECRET_KEY:
            missing.append('BINANCE_SECRET_KEY')

        if missing:
            print(f"⚠️  Warning: Missing environment variables: {', '.join(missing)}")
            print("   The system will run with limited functionality.")
            return False

        return True

    @classmethod
    def get_config_summary(cls):
        """Get configuration summary (without secrets)"""
        return {
            'exchange': 'Binance',
            'testnet': cls.BINANCE_TESTNET,
            'has_binance_keys': bool(cls.BINANCE_API_KEY and cls.BINANCE_SECRET_KEY),
            'has_deepseek': bool(cls.DEEPSEEK_API_KEY),
            'has_grok': bool(cls.GROK_API_KEY),
            'has_perplexity': bool(cls.PERPLEXITY_API_KEY),
            'trading_pairs': cls.DEFAULT_TRADING_PAIRS,
            'update_interval': cls.UPDATE_INTERVAL_SECONDS,
            'dashboard_port': cls.DASHBOARD_PORT
        }

# Validate configuration on import
EnvConfig.validate()
