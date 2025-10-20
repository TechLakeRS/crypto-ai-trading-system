# Configuration Guide for Multi-AI Crypto Trading System

## Environment Variables Setup

Create a `.env` file in the project root with the following variables:

### Exchange API Keys (for real data)
```
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET=your_coinbase_secret_here
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_SECRET=your_kraken_secret_here
```

### AI Model API Keys (when implementing real connections)
```
GROK_API_KEY=your_grok_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Social Media APIs (optional)
```
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_SECRET=your_twitter_access_secret_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
```

### On-chain Data APIs (optional)
```
GLASSNODE_API_KEY=your_glassnode_api_key_here
SANTIMENT_API_KEY=your_santiment_api_key_here
CRYPTOQUANT_API_KEY=your_cryptoquant_api_key_here
```

### Notification Services (optional)
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_email_password_here
```

### Database Configuration (optional)
```
DATABASE_URL=postgresql://user:password@localhost:5432/crypto_trading
REDIS_URL=redis://localhost:6379/0
MONGODB_URL=mongodb://localhost:27017/crypto_trading
```

### System Configuration
```
LOG_LEVEL=INFO
TESTNET=true
MAX_POSITION_SIZE_PCT=10.0
MAX_DAILY_LOSS_PCT=5.0
ANALYSIS_INTERVAL=300
```

## Getting API Keys

### Exchange APIs
1. **Binance**: Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. **Coinbase**: Go to [Coinbase Pro API](https://pro.coinbase.com/profile/api)
3. **Kraken**: Go to [Kraken API](https://www.kraken.com/features/api)

### AI Model APIs
1. **OpenAI**: Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Anthropic Claude**: Get API key from [Anthropic Console](https://console.anthropic.com/)
3. **Google Gemini**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Social Media APIs
1. **Twitter**: Apply for API access at [Twitter Developer Portal](https://developer.twitter.com/)
2. **Reddit**: Create app at [Reddit App Preferences](https://www.reddit.com/prefs/apps)

### On-chain Data APIs
1. **Glassnode**: Get API key from [Glassnode](https://glassnode.com/)
2. **Santiment**: Get API key from [Santiment](https://santiment.net/)
3. **CryptoQuant**: Get API key from [CryptoQuant](https://cryptoquant.com/)

## Security Notes

- Never commit your `.env` file to version control
- Use read-only API keys for data collection
- Enable IP whitelisting when possible
- Regularly rotate your API keys
- Use testnet for development and testing
- Start with small position sizes

## Testing Without API Keys

The system can run with mock data for testing purposes. Simply run:

```bash
python example.py
```

This will use simulated data for all external APIs.
