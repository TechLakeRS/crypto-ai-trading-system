# Crypto AI Trading Dashboard - Setup Guide

## Quick Start

This guide will help you get the trading dashboard up and running.

## Prerequisites

- Python 3.8 or higher
- Binance account (with API keys)
- Internet connection

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Keys

Your Binance API credentials are already configured in the `.env` file:

```
BINANCE_API_KEY=gfyCln8X9xSqOhCqnVpJl5VzMwFvb4TEpfph6IKpJyrazey1027TNCLxm6iR5hGO
BINANCE_SECRET_KEY=TzcRvn1unOaT7ZmKehFBCrteHNnY6Fc3hDvFP2XNHc1WXk36mhW80cJAPfOoz7wm
BINANCE_TESTNET=False
```

### Add AI Model API Keys (Optional but Recommended)

To enable AI-powered analysis, you need to add API keys for:

1. **DeepSeek** - For financial market analysis
2. **Grok (xAI)** - For Twitter/social sentiment analysis
3. **Perplexity.ai** - For market research

Edit the `.env` file and add your API keys:

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GROK_API_KEY=your_grok_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

### How to Get API Keys

#### DeepSeek
1. Visit https://platform.deepseek.com
2. Sign up and navigate to API section
3. Generate a new API key

#### Grok (xAI)
1. Visit https://x.ai
2. Sign up for API access
3. Generate an API key

#### Perplexity.ai
1. Visit https://www.perplexity.ai
2. Sign up for API access
3. Generate an API key

## Step 3: Run the Dashboard

Simply run:

```bash
python run_dashboard.py
```

Or alternatively:

```bash
python -m src.dashboard.main
```

The dashboard will start on http://127.0.0.1:8000

## Step 4: Open in Browser

Open your web browser and navigate to:

```
http://127.0.0.1:8000
```

You should see the crypto trading dashboard with:
- Real-time price data for BTC, ETH, SOL, and BNB
- 24-hour price changes
- Trading signals (when available)
- Confluence indicators
- Stop Loss and Take Profit levels

## Dashboard Features

### Market Overview
- Real-time cryptocurrency prices from Binance
- 24-hour percentage change (color-coded: green for up, red for down)
- Volume information
- Automatic updates every 60 seconds

### Trading Signals
- AI-powered trading signals
- Confidence levels for each signal
- Multiple confluence indicators
- Entry price, Stop Loss, and Take Profit levels
- Risk/Reward ratios
- Position sizing recommendations

### WebSocket Connection
- Live updates without page refresh
- Connection status indicator
- Automatic reconnection if disconnected

## Customization

### Change Trading Pairs

Edit the `.env` file:

```
DEFAULT_TRADING_PAIRS=BTC/USDT,ETH/USDT,SOL/USDT,ADA/USDT,DOT/USDT
```

### Change Update Interval

Edit the `.env` file:

```
UPDATE_INTERVAL_SECONDS=30  # Update every 30 seconds
```

### Change Dashboard Port

Edit the `.env` file:

```
DASHBOARD_PORT=3000  # Use port 3000 instead of 8000
```

## Troubleshooting

### Dashboard won't start

1. **Check if port is already in use:**
   ```bash
   # Change the port in .env file
   DASHBOARD_PORT=8080
   ```

2. **Verify Python dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Check API credentials:**
   - Ensure Binance API keys are correct
   - Make sure API keys have proper permissions

### No price data showing

1. **Check internet connection**
2. **Verify Binance API keys are valid**
3. **Check browser console for errors** (F12 → Console tab)

### WebSocket disconnected

- The dashboard will automatically try to reconnect
- Check your network connection
- Ensure the backend server is running

## Next Steps

### Enable AI Analysis

To enable full AI-powered trading analysis:

1. **Add DeepSeek API key** - For deep market analysis
2. **Add Grok API key** - For Twitter sentiment
3. **Add Perplexity API key** - For market research

Once these are configured, the system will:
- Analyze market sentiment from multiple sources
- Generate AI-powered trading signals
- Provide multi-model consensus recommendations
- Track whale movements and on-chain data
- Perform continuous learning

### Modules to Implement Next

The following features are planned and can be integrated:

1. **DeepSeek Integration** - AI-powered financial analysis
2. **Perplexity.ai Integration** - Market research and news analysis
3. **Grok Integration** - Twitter and social media sentiment
4. **Multi-Model Comparison** - Show what each AI model recommends
5. **Continuous Learning** - AI models learn from past signals

## Safety Reminders

- **Start with paper trading** - Test with small amounts first
- **Never invest more than you can afford to lose**
- **Cryptocurrency markets are highly volatile**
- **AI signals are recommendations, not guarantees**
- **Always do your own research**
- **Use proper risk management**

## Support

For issues or questions:
- Check the TODO.md file for current development status
- Review the README.md for system architecture details
- Check the logs in the terminal for error messages

## Development Mode

To run in development mode with auto-reload:

```bash
# Set DEBUG_MODE=True in .env
DEBUG_MODE=True
```

Then run:
```bash
python run_dashboard.py
```

The server will automatically reload when you make changes to the code.

---

**Happy Trading! 🚀**

*Remember: This system provides signals for manual execution only. Always verify recommendations with your own analysis.*
