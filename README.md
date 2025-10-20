# Multi-AI Cryptocurrency Trading System

A comprehensive cryptocurrency trading platform that leverages multiple AI models (Grok, Claude, Deepseek) for sentiment analysis, technical analysis verification, and consensus-based signal generation. Designed for manual trade execution with AI-powered insights.

## 🏗️ System Architecture

```
Trading System
├── Data Collection Layer
│   ├── Exchange Connector (Binance, Coinbase, Kraken)
│   ├── On-chain Data Collector (Blockchain metrics, whale movements)
│   └── Social Media Collector (Twitter/X, Reddit)
│
├── AI Analysis Layer
│   ├── Grok (Twitter/X sentiment specialist)
│   ├── Claude (Technical verification & analysis)
│   └── Deepseek (DeFi & on-chain specialist)
│
├── Technical Analysis Engine
│   ├── Crypto-optimized indicators
│   ├── Pattern detection
│   └── Multi-timeframe analysis
│
├── Signal Generation & Consensus
│   ├── AI consensus mechanism
│   ├── Signal aggregation
│   └── Confidence scoring
│
└── Risk Management & Alerts
    ├── Position sizing
    ├── Risk metrics
    └── Manual execution alerts
```

## 🚀 Key Features

### 1. Multi-Exchange Data Aggregation
- Real-time OHLCV data from multiple exchanges
- Order book aggregation for best execution prices
- Volume analysis and liquidity assessment
- Funding rates and futures metrics

### 2. Multi-AI Sentiment Analysis
- **Grok**: Specializes in Twitter/X sentiment analysis
  - Influencer tracking
  - Trend detection
  - Social momentum indicators

- **Claude**: Technical and fundamental verification
  - Cross-validates other AI signals
  - Risk assessment
  - Market structure analysis

- **Deepseek**: DeFi and on-chain focus
  - Smart money movements
  - DeFi metrics analysis
  - Whale activity tracking

### 3. Advanced Technical Analysis
- **Crypto-Optimized Indicators**:
  - Adjusted RSI levels (80/20 for crypto vs 70/30 for forex)
  - Wider Bollinger Bands (2.5σ vs 2.0σ)
  - Custom MACD settings for different timeframes

- **Trading Modes**:
  - Scalping (1-5 min)
  - Day Trading (5-15 min)
  - Swing Trading (15min-1hr)
  - Position Trading (multi-hour/day)

### 4. On-Chain Analytics
- Exchange flow tracking (inflows/outflows)
- Whale movement detection
- Stablecoin flow analysis
- Network activity metrics
- DeFi TVL and liquidation tracking

### 5. Risk Management
- Volatility-adjusted position sizing
- Maximum 10% per position
- Dynamic stop-loss based on ATR
- Correlation-based portfolio management
- Maximum daily loss limits

## 📦 Installation

### Quick Start (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd crypto-ai-trading-system

# Run the quick start script
python quick_start.py
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional for testing)
cp .env.example .env
# Edit .env with your API keys
```

## 🚀 How to Run

### Option 1: Quick Start Script
```bash
python quick_start.py
```
This will guide you through installation and running your first analysis.

### Option 2: Simple Example
```bash
python example.py
```
Runs a basic BTC analysis with technical indicators and AI sentiment.

### Option 3: Main System
```bash
# Single analysis
python main.py --single --symbol BTC/USDT

# Multiple symbols
python main.py --single --symbols BTC/USDT ETH/USDT SOL/USDT

# Continuous monitoring
python main.py --monitor --symbols BTC/USDT ETH/USDT --interval 300

# Help
python main.py --help
```

### Command Line Options
- `--single`: Run single analysis (default)
- `--monitor`: Run continuous monitoring mode
- `--symbol SYMBOL`: Analyze single symbol (e.g., BTC/USDT)
- `--symbols SYMBOL1 SYMBOL2`: Analyze multiple symbols
- `--timeframe 5m`: Set timeframe (1m, 5m, 15m, 1h, 4h, 1d)
- `--interval 300`: Analysis interval in seconds for monitoring
- `--exchange binance`: Choose exchange (binance, coinbase, kraken)
- `--mainnet`: Use mainnet instead of testnet
- `--verbose`: Enable detailed logging
- `--quiet`: Suppress output except errors

## 🔧 Configuration

Edit `src/config/system_config.py` to configure:
- Exchange API credentials
- AI model endpoints and API keys
- Risk parameters
- Trading pairs
- Alert channels

## 🎯 Trading Strategies

### Strategy 1: Momentum with AI Consensus
```python
# Conditions for entry:
1. At least 2/3 AI models agree on direction
2. RSI showing momentum (not overbought/oversold)
3. Volume confirmation (1.5x average)
4. MACD alignment
```

### Strategy 2: Mean Reversion at Extremes
```python
# Conditions for entry:
1. Price at Bollinger Band extremes
2. RSI divergence present
3. Social sentiment at extremes (Fear < 20 or Greed > 80)
4. On-chain shows opposing flow
```

### Strategy 3: Whale Following
```python
# Conditions for entry:
1. Large on-chain movements detected
2. Exchange outflows > inflows significantly
3. Technical setup confirms
4. AI consensus aligns with whale direction
```

## 📊 Signal Generation Process

1. **Data Collection** (Every 60 seconds)
   - Fetch latest price data from exchanges
   - Collect social media posts
   - Get on-chain metrics

2. **AI Analysis** (Every 5 minutes)
   - Each AI model analyzes their specialized data
   - Confidence scores calculated
   - Divergence detection

3. **Consensus Building**
   - Weighted average of AI sentiments
   - Agreement score calculation
   - Confidence adjustment based on agreement

4. **Signal Generation**
   - Technical indicators checked
   - AI consensus evaluated
   - Risk parameters verified
   - Alert sent for manual execution

## 🎨 Usage Example

```python
import asyncio
from src.data_collection.exchange_connector import ExchangeConnector
from src.sentiment.multi_ai_sentiment import MultiAISentimentAnalyzer
from src.technical.indicators import CryptoTechnicalAnalyzer

async def analyze_btc():
    # Initialize components
    exchange = ExchangeConnector('binance')
    ai_analyzer = MultiAISentimentAnalyzer()
    technical = CryptoTechnicalAnalyzer(mode='day_trading')

    # Fetch data
    ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '5m')

    # Technical analysis
    ohlcv_with_indicators = technical.calculate_all_indicators(ohlcv)
    tech_signals = technical.generate_signals(ohlcv_with_indicators)

    # AI sentiment analysis
    market_data = {
        'symbol': 'BTC',
        'technical': ohlcv_with_indicators.iloc[-1].to_dict(),
        'social': await collect_social_data('BTC')
    }

    consensus = await ai_analyzer.analyze_comprehensive_sentiment(market_data)

    # Make decision
    if consensus.recommendation in ['buy', 'strong_buy'] and len(tech_signals) > 2:
        print(f"🟢 BUY Signal: {consensus.overall_sentiment:.2f} confidence: {consensus.confidence:.2f}")
    elif consensus.recommendation in ['sell', 'strong_sell'] and len(tech_signals) > 2:
        print(f"🔴 SELL Signal: {consensus.overall_sentiment:.2f} confidence: {consensus.confidence:.2f}")
    else:
        print(f"⚪ NEUTRAL: Waiting for better setup")

# Run analysis
asyncio.run(analyze_btc())
```

## 🚨 Alert System

Alerts are generated when:
- Strong consensus from AI models (>70% confidence)
- Multiple technical indicators align
- Significant on-chain movements detected
- Social sentiment reaches extremes

Alerts include:
- Signal strength (1-10 scale)
- Recommended position size
- Entry price levels
- Stop loss and take profit targets
- Key reasons for the signal

## 📈 Performance Metrics

Target metrics based on the strategies:
- **Win Rate**: 55-65% (adjusted for crypto volatility)
- **Risk:Reward**: 1:1.5 to 1:3
- **Sharpe Ratio**: >1.0
- **Max Drawdown**: <30-40%
- **Monthly Return**: 5-15% (with proper risk management)

## ⚠️ Risk Warnings

1. **Cryptocurrency markets are highly volatile** - Position sizes adjusted accordingly
2. **Exchange risk** - Never leave all funds on exchanges
3. **AI models can be wrong** - Always verify signals manually
4. **Correlation risk** - Crypto assets often move together
5. **Regulatory risk** - Rules can change suddenly
6. **Technical failures** - Have backup plans for system outages

## 🔍 Monitoring & Maintenance

- **Daily**: Check AI model accuracy scores
- **Weekly**: Review strategy performance
- **Monthly**: Rebalance AI model weights based on performance
- **Quarterly**: Full strategy review and optimization

## 📝 Important Notes

- This system provides **signals for manual execution only**
- Always verify AI recommendations with your own analysis
- Start with small position sizes until comfortable with the system
- Keep detailed records for tax purposes
- Never risk more than you can afford to lose

## 🛠️ Future Enhancements

- [ ] Add more AI models (GPT-4, Gemini, local LLMs)
- [ ] Implement backtesting framework
- [ ] Add portfolio optimization
- [ ] Create web dashboard for monitoring
- [ ] Integrate with trading execution APIs
- [ ] Add machine learning for strategy improvement
- [ ] Implement automated risk management

## 📚 Based On Research

This system implements strategies from:
1. **Cryptocurrency Trading and Data Analysis Skill** - Comprehensive technical framework
2. **Crypto Trading Strategy October 2025** - Adapted forex strategies for crypto markets

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines before submitting PRs.

## ⚖️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies carries significant risk. Past performance does not guarantee future results. Always do your own research and never invest more than you can afford to lose.

## 📄 License

MIT License - See LICENSE file for details