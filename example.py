#!/usr/bin/env python3
"""
Simple example script for the Multi-AI Crypto Trading System
Demonstrates basic usage without command line arguments
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.system_config import SystemConfig
from data_collection.exchange_connector import ExchangeConnector
from technical.indicators import CryptoTechnicalAnalyzer
from sentiment.multi_ai_sentiment import MultiAISentimentAnalyzer

async def simple_btc_analysis():
    """Simple BTC analysis example"""
    print("🚀 Multi-AI Crypto Trading System - Simple Example")
    print("=" * 60)
    
    try:
        # Initialize components
        print("📡 Initializing components...")
        exchange = ExchangeConnector('binance', testnet=False)
        technical = CryptoTechnicalAnalyzer(mode='day_trading')
        ai_analyzer = MultiAISentimentAnalyzer()
        
        # Fetch BTC data
        print("📊 Fetching BTC/USDT data...")
        ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '5m', limit=500)
        
        if ohlcv.empty:
            print("❌ No data available")
            return
        
        # Calculate technical indicators
        print("📈 Calculating technical indicators...")
        ohlcv_with_indicators = technical.calculate_all_indicators(ohlcv)
        
        # Generate technical signals
        signals = technical.generate_signals(ohlcv_with_indicators)
        
        # Get latest data
        latest = ohlcv_with_indicators.iloc[-1]
        
        # Display results
        print(f"\n💰 BTC/USDT Analysis Results:")
        print(f"Current Price: ${latest['close']:,.2f}")
        print(f"RSI: {latest['rsi']:.2f}")
        print(f"MACD: {latest['macd']:.4f}")
        print(f"Volume Ratio: {latest['volume_ratio']:.2f}x")
        print(f"ATR: {latest['atr_percent']:.2f}%")
        
        print(f"\n🎯 Technical Signals ({len(signals)}):")
        for signal in signals:
            print(f"  • {signal.signal_type.upper()}: {signal.reasoning}")
        
        # AI sentiment analysis (mock data)
        print(f"\n🤖 AI Sentiment Analysis:")
        market_data = {
            'symbol': 'BTC',
            'technical': latest.to_dict(),
            'social': {'twitter': [], 'reddit': []},
            'onchain': {'exchange_flows': 0, 'active_addresses': 1000000},
            'news': [],
            'defi': {},
            'whale_movements': []
        }
        
        consensus = await ai_analyzer.analyze_comprehensive_sentiment(market_data)
        if consensus:
            print(f"Overall Sentiment: {consensus.overall_sentiment:.2f}")
            print(f"Confidence: {consensus.confidence:.1%}")
            print(f"Recommendation: {consensus.recommendation.upper()}")
            print(f"Key Insights: {', '.join(consensus.key_insights[:3])}")
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(simple_btc_analysis())
