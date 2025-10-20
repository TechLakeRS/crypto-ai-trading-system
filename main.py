#!/usr/bin/env python3
"""
Multi-AI Cryptocurrency Trading System - Main Runner
Orchestrates data collection, AI analysis, technical indicators, and signal generation
"""

import asyncio
import argparse
import logging
import sys
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from config.system_config import SystemConfig, TradingMode, AIModel
from data_collection.exchange_connector import ExchangeConnector, MultiExchangeAggregator
from technical.indicators import CryptoTechnicalAnalyzer
from sentiment.multi_ai_sentiment import MultiAISentimentAnalyzer
from signals.signal_generator import SignalGenerator, SignalType
from risk.risk_manager import RiskManager
from sentiment.social_collector import SocialMediaAggregator
from data_collection.onchain_collector import OnChainDataCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemRunner:
    """Main orchestrator for the crypto AI trading system"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.running = False
        self.last_analysis_time = None
        
        # Initialize components
        self.exchange_connector = None
        self.multi_exchange = None
        self.technical_analyzer = None
        self.ai_sentiment_analyzer = None
        self.signal_generator = None
        self.risk_manager = None
        self.social_collector = None
        self.onchain_collector = None
        
        # Data storage
        self.market_data_cache = {}
        self.signal_history = []
        self.active_signals = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.start_time = None
        
    async def initialize(self, exchange_id: str = 'binance', testnet: bool = True):
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Multi-AI Crypto Trading System...")
            
            # Initialize exchange connector
            logger.info(f"📡 Connecting to {exchange_id} {'testnet' if testnet else 'mainnet'}...")
            self.exchange_connector = ExchangeConnector(exchange_id, testnet)
            
            # Initialize multi-exchange aggregator
            self.multi_exchange = MultiExchangeAggregator(['binance', 'coinbase', 'kraken'])
            
            # Initialize technical analyzer
            self.technical_analyzer = CryptoTechnicalAnalyzer(mode='day_trading')
            
            # Initialize AI sentiment analyzer
            self.ai_sentiment_analyzer = MultiAISentimentAnalyzer()
            
            # Initialize signal generator
            self.signal_generator = SignalGenerator()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(self.config.risk)
            
            # Initialize data collectors
            self.social_collector = SocialMediaAggregator()
            self.onchain_collector = OnChainDataCollector()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    async def analyze_symbol(self, symbol: str, timeframe: str = '5m') -> Optional[Dict]:
        """Perform comprehensive analysis for a single symbol"""
        try:
            logger.info(f"🔍 Analyzing {symbol} on {timeframe} timeframe...")
            
            # 1. Fetch market data
            ohlcv = await self.exchange_connector.fetch_ohlcv(symbol, timeframe, limit=500)
            if ohlcv.empty:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # 2. Calculate technical indicators
            ohlcv_with_indicators = self.technical_analyzer.calculate_all_indicators(ohlcv)
            
            # 3. Generate technical signals
            technical_signals = self.technical_analyzer.generate_signals(ohlcv_with_indicators)
            
            # 4. Collect social media data
            social_data = await self.social_collector.collect_all_social_data(symbol)
            
            # 5. Collect on-chain data
            onchain_metrics = await self.onchain_collector.fetch_blockchain_metrics(symbol)
            onchain_data = {
                'exchange_flows': onchain_metrics.net_exchange_flow if onchain_metrics else 0,
                'active_addresses': onchain_metrics.active_addresses if onchain_metrics else 1000000,
                'whale_movements': [],  # Simplified for now
                'fear_greed': await self.onchain_collector.fetch_fear_greed_index()
            }
            
            # 6. Prepare data for AI analysis
            market_data_for_ai = {
                'symbol': symbol,
                'technical': ohlcv_with_indicators.iloc[-1].to_dict(),
                'social': social_data,
                'onchain': onchain_data,
                'news': [],  # Would be populated by news collector
                'defi': {},  # Would be populated by DeFi collector
                'whale_movements': onchain_data.get('whale_movements', [])
            }
            
            # 7. Get AI consensus
            ai_consensus = await self.ai_sentiment_analyzer.analyze_comprehensive_sentiment(market_data_for_ai)
            
            # 8. Generate comprehensive trading signal
            trading_signal = None
            if ai_consensus:
                trading_signal = await self.signal_generator.generate_signal(
                    symbol=symbol,
                    technical_data=ohlcv_with_indicators.iloc[-1].to_dict(),
                    ai_consensus={
                        'overall_sentiment': ai_consensus.overall_sentiment,
                        'confidence': ai_consensus.confidence,
                        'agreement_score': ai_consensus.agreement_score,
                        'key_insights': ai_consensus.key_insights
                    },
                    on_chain_data=onchain_data,
                    social_data=social_data,
                    market_data=ohlcv_with_indicators,
                    risk_manager=self.risk_manager
                )
            
            # 9. Compile analysis results
            analysis_result = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': ohlcv_with_indicators['close'].iloc[-1],
                'technical_indicators': {
                    'rsi': ohlcv_with_indicators['rsi'].iloc[-1],
                    'macd': ohlcv_with_indicators['macd'].iloc[-1],
                    'bb_percent': ohlcv_with_indicators['bb_percent'].iloc[-1],
                    'volume_ratio': ohlcv_with_indicators['volume_ratio'].iloc[-1],
                    'atr_percent': ohlcv_with_indicators['atr_percent'].iloc[-1]
                },
                'technical_signals': technical_signals,
                'ai_consensus': ai_consensus,
                'trading_signal': trading_signal,
                'social_sentiment': social_data,
                'onchain_metrics': onchain_data
            }
            
            # Store in cache
            self.market_data_cache[symbol] = analysis_result
            
            # Store signal if generated
            if trading_signal:
                self.active_signals[symbol] = trading_signal
                self.signal_history.append(trading_signal)
            
            self.analysis_count += 1
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def display_analysis_result(self, result: Dict):
        """Display formatted analysis results"""
        if not result:
            return
        
        symbol = result['symbol']
        price = result['current_price']
        indicators = result['technical_indicators']
        
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS RESULTS: {symbol}")
        print(f"{'='*60}")
        print(f"💰 Current Price: ${price:,.2f}")
        print(f"⏰ Analysis Time: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Technical indicators
        print(f"\n📈 TECHNICAL INDICATORS:")
        print(f"  RSI: {indicators['rsi']:.2f}")
        print(f"  MACD: {indicators['macd']:.4f}")
        print(f"  BB Position: {indicators['bb_percent']:.1%}")
        print(f"  Volume Ratio: {indicators['volume_ratio']:.2f}x")
        print(f"  ATR: {indicators['atr_percent']:.2f}%")
        
        # Technical signals
        tech_signals = result['technical_signals']
        if tech_signals:
            print(f"\n🎯 TECHNICAL SIGNALS ({len(tech_signals)}):")
            for signal in tech_signals:
                print(f"  • {signal.signal_type.upper()}: {signal.reasoning}")
        else:
            print(f"\n🎯 TECHNICAL SIGNALS: None")
        
        # AI consensus
        ai_consensus = result['ai_consensus']
        if ai_consensus:
            print(f"\n🤖 AI CONSENSUS:")
            print(f"  Overall Sentiment: {ai_consensus.overall_sentiment:.2f}")
            print(f"  Confidence: {ai_consensus.confidence:.1%}")
            print(f"  Agreement: {ai_consensus.agreement_score:.1%}")
            print(f"  Recommendation: {ai_consensus.recommendation.upper()}")
            print(f"  Key Insights: {', '.join(ai_consensus.key_insights[:3])}")
        
        # Trading signal
        trading_signal = result['trading_signal']
        if trading_signal:
            print(f"\n🚨 TRADING SIGNAL:")
            print(f"  Signal: {trading_signal.signal_type.value.upper()}")
            print(f"  Confidence: {trading_signal.confidence:.1%}")
            print(f"  Strength: {trading_signal.strength:.1%}")
            print(f"  Entry: ${trading_signal.entry_price:,.2f}")
            print(f"  Stop Loss: ${trading_signal.stop_loss:,.2f}")
            print(f"  Take Profit: ${trading_signal.take_profit_levels[0][1]:,.2f}")
            print(f"  Risk/Reward: 1:{trading_signal.risk_reward_ratio:.1f}")
            print(f"  Position Size: {trading_signal.position_size_pct:.1f}%")
            print(f"  Priority: {trading_signal.priority}/10")
            print(f"  Reason: {trading_signal.primary_reason}")
        else:
            print(f"\n🚨 TRADING SIGNAL: None (conditions not met)")
        
        print(f"{'='*60}")
    
    async def run_continuous_monitoring(self, symbols: List[str], interval: int = 300):
        """Run continuous monitoring mode"""
        logger.info(f"🔄 Starting continuous monitoring for {symbols}")
        logger.info(f"⏱️ Analysis interval: {interval} seconds")
        
        self.running = True
        self.start_time = datetime.now()
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Shutdown signal received...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                cycle_start = time.time()
                
                logger.info(f"🔄 Starting analysis cycle #{self.analysis_count + 1}")
                
                # Analyze all symbols
                for symbol in symbols:
                    try:
                        result = await self.analyze_symbol(symbol)
                        if result:
                            self.display_analysis_result(result)
                    except Exception as e:
                        logger.error(f"Error in continuous monitoring for {symbol}: {e}")
                
                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, interval - cycle_duration)
                
                if self.running:
                    logger.info(f"😴 Sleeping for {sleep_time:.1f} seconds...")
                    await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        finally:
            self.running = False
            self.display_performance_summary()
    
    def display_performance_summary(self):
        """Display performance summary"""
        if self.start_time:
            duration = datetime.now() - self.start_time
            print(f"\n{'='*60}")
            print(f"📊 PERFORMANCE SUMMARY")
            print(f"{'='*60}")
            print(f"⏱️ Total Runtime: {duration}")
            print(f"🔍 Analysis Cycles: {self.analysis_count}")
            print(f"📈 Active Signals: {len(self.active_signals)}")
            print(f"📚 Signal History: {len(self.signal_history)}")
            
            if self.analysis_count > 0:
                avg_cycle_time = duration.total_seconds() / self.analysis_count
                print(f"⚡ Average Cycle Time: {avg_cycle_time:.1f}s")
            
            print(f"{'='*60}")
    
    async def run_single_analysis(self, symbols: List[str]):
        """Run single analysis for specified symbols"""
        logger.info(f"🔍 Running single analysis for {symbols}")
        
        for symbol in symbols:
            result = await self.analyze_symbol(symbol)
            if result:
                self.display_analysis_result(result)
            else:
                logger.warning(f"No analysis result for {symbol}")
    
    async def run_backtest(self, symbol: str, start_date: str, end_date: str):
        """Run backtest analysis"""
        logger.info(f"📊 Starting backtest for {symbol} from {start_date} to {end_date}")
        
        try:
            # Import backtest engine
            from backtest import BacktestEngine
            
            # Initialize backtest engine
            backtest_engine = BacktestEngine(self.config)
            
            # Run backtest
            results = await backtest_engine.run_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='5m',
                initial_capital=10000.0
            )
            
            if results:
                # Print results summary
                print(f"\n📊 BACKTEST RESULTS")
                print(f"{'='*60}")
                print(f"💰 Final Capital: ${results.final_capital:,.2f}")
                print(f"📈 Total Return: {results.total_return:.2%}")
                print(f"🔢 Total Trades: {results.total_trades}")
                print(f"✅ Winning Trades: {results.winning_trades}")
                print(f"❌ Losing Trades: {results.losing_trades}")
                print(f"🎯 Win Rate: {results.win_rate:.2%}")
                print(f"💵 Average Win: ${results.avg_win:.2f}")
                print(f"💸 Average Loss: ${results.avg_loss:.2f}")
                print(f"⚖️ Profit Factor: {results.profit_factor:.2f}")
                print(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
                print(f"📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
                
                # Print trade log
                backtest_engine.print_trade_log(results)
                
                # Create visualization
                print(f"\n📈 Generating visualization...")
                backtest_engine.plot_results(results, symbol)
            else:
                logger.error("Backtest failed")
                
        except Exception as e:
            logger.error(f"Backtest error: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-AI Cryptocurrency Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol BTC/USDT                    # Single analysis
  python main.py --symbols BTC/USDT ETH/USDT         # Multiple symbols
  python main.py --monitor --symbols BTC/USDT ETH/USDT # Continuous monitoring
  python main.py --backtest BTC/USDT --start 2024-01-01 --end 2024-01-31
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single', action='store_true', 
                           help='Run single analysis (default)')
    mode_group.add_argument('--monitor', action='store_true',
                           help='Run continuous monitoring mode')
    mode_group.add_argument('--backtest', metavar='SYMBOL',
                           help='Run backtest analysis for symbol')
    
    # Symbol selection
    parser.add_argument('--symbol', metavar='SYMBOL',
                       help='Single symbol to analyze (e.g., BTC/USDT)')
    parser.add_argument('--symbols', nargs='+', metavar='SYMBOL',
                       help='Multiple symbols to analyze')
    
    # Analysis parameters
    parser.add_argument('--timeframe', default='5m',
                       choices=['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d'],
                       help='Timeframe for analysis (default: 5m)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Analysis interval in seconds for monitoring mode (default: 300)')
    
    # Exchange settings
    parser.add_argument('--exchange', default='binance',
                       choices=['binance', 'coinbase', 'kraken'],
                       help='Exchange to use (default: binance)')
    parser.add_argument('--mainnet', action='store_true',
                       help='Use mainnet instead of testnet')
    
    # Backtest parameters
    parser.add_argument('--start', metavar='DATE',
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', metavar='DATE',
                       help='End date for backtest (YYYY-MM-DD)')
    
    # System settings
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output except errors')
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Initialize system configuration
    config = SystemConfig()
    
    # Initialize trading system
    runner = TradingSystemRunner(config)
    
    # Initialize components
    success = await runner.initialize(
        exchange_id=args.exchange,
        testnet=not args.mainnet
    )
    
    if not success:
        logger.error("Failed to initialize system")
        sys.exit(1)
    
    try:
        # Determine symbols to analyze
        symbols = []
        if args.symbol:
            symbols = [args.symbol]
        elif args.symbols:
            symbols = args.symbols
        elif args.backtest:
            symbols = [args.backtest]
        else:
            symbols = config.trading_pairs  # Use default pairs
        
        # Run appropriate mode
        if args.monitor:
            await runner.run_continuous_monitoring(symbols, args.interval)
        elif args.backtest:
            if not args.start or not args.end:
                logger.error("Backtest requires --start and --end dates")
                sys.exit(1)
            await runner.run_backtest(args.backtest, args.start, args.end)
        else:
            await runner.run_single_analysis(symbols)
            
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 System stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
