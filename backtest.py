#!/usr/bin/env python3
"""
Backtesting Engine for Multi-AI Crypto Trading System
Fetches historical data from Binance and simulates trade execution
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import json
from dataclasses import dataclass
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.system_config import SystemConfig
from data_collection.exchange_connector import ExchangeConnector
from technical.indicators import CryptoTechnicalAnalyzer
from sentiment.multi_ai_sentiment import MultiAISentimentAnalyzer
from signals.signal_generator import SignalGenerator, SignalType
from risk.risk_manager import RiskManager
from sentiment.social_collector import SocialMediaAggregator
from data_collection.onchain_collector import OnChainDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade record"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    value: float
    signal_type: SignalType
    signal_strength: float
    signal_confidence: float
    stop_loss: float
    take_profit: float
    reasoning: str
    fees: float
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_timestamp: datetime = None
    exit_reason: str = ""

@dataclass
class BacktestResults:
    """Backtesting results summary"""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.DataFrame

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.exchange = ExchangeConnector('binance', testnet=False)
        self.technical_analyzer = CryptoTechnicalAnalyzer(mode='day_trading')
        self.ai_analyzer = MultiAISentimentAnalyzer()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(config.risk)
        self.social_collector = SocialMediaAggregator()
        self.onchain_collector = OnChainDataCollector()
        
        # Backtest state
        self.capital = 10000.0  # Starting capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.current_date = None
        
    async def run_backtest(self, 
                          symbol: str, 
                          start_date: str, 
                          end_date: str,
                          timeframe: str = '5m',
                          initial_capital: float = 10000.0) -> BacktestResults:
        """Run complete backtest"""
        
        logger.info(f"🚀 Starting backtest for {symbol}")
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info(f"💰 Initial Capital: ${initial_capital:,.2f}")
        
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []
        
        # Convert dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Fetch historical data
        logger.info("📊 Fetching historical data...")
        historical_data = await self._fetch_historical_data(symbol, start_dt, end_dt, timeframe)
        
        if historical_data.empty:
            logger.error("No historical data available")
            return None
        
        logger.info(f"📈 Loaded {len(historical_data)} candles")
        
        # Calculate technical indicators
        logger.info("🔧 Calculating technical indicators...")
        data_with_indicators = self.technical_analyzer.calculate_all_indicators(historical_data)
        
        # Run backtest simulation
        logger.info("⚡ Running backtest simulation...")
        await self._simulate_trading(data_with_indicators, symbol)
        
        # Calculate results
        logger.info("📊 Calculating performance metrics...")
        results = self._calculate_results(start_dt, end_dt, initial_capital)
        
        return results
    
    async def _fetch_historical_data(self, symbol: str, start_date: datetime, 
                                   end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Fetch historical OHLCV data from Binance"""
        
        # Calculate number of candles needed
        timeframe_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes_per_candle = timeframe_minutes.get(timeframe, 5)
        total_minutes = (end_date - start_date).total_seconds() / 60
        total_candles = int(total_minutes / minutes_per_candle)
        
        # Binance API limit is 1000 candles per request
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            try:
                # Fetch data in chunks
                chunk_data = await self.exchange.fetch_ohlcv(
                    symbol, timeframe, limit=min(1000, total_candles)
                )
                
                if chunk_data.empty:
                    break
                
                all_data.append(chunk_data)
                
                # Move to next chunk
                current_start = chunk_data.index[-1] + timedelta(minutes=minutes_per_candle)
                total_candles -= len(chunk_data)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break
        
        if all_data:
            return pd.concat(all_data).drop_duplicates().sort_index()
        else:
            return pd.DataFrame()
    
    async def _simulate_trading(self, data: pd.DataFrame, symbol: str):
        """Simulate trading based on signals"""
        
        logger.info(f"🎯 Simulating trades for {symbol}")
        
        for i in range(100, len(data)):  # Start after 100 candles for indicators
            current_data = data.iloc[:i+1]
            current_candle = data.iloc[i]
            self.current_date = current_data.index[-1]
            
            # Skip if we have an open position
            if self.position is not None:
                # Check exit conditions
                exit_signal = self._check_exit_conditions(current_candle, current_data)
                if exit_signal:
                    await self._close_position(current_candle, exit_signal)
                continue
            
            # Generate signal for current candle
            signal = await self._generate_signal_for_candle(current_data, symbol)
            
            if signal and signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                await self._open_position(signal, current_candle)
            elif signal and signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                # For short selling (simplified - just skip for now)
                continue
            
            # Update equity curve
            self._update_equity_curve(current_candle)
    
    async def _generate_signal_for_candle(self, data: pd.DataFrame, symbol: str):
        """Generate signal for a specific candle"""
        
        try:
            # Get latest technical data
            latest_technical = data.iloc[-1].to_dict()
            
            # Generate technical signals
            technical_signals = self.technical_analyzer.generate_signals(data)
            
            # Mock social and on-chain data (in real implementation, would use historical data)
            social_data = await self.social_collector.collect_all_social_data(symbol)
            onchain_metrics = await self.onchain_collector.fetch_blockchain_metrics(symbol)
            onchain_data = {
                'exchange_flows': onchain_metrics.net_exchange_flow if onchain_metrics else 0,
                'active_addresses': onchain_metrics.active_addresses if onchain_metrics else 1000000,
                'whale_movements': [],
                'fear_greed': await self.onchain_collector.fetch_fear_greed_index()
            }
            
            # Prepare AI data
            market_data_for_ai = {
                'symbol': symbol,
                'technical': latest_technical,
                'social': social_data,
                'onchain': onchain_data,
                'news': [],
                'defi': {},
                'whale_movements': []
            }
            
            # Get AI consensus
            ai_consensus = await self.ai_analyzer.analyze_comprehensive_sentiment(market_data_for_ai)
            
            if not ai_consensus:
                return None
            
            # Generate comprehensive signal
            signal = await self.signal_generator.generate_signal(
                symbol=symbol,
                technical_data=latest_technical,
                ai_consensus={
                    'overall_sentiment': ai_consensus.overall_sentiment,
                    'confidence': ai_consensus.confidence,
                    'agreement_score': ai_consensus.agreement_score,
                    'key_insights': ai_consensus.key_insights
                },
                on_chain_data=onchain_data,
                social_data=social_data,
                market_data=data,
                risk_manager=self.risk_manager
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    async def _open_position(self, signal, candle):
        """Open a new position"""
        
        # Calculate position size
        position_sizing = self.risk_manager.calculate_position_size(
            symbol=signal.symbol,
            confidence=signal.confidence,
            volatility=candle.get('atr_percent', 0.03),
            signal_strength=signal.strength
        )
        
        # Calculate quantity
        quantity = (self.capital * position_sizing['position_size_pct'] / 100) / candle['close']
        
        # Calculate fees
        fees = self.exchange.calculate_fees(quantity * candle['close'])
        
        # Create trade
        trade = Trade(
            timestamp=self.current_date,
            symbol=signal.symbol,
            side='buy',
            price=candle['close'],
            quantity=quantity,
            value=quantity * candle['close'],
            signal_type=signal.signal_type,
            signal_strength=signal.strength,
            signal_confidence=signal.confidence,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit_levels[0][1] if signal.take_profit_levels else candle['close'] * 1.02,
            reasoning=signal.primary_reason,
            fees=fees
        )
        
        # Update capital
        self.capital -= (trade.value + fees)
        
        # Set position
        self.position = trade
        
        logger.info(f"📈 OPENED POSITION: {signal.symbol} at ${candle['close']:.2f}")
        logger.info(f"   Quantity: {quantity:.6f}, Value: ${trade.value:.2f}")
        logger.info(f"   Stop Loss: ${signal.stop_loss:.2f}, Take Profit: ${trade.take_profit:.2f}")
        logger.info(f"   Reason: {signal.primary_reason}")
    
    def _check_exit_conditions(self, candle, data):
        """Check if position should be closed"""
        
        if not self.position:
            return None
        
        current_price = candle['close']
        
        # Stop loss
        if current_price <= self.position.stop_loss:
            return "stop_loss"
        
        # Take profit
        if current_price >= self.position.take_profit:
            return "take_profit"
        
        # Time-based exit (simplified - close after 24 hours)
        if (self.current_date - self.position.timestamp).total_seconds() > 86400:
            return "time_exit"
        
        # Signal reversal (simplified)
        if len(data) > 10:
            recent_signals = self.technical_analyzer.generate_signals(data.tail(10))
            sell_signals = [s for s in recent_signals if s.signal_type == 'sell']
            if len(sell_signals) >= 2:  # Multiple sell signals
                return "signal_reversal"
        
        return None
    
    async def _close_position(self, candle, exit_reason):
        """Close the current position"""
        
        if not self.position:
            return
        
        current_price = candle['close']
        
        # Calculate P&L
        pnl = (current_price - self.position.price) * self.position.quantity
        
        # Calculate fees
        fees = self.exchange.calculate_fees(self.position.value)
        
        # Update trade
        self.position.exit_price = current_price
        self.position.exit_timestamp = self.current_date
        self.position.exit_reason = exit_reason
        self.position.pnl = pnl - fees
        
        # Update capital
        self.capital += (self.position.value + pnl - fees)
        
        # Add to trades list
        self.trades.append(self.position)
        
        logger.info(f"📉 CLOSED POSITION: {self.position.symbol} at ${current_price:.2f}")
        logger.info(f"   P&L: ${pnl:.2f}, Fees: ${fees:.2f}, Net: ${self.position.pnl:.2f}")
        logger.info(f"   Exit Reason: {exit_reason}")
        
        # Clear position
        self.position = None
    
    def _update_equity_curve(self, candle):
        """Update equity curve"""
        
        current_equity = self.capital
        
        # Add unrealized P&L if position is open
        if self.position:
            unrealized_pnl = (candle['close'] - self.position.price) * self.position.quantity
            current_equity += unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': self.current_date,
            'equity': current_equity,
            'price': candle['close']
        })
    
    def _calculate_results(self, start_date: datetime, end_date: datetime, initial_capital: float) -> BacktestResults:
        """Calculate backtest performance metrics"""
        
        if not self.trades:
            logger.warning("No trades executed during backtest period")
            return BacktestResults(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=self.capital,
                total_return=(self.capital - initial_capital) / initial_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                max_drawdown=0,
                sharpe_ratio=0,
                trades=[],
                equity_curve=pd.DataFrame(self.equity_curve)
            )
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trades)
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = [t.pnl / initial_capital for t in self.trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return BacktestResults(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=self.capital,
            total_return=(self.capital - initial_capital) / initial_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=pd.DataFrame(self.equity_curve)
        )
    
    def plot_results(self, results: BacktestResults, symbol: str):
        """Create comprehensive backtest visualization"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Backtest Results: {symbol}', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        if not results.equity_curve.empty:
            axes[0, 0].plot(results.equity_curve['timestamp'], results.equity_curve['equity'], 
                           linewidth=2, color='blue')
            axes[0, 0].axhline(y=results.initial_capital, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Mark trade entries
            for trade in results.trades:
                axes[0, 0].axvline(x=trade.timestamp, color='green', alpha=0.3, linestyle=':')
                axes[0, 0].axvline(x=trade.exit_timestamp, color='red', alpha=0.3, linestyle=':')
        
        # 2. Trade P&L Distribution
        if results.trades:
            pnl_values = [t.pnl for t in results.trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
            axes[0, 1].bar(range(len(pnl_values)), pnl_values, color=colors, alpha=0.7)
            axes[0, 1].set_title('Trade P&L Distribution')
            axes[0, 1].set_xlabel('Trade Number')
            axes[0, 1].set_ylabel('P&L ($)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Win/Loss Pie Chart
        if results.total_trades > 0:
            sizes = [results.winning_trades, results.losing_trades]
            labels = ['Winning Trades', 'Losing Trades']
            colors = ['lightgreen', 'lightcoral']
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Win/Loss Ratio')
        
        # 4. Performance Metrics Table
        metrics_data = [
            ['Total Return', f"{results.total_return:.2%}"],
            ['Total Trades', f"{results.total_trades}"],
            ['Win Rate', f"{results.win_rate:.2%}"],
            ['Avg Win', f"${results.avg_win:.2f}"],
            ['Avg Loss', f"${results.avg_loss:.2f}"],
            ['Profit Factor', f"{results.profit_factor:.2f}"],
            ['Max Drawdown', f"{results.max_drawdown:.2%}"],
            ['Sharpe Ratio', f"{results.sharpe_ratio:.2f}"]
        ]
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=metrics_data, 
                                colLabels=['Metric', 'Value'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Performance Metrics')
        
        # 5. Trade Timeline
        if results.trades:
            trade_times = [t.timestamp for t in results.trades]
            trade_pnl = [t.pnl for t in results.trades]
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnl]
            
            axes[2, 0].scatter(trade_times, trade_pnl, c=colors, alpha=0.7, s=50)
            axes[2, 0].set_title('Trade Timeline')
            axes[2, 0].set_ylabel('P&L ($)')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Exit Reasons
        if results.trades:
            exit_reasons = {}
            for trade in results.trades:
                reason = trade.exit_reason
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            axes[2, 1].bar(exit_reasons.keys(), exit_reasons.values(), alpha=0.7)
            axes[2, 1].set_title('Exit Reasons')
            axes[2, 1].set_ylabel('Count')
            axes[2, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_trade_log(self, results: BacktestResults):
        """Print detailed trade log"""
        
        print(f"\n{'='*80}")
        print(f"📊 BACKTEST TRADE LOG")
        print(f"{'='*80}")
        
        if not results.trades:
            print("❌ No trades executed during backtest period")
            return
        
        for i, trade in enumerate(results.trades, 1):
            print(f"\n🔹 TRADE #{i}")
            print(f"   Symbol: {trade.symbol}")
            print(f"   Entry: {trade.timestamp.strftime('%Y-%m-%d %H:%M')} at ${trade.price:.2f}")
            print(f"   Exit: {trade.exit_timestamp.strftime('%Y-%m-%d %H:%M')} at ${trade.exit_price:.2f}")
            print(f"   Quantity: {trade.quantity:.6f}")
            print(f"   Signal: {trade.signal_type.value.upper()} (Confidence: {trade.signal_confidence:.1%})")
            print(f"   P&L: ${trade.pnl:.2f}")
            print(f"   Exit Reason: {trade.exit_reason}")
            print(f"   Reasoning: {trade.reasoning}")
            print(f"   {'─'*60}")

async def main():
    """Main backtest execution"""
    
    # Configuration
    symbol = "BTC/USDT"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    timeframe = "5m"
    initial_capital = 10000.0
    
    print(f"🚀 Multi-AI Crypto Trading System - Backtest")
    print(f"{'='*60}")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"{'='*60}")
    
    # Initialize system
    config = SystemConfig()
    backtest_engine = BacktestEngine(config)
    
    # Run backtest
    results = await backtest_engine.run_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        initial_capital=initial_capital
    )
    
    if not results:
        print("❌ Backtest failed")
        return
    
    # Print results
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

if __name__ == "__main__":
    asyncio.run(main())
