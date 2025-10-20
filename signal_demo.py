#!/usr/bin/env python3
"""
Signal Demo - Shows exactly where trades would be executed
Uses relaxed signal conditions to demonstrate trade execution points
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.system_config import SystemConfig
from data_collection.exchange_connector import ExchangeConnector
from technical.indicators import CryptoTechnicalAnalyzer
from signals.signal_generator import SignalGenerator, SignalType

class SignalDemo:
    """Demonstrates signal generation with relaxed conditions"""
    
    def __init__(self):
        self.exchange = ExchangeConnector('binance', testnet=False)
        self.technical_analyzer = CryptoTechnicalAnalyzer(mode='day_trading')
        self.signal_generator = SignalGenerator()
        
        # Relaxed signal conditions for demo
        self.relaxed_config = {
            'min_confidence': 0.3,  # Lowered from 0.6
            'min_sources': 1,        # Lowered from 2
            'min_ai_agreement': 0.3,  # Lowered from 0.6
            'signal_thresholds': {
                'strong_buy': 0.5,   # Lowered from 0.8
                'buy': 0.3,          # Lowered from 0.6
                'neutral': 0.1,     # Lowered from 0.4
                'sell': -0.3,        # Raised from -0.6
                'strong_sell': -0.5  # Raised from -0.8
            }
        }
    
    async def demo_signals(self, symbol: str = "BTC/USDT", days: int = 7):
        """Demonstrate signal generation with real data"""
        
        print(f"🎯 SIGNAL DEMONSTRATION")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Period: Last {days} days")
        print(f"Relaxed Conditions: Confidence ≥30%, Sources ≥1")
        print(f"{'='*60}")
        
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"📊 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get historical data
        data = await self.exchange.fetch_ohlcv(symbol, '5m', limit=2000)
        
        if data.empty:
            print("❌ No data available")
            return
        
        print(f"✅ Loaded {len(data)} candles")
        
        # Calculate technical indicators
        data_with_indicators = self.technical_analyzer.calculate_all_indicators(data)
        
        # Find signal points
        signals = []
        
        print(f"\n🔍 Scanning for signals with relaxed conditions...")
        
        for i in range(50, len(data_with_indicators)):  # Start after 50 candles
            current_data = data_with_indicators.iloc[:i+1]
            current_candle = data_with_indicators.iloc[i]
            
            # Check for technical signals
            technical_signals = self.technical_analyzer.generate_signals(current_data)
            
            # Check RSI conditions
            rsi_signal = None
            if 'rsi' in current_candle:
                rsi = current_candle['rsi']
                if rsi < 35:  # Relaxed oversold
                    rsi_signal = ('BUY', f'RSI oversold at {rsi:.1f}')
                elif rsi > 65:  # Relaxed overbought
                    rsi_signal = ('SELL', f'RSI overbought at {rsi:.1f}')
            
            # Check MACD conditions
            macd_signal = None
            if 'macd' in current_candle and 'macd_signal' in current_candle:
                if current_candle['macd'] > current_candle['macd_signal']:
                    macd_signal = ('BUY', 'MACD bullish')
                else:
                    macd_signal = ('SELL', 'MACD bearish')
            
            # Check Bollinger Bands
            bb_signal = None
            if 'bb_percent' in current_candle:
                bb = current_candle['bb_percent']
                if bb < 0.1:  # Near lower band
                    bb_signal = ('BUY', f'Near lower BB ({bb:.2f})')
                elif bb > 0.9:  # Near upper band
                    bb_signal = ('SELL', f'Near upper BB ({bb:.2f})')
            
            # Check Moving Average alignment
            ma_signal = None
            if all(col in current_candle for col in ['close', 'ema_20', 'ema_50']):
                close = current_candle['close']
                ema20 = current_candle['ema_20']
                ema50 = current_candle['ema_50']
                
                if close > ema20 > ema50:
                    ma_signal = ('BUY', 'Bullish MA alignment')
                elif close < ema20 < ema50:
                    ma_signal = ('SELL', 'Bearish MA alignment')
            
            # Collect all signals
            all_signals = []
            if rsi_signal:
                all_signals.append(rsi_signal)
            if macd_signal:
                all_signals.append(macd_signal)
            if bb_signal:
                all_signals.append(bb_signal)
            if ma_signal:
                all_signals.append(ma_signal)
            
            # Count signal types
            buy_signals = [s for s in all_signals if s[0] == 'BUY']
            sell_signals = [s for s in all_signals if s[0] == 'SELL']
            
            # Determine overall signal
            signal_type = None
            signal_reason = ""
            
            if len(buy_signals) >= 1:  # At least 1 buy signal
                signal_type = 'BUY'
                signal_reason = '; '.join([s[1] for s in buy_signals])
            elif len(sell_signals) >= 1:  # At least 1 sell signal
                signal_type = 'SELL'
                signal_reason = '; '.join([s[1] for s in sell_signals])
            
            # Record signal if found
            if signal_type:
                signals.append({
                    'timestamp': current_candle.name,
                    'price': current_candle['close'],
                    'signal_type': signal_type,
                    'reason': signal_reason,
                    'rsi': current_candle.get('rsi', 0),
                    'macd': current_candle.get('macd', 0),
                    'macd_signal': current_candle.get('macd_signal', 0),
                    'bb_percent': current_candle.get('bb_percent', 0),
                    'volume_ratio': current_candle.get('volume_ratio', 0),
                    'all_signals': all_signals
                })
        
        # Display results
        self._display_signals(signals, symbol)
        
        # Create visualization
        self._plot_signals(data_with_indicators, signals, symbol)
        
        return signals
    
    def _display_signals(self, signals, symbol):
        """Display found signals"""
        
        print(f"\n📈 SIGNAL ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        if not signals:
            print("❌ No signals found with relaxed conditions")
            return
        
        print(f"🎯 Found {len(signals)} signals for {symbol}")
        print(f"{'='*80}")
        
        buy_signals = [s for s in signals if s['signal_type'] == 'BUY']
        sell_signals = [s for s in signals if s['signal_type'] == 'SELL']
        
        print(f"📈 BUY Signals: {len(buy_signals)}")
        print(f"📉 SELL Signals: {len(sell_signals)}")
        print(f"{'='*80}")
        
        # Show detailed signals
        for i, signal in enumerate(signals, 1):
            print(f"\n🔹 SIGNAL #{i}")
            print(f"   Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   Price: ${signal['price']:.2f}")
            print(f"   Type: {signal['signal_type']}")
            print(f"   RSI: {signal['rsi']:.1f}")
            print(f"   MACD: {signal['macd']:.4f} (Signal: {signal['macd_signal']:.4f})")
            print(f"   BB%: {signal['bb_percent']:.2f}")
            print(f"   Volume Ratio: {signal['volume_ratio']:.2f}x")
            print(f"   Reason: {signal['reason']}")
            print(f"   {'─'*60}")
    
    def _plot_signals(self, data, signals, symbol):
        """Create signal visualization"""
        
        print(f"\n📊 Generating signal visualization...")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Signal Analysis: {symbol}', fontsize=16, fontweight='bold')
        
        # Price chart with signals
        axes[0].plot(data.index, data['close'], linewidth=1, color='blue', alpha=0.7)
        
        # Mark buy signals
        buy_signals = [s for s in signals if s['signal_type'] == 'BUY']
        if buy_signals:
            buy_times = [s['timestamp'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            axes[0].scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='BUY', zorder=5)
        
        # Mark sell signals
        sell_signals = [s for s in signals if s['signal_type'] == 'SELL']
        if sell_signals:
            sell_times = [s['timestamp'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            axes[0].scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='SELL', zorder=5)
        
        axes[0].set_title('Price Chart with Signals')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI chart
        axes[1].plot(data.index, data['rsi'], linewidth=1, color='purple')
        axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        axes[1].axhline(y=65, color='orange', linestyle=':', alpha=0.5, label='Relaxed Overbought')
        axes[1].axhline(y=35, color='orange', linestyle=':', alpha=0.5, label='Relaxed Oversold')
        axes[1].set_title('RSI Indicator')
        axes[1].set_ylabel('RSI')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MACD chart
        axes[2].plot(data.index, data['macd'], linewidth=1, color='blue', label='MACD')
        axes[2].plot(data.index, data['macd_signal'], linewidth=1, color='red', label='Signal')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('MACD Indicator')
        axes[2].set_ylabel('MACD')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print trade simulation
        self._simulate_trades(signals)
    
    def _simulate_trades(self, signals, leverage=1.0):
        """Simulate trades based on signals with leverage"""
        
        print(f"\n💰 TRADE SIMULATION (Leverage: {leverage}x)")
        print(f"{'='*60}")
        
        if not signals:
            print("❌ No trades to simulate")
            return
        
        capital = 10000.0
        position = None
        trades = []
        
        for signal in signals:
            if signal['signal_type'] == 'BUY' and position is None:
                # Open long position with leverage
                position_size = capital * 0.1 * leverage  # 10% of capital with leverage
                quantity = position_size / signal['price']
                margin_required = (capital * 0.1)  # Margin = 10% of capital
                
                position = {
                    'entry_price': signal['price'],
                    'quantity': quantity,
                    'entry_time': signal['timestamp'],
                    'entry_reason': signal['reason'],
                    'leverage': leverage,
                    'margin_used': margin_required
                }
                capital -= margin_required  # Only margin is locked up
                
                print(f"📈 BUY at ${signal['price']:.2f} (Leverage: {leverage}x)")
                print(f"   Quantity: {quantity:.6f}")
                print(f"   Margin Used: ${margin_required:.2f}")
                print(f"   Position Value: ${position_size:.2f}")
                print(f"   Reason: {signal['reason']}")
                
            elif signal['signal_type'] == 'SELL' and position is not None:
                # Close long position with leverage
                price_change = signal['price'] - position['entry_price']
                leveraged_pnl = price_change * position['quantity'] * position['leverage']
                
                # Return margin + leveraged P&L
                capital += position['margin_used'] + leveraged_pnl
                
                trade = {
                    'entry_price': position['entry_price'],
                    'exit_price': signal['price'],
                    'quantity': position['quantity'],
                    'pnl': leveraged_pnl,
                    'leverage': position['leverage'],
                    'margin_used': position['margin_used'],
                    'entry_time': position['entry_time'],
                    'exit_time': signal['timestamp'],
                    'duration': signal['timestamp'] - position['entry_time'],
                    'entry_reason': position['entry_reason'],
                    'exit_reason': signal['reason']
                }
                trades.append(trade)
                
                print(f"📉 SELL at ${signal['price']:.2f}")
                print(f"   Price Change: ${price_change:.2f}")
                print(f"   Leveraged P&L: ${leveraged_pnl:.2f}")
                print(f"   Reason: {signal['reason']}")
                print(f"   Duration: {trade['duration']}")
                
                position = None
        
        # Close any remaining position
        if position is not None:
            last_signal = signals[-1]
            price_change = last_signal['price'] - position['entry_price']
            leveraged_pnl = price_change * position['quantity'] * position['leverage']
            capital += position['margin_used'] + leveraged_pnl
            
            trade = {
                'entry_price': position['entry_price'],
                'exit_price': last_signal['price'],
                'quantity': position['quantity'],
                'pnl': leveraged_pnl,
                'leverage': position['leverage'],
                'margin_used': position['margin_used'],
                'entry_time': position['entry_time'],
                'exit_time': last_signal['timestamp'],
                'duration': last_signal['timestamp'] - position['entry_time'],
                'entry_reason': position['entry_reason'],
                'exit_reason': 'End of period'
            }
            trades.append(trade)
            
            print(f"📉 CLOSE at ${last_signal['price']:.2f}")
            print(f"   Price Change: ${price_change:.2f}")
            print(f"   Leveraged P&L: ${leveraged_pnl:.2f}")
            print(f"   Reason: End of period")
        
        # Summary
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] < 0])
            total_return = (capital - 10000) / 10000 * 100
            
            print(f"\n📊 TRADE SUMMARY")
            print(f"{'='*60}")
            print(f"💰 Final Capital: ${capital:.2f}")
            print(f"📈 Total P&L: ${total_pnl:.2f}")
            print(f"📊 Total Return: {total_return:.2f}%")
            print(f"🔢 Total Trades: {len(trades)}")
            print(f"✅ Winning Trades: {winning_trades}")
            print(f"❌ Losing Trades: {losing_trades}")
            print(f"🎯 Win Rate: {winning_trades/len(trades)*100:.1f}%")
            print(f"⚡ Leverage Used: {leverage}x")
            
            if trades:
                avg_pnl = total_pnl / len(trades)
                print(f"💵 Average P&L: ${avg_pnl:.2f}")
                
                # Calculate max drawdown
                peak_capital = 10000
                max_drawdown = 0
                for trade in trades:
                    if trade['pnl'] > 0:
                        peak_capital += trade['pnl']
                    else:
                        drawdown = (peak_capital - (peak_capital + trade['pnl'])) / peak_capital * 100
                        max_drawdown = max(max_drawdown, drawdown)
                
                print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        else:
            print(f"\n📊 No completed trades")

async def main():
    """Run signal demonstration with different leverage levels"""
    
    demo = SignalDemo()
    
    print("🚀 LEVERAGE COMPARISON ANALYSIS")
    print("="*80)
    
    # Test different leverage levels
    leverage_levels = [1, 5, 10, 20]
    
    for leverage in leverage_levels:
        print(f"\n{'='*20} TESTING {leverage}x LEVERAGE {'='*20}")
        signals = await demo.demo_signals("BTC/USDT", days=7)
        
        if signals:
            # Run simulation with leverage
            demo._simulate_trades(signals, leverage=leverage)
        
        print(f"\n{'='*60}")
        print("Press Enter to continue to next leverage level...")
        input()

if __name__ == "__main__":
    asyncio.run(main())
