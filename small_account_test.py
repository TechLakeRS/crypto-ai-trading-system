#!/usr/bin/env python3
"""
Small Account Leverage Analysis - Starting with $500
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from signal_demo import SignalDemo

class SmallAccountDemo(SignalDemo):
    """Signal demo for small accounts"""
    
    def _simulate_trades(self, signals, leverage=1.0, initial_capital=500.0):
        """Simulate trades with custom starting capital"""
        
        print(f"\n💰 TRADE SIMULATION (Starting Capital: ${initial_capital:.2f}, Leverage: {leverage}x)")
        print(f"{'='*70}")
        
        if not signals:
            print("❌ No trades to simulate")
            return
        
        capital = initial_capital
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
            total_return = (capital - initial_capital) / initial_capital * 100
            
            print(f"\n📊 TRADE SUMMARY")
            print(f"{'='*70}")
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
                peak_capital = initial_capital
                max_drawdown = 0
                for trade in trades:
                    if trade['pnl'] > 0:
                        peak_capital += trade['pnl']
                    else:
                        drawdown = (peak_capital - (peak_capital + trade['pnl'])) / peak_capital * 100
                        max_drawdown = max(max_drawdown, drawdown)
                
                print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
                
                # Show best and worst trades
                best_trade = max(trades, key=lambda x: x['pnl'])
                worst_trade = min(trades, key=lambda x: x['pnl'])
                
                print(f"\n🏆 BEST TRADE:")
                print(f"   P&L: ${best_trade['pnl']:.2f}")
                print(f"   Entry: ${best_trade['entry_price']:.2f} → Exit: ${best_trade['exit_price']:.2f}")
                print(f"   Duration: {best_trade['duration']}")
                print(f"   Reason: {best_trade['entry_reason']}")
                
                print(f"\n💔 WORST TRADE:")
                print(f"   P&L: ${worst_trade['pnl']:.2f}")
                print(f"   Entry: ${worst_trade['entry_price']:.2f} → Exit: ${worst_trade['exit_price']:.2f}")
                print(f"   Duration: {worst_trade['duration']}")
                print(f"   Reason: {worst_trade['entry_reason']}")
        else:
            print(f"\n📊 No completed trades")

async def compare_small_account():
    """Compare different leverage levels with $500 starting capital"""
    
    demo = SmallAccountDemo()
    
    print("🚀 SMALL ACCOUNT LEVERAGE ANALYSIS")
    print("="*80)
    print("Starting Capital: $500")
    print("="*80)
    
    # Get signals once
    print("📊 Fetching signals...")
    signals = await demo.demo_signals("BTC/USDT", days=7)
    
    if not signals:
        print("❌ No signals found")
        return
    
    print(f"\n✅ Found {len(signals)} signals")
    
    # Test different leverage levels
    leverage_levels = [1, 5, 10, 20, 50]
    
    for leverage in leverage_levels:
        print(f"\n{'='*20} TESTING {leverage}x LEVERAGE {'='*20}")
        demo._simulate_trades(signals, leverage=leverage, initial_capital=500.0)
        
        if leverage < leverage_levels[-1]:  # Don't pause after last test
            print(f"\n{'='*70}")
            print("Press Enter to continue to next leverage level...")
            input()

if __name__ == "__main__":
    asyncio.run(compare_small_account())
