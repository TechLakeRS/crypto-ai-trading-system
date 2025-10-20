#!/usr/bin/env python3
"""
Small Account Leverage Analysis - Starting with $500 (No Input Required)
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
        
        # Summary
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            losing_trades = len([t for t in trades if t['pnl'] < 0])
            total_return = (capital - initial_capital) / initial_capital * 100
            
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
                
                print(f"🏆 Best Trade: ${best_trade['pnl']:.2f} ({best_trade['entry_price']:.2f} → {best_trade['exit_price']:.2f})")
                print(f"💔 Worst Trade: ${worst_trade['pnl']:.2f} ({worst_trade['entry_price']:.2f} → {worst_trade['exit_price']:.2f})")
        else:
            print(f"📊 No completed trades")

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
    
    print(f"✅ Found {len(signals)} signals")
    
    # Test different leverage levels
    leverage_levels = [1, 5, 10, 20, 50]
    results = []
    
    for leverage in leverage_levels:
        print(f"\n{'='*20} TESTING {leverage}x LEVERAGE {'='*20}")
        
        # Simulate trades
        capital = 500.0
        position = None
        trades = []
        
        for signal in signals:
            if signal['signal_type'] == 'BUY' and position is None:
                position_size = capital * 0.1 * leverage
                quantity = position_size / signal['price']
                margin_required = (capital * 0.1)
                
                position = {
                    'entry_price': signal['price'],
                    'quantity': quantity,
                    'entry_time': signal['timestamp'],
                    'entry_reason': signal['reason'],
                    'leverage': leverage,
                    'margin_used': margin_required
                }
                capital -= margin_required
                
            elif signal['signal_type'] == 'SELL' and position is not None:
                price_change = signal['price'] - position['entry_price']
                leveraged_pnl = price_change * position['quantity'] * position['leverage']
                capital += position['margin_used'] + leveraged_pnl
                
                trade = {
                    'entry_price': position['entry_price'],
                    'exit_price': signal['price'],
                    'pnl': leveraged_pnl,
                    'leverage': position['leverage']
                }
                trades.append(trade)
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
                'pnl': leveraged_pnl,
                'leverage': position['leverage']
            }
            trades.append(trade)
        
        # Calculate results
        if trades:
            total_pnl = sum(t['pnl'] for t in trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            total_return = (capital - 500.0) / 500.0 * 100
            
            results.append({
                'leverage': leverage,
                'final_capital': capital,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'win_rate': winning_trades/len(trades)*100,
                'trades': len(trades)
            })
            
            print(f"💰 Final Capital: ${capital:.2f}")
            print(f"📈 Total P&L: ${total_pnl:.2f}")
            print(f"📊 Total Return: {total_return:.2f}%")
            print(f"🎯 Win Rate: {winning_trades/len(trades)*100:.1f}%")
            print(f"🔢 Total Trades: {len(trades)}")
        else:
            print("📊 No completed trades")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("📊 LEVERAGE COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Leverage':<10} {'Final Capital':<15} {'Total Return':<15} {'Win Rate':<10} {'Trades':<8}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result['leverage']}x{'':<6} ${result['final_capital']:<14.2f} {result['total_return']:<14.2f}% {result['win_rate']:<9.1f}% {result['trades']:<8}")
    
    # Find best performing leverage
    if results:
        best_leverage = max(results, key=lambda x: x['total_return'])
        print(f"\n🏆 BEST PERFORMING LEVERAGE: {best_leverage['leverage']}x")
        print(f"   Final Capital: ${best_leverage['final_capital']:.2f}")
        print(f"   Total Return: {best_leverage['total_return']:.2f}%")
        print(f"   Win Rate: {best_leverage['win_rate']:.1f}%")

if __name__ == "__main__":
    asyncio.run(compare_small_account())
