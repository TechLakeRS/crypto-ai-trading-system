#!/usr/bin/env python3
"""
Leverage Analysis - Compare 1x vs 10x leverage performance
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from signal_demo import SignalDemo

async def compare_leverage():
    """Compare 1x vs 10x leverage performance"""
    
    demo = SignalDemo()
    
    print("🚀 LEVERAGE COMPARISON: 1x vs 10x")
    print("="*80)
    
    # Get signals once
    print("📊 Fetching signals...")
    signals = await demo.demo_signals("BTC/USDT", days=7)
    
    if not signals:
        print("❌ No signals found")
        return
    
    print(f"\n✅ Found {len(signals)} signals")
    
    # Test 1x leverage
    print(f"\n{'='*20} 1x LEVERAGE (NO LEVERAGE) {'='*20}")
    demo._simulate_trades(signals, leverage=1.0)
    
    print(f"\n{'='*20} 10x LEVERAGE {'='*20}")
    demo._simulate_trades(signals, leverage=10.0)
    
    print(f"\n{'='*20} 20x LEVERAGE {'='*20}")
    demo._simulate_trades(signals, leverage=20.0)

if __name__ == "__main__":
    asyncio.run(compare_leverage())
