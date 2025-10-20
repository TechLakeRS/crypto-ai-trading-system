#!/usr/bin/env python3
"""
Quick Start Script for Multi-AI Crypto Trading System
Installs dependencies and runs a basic analysis
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Multi-AI Crypto Trading System - Quick Start")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    print("\n🎯 Choose an option:")
    print("1. Run simple example (recommended for first time)")
    print("2. Run main system with BTC analysis")
    print("3. Run main system in monitoring mode")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Running simple example...")
        run_command("python example.py", "Running simple BTC analysis")
    elif choice == "2":
        print("\n🚀 Running main system...")
        run_command("python main.py --single --symbol BTC/USDT", "Running BTC analysis")
    elif choice == "3":
        print("\n🚀 Starting monitoring mode (press Ctrl+C to stop)...")
        run_command("python main.py --monitor --symbols BTC/USDT ETH/USDT", "Starting monitoring")
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")
    
    print("\n📚 For more options, run:")
    print("  python main.py --help")

if __name__ == "__main__":
    main()
