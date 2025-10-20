"""
Startup script for Crypto AI Trading Dashboard
Run this to start the dashboard server
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

# Import and run the dashboard
from src.dashboard.main import main

if __name__ == "__main__":
    print("=" * 60)
    print("  Crypto AI Trading Dashboard")
    print("=" * 60)
    print()
    print("Starting the dashboard server...")
    print()

    main()
