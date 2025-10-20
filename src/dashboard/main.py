"""
FastAPI Dashboard Server
Real-time cryptocurrency trading dashboard with AI-powered signals
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
from datetime import datetime
from typing import List, Dict
import logging
from pathlib import Path

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.env_loader import EnvConfig
from src.data_collection.exchange_connector import ExchangeConnector
from src.signals.signal_generator import SignalGenerator
from src.risk.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crypto AI Trading Dashboard",
    description="Real-time cryptocurrency trading signals powered by multiple AI models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
exchange_connector = None
signal_generator = None
risk_manager = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                dead_connections.append(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global exchange_connector, signal_generator, risk_manager

    logger.info("Starting Crypto AI Trading Dashboard...")
    logger.info(f"Configuration: {EnvConfig.get_config_summary()}")

    # Initialize exchange connector (Binance only)
    try:
        exchange_connector = ExchangeConnector(
            exchange_id='binance',
            testnet=EnvConfig.BINANCE_TESTNET
        )
        # Set API keys if available
        if EnvConfig.BINANCE_API_KEY and EnvConfig.BINANCE_SECRET_KEY:
            exchange_connector.exchange.apiKey = EnvConfig.BINANCE_API_KEY
            exchange_connector.exchange.secret = EnvConfig.BINANCE_SECRET_KEY
            logger.info("Binance API keys configured")
    except Exception as e:
        logger.error(f"Failed to initialize exchange connector: {e}")
        exchange_connector = None

    # Initialize signal generator and risk manager
    signal_generator = SignalGenerator()
    risk_manager = RiskManager()

    logger.info("Dashboard services initialized successfully!")

@app.get("/")
async def read_root():
    """Serve the dashboard HTML page"""
    dashboard_path = Path(__file__).parent / "static" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return HTMLResponse(content="<h1>Dashboard loading...</h1>", status_code=200)

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "status": "ok",
        "config": EnvConfig.get_config_summary(),
        "exchange_status": "connected" if exchange_connector else "disconnected"
    }

@app.get("/api/pairs")
async def get_trading_pairs():
    """Get configured trading pairs"""
    return {
        "pairs": EnvConfig.DEFAULT_TRADING_PAIRS,
        "total": len(EnvConfig.DEFAULT_TRADING_PAIRS)
    }

@app.get("/api/market/{symbol}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    if not exchange_connector:
        raise HTTPException(status_code=503, detail="Exchange connector not available")

    try:
        # Normalize symbol format
        if '/' not in symbol:
            symbol = f"{symbol}/USDT"

        # Fetch current ticker
        ticker = await exchange_connector.fetch_ticker(symbol)

        # Fetch recent OHLCV
        ohlcv = await exchange_connector.fetch_ohlcv(symbol, '5m', limit=100)

        # Calculate basic metrics
        price_change_pct = ticker.get('change_24h', 0)

        return {
            "symbol": symbol,
            "current_price": ticker['last'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "volume_24h": ticker['volume_24h'],
            "high_24h": ticker['high_24h'],
            "low_24h": ticker['low_24h'],
            "change_24h_pct": price_change_pct,
            "timestamp": ticker['timestamp'].isoformat(),
            "candles": ohlcv.tail(20).to_dict('records') if not ohlcv.empty else []
        }

    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals")
async def get_signals():
    """Get current active trading signals"""
    if not signal_generator:
        raise HTTPException(status_code=503, detail="Signal generator not available")

    try:
        active_signals = signal_generator.get_active_signals()

        signals_data = []
        for signal in active_signals:
            signals_data.append({
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "strength": signal.strength,
                "current_price": signal.current_price,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit_levels,
                "risk_reward": signal.risk_reward_ratio,
                "position_size_pct": signal.position_size_pct,
                "primary_reason": signal.primary_reason,
                "supporting_factors": signal.supporting_factors,
                "risk_factors": signal.risk_factors,
                "timestamp": signal.timestamp.isoformat(),
                "priority": signal.priority,
                "confluences": len(signal.sources)
            })

        return {
            "signals": signals_data,
            "total": len(signals_data),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "exchange": "connected" if exchange_connector else "disconnected",
            "signals": "active" if signal_generator else "inactive",
            "risk_manager": "active" if risk_manager else "inactive"
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping/pong, subscriptions, etc.)
                data = await websocket.receive_text()
                logger.info(f"Received from client: {data}")

                # Echo back or handle specific commands
                await websocket.send_json({
                    "type": "ack",
                    "message": "Message received",
                    "timestamp": datetime.now().isoformat()
                })

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        manager.disconnect(websocket)

async def broadcast_market_updates():
    """Background task to broadcast market updates to all connected clients"""
    while True:
        try:
            if exchange_connector and manager.active_connections:
                # Fetch data for all trading pairs
                market_data = {}

                for pair in EnvConfig.DEFAULT_TRADING_PAIRS:
                    try:
                        ticker = await exchange_connector.fetch_ticker(pair)
                        market_data[pair] = {
                            "price": ticker['last'],
                            "change_24h": ticker.get('change_24h', 0),
                            "volume": ticker.get('volume_24h', 0)
                        }
                    except Exception as e:
                        logger.error(f"Error fetching {pair}: {e}")

                # Broadcast to all connected clients
                if market_data:
                    await manager.broadcast({
                        "type": "market_update",
                        "data": market_data,
                        "timestamp": datetime.now().isoformat()
                    })

        except Exception as e:
            logger.error(f"Error in market updates broadcast: {e}")

        # Wait before next update
        await asyncio.sleep(EnvConfig.UPDATE_INTERVAL_SECONDS)

@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(broadcast_market_updates())
    logger.info("Background tasks started")

def main():
    """Run the dashboard server"""
    logger.info(f"Starting server on {EnvConfig.DASHBOARD_HOST}:{EnvConfig.DASHBOARD_PORT}")
    logger.info(f"Dashboard will be available at: http://{EnvConfig.DASHBOARD_HOST}:{EnvConfig.DASHBOARD_PORT}")

    uvicorn.run(
        "src.dashboard.main:app",
        host=EnvConfig.DASHBOARD_HOST,
        port=EnvConfig.DASHBOARD_PORT,
        reload=EnvConfig.DEBUG_MODE,
        log_level="info"
    )

if __name__ == "__main__":
    main()
