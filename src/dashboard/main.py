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
from src.signals.free_signal_engine import FreeSignalEngine
from src.signals.dp_signal_engine import DPSignalEngine
from src.signals.hybrid_signal_engine import HybridSignalEngine
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

# Mount static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global instances
exchange_connector = None
signal_generator = None
free_signal_engine = None    # Model 1 (manual.md)
dp_signal_engine = None      # Model 2 (dp.md)
hybrid_signal_engine = None  # Model 3 (hybrid)
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
    global exchange_connector, signal_generator, free_signal_engine, dp_signal_engine, hybrid_signal_engine, risk_manager

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

    # Initialize signal generators and risk manager
    signal_generator = SignalGenerator()
    free_signal_engine = FreeSignalEngine()      # Model 1 (manual.md - 5min day trading)
    dp_signal_engine = DPSignalEngine()          # Model 2 (dp.md - 3min scalping)
    hybrid_signal_engine = HybridSignalEngine()  # Model 3 (hybrid - best of both)
    risk_manager = RiskManager()

    logger.info("Dashboard services initialized successfully!")
    logger.info("Model 1: manual.md (5-min day trading)")
    logger.info("Model 2: dp.md (3-min scalping)")
    logger.info("Model 3: HYBRID (multi-timeframe balanced)")

@app.get("/")
async def read_root():
    """Serve the dashboard HTML page"""
    dashboard_path = Path(__file__).parent / "static" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return HTMLResponse(content="<h1>Dashboard loading...</h1>", status_code=200)

@app.get("/styles.css")
async def get_styles():
    """Serve CSS file"""
    css_path = Path(__file__).parent / "static" / "styles.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/app.js")
async def get_app_js():
    """Serve JavaScript file"""
    js_path = Path(__file__).parent / "static" / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")

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

@app.get("/api/market/{symbol:path}")
async def get_market_data(symbol: str):
    """Get current market data for a symbol"""
    if not exchange_connector or not free_signal_engine:
        raise HTTPException(status_code=503, detail="Services not available")

    try:
        # Normalize symbol format
        if '/' not in symbol:
            symbol = f"{symbol}/USDT"

        # Extract base symbol for sentiment
        base_symbol = symbol.split('/')[0]

        # Fetch current ticker
        ticker = await exchange_connector.fetch_ticker(symbol)

        # Fetch sentiment data (for display only - not used in signals)
        sentiment_data = await free_signal_engine.sentiment_aggregator.get_market_sentiment(base_symbol)

        # Calculate sentiment score for display
        sentiment_score = 0
        sentiment_text = "Neutral"
        if sentiment_data:
            score = sentiment_data.get('overall_sentiment', 0)
            sentiment_score = int((score + 1) * 50)  # Convert -1 to +1 → 0 to 100

            if score > 0.3:
                sentiment_text = "Bullish"
            elif score < -0.3:
                sentiment_text = "Bearish"

        # Calculate technical confluences (for signal generation)
        confluences = 0
        try:
            # Quick technical check - fetch small OHLCV
            ohlcv = await exchange_connector.fetch_ohlcv(symbol, '5m', limit=100)
            if not ohlcv.empty:
                # Calculate indicators
                ohlcv = free_signal_engine.technical_analyzer.calculate_all_indicators(ohlcv)

                # Quick confluence check (Manual.md: 3 of 5 sources)
                latest = ohlcv.iloc[-1]

                # Source 1: Indicator confirmation (RSI 75/25 for 5-min from Manual.md)
                if 'rsi' in latest:
                    if latest['rsi'] < 25 or latest['rsi'] > 75:
                        confluences += 1

                # Source 2: Indicator confirmation (MACD crossover)
                if 'macd' in latest and 'macd_signal' in latest:
                    if abs(latest['macd'] - latest['macd_signal']) > 0:
                        confluences += 1

                # Source 3: Key support/resistance (Bollinger Bands)
                if 'bb_upper' in latest and 'bb_lower' in latest:
                    if latest['close'] < latest['bb_lower'] or latest['close'] > latest['bb_upper']:
                        confluences += 1

                # Source 4: Volume confirmation (Manual.md: 1.5× average)
                if 'volume_ratio' in latest:
                    if latest['volume_ratio'] >= 1.5:
                        confluences += 1

                # Source 5: Trend alignment (EMA alignment)
                if 'ema_20' in latest and 'ema_50' in latest:
                    if (latest['close'] > latest['ema_20'] > latest['ema_50']) or \
                       (latest['close'] < latest['ema_20'] < latest['ema_50']):
                        confluences += 1
        except Exception as e:
            logger.warning(f"Error calculating confluences for {symbol}: {e}")

        return {
            "symbol": symbol,
            "current_price": ticker['last'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "volume_24h": ticker['volume_24h'],
            "high_24h": ticker['high_24h'],
            "low_24h": ticker['low_24h'],
            "change_24h_pct": ticker.get('change_24h', 0),
            "timestamp": ticker['timestamp'].isoformat(),
            "sentiment_score": sentiment_score,
            "sentiment_text": sentiment_text,
            "confluences": confluences
        }

    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/signals")
async def get_signals():
    """Get current active trading signals"""
    if not free_signal_engine:
        raise HTTPException(status_code=503, detail="Signal engine not available")

    try:
        active_signals = free_signal_engine.get_active_signals()

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

@app.get("/api/signals/all")
async def get_all_models_signals():
    """Get signals from all 3 models for comparison"""
    if not free_signal_engine or not dp_signal_engine or not hybrid_signal_engine:
        raise HTTPException(status_code=503, detail="Signal engines not available")

    try:
        # Model 1 signals (manual.md - 5-minute day trading)
        model1_signals = free_signal_engine.get_active_signals()
        model1_data = []
        for signal in model1_signals:
            model1_data.append({
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
                "confluences": len(signal.sources),
                "model": "Model 1 (manual.md)",
                "timeframe": "5-minute",
                "approach": "Day Trading"
            })

        # Model 2 signals (dp.md - 3-minute scalping)
        model2_signals = dp_signal_engine.get_active_signals()
        model2_data = []
        for signal in model2_signals:
            metadata = signal.metadata or {}
            model2_data.append({
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
                "confluences": len(signal.sources),
                "model": "Model 2 (dp.md)",
                "timeframe": "3-minute",
                "approach": "Scalping",
                "leverage": metadata.get('leverage', 10),
                "invalidation_condition": metadata.get('invalidation_condition', 'N/A'),
                "invalidation_price": metadata.get('invalidation_price', 0),
                "risk_usd": metadata.get('risk_usd', 0),
                "funding_rate": metadata.get('funding_rate'),
                "rsi_7_period": metadata.get('rsi_7_period', 0)
            })

        # Model 3 signals (hybrid - best of both)
        model3_signals = hybrid_signal_engine.get_active_signals()
        model3_data = []
        for signal in model3_signals:
            metadata = signal.metadata or {}
            model3_data.append({
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
                "confluences": metadata.get('confluence_count', len(signal.sources)),
                "model": "Model 3 (hybrid)",
                "timeframe": "Multi (3m+5m+4h)",
                "approach": "Balanced",
                "leverage": metadata.get('leverage', 5),
                "invalidation_condition": metadata.get('invalidation_condition', 'N/A'),
                "invalidation_price": metadata.get('invalidation_price', 0),
                "risk_usd": metadata.get('risk_usd', 0),
                "funding_rate": metadata.get('funding_rate'),
                "fast_rsi": metadata.get('fast_rsi_7', 0),
                "slow_rsi": metadata.get('slow_rsi_14', 0),
                "volatility": metadata.get('volatility_pct', 0),
                "stop_type": metadata.get('stop_type', 'adaptive')
            })

        return {
            "model1": {
                "name": "Manual.md",
                "timeframe": "5m",
                "signals": model1_data,
                "total": len(model1_data)
            },
            "model2": {
                "name": "DP.md",
                "timeframe": "3m",
                "signals": model2_data,
                "total": len(model2_data)
            },
            "model3": {
                "name": "HYBRID",
                "timeframe": "Multi",
                "signals": model3_data,
                "total": len(model3_data)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error fetching all model signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "exchange": "connected" if exchange_connector else "disconnected",
            "free_signals": "active" if free_signal_engine else "inactive",
            "risk_manager": "active" if risk_manager else "inactive"
        },
        "signal_engine": "FREE (no API costs)"
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

async def generate_signals_task():
    """Background task to generate trading signals from ALL 3 models"""
    # Wait for initial startup
    await asyncio.sleep(10)

    while True:
        try:
            if exchange_connector and free_signal_engine and dp_signal_engine and hybrid_signal_engine:
                logger.info("Generating signals from all 3 models for all trading pairs...")

                for pair in EnvConfig.DEFAULT_TRADING_PAIRS:
                    try:
                        # MODEL 1: Fetch 5-minute OHLCV for day trading
                        ohlcv_5m = await exchange_connector.fetch_ohlcv(pair, '5m', limit=500)

                        if not ohlcv_5m.empty:
                            current_price = ohlcv_5m['close'].iloc[-1]

                            # Generate Model 1 signal
                            signal_model1 = await free_signal_engine.generate_signal(
                                symbol=pair,
                                market_data=ohlcv_5m,
                                current_price=current_price
                            )

                            if signal_model1:
                                logger.info(f"Model 1: {signal_model1.signal_type.value} signal for {pair} "
                                          f"(confluences: {len(signal_model1.sources)})")

                        # MODEL 2: Fetch 3-minute and 4-hour OHLCV for scalping
                        ohlcv_3m = await exchange_connector.fetch_ohlcv(pair, '3m', limit=500)
                        ohlcv_4h = await exchange_connector.fetch_ohlcv(pair, '4h', limit=100)

                        if not ohlcv_3m.empty and not ohlcv_4h.empty:
                            current_price = ohlcv_3m['close'].iloc[-1]

                            # Fetch funding rate (if available)
                            try:
                                funding_rate = exchange_connector.exchange.fetch_funding_rate(pair.replace('/', ''))
                                funding_rate_value = funding_rate.get('fundingRate', None)
                            except Exception:
                                funding_rate_value = None

                            # Fetch open interest (if available)
                            try:
                                oi_data = exchange_connector.exchange.fetch_open_interest(pair.replace('/', ''))
                                open_interest = oi_data.get('openInterest', None)
                            except Exception:
                                open_interest = None

                            # Generate Model 2 signal
                            signal_model2 = await dp_signal_engine.generate_signal(
                                symbol=pair,
                                market_data_3m=ohlcv_3m,
                                market_data_4h=ohlcv_4h,
                                current_price=current_price,
                                funding_rate=funding_rate_value,
                                open_interest=open_interest
                            )

                            if signal_model2:
                                logger.info(f"Model 2: {signal_model2.signal_type.value} signal for {pair} "
                                          f"(confidence: {signal_model2.confidence:.2f}, "
                                          f"leverage: {signal_model2.metadata.get('leverage', 10)}×)")

                        # MODEL 3: Hybrid (uses 3m, 5m, and 4h data)
                        # We already have ohlcv_3m, ohlcv_5m, ohlcv_4h from Model 1 and Model 2
                        if not ohlcv_5m.empty and not ohlcv_3m.empty and not ohlcv_4h.empty:
                            current_price = ohlcv_5m['close'].iloc[-1]

                            # Generate Model 3 signal (hybrid)
                            signal_model3 = await hybrid_signal_engine.generate_signal(
                                symbol=pair,
                                market_data_3m=ohlcv_3m,
                                market_data_5m=ohlcv_5m,
                                market_data_4h=ohlcv_4h,
                                current_price=current_price,
                                funding_rate=funding_rate_value,
                                open_interest=open_interest
                            )

                            if signal_model3:
                                logger.info(f"Model 3: {signal_model3.signal_type.value} signal for {pair} "
                                          f"(confidence: {signal_model3.confidence:.2f}, "
                                          f"leverage: {signal_model3.metadata.get('leverage', 5)}×, "
                                          f"confluences: {signal_model3.metadata.get('confluence_count', 0)}/9)")

                    except Exception as e:
                        logger.error(f"Error generating signals for {pair}: {e}")

                model1_count = len(free_signal_engine.get_active_signals())
                model2_count = len(dp_signal_engine.get_active_signals())
                model3_count = len(hybrid_signal_engine.get_active_signals())
                logger.info(f"Signal generation complete. Model 1: {model1_count}, Model 2: {model2_count}, Model 3: {model3_count} signals")

        except Exception as e:
            logger.error(f"Error in signal generation task: {e}")

        # Generate signals every 3 minutes (to match Model 2's timeframe)
        await asyncio.sleep(180)

@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(broadcast_market_updates())
    asyncio.create_task(generate_signals_task())
    logger.info("Background tasks started (market updates + signal generation)")

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
