"""
Risk Management Module for Crypto Trading
Implements position sizing, portfolio management, and risk metrics
Based on strategies from markdown files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for different market conditions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class Position:
    """Trading position data structure"""
    symbol: str
    entry_price: float
    current_price: float
    quantity: float
    position_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: datetime
    position_type: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    risk_amount: float

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    total_exposure: float
    total_exposure_pct: float
    portfolio_var: float  # Value at Risk
    portfolio_cvar: float  # Conditional VaR
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    correlation_risk: float
    concentration_risk: float
    kelly_fraction: float

class RiskManager:
    """Comprehensive risk management system for crypto trading"""

    def __init__(self, initial_capital: float = 10000, config: Dict = None, mode: str = 'standard'):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.mode = mode  # 'conservative', 'standard', 'aggressive'

        # Risk parameters from Manual.md - Crypto-specific
        # Note: stop_loss_pct and take_profit_ratio are now dynamic based on mode
        self.config = config or {
            'mode': mode,  # Trading mode
            'max_position_size_pct': 10.0,  # Max 10% per position
            'max_total_exposure_pct': 60.0,  # Max 60% total exposure
            'max_correlation_exposure': 0.7,  # Max correlation between positions
            'risk_per_trade_pct': 0.75,  # Manual.md: 0.5-1% risk per trade (using 0.75%)
            'max_daily_loss_pct': 5.0,  # Circuit breaker at 5% daily loss
            'max_drawdown_pct': 30.0,  # Manual.md: 30-40% max drawdown for crypto
            'min_liquidity_ratio': 0.2,  # Manual.md: 20% in stablecoins
            'volatility_scalar': 1.5,  # Volatility adjustment for crypto
            'confidence_threshold': 0.7,  # Minimum confidence for full position
            'atr_stop_multiplier': 1.5,  # Manual.md: minimum 1.5× ATR for stops
        }

        # Portfolio tracking
        self.positions = {}
        self.closed_positions = []
        self.daily_pnl = []
        self.equity_curve = [initial_capital]
        self.trade_history = []

        # Risk tracking
        self.daily_loss = 0
        self.max_portfolio_value = initial_capital
        self.current_drawdown = 0
        self.risk_events = []

        # Correlation matrix for portfolio
        self.correlation_matrix = pd.DataFrame()

    @property
    def stop_loss_pct(self) -> float:
        """Dynamic stop loss % based on trading mode (Model 1)"""
        modes = {
            'conservative': 2.0,  # Tighter stop, more cautious
            'standard': 2.5,      # Default day trading stop
            'aggressive': 3.0     # Wider stop for volatility
        }
        return modes.get(self.mode, 2.5)

    @property
    def take_profit_ratio(self) -> float:
        """Dynamic base R:R ratio based on mode (Model 1)"""
        modes = {
            'conservative': 1.5,  # Smaller targets, quicker exits
            'standard': 2.0,      # Default 1:2 risk/reward
            'aggressive': 2.5     # Larger targets for bigger wins
        }
        return modes.get(self.mode, 2.0)

    def calculate_position_size(self, symbol: str, confidence: float,
                              volatility: float, signal_strength: float,
                              account_balance: Optional[float] = None) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion with crypto adjustments
        Based on strategies from markdown files
        """
        if account_balance is None:
            account_balance = self.current_capital

        # Base position size using Kelly Criterion
        win_probability = 0.5 + (confidence * 0.3)  # Convert confidence to win probability
        avg_win = self.take_profit_ratio  # Use dynamic property
        avg_loss = 1.0

        # Kelly formula: f = (p * b - q) / b
        # where p = win prob, q = loss prob, b = win/loss ratio
        kelly_fraction = (win_probability * avg_win - (1 - win_probability)) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% (crypto safety)

        # Adjust for volatility (higher volatility = smaller position)
        volatility_adjustment = 1 / (1 + volatility * self.config['volatility_scalar'])

        # Adjust for signal strength
        signal_adjustment = 0.5 + (signal_strength * 0.5)  # 50% to 100% based on signal

        # Adjust for confidence
        confidence_adjustment = 0.5 + (confidence * 0.5) if confidence > self.config['confidence_threshold'] else 0.5

        # Calculate final position size
        position_size_pct = kelly_fraction * volatility_adjustment * signal_adjustment * confidence_adjustment

        # Apply maximum position size constraint
        position_size_pct = min(position_size_pct, self.config['max_position_size_pct'] / 100)

        # Check portfolio constraints
        current_exposure_pct = self._calculate_total_exposure() / account_balance
        remaining_exposure = (self.config['max_total_exposure_pct'] / 100) - current_exposure_pct
        position_size_pct = min(position_size_pct, remaining_exposure)

        # Calculate position value and units
        position_value = account_balance * position_size_pct

        # Risk-based position sizing (ensure we don't risk more than stop loss allows)
        max_risk_amount = account_balance * (self.stop_loss_pct / 100)  # Use dynamic property
        if position_value * (self.stop_loss_pct / 100) > max_risk_amount:
            position_value = max_risk_amount / (self.stop_loss_pct / 100)

        return {
            'position_size_pct': position_size_pct * 100,
            'position_value': position_value,
            'kelly_fraction': kelly_fraction,
            'volatility_adjustment': volatility_adjustment,
            'confidence_adjustment': confidence_adjustment,
            'max_risk_amount': max_risk_amount,
            'reasoning': self._generate_position_reasoning(
                kelly_fraction, volatility, confidence, signal_strength
            )
        }

    def calculate_stop_loss(self, entry_price: float, position_type: str,
                           atr: float, support_level: Optional[float] = None,
                           volatility_pct: Optional[float] = None) -> float:
        """
        Calculate dynamic stop loss based on market conditions
        Manual.md: Day trading needs tight stops (2-3%), not wide ATR-based stops
        """
        # Percentage-based stop for day trading (PRIMARY METHOD - use this)
        pct_stop = entry_price * (self.stop_loss_pct / 100)  # Use dynamic property

        # ATR-based validation (only use if percentage stop is unreasonable)
        atr_multiplier = 1.2  # Reasonable multiplier for day trading
        if volatility_pct and volatility_pct > 5:
            atr_multiplier = 1.5  # Wider in high volatility
        atr_stop = atr * atr_multiplier

        # Cap ATR stop between 1% (minimum for day trading) and 4% (maximum)
        min_stop_distance = entry_price * 0.01  # Don't go below 1%
        max_stop_distance = entry_price * 0.04  # Don't go above 4%
        atr_stop = min(max(atr_stop, min_stop_distance), max_stop_distance)

        # Use percentage stop as base, only adjust if ATR suggests it's unreasonable
        # If ATR stop is much wider than percentage (e.g., high volatility), use percentage
        # If ATR stop is slightly wider, use ATR (gives breathing room)
        if atr_stop > pct_stop * 1.5:
            # ATR suggests much wider stop, stick with percentage (more conservative)
            base_stop_distance = pct_stop
        else:
            # ATR is reasonable, use it for slight breathing room
            base_stop_distance = max(pct_stop, atr_stop * 0.9)

        # Consider support/resistance if available
        if support_level:
            if position_type == 'long':
                support_distance = abs(entry_price - support_level)
            else:
                support_distance = abs(support_level - entry_price)

            # Use support if it provides reasonable stop (not too wide)
            if support_distance <= max_stop_distance:
                # Support gives breathing room, use it if it's slightly wider but still reasonable
                base_stop_distance = max(base_stop_distance, support_distance * 0.9)

        # Calculate final stop loss
        if position_type == 'long':
            stop_loss = entry_price - base_stop_distance
        else:
            stop_loss = entry_price + base_stop_distance

        return round(stop_loss, 2)

    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                            position_type: str, resistance_level: Optional[float] = None,
                            risk_reward_ratio: Optional[float] = None,
                            market_data: Optional[pd.Series] = None,
                            htf_data: Optional[pd.Series] = None) -> List[float]:
        """
        Calculate dynamic take profit levels with partial exit strategy

        Auto-adjusts based on market conditions:
        - Volatility (ATR): High vol = tighter targets, Low vol = wider targets
        - Trend strength: Strong trend = extend targets
        - Volume: High volume = more confident, wider targets
        """
        if risk_reward_ratio is None:
            risk_reward_ratio = self.take_profit_ratio  # Use dynamic property

        risk_amount = abs(entry_price - stop_loss)

        # Dynamic adjustments based on market conditions
        volatility_multiplier = 1.0
        trend_multiplier = 1.0
        volume_multiplier = 1.0

        # Volatility adjustment (ATR-based)
        if htf_data is not None:
            atr_htf = htf_data.get('atr', 0)
            if atr_htf > 0:
                atr_pct = (atr_htf / entry_price) * 100
                if atr_pct > 4:  # High volatility
                    volatility_multiplier = 0.85  # Tighter targets in choppy markets
                elif atr_pct < 2:  # Low volatility
                    volatility_multiplier = 1.15  # Wider targets in calm markets

        # Trend strength and volume adjustments
        if market_data is not None:
            # Trend strength adjustment
            ema_20 = market_data.get('ema_20', entry_price)
            ema_50 = market_data.get('ema_50', entry_price)

            if position_type == 'long' and entry_price > ema_20 > ema_50:
                trend_multiplier = 1.1  # Strong uptrend, extend target
            elif position_type == 'short' and entry_price < ema_20 < ema_50:
                trend_multiplier = 1.1  # Strong downtrend, extend target

            # Volume adjustment
            volume_ratio = market_data.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # High volume confirmation
                volume_multiplier = 1.1
            elif volume_ratio < 0.8:  # Weak volume
                volume_multiplier = 0.9

        # Apply all adjustments to base R:R ratio
        adjusted_rr = risk_reward_ratio * volatility_multiplier * trend_multiplier * volume_multiplier

        # Clamp to reasonable range for day trading (1.5x to 3x)
        adjusted_rr = max(1.5, min(adjusted_rr, 3.0))

        # Multiple take profit levels for scaling out
        tp_levels = []

        if position_type == 'long':
            # TP1: 1:1 risk/reward (secure breakeven) - 40% position
            tp1 = entry_price + risk_amount
            tp_levels.append(('TP1_40%', round(tp1, 2)))

            # TP2: Adjusted risk/reward ratio - 40% position
            tp2 = entry_price + (risk_amount * adjusted_rr)
            tp_levels.append(('TP2_40%', round(tp2, 2)))

            # TP3: Extended target - 20% ride with trailing stop
            tp3 = entry_price + (risk_amount * adjusted_rr * 1.3)
            if resistance_level and resistance_level > entry_price:
                tp3 = min(tp3, resistance_level * 0.98)  # Just below resistance
            tp_levels.append(('TP3_20%_trail', round(tp3, 2)))

        else:  # short position
            tp1 = entry_price - risk_amount
            tp_levels.append(('TP1_40%', round(tp1, 2)))

            tp2 = entry_price - (risk_amount * adjusted_rr)
            tp_levels.append(('TP2_40%', round(tp2, 2)))

            tp3 = entry_price - (risk_amount * adjusted_rr * 1.3)
            if resistance_level and resistance_level < entry_price:
                tp3 = max(tp3, resistance_level * 1.02)
            tp_levels.append(('TP3_20%_trail', round(tp3, 2)))

        return tp_levels

    def check_correlation_risk(self, symbol: str, correlation_data: pd.DataFrame) -> Dict:
        """
        Check if adding position increases correlation risk
        """
        if not self.positions:
            return {'correlation_risk': 0, 'warning': None}

        # Get correlations with existing positions
        existing_symbols = list(self.positions.keys())

        if symbol not in correlation_data.columns:
            return {'correlation_risk': 0, 'warning': 'No correlation data available'}

        max_correlation = 0
        correlated_positions = []

        for existing_symbol in existing_symbols:
            if existing_symbol in correlation_data.columns:
                correlation = correlation_data.loc[symbol, existing_symbol]
                if abs(correlation) > self.config['max_correlation_exposure']:
                    correlated_positions.append({
                        'symbol': existing_symbol,
                        'correlation': correlation
                    })
                max_correlation = max(max_correlation, abs(correlation))

        risk_level = 'low' if max_correlation < 0.5 else 'medium' if max_correlation < 0.7 else 'high'

        return {
            'correlation_risk': max_correlation,
            'risk_level': risk_level,
            'correlated_positions': correlated_positions,
            'recommendation': 'reduce_size' if max_correlation > self.config['max_correlation_exposure'] else 'proceed'
        }

    def calculate_portfolio_risk_metrics(self, returns_data: pd.DataFrame) -> RiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        """
        if returns_data.empty or len(returns_data) < 2:
            return self._default_risk_metrics()

        # Calculate returns statistics
        returns = returns_data['returns'] if 'returns' in returns_data.columns else returns_data

        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5)

        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()

        # Sharpe Ratio (crypto adjusted for higher volatility)
        risk_free_rate = 0.02 / 365  # Daily risk-free rate
        excess_returns = returns - risk_free_rate
        sharpe_ratio = np.sqrt(365) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

        # Sortino Ratio (downside risk only)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
        sortino_ratio = np.sqrt(365) * excess_returns.mean() / downside_std if downside_std > 0 else 0

        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]

        # Portfolio exposure
        total_exposure = self._calculate_total_exposure()
        total_exposure_pct = (total_exposure / self.current_capital) * 100

        # Correlation risk (average correlation between positions)
        correlation_risk = self._calculate_portfolio_correlation()

        # Concentration risk (Herfindahl index)
        concentration_risk = self._calculate_concentration_risk()

        # Kelly fraction for portfolio
        kelly_fraction = self._calculate_portfolio_kelly(returns)

        return RiskMetrics(
            total_exposure=total_exposure,
            total_exposure_pct=total_exposure_pct,
            portfolio_var=var_95,
            portfolio_cvar=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            kelly_fraction=kelly_fraction
        )

    def check_risk_limits(self) -> Dict:
        """
        Check if any risk limits are breached
        """
        warnings = []
        stop_trading = False

        # Check daily loss limit
        if abs(self.daily_loss) > self.config['max_daily_loss_pct']:
            warnings.append({
                'type': 'daily_loss_limit',
                'severity': 'critical',
                'message': f"Daily loss limit breached: {self.daily_loss:.2f}%",
                'action': 'stop_trading'
            })
            stop_trading = True

        # Check maximum drawdown
        if abs(self.current_drawdown) > self.config['max_drawdown_pct']:
            warnings.append({
                'type': 'max_drawdown',
                'severity': 'critical',
                'message': f"Maximum drawdown breached: {self.current_drawdown:.2f}%",
                'action': 'reduce_exposure'
            })

        # Check total exposure
        total_exposure_pct = (self._calculate_total_exposure() / self.current_capital) * 100
        if total_exposure_pct > self.config['max_total_exposure_pct']:
            warnings.append({
                'type': 'exposure_limit',
                'severity': 'high',
                'message': f"Total exposure too high: {total_exposure_pct:.2f}%",
                'action': 'no_new_positions'
            })

        # Check liquidity ratio
        cash_ratio = 1 - (total_exposure_pct / 100)
        if cash_ratio < self.config['min_liquidity_ratio']:
            warnings.append({
                'type': 'liquidity',
                'severity': 'medium',
                'message': f"Low liquidity: {cash_ratio:.2%} cash remaining",
                'action': 'increase_cash_reserves'
            })

        return {
            'warnings': warnings,
            'stop_trading': stop_trading,
            'risk_score': len(warnings),
            'can_trade': not stop_trading and len(warnings) < 3
        }

    def add_position(self, symbol: str, entry_price: float, quantity: float,
                    position_type: str, stop_loss: float, take_profit: List) -> Dict:
        """
        Add a new position to the portfolio
        """
        position_value = entry_price * quantity
        risk_amount = abs(entry_price - stop_loss) * quantity

        # Check if position can be added
        risk_check = self.check_risk_limits()
        if not risk_check['can_trade']:
            return {
                'success': False,
                'reason': 'Risk limits prevent new position',
                'warnings': risk_check['warnings']
            }

        # Add position
        self.positions[symbol] = Position(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            quantity=quantity,
            position_value=position_value,
            unrealized_pnl=0,
            unrealized_pnl_pct=0,
            entry_time=datetime.now(),
            position_type=position_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount
        )

        # Log the trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'action': 'open',
            'symbol': symbol,
            'price': entry_price,
            'quantity': quantity,
            'type': position_type,
            'risk_amount': risk_amount
        })

        return {
            'success': True,
            'position': self.positions[symbol],
            'portfolio_exposure': self._calculate_total_exposure(),
            'remaining_buying_power': self.current_capital - self._calculate_total_exposure()
        }

    def update_positions(self, price_updates: Dict[str, float]) -> Dict:
        """
        Update position values with current prices
        """
        total_pnl = 0
        position_updates = {}

        for symbol, position in self.positions.items():
            if symbol in price_updates:
                current_price = price_updates[symbol]
                position.current_price = current_price

                # Calculate P&L
                if position.position_type == 'long':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:  # short
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

                position.unrealized_pnl_pct = (position.unrealized_pnl / position.position_value) * 100
                position.position_value = current_price * position.quantity

                total_pnl += position.unrealized_pnl

                # Check stop loss
                if self._check_stop_loss(position, current_price):
                    position_updates[symbol] = 'stop_loss_hit'

                # Check take profit levels
                tp_hit = self._check_take_profit(position, current_price)
                if tp_hit:
                    position_updates[symbol] = f'take_profit_hit: {tp_hit}'

        # Update portfolio metrics
        self.current_capital = self.initial_capital + total_pnl + sum(
            p['pnl'] for p in self.closed_positions
        )

        # Update daily P&L
        if self.daily_pnl and self.daily_pnl[-1]['date'] == datetime.now().date():
            self.daily_pnl[-1]['pnl'] = total_pnl
        else:
            self.daily_pnl.append({'date': datetime.now().date(), 'pnl': total_pnl})

        # Update equity curve
        self.equity_curve.append(self.current_capital)

        # Update drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.current_capital)
        self.current_drawdown = ((self.current_capital - self.max_portfolio_value) /
                                self.max_portfolio_value) * 100

        return {
            'total_unrealized_pnl': total_pnl,
            'current_capital': self.current_capital,
            'position_updates': position_updates,
            'current_drawdown': self.current_drawdown
        }

    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        return sum(p.position_value for p in self.positions.values())

    def _calculate_portfolio_correlation(self) -> float:
        """Calculate average correlation between positions"""
        if len(self.positions) < 2:
            return 0

        # Simplified - in production would use actual correlation matrix
        return 0.5  # Placeholder

    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk using Herfindahl index"""
        if not self.positions:
            return 0

        total_value = self._calculate_total_exposure()
        if total_value == 0:
            return 0

        weights = [p.position_value / total_value for p in self.positions.values()]
        herfindahl = sum(w**2 for w in weights)

        return herfindahl

    def _calculate_portfolio_kelly(self, returns: pd.Series) -> float:
        """Calculate optimal portfolio allocation using Kelly Criterion"""
        if len(returns) < 30:
            return 0.1  # Default conservative allocation

        mean_return = returns.mean()
        variance = returns.var()

        if variance == 0:
            return 0.1

        kelly = mean_return / variance
        kelly = max(0, min(kelly, 0.25))  # Cap at 25% for crypto safety

        return kelly

    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if position.position_type == 'long':
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss

    def _check_take_profit(self, position: Position, current_price: float) -> Optional[str]:
        """Check if any take profit level is hit"""
        for tp_label, tp_price in position.take_profit:
            if position.position_type == 'long' and current_price >= tp_price:
                return tp_label
            elif position.position_type == 'short' and current_price <= tp_price:
                return tp_label
        return None

    def _generate_position_reasoning(self, kelly: float, volatility: float,
                                   confidence: float, signal_strength: float) -> str:
        """Generate reasoning for position sizing"""
        reasons = []

        if kelly > 0.15:
            reasons.append(f"Strong Kelly fraction: {kelly:.2%}")
        elif kelly > 0.05:
            reasons.append(f"Moderate Kelly fraction: {kelly:.2%}")
        else:
            reasons.append(f"Weak Kelly fraction: {kelly:.2%}")

        if volatility > 0.05:
            reasons.append(f"High volatility adjustment needed")

        if confidence > 0.8:
            reasons.append(f"High confidence signal")
        elif confidence > 0.6:
            reasons.append(f"Moderate confidence signal")
        else:
            reasons.append(f"Low confidence - reduced position")

        if signal_strength > 0.8:
            reasons.append(f"Strong technical alignment")

        return " | ".join(reasons)

    def _default_risk_metrics(self) -> RiskMetrics:
        """Return default risk metrics when data insufficient"""
        return RiskMetrics(
            total_exposure=self._calculate_total_exposure(),
            total_exposure_pct=(self._calculate_total_exposure() / self.current_capital) * 100,
            portfolio_var=0,
            portfolio_cvar=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            current_drawdown=0,
            correlation_risk=0,
            concentration_risk=0,
            kelly_fraction=0.1
        )

    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'pnl': self.current_capital - self.initial_capital,
                'pnl_pct': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            },
            'positions': {
                'open': len(self.positions),
                'closed': len(self.closed_positions),
                'total_exposure': self._calculate_total_exposure(),
                'exposure_pct': (self._calculate_total_exposure() / self.current_capital) * 100
            },
            'risk': {
                'current_drawdown': self.current_drawdown,
                'max_drawdown': min(self.equity_curve) if self.equity_curve else 0,
                'daily_pnl': self.daily_pnl[-1] if self.daily_pnl else {'date': None, 'pnl': 0},
                'risk_warnings': self.check_risk_limits()['warnings']
            },
            'can_trade': self.check_risk_limits()['can_trade']
        }