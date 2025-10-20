# Model 3: Hybrid - Thinking Process

## What Model 3 Does

Takes the best parts of both models and creates a balanced "soup":

### From Model 1 (manual.md):
- ✅ RSI 14-period for reliability
- ✅ MACD 5-13-8 (proven for crypto)
- ✅ Bollinger Bands 2.5σ
- ✅ Volume 1.5× confirmation
- ✅ EMA 20/50 trend alignment

### From Model 2 (dp.md):
- ✅ RSI 7-period for fast entries
- ✅ Multi-timeframe (3m + 5m + 4H)
- ✅ Funding rate analysis
- ✅ 4H context for big picture
- ✅ Invalidation conditions

### New Hybrid Features:
- **Dual RSI**: Uses BOTH RSI-7 (fast) AND RSI-14 (slow) for confirmation
- **Triple timeframe**: 3m (scalp) + 5m (day) + 4H (context)
- **Moderate leverage**: 3-8× instead of 0× or 15×
- **Adaptive stops**: Tight (2.5%) in low volatility, Wide (5%) in high volatility
- **Stricter confluences**: Needs 4 of 9 sources (vs 3/5 in Model 1)

## Confluence System (9 Sources)

1. **Fast RSI (7)** - Quick scalping signals
2. **Slow RSI (14)** - Reliable day trading signals
3. **MACD** - Momentum confirmation
4. **Bollinger Bands** - Support/resistance
5. **Volume** - 1.5× confirmation
6. **5m Trend** - EMA alignment
7. **4H Trend** - Higher timeframe context
8. **Funding Rate** - Sentiment indicator
9. **Open Interest** - Market participation confirmation

**Requirement**: Minimum 4/9 must agree (stricter than Model 1's 3/5)

## Risk Management

### Leverage (3-8×)
- Low volatility (<3%) + high confidence (>0.7) = 7-8× leverage
- High volatility (>4%) = 3× leverage only
- Not extreme like Model 2 (15×), not zero like Model 1

### Stop Loss (Adaptive)
- **Tight stops (2.5%)**: When volatility < 3%
- **Wide stops (5%)**: When volatility > 3%
- Uses 4H ATR for volatility measurement

### Position Size
- 0.85% risk per trade (between Model 1's 0.75% and Model 2's 1.0%)

### Invalidation
- 2% hard stop (tighter than Model 2's 3%)
- Monitors both 3m and 5m candle closes

## Confidence Calculation

```
Base confidence = Sum of confluence strengths
Bonus = (confluences / 9) × 0.2
Final = Base + Bonus (capped at 0.95)
```

Example:
- 6/9 confluences active
- Total strength: 0.65
- Bonus: (6/9) × 0.2 = 0.13
- Final: 0.65 + 0.13 = 0.78 confidence

## Take Profit (Dynamic)

Risk:Reward ratio based on confidence:
- Low confidence (0.67): 1.5× risk
- Mid confidence (0.75): 2.25× risk
- High confidence (0.85): 2.78× risk
- Maximum: 3.0× risk

## When Hybrid Signals Trigger

**Example BTC Long**:
```
✅ Fast RSI-7: 21 (oversold on 3m)
✅ Slow RSI-14: 24 (oversold on 5m)
✅ MACD: Bullish crossover on 5m
✅ Bollinger: Price below lower band
✅ Volume: 1.8× average
✅ 5m Trend: Price > EMA20 > EMA50
❌ 4H Trend: Not aligned (EMA20 < EMA50)
✅ Funding: Low/neutral
✅ Open Interest: Rising (27,238 > avg 27,227)

Result: 7/9 confluences = VALID SIGNAL
Confidence: 0.80
Leverage: 6×
Stop: Adaptive (volatility dependent)
```

## Why This Works

1. **Double confirmation**: Both fast and slow RSI must agree
2. **Multi-timeframe**: See the full picture (scalp + day + context)
3. **Balanced risk**: Not too conservative, not too aggressive
4. **Adaptive**: Adjusts to market volatility automatically
5. **Strict requirements**: 4/9 confluences means high quality signals only
6. **Market participation**: Open interest confirms real conviction behind moves
