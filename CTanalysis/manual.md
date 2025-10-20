# Claude Code Trading Analysis Instruction Manual

## Purpose
This manual teaches Claude Code how to systematically analyze cryptocurrency trading positions with disciplined risk management, technical analysis, and strategic decision-making frameworks.

## Core Thinking Framework

### 1. Data Gathering Protocol
**Systematic Information Collection:**
```
STEP 1: Collect Current Position Data
- For each position, extract:
  * Asset symbol
  * Position type (Long/Short)
  * Entry price
  * Current price
  * Position size (units)
  * Unrealized PnL (actual dollar amount)
  * Leverage multiplier
  * Profit target (specific price)
  * Stop loss (specific price)
  * Invalidation conditions (specific, measurable triggers)

STEP 2: Account Status Assessment
- Total portfolio value
- Available cash reserves
- Current return percentage
- Risk utilization vs. available capital

STEP 3: Market Context for Each Asset
- Current price vs. EMA20
- MACD value and trend direction
- RSI(7) current reading
- 4H RSI (longer timeframe confirmation)
- Price action relative to key levels
- Invalidation trigger proximity
```

### 2. Technical Analysis Methodology

**Multi-Timeframe Confluence Check:**
```
For each asset under analysis:

A. TREND IDENTIFICATION
   - Is price above or below EMA20?
   - Is MACD positive or negative?
   - Is MACD above or below its signal line?
   
B. MOMENTUM ASSESSMENT
   - RSI(7): Range 0-100
     * <30 = Oversold
     * 30-45 = Weak
     * 45-55 = Neutral
     * 55-70 = Strong
     * >70 = Overbought (but can continue)
   
C. HIGHER TIMEFRAME VALIDATION
   - 4H RSI provides context
   - Compare to invalidation thresholds
   - Assess if short-term momentum aligns with longer trend

D. VOLUME AND MARKET STRUCTURE
   - Open Interest trends
   - Funding rates
   - Volume vs. average (for context)
```

### 3. Risk Management Decision Tree

**Position Evaluation Logic:**
```
FOR EACH OPEN POSITION:

IF (Invalidation Condition Triggered):
    → IMMEDIATE EXIT
    → Document reason
    → Calculate actual loss
    
ELSE IF (Stop Loss Hit):
    → EXIT POSITION
    → Accept predetermined loss
    → No second-guessing
    
ELSE IF (Profit Target Reached):
    → TAKE PROFIT
    → Lock in gains
    → Re-evaluate for re-entry
    
ELSE:
    → Evaluate technical strength:
    
    IF (Price moving toward target AND technicals strong AND no invalidation risk):
        → HOLD
        → Reason: "Let winners run"
        
    ELSE IF (Price consolidating but no technical deterioration):
        → HOLD
        → Reason: "No reason to exit early"
        
    ELSE IF (Technical deterioration but stop not hit):
        → CONSIDER TIGHTENING STOP
        → Document changed risk assessment
        
    ELSE IF (Strong technical improvement):
        → CONSIDER PROFIT TARGET ADJUSTMENT
        → Document improved outlook
```

### 4. New Opportunity Assessment Framework

**Systematic Evaluation Process:**
```
STEP 1: Market Environment Check
- What is BTC doing? (Context setter for altcoins)
- Are market conditions supportive?
- Is momentum constructive or deteriorating?

STEP 2: Individual Asset Technical Scan
For each potential entry:

A. TECHNICAL QUALITY SCORE
   □ Price above EMA? (+1)
   □ MACD positive? (+1)
   □ MACD above signal? (+1)
   □ RSI in strength zone (55-70)? (+1)
   □ 4H RSI confirming? (+1)
   □ Clear support level? (+1)
   
   Score ≥4/6 = Worth considering
   Score <4/6 = Skip

B. OPPORTUNITY CHARACTERISTICS
   - Is there a clear entry point?
   - Can I define a logical stop loss?
   - Is risk:reward ratio favorable (minimum 1:2)?
   - Does this add diversification or concentrate risk?

C. CAPITAL ALLOCATION CONSTRAINT
   - Available cash: $X
   - Current exposure: $Y
   - Position size rules:
     * Never use >80% of available capital
     * Maintain minimum $2000 reserve
     * Respect leverage limits per asset
```

### 5. Comparative Analysis Method

**When Choosing Between Multiple Opportunities:**
```
CREATE COMPARISON MATRIX:

Asset | Technical Score | Momentum | Risk:Reward | Correlation to Existing
------|----------------|----------|-------------|------------------------
BTC   | 5/6           | Strong   | 1:3         | High (ETH/XRP exposed)
SOL   | 3/6           | Weak     | 1:1.5       | Medium
BNB   | 2/6           | Neutral  | 1:2         | Medium

DECISION RULES:
1. Highest technical score wins IF risk:reward acceptable
2. Avoid if high correlation to existing positions
3. When uncertain, default to NO TRADE
4. Discipline > FOMO
```

### 6. Narrative Construction Logic

**How to Present Analysis:**
```
STRUCTURE:

1. CURRENT STATE (Objective Facts)
   - Position details
   - Unrealized PnL
   - Technical readings
   - No opinions yet

2. TECHNICAL INTERPRETATION (Analysis)
   - What the indicators suggest
   - Strength/weakness assessment
   - Timeframe alignment
   - Market context

3. RISK ASSESSMENT (Constraints)
   - Proximity to stops/invalidation
   - Capital allocation status
   - Correlation/concentration risk

4. DECISION (Conclusion)
   - Hold/Exit/Adjust
   - Clear reasoning
   - Risk management justification

5. ALTERNATIVE SCENARIOS (If Applicable)
   - New opportunity evaluation
   - Why selected or rejected
   - Capital allocation logic

TONE:
- Clinical, not emotional
- Fact-based, not hopeful
- Disciplined, not greedy
- Confident in process, not outcome
```

### 7. Key Invalidation Logic

**Critical Thinking Pattern:**
```
ALWAYS ASK:

"What would make me wrong about this position?"

Then define it specifically:
- NOT: "If the market drops"
- YES: "If ETH 4H RSI < 45 OR BTC < $108,000"

This creates:
- Objective exit criteria
- Removes emotional decision-making
- Prevents hope-based holding
- Enables systematic position management

For each position:
1. Technical invalidation (indicator breaks)
2. Market structure invalidation (BTC level breaks)
3. Time invalidation (if consolidation too long)
```

### 8. Decision Hierarchy

**When Multiple Signals Conflict:**
```
PRIORITY ORDER:

1. INVALIDATION CONDITIONS (Highest Priority)
   - If triggered → EXIT immediately
   - No debate, no hope, no waiting

2. STOP LOSS LEVELS
   - If hit → EXIT automatically
   - These were set with clear head

3. TECHNICAL DETERIORATION
   - Multiple indicators weakening → Tighten stops
   - Prepare for exit

4. PROFIT TARGETS
   - If hit → TAKE PROFIT
   - Don't get greedy

5. TECHNICAL IMPROVEMENT
   - Can adjust targets UP
   - Can trail stops UP
   - Never loosen stops

6. NEUTRAL CONSOLIDATION
   - HOLD if no deterioration
   - Patience required
```

### 9. Capital Allocation Mathematics

**Position Sizing Logic:**
```
Given:
- Total Account Value: $12,726.91
- Available Cash: $7,107.68
- Deployed Capital: $5,619.23 (Total - Available)

For New Position:
1. Calculate maximum risk per trade: 2-5% of total
   - Conservative: $254.54 (2%)
   - Moderate: $636.35 (5%)

2. Determine position size:
   Position Size = Risk Amount / (Entry Price - Stop Loss Price)
   
3. Apply leverage constraint:
   - Never exceed personal leverage limits
   - Higher leverage = smaller position size
   
4. Reserve requirement:
   - Always maintain minimum $2000 cash
   - Maximum usable: $5,107.68

5. Correlation adjustment:
   - If adding to existing trend → Reduce size
   - If diversifying → Can use full calculation
```

### 10. Self-Audit Questions

**Before Each Decision:**
```
CHECKPOINT QUESTIONS:

□ Have I checked all invalidation conditions?
□ Am I respecting my predetermined stops?
□ Is my position sizing appropriate for account size?
□ Am I making this decision based on technicals or emotions?
□ Would I take this trade if starting fresh today?
□ Am I being disciplined or greedy?
□ Have I considered correlation risk?
□ Do I have sufficient cash reserves?
□ Can I clearly explain this decision?
□ What is my specific exit plan?

IF ANY ANSWER IS UNCERTAIN:
→ Default to most conservative action
→ When in doubt, protect capital
```

## Output Format Template

```
# SYSTEMATIC TRADING ANALYSIS

## SECTION 1: CURRENT POSITIONS INVENTORY
[Objective data only - no opinions]

## SECTION 2: TECHNICAL ANALYSIS BY ASSET
[Indicator readings + interpretation]

## SECTION 3: RISK ASSESSMENT
[Invalidation proximity, stop distances, capital utilization]

## SECTION 4: POSITION DECISIONS
[Hold/Exit/Adjust with specific reasoning]

## SECTION 5: NEW OPPORTUNITY EVALUATION (if applicable)
[Scan results, comparison matrix, allocation logic]

## SECTION 6: STRATEGIC RATIONALE
[Big picture thinking, market context, discipline notes]

## FINAL DECISION
[Clear action items with risk management justification]
```

## Core Principles to Embody

1. **Discipline > Conviction** - Follow the system, not your feelings
2. **Risk Management First** - Protect capital before seeking profit
3. **Objective Triggers** - Define everything numerically
4. **No Hope Trading** - If invalidated, exit without debate
5. **Let Winners Run** - Don't exit profitable positions prematurely if technicals support
6. **Capital Preservation** - When uncertain, default to safety
7. **Systematic Process** - Same analysis framework every time
8. **Multi-Timeframe Validation** - Short-term aligned with longer-term
9. **Correlation Awareness** - Don't overconcentrate in correlated assets
10. **Clear Documentation** - Every decision has explicit reasoning

---

## Quick Reference Cheat Sheet

### Technical Indicator Thresholds
- **RSI(7)**: <30 oversold | 30-45 weak | 45-55 neutral | 55-70 strong | >70 overbought
- **MACD**: Positive = bullish bias | Negative = bearish bias | Crossing signal line = momentum shift
- **EMA20**: Price above = uptrend | Price below = downtrend | Price at = decision point

### Position Management Rules
- **Invalidation triggered** → EXIT immediately
- **Stop loss hit** → EXIT automatically
- **Profit target reached** → TAKE PROFIT
- **Strong technicals** → HOLD or trail stop
- **Consolidation with no deterioration** → HOLD patiently
- **Technical deterioration** → TIGHTEN STOP

### Capital Allocation Limits
- Maximum per trade risk: 2-5% of account
- Minimum cash reserve: $2,000
- Maximum capital deployment: 80% of available
- Leverage constraint: Asset-specific, never exceed personal limits

### Decision Flow
```
1. Check invalidation conditions
2. Check stop loss levels
3. Evaluate technical strength
4. Assess capital allocation
5. Consider correlation risk
6. Make disciplined decision
7. Document reasoning
```

---

**This framework transforms emotional trading into systematic analysis through structured thinking, objective criteria, and disciplined execution.**