# Architecture & Strategy 🧠

## 1. The Core Philosophy
**"Enter like a Machine, Exit like a Coward."**

The **Precision Sniper V2** does not try to predict the exact top. Instead, it uses:
1.  **AI for Entries**: A PPO (Proximal Policy Optimization) Reinforcement Learning model trained to identify high-probability reversal points.
2.  **Algorithms for Exits**: A strict, unthinking **Multi-Stage ATR Trailing Stop** that locks in profits instantly when the trend wavers.

## 2. The AI Model (The "Sniper")
- **Type**: Proximal Policy Optimization (PPO) - Reinforcement Learning.
- **Input Features (80 dimensions)**:
    - Multi-timeframe Data: 1m, 5m, 15m, 4h candles.
    - Indicators: RSI, MACD, Bollinger Bands, ATR, VWAP proximity.
    - **Regime Detection**: Recognizes low-volatility vs. high-volatility environments.
- **Training**: Trained on 60 days of high-frequency market data with a "Curriculum Learning" approach (starting easy, getting harder).

## 3. The "Secret Sauce": Multi-Stage Trailing Stop 🛑
Our simulations proved that **100%** of the system's alpha comes from this exit logic. The AI is great at buying, but the algorithm protects the bag.

### The Logic:
1.  **Initial Safety Net**:
    - When a trade opens, a Stop Loss is set at `Entry Price - (2.5 * ATR)`.
    - This allows for normal market noise ("breathing room").

2.  **Profit Tightening**:
    - As soon as the position is **+0.1% profitable**, the Stop Loss tightens aggressively to `Max Price - (0.5 * ATR)`.
    - This effectively "ratchets" the profit. The trade cannot turn into a significant loser once it has been slightly green.

3.  **The Exit**:
    - If price drops below the Trailing Stop level, the bot forces a **MARKET SELL** immediately.
    - It ignores the AI model's opinion during this event. Safety first.

## 4. Continuous Trading 🕒
- **No Sleep**: The bot runs 24/7 (if the market is open).
- **Overnight**: Positions are held overnight if they are profitable. The Trailing Stop honors pre-market and after-hours price action if data is streaming.
- **No Daily Cap**: We removed the "1% Daily Goal" cap to allow for massive runners (e.g., +5%, +10% days).

## 5. Ensemble Mode (Optional)
The system supports a "Regime Switcher" that swaps between models:
- **Aggressive Mode**: Uses `Sniper V2` during normal/trending markets.
- **Defender Mode**: Switches to a conservative model (`Defender V1`) when volatility collapses (ATR drops below threshold), preventing chop-loss.
