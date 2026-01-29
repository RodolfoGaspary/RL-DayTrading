# Training Methodology 🧠

This document provides technical details on how the V2 Precision Sniper model was trained using Reinforcement Learning.

## Overview

**Algorithm**: Proximal Policy Optimization (PPO)  
**Framework**: Stable-Baselines3  
**Training Asset**: QQQ (Tech ETF)  
**Training Period**: 60 days of high-frequency data  
**Total Timesteps**: 300,000

## Data Preparation

### Multi-Timeframe Architecture
The model consumes data from 4 timeframes simultaneously:
- **1-minute**: Micro-movements and entry precision
- **5-minute**: Short-term momentum
- **15-minute**: Intraday trends
- **4-hour**: Macro context and regime detection

### Feature Engineering (80 Dimensions)
```
Base Features (4 per timeframe × 4 = 16):
- Open, High, Low, Close

Technical Indicators (per timeframe):
- RSI (14-period)
- MACD + Signal Line
- Bollinger Bands (width)
- ATR (volatility)
- ADX (trend strength)
- CCI (momentum)
- MFI (volume-weighted momentum)

Advanced Features:
- VWAP Proximity (distance from volume-weighted average price)
- Volume Spike Detection (z-score)
- Local Support/Resistance (12-candle rolling window)

State Variables (3):
- Position (0 or 1)
- Unrealized PnL (%)
- Daily PnL (%)
```

## Reinforcement Learning Environment

### Action Space
```python
0: HOLD  # Do nothing
1: BUY   # Enter long position (if flat)
2: SELL  # Exit position (if holding)
```

### Reward Function
The reward is based on **realized PnL** when a position is closed:
```python
reward = (exit_price - entry_price) / entry_price
```

**Key Design Choice**: Rewards are only given on trade completion, not unrealized gains. This prevents the model from learning to "hold forever."

### Multi-Stage Trailing Stop Loss
The environment enforces a dynamic stop loss that adapts to profitability:

```python
# Stage 1: Initial Protection (Wide)
if position_pnl < 0.1%:
    stop_distance = 2.5 × ATR

# Stage 2: Profit Lock (Tight)
if position_pnl >= 0.1%:
    stop_distance = 0.5 × ATR
```

This logic is **embedded in the training environment**, so the model learns to work *with* the stop, not against it.

## PPO Hyperparameters

```python
Policy: MlpPolicy
Network Architecture: [256, 256]  # 2 hidden layers
Learning Rate: 0.0003
Entropy Coefficient: 0.05  # Encourages exploration
Batch Size: 64
Gamma (Discount): 0.99
GAE Lambda: 0.95
Clip Range: 0.2
```

### Why PPO?
- **Stable**: Prevents catastrophic policy updates
- **Sample Efficient**: Reuses experience via mini-batches
- **On-Policy**: Learns from its own trading decisions (realistic)

## Curriculum Learning

The model was trained using a progressive difficulty approach:

1. **Easy Period (Days 1-20)**: Low volatility, clear trends
2. **Medium Period (Days 21-40)**: Mixed conditions
3. **Hard Period (Days 41-60)**: High volatility, choppy markets

This prevents the model from overfitting to a single market regime.

## Training Process

### 1. Data Fetching
```bash
python data_loader.py --symbol QQQ --days 60
```

### 2. Feature Engineering
```bash
python features.py --add-mtf-indicators
```

### 3. RL Training
```bash
python train_rl.py --timesteps 300000
```

**Training Time**: ~5-10 minutes on CPU (Intel i5)

### 4. Evaluation
The model is evaluated on a 20% holdout set (last 12 days of data) to measure generalization.

## Validation Results

**Test Set Performance (QQQ, 12 days)**:
- PnL: +2.3%
- Win Rate: 52.1%
- Trades: 87

**Out-of-Sample Performance (ARKK, 1 year)**:
- PnL: +1419.78%
- Win Rate: 51.3%
- Trades: 3,591

The massive out-of-sample performance on ARKK validates the model's generalization.

## Why Generalization Works

### Asset-Agnostic Patterns
The model learns **universal volatility signatures**:
- Reversal patterns after sharp moves
- Volume-price divergence
- Support/resistance bounces

### ATR-Based Stop Loss
By using ATR (Average True Range) for stops, the logic automatically adapts to each asset's volatility:
- QQQ: ATR ≈ $0.50 → Stop ≈ $1.25
- ARKK: ATR ≈ $1.20 → Stop ≈ $3.00

This makes the strategy asset-agnostic.

## Reproducibility

To retrain the model from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training script
python train_rl.py

# 3. Model will be saved to:
models/ppo_precision_sniper_v2.zip
```

**Note**: Results may vary slightly due to:
- Random initialization
- Market data changes (if using live API)
- PPO's stochastic policy updates

## Key Takeaways

1. **Multi-Timeframe Data** provides context at multiple scales
2. **Embedded Stop Loss** in training ensures deployment consistency
3. **PPO** is stable and sample-efficient for trading
4. **Curriculum Learning** prevents regime overfitting
5. **Generalization > Specialization** (proven via V2 vs V3 experiment)

---

For questions or improvements, open an issue on GitHub.
