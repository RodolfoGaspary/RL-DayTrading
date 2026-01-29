# Active Models Summary

## 🏆 Production-Ready Models

### 1. Ultra RF (W60) - **PRIMARY MODEL**
- **File**: `ultra_rf_(w60).pth`
- **Performance**: +8.70% annual return (360-day simulation)
- **Trades/Day**: ~3.1
- **Win Rate**: 39.9%
- **Lookback Window**: 60 candles (1 hour of context)
- **Threshold**: 0.10% price move in 5 minutes
- **Status**: ✅ **READY FOR DEPLOYMENT**
- **Use Case**: Primary 1-minute scalping model

### 2. Ultra RF + Temporal - **SECONDARY MODEL**
- **File**: `ultra_rf_+_temporal.pth`
- **Performance**: +3.89% annual return (360-day simulation)
- **Trades/Day**: ~2.2
- **Win Rate**: 40.5%
- **Features**: Base features + hour/day/session patterns
- **Status**: ✅ Promising for time-based filtering
- **Use Case**: Alternative model with better selectivity

### 3. Apex RF - **ULTRA-SELECTIVE**
- **File**: `apex_rf.pth`
- **Performance**: +0.56% annual return (360-day simulation)
- **Trades/Day**: ~0.3
- **Win Rate**: 42.6%
- **Threshold**: 0.25% price move in 5 minutes
- **Status**: ⚠️ Marginal profitability
- **Use Case**: Experimental ultra-selective variant

## 📊 Legacy Models (5-Minute Timeframe)

### 4. Challenger TQQQ
- **File**: `challenger_tqqq.pth`
- **Timeframe**: 5-minute candles
- **Status**: Not tested in recent campaign
- **Use Case**: 5-minute QQQ/TQQQ sniper

### 5. Watcher Transformer
- **File**: `watcher_transformer.pth`
- **Timeframe**: 5-minute candles
- **Status**: Not tested in recent campaign
- **Use Case**: 5-minute transformer variant

### 6. Ultra RF (Baseline)
- **File**: `rf_1m_ultra_hardened.pth`
- **Performance**: +10.14% annual return (360-day simulation)
- **Lookback Window**: 20 candles
- **Status**: ✅ Strong performer, but W60 variant preferred
- **Use Case**: Baseline reference model

---

## Deployment Recommendation

**Deploy Ultra RF (W60)** as the primary model for live paper trading:

```python
# Configuration
MODEL_PATH = "models/ultra_rf_(w60).pth"
LOOKBACK_WINDOW = 60
THRESHOLD = 0.50  # Confidence threshold for trading
ATR_TRAIL_MULT = 3.0
```

**Expected Performance:**
- Annual Return: +8.70%
- Daily Trades: 3-4
- Win Rate: ~40%
- Max Drawdown: ~24%

**Risk Management:**
- Position Size: 1-2% of capital per trade
- Daily Loss Limit: -3%
- Weekly Retraining: Every Sunday with latest 14 days of data

---

## Model Maintenance

### Weekly Retraining Protocol
1. Fetch latest 14 days of 1-minute QQQ data
2. Run `train_ultra_hardened.py` with 60-candle lookback
3. Validate accuracy > 97%
4. Replace `ultra_rf_(w60).pth` with new model
5. Run 10-day backtest to verify performance

### Performance Monitoring
- Track daily win rate (alert if < 35% for 3 consecutive days)
- Monitor equity curve for drawdowns > 25%
- Log all trades to `logs/` directory

---

## Obsolete Models

All underperforming models have been moved to `obsolete/models/`:
- 20 experimental variants from optimization campaign
- Legacy hardened models (-41% to -59% returns)
- Failed leading indicator experiments

See `obsolete/README.md` for details.
