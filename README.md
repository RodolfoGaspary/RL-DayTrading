# RL Day Trading Bot (Precision Sniper) 🦅

**An Advanced AI-Powered Trading System for High-Volatility Assets (QQQ, ARKK, SOXX)**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Stable-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Live%20Demo%20Testing-yellow)

> [!IMPORTANT]
> **DISCLAIMER: LIVE DEMO TESTING & SIMULATED RESULTS**
> This bot is currently being tested on a **Live Paper Trading (Demo)** account. The performance metrics listed below are derived from **Historical Simulations** using high-fidelity 1-minute candle data. While the logic used in simulation is identical to the live deployment, these results represent theoretical performance and do not guarantee future real-world returns.

## 🚀 Performance Highlights (1-Year Verification)

| Asset | PnL | Trades | Win Methodology |
|-------|-----|--------|-----------------|
| **ARKK** | **+1419.8%** | 3,591 | 100% Trailing Stop |
| **SOXX** | **+1277.6%** | 4,360 | 100% Trailing Stop |
| **QQQ** | **+647.5%** | 10,023 | 100% Trailing Stop |

*Verified via 1-minute candle simulation (365 days).*

## 📚 Documentation

- **[Strategy Overview](STRATEGY.md)**: How the bot works
- **[Training Methodology](TRAINING.md)**: Technical details on RL training
- **[Performance Results](PERFORMANCE.md)**: 1-year simulation data
- **[Deployment Guide](DEPLOYMENT.md)**: Run on Windows or Cloud VPS

## ⚡ Quick Start

### 1. Requirements
- Python 3.10+
- Alpaca Paper Trading Account (API Key & Secret)

### 2. Install
```bash
git clone https://github.com/yourusername/rl_day_trading_bot
cd rl-day-trading-bot
pip install -r requirements.txt
```

### 3. Run (Windows Dashboard)
```bash
python dashboard.py
```

### 4. Run (Headless / VPS)
```bash
python deploy_bot.py --bot v2 --symbol ARKK
```

---
*Disclaimer: This software is for educational purposes only. Past performance does not guarantee future results. Use at your own risk.*
