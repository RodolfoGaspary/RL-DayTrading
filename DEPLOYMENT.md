# Deployment Guide 🛠️

## Option A: Windows Dashboard (Local)
**Best for**: Monitoring, Visualization, Manual Intervention.

1.  **Open Terminal** in the project folder.
2.  **Run**:
    ```bash
    python dashboard.py
    ```
3.  **Features**:
    - **Visual**: Live graphs of Price, Buy/Sell markers, and PnL.
    - **Controls**: Start/Stop specific bots.
    - **Logs**: Real-time console output visible in the GUI.

---

## Option B: Cloud VPS (Headless / 24/7)
**Best for**: "Set and Forget", Production, Reliability.
**Recommended Provider**: **Oracle Cloud "Always Free"** (ARM Instance: 4 CPU, 24GB RAM).

### 1. Server Setup (Ubuntu 24.04)
Connect to your VPS via SSH (`ssh ubuntu@your-ip`).

### 2. Install Dependencies
```bash
sudo apt update && sudo apt install -y python3-pip git screen
git clone https://github.com/your-repo/rl-trading-bot.git
cd rl-trading-bot
pip3 install -r requirements.txt
```

### 3. Run with `screen`
Use `screen` to keep the bot running even if you disconnect.

```bash
# Create a session named 'arkk_bot'
screen -S arkk_bot

# Run the bot
python3 deploy_bot.py --bot v2 --symbol ARKK --key YOUR_KEY --secret YOUR_SECRET --url https://paper-api.alpaca.markets
```

### 4. Detach and Reattach
- **Detach**: Press `Ctrl+A`, then `d`. (The bot keeps running).
- **Reattach**: `screen -r arkk_bot`

### 5. Managing Multiple Bots
Just create multiple screens:
```bash
screen -S qqq_bot
# run QQQ bot...
# detach

screen -S soxx_bot
# run SOXX bot...
# detach
```

## 🔧 Configuration Arguments
`deploy_bot.py` accepts the following arguments:
- `--bot`: Model version (`v2`, `ensemble`, `rf_w60`).
- `--symbol`: Asset ticker (`ARKK`, `QQQ`, `TSLA`).
- `--daily_target`: Profit target percentage (e.g., `0.01` for 1%). Set to `999` to disable cap (Recommended).
- `--key`, `--secret`, `--url`: API Credentials (if not set in `config.py`).
