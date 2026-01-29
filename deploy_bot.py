import os
import time
import pandas as pd
import numpy as np
import schedule
import sys
import argparse
from math import floor
from datetime import datetime
from alpaca_trade_api.rest import REST, TimeFrame
from config import API_KEY, SECRET_KEY, BASE_URL, SYMBOL, TRADE_AMOUNT
from data_loader import fetch_latest_multi_bars, fetch_latest_bars
from features import add_multi_timeframe_indicators, add_technical_indicators
from stable_baselines3 import PPO
from model_factory import RandomForestModel

# --- Configuration ---
DAILY_TARGET_PCT = 0.01  # Snowball Protocol: 1% Target
RF_THRESHOLD = 0.50

# Model Registry
MODELS = {
    "rf_w60": {
        "name": "Ultra RF (W60)",
        "type": "RF",
        "path": "models/ultra_rf_(w60).pth",
        "lookback": 60,
        "features": ['close', 'rsi', 'macd', 'macd_signal', 'bb_width', 'atr', 'log_return', 'adx', 'cci', 'mfi']
    },
    "v1": {
        "name": "Precision Sniper v1",
        "type": "RL",
        "path": "models/ppo_precision_sniper.zip",
        "expected_obs": 72
    },
    "v2": {
        "name": "Precision Sniper v2",
        "type": "RL",
        "path": "models/ppo_precision_sniper_v2.zip",
        "expected_obs": 80
    },
    "v5": {
        "name": "ARKK Optimized v5",
        "type": "RL",
        "path": "models/ppo_arkk_optimized_v5.zip",
        "expected_obs": 80
    },
    "tqqq_v2": {
        "name": "TQQQ Precision Sniper",
        "type": "RL",
        "path": "models/ppo_tqqq_precision_sniper.zip",
        "expected_obs": 80
    },
    "ensemble": {
        "name": "Curriculum Switcher (Sniper + Defender)",
        "type": "ENSEMBLE",
        "aggressive": "v2",
        "defender": "models/ppo_dynamic_defender_v2.zip",
        "low_vol_thresh": 0.000185 # From analyze_regimes.py
    },
    "v6": {
        "name": "QQQ V6",
        "type": "RL",
        "path": "models/ppo_ultimate_v6.zip",
        "expected_obs": 80
    }
}

class RegimeSwitcher:
    def __init__(self, low_thresh):
        self.low_thresh = low_thresh
        self.current_regime = "Normal/Trend"
        
    def get_model_key(self, df):
        # Calculate 60-min relative ATR
        # (1m_atr / 1m_close)
        rel_atr = (df['1m_atr'] / df['1m_close']).tail(60).mean()
        
        if rel_atr <= self.low_thresh:
            self.current_regime = "LOW VOL (Defender Mode)"
            return "defender"
        else:
            self.current_regime = "NORMAL/TREND (Aggressive Mode)"
            return "aggressive"

# Initialize Alpaca (Moved to main for dynamic override)
api = None

# Global State
start_equity = 0
daily_stop_hit = False
max_price = 0

def get_equity():
    try:
        account = api.get_account()
        return float(account.equity)
    except Exception as e:
        print(f"Error fetching account info: {e}")
        return 0

def get_current_position():
    try:
        pos = api.get_position(SYMBOL)
        return float(pos.qty), float(pos.avg_entry_price), float(pos.unrealized_plpc)
    except:
        return 0, 0, 0

def trading_step(model_data, model_instances):
    global start_equity, daily_stop_hit, max_price
    
    now = datetime.now()
    
    # --- Market Hours Logic ---
    # Eastern Time assumed as system time (or handled by Alpaca check, but we enforce locally)
    # Open: 09:30, Close: 16:00, Liquidate: 15:55
    current_hhmm = now.hour * 100 + now.minute
    
    # Removed Market Hours Check as per user request (Continuous Trading)

    
    # ... (Daily Reset and Position/Equity checks remain same) ...
    # Handle both single model and dict of instances for Ensemble
    
    # 2. Strategy Specific Execution
    action = 0 # 0: Nothing, 1: Buy, 2: Sell
    df_for_log = None
    
    active_bot_name = model_data['name']
    
    if model_data['type'] == "ENSEMBLE":
        # 0. Fetch MTF Data for Regime Detection
        df_raw = fetch_latest_multi_bars(SYMBOL, limit=100)
        df_mtf = add_multi_timeframe_indicators(df_raw, sr_window=12)
        if df_mtf is None or df_mtf.empty:
            print(f"[{now.strftime('%H:%M:%S')}] ⏳ Waiting for data...", end='\r')
            return
        
        # 1. Regime Detection
        bot_key = model_instances['switcher'].get_model_key(df_mtf)
        model_instance = model_instances[bot_key]
        active_bot_name = f"ENSEMBLE ({model_instances['switcher'].current_regime})"
        
        # 2. Prepare Observation
        current_row = df_mtf.iloc[-1]
        features = current_row.drop('timestamp', errors='ignore').values.astype(np.float32)
        
        # If we selected defender or v1, check shape
        expected_shape = model_instance.observation_space.shape[0]
        if expected_shape == 72:
             vol_features = [c for c in df_mtf.columns if 'vol_spike' in c or 'vwap_prox' in c]
             reduced_features = current_row.drop(labels=vol_features, errors='ignore').drop('timestamp', errors='ignore').values.astype(np.float32)
             features = reduced_features

        pos_qty, _, unrealized_plpc = get_current_position()
        current_equity = get_equity()
        daily_pnl_pct = (current_equity - start_equity) / start_equity if start_equity > 0 else 0
        
        obs = np.append(features, [float(1 if pos_qty > 0 else 0), float(unrealized_plpc), float(daily_pnl_pct)]).astype(np.float32)
        action, _ = model_instance.predict(obs, deterministic=True)
        df_for_log = df_mtf
        
    elif model_data['type'] == "RF":
        # ... (RF logic remains same) ...
        df = fetch_latest_bars(SYMBOL, limit=200)
        df = add_technical_indicators(df)
        if len(df) < model_data['lookback']: return
        X_input = df[model_data['features']].tail(model_data['lookback']).values
        X_input = np.array([X_input])
        action = 1 if model_instances.get_confidence(X_input)[0] >= RF_THRESHOLD else 0
        df_for_log = df

    elif model_data['type'] == "RL":
        # ... (RL logic remains same) ...
        df_raw = fetch_latest_multi_bars(SYMBOL, limit=150)
        # Safety check for data
        if df_raw.empty or len(df_raw) < 20:
            print(f"[{datetime.now()}] Insufficient data fetched ({len(df_raw)} candles). Skipping step.")
            return

        df = add_multi_timeframe_indicators(df_raw, sr_window=12)
        if df is None or df.empty:
            print(f"[{now.strftime('%H:%M:%S')}] ⏳ Waiting for data...", end='\r')
            return
        current_row = df.iloc[-1]
        features = current_row.drop('timestamp', errors='ignore').values.astype(np.float32)
        if model_data.get('expected_obs') == 72:
            vol_features = [c for c in df.columns if 'vol_spike' in c or 'vwap_prox' in c]
            features = current_row.drop(labels=vol_features, errors='ignore').drop('timestamp', errors='ignore').values.astype(np.float32)
        
        pos_qty, _, unrealized_plpc = get_current_position()
        current_equity = get_equity()
        daily_pnl_pct = (current_equity - start_equity) / start_equity if start_equity > 0 else 0
        obs = np.append(features, [float(1 if pos_qty > 0 else 0), float(unrealized_plpc), float(daily_pnl_pct)]).astype(np.float32)
        action, _ = model_instances.predict(obs, deterministic=True)
        df_for_log = df

    # Order Execution Logic ...
    if df_for_log is None or df_for_log.empty: return
    current_price = df_for_log['close'].iloc[-1]
    pos_qty, _, unrealized_plpc = get_current_position()
    current_equity = get_equity()
    daily_pnl_pct = (current_equity - start_equity) / start_equity if start_equity > 0 else 0

    # Max Price Tracking for Trailing Stop
    # --- Prepare telemetry logging early so trailing-stop can write to it ---
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    safe_bot_name = active_bot_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    log_file = f"{log_dir}/telemetry_{SYMBOL}_{safe_bot_name}.csv"
    file_exists = os.path.isfile(log_file)

    global max_price
    if pos_qty == 0:
        max_price = 0
    else:
        # If we just started or restarted, init max_price
        if max_price == 0: 
            max_price = current_price
        
        # Update High
        if current_price > max_price:
            max_price = current_price
            
        # --- TRAILING STOP LOGIC ---
        # 1. Get ATR
        # We need to ensure we have ATR. Both DF paths above (RF and RL) add indicators.
        # Check if 'atr' or '1m_atr' exists in df_for_log
        current_atr = 0
        if '1m_atr' in df_for_log.columns:
            current_atr = df_for_log['1m_atr'].iloc[-1]
        elif 'atr' in df_for_log.columns:
            current_atr = df_for_log['atr'].iloc[-1]
            
        if current_atr > 0:
            # 2. Multi-Stage Logic (Match Training Env)
            # Default: 2.5x ATR (Wide) -> 0.5x ATR (Tight) if >0.1% Profit
            current_trail_mult = 2.5
            if unrealized_plpc >= 0.001:
                current_trail_mult = 0.5
            
            stop_price = max_price - (current_atr * current_trail_mult)
            
            # 3. Check Stop
            if current_price <= stop_price:
                print(f"🛑 [{active_bot_name}] TRAILING STOP HIT! Price: {current_price:.2f} (High: {max_price:.2f}, Stop: {stop_price:.2f})")
                print(f"   Reason: Reversal of {current_trail_mult}x ATR ({current_atr:.4f})")
                action = 2 # Force Sell
                # Add logging info
                with open(log_file, "a") as f:
                    # Mark action as 2 (Sell) but maybe add a note? standard csv format doesn't have 'reason'
                    # We just log it as a standard sell for now, the console log captures the reason.
                    pass

    if action == 1 and pos_qty == 0:
        print(f"🤖 [{active_bot_name}] BUY SIGNAL for {SYMBOL} at {current_price:.2f}")
        try:
            # 1. Proactive Maintenance: Cancel all open orders FOR THIS SYMBOL to release held Buying Power
            open_orders = api.list_orders(status='open')
            my_orders = [o for o in open_orders if o.symbol == SYMBOL]
            if my_orders:
                print(f"🧹 Finding {len(my_orders)} open orders for {SYMBOL}. Cancelling to release BP...")
                for o in my_orders:
                    api.cancel_order(o.id)
                time.sleep(2) 
            
            # 2. Re-fetch fresh account state with ALL metrics
            account = api.get_account()
            bp = float(account.buying_power)
            nm_bp = float(account.non_marginable_buying_power)
            cash = float(account.cash)
            equity = float(account.equity)
            
            # Check for withdrawable/unsettled (Crucial for Cash Accounts)
            withdrawable = float(account.cash_withdrawable) if hasattr(account, 'cash_withdrawable') else cash
            
            print(f"--- 🔍 ADVANCED DIAGNOSTICS ({SYMBOL}) ---")
            print(f"  Account Status: {account.status} | Shorting: {account.shorting_enabled}")
            print(f"  Equity: ${equity:.2f} | Total Cash: ${cash:.2f}")
            print(f"  Withdrawable/Settled: ${withdrawable:.2f}")
            print(f"  Buying Power: ${bp:.2f} | Non-Marg BP: ${nm_bp:.2f}")
            print(f"  DayTrade BP: ${account.daytrading_buying_power if hasattr(account, 'daytrading_buying_power') else 'N/A'}")
            
            if bp < 10 and cash > 90:
                print("⚠️ DETECTION: Cash is present but Buying Power is near zero.")
                print("   This is usually caused by 'Unsettled Funds' (T+1/T+2) or a pending transfer.")

            # 3. Check for other positions
            all_pos = api.list_positions()
            if len(all_pos) > 0:
                print(f"⚠️ Other active positions found: {[p.symbol for p in all_pos]}")
            
            # 4. Calculation Logic
            # We use the Non-Marginable BP if available (for fractional), else settled cash
            base_bp = nm_bp if nm_bp > 0 else withdrawable
            
            # If the user has $100 cash, we WANT to use it. 
            # If Alpaca reports $0.89 BP, we literally cannot place the order.
            
            buffer = 0.85 # 15% safety buffer
            qty = (base_bp * buffer) / current_price
            qty = floor(qty * 10000) / 10000.0 
            
            if qty * current_price >= 1.0:
                print(f"  >>> Attempting Fractional Order: {qty} shares (~${qty * current_price:.2f})")
                
                # Check for post-market
                if now.hour >= 16 or now.hour < 9:
                    print("  ⚠️ WARNING: Market is currently closed. Market orders will be held or may fail.")
                
                order = api.submit_order(
                    symbol=SYMBOL, 
                    qty=qty, 
                    side='buy', 
                    type='market', 
                    time_in_force='day'
                )
                print(f"  ✅ Order Submitted! ID: {order.id} | Status: {order.status} | Symbol: {order.symbol}")
            else:
                print(f"⚠️ Order too small (${qty * current_price:.2f} based on BP ${base_bp:.2f}). Minimum $1.00 required.")
                if base_bp < 2.0:
                    print(f"❌ ACCOUNT STALL: Alpaca is only allowing ${base_bp:.2f} of your ${cash:.2f} total cash to be used right now.")
                    print("   PRO-TIP: Check if you have 'Paper Trading' set to 'Cash' account instead of 'Margin'.")
        except Exception as e:
            print(f"Trade failed: {e}")
            
    elif action == 2 and pos_qty > 0:
        print(f"🤖 [{active_bot_name}] SELL {SYMBOL} at {current_price:.2f} (PnL: {unrealized_plpc:.2%})")
        try:
            api.submit_order(symbol=SYMBOL, qty=pos_qty, side='sell', type='market', time_in_force='day')
        except Exception as e:
            print(f"Exit failed: {e}")

    # Heartbeat
    if pos_qty > 0:
        print(f"[{now.strftime('%H:%M:%S')}] {active_bot_name} | HOLDING | PnL: {unrealized_plpc:.2%} | Daily: {daily_pnl_pct:.2%}")
    else:
        print(f"[{now.strftime('%H:%M:%S')}] {active_bot_name} | ANALYZING | Price: {current_price:.2f} | Daily: {daily_pnl_pct:.2%}")
        
    # --- Telemetry Logging via CSV ---
    with open(log_file, "a") as f:
        if not file_exists:
            f.write("Timestamp,Price,DailyPnL,UnrealizedPnL,Action,Equity\n")
        # Action: 0=Hold, 1=Buy, 2=Sell
        # Timestamp: ISO format
        f.write(f"{now.isoformat()},{current_price},{daily_pnl_pct},{unrealized_plpc},{action},{current_equity}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy a NASDAQ Trading Bot")
    parser.add_argument("--bot", choices=["rf_w60", "v1", "v2", "v5", "v6", "ensemble", "tqqq_v2"], required=True, help="Which bot to deploy")
    parser.add_argument("--symbol", type=str, default=SYMBOL, help="Symbol to trade (overrides config)")
    parser.add_argument("--daily_target", type=float, default=0.01, help="Daily profit target (default: 0.01). Set to 999 to disable.")
    parser.add_argument("--key", type=str, help="Alpaca API Key (overrides config)")
    parser.add_argument("--secret", type=str, help="Alpaca API Secret (overrides config)")
    parser.add_argument("--url", type=str, help="Alpaca API Base URL (overrides config)")
    args = parser.parse_args()
    
    # Credentials override
    k = args.key if args.key else API_KEY
    s = args.secret if args.secret else SECRET_KEY
    u = args.url if args.url else BASE_URL
    DAILY_TARGET_PCT = args.daily_target

    # Credentials processed (no enforced symbol overrides)
    
    # Check for placeholder values
    placeholders = ["YOUR_API_KEY", "YOUR_ACCOUNT", "YOUR_SECRET"]
    if any(p in k.upper() for p in placeholders) or k == "":
        print(f"\n❌ ERROR: Invalid API Key Detected: '{k}'")
        print("Please edit start_all_bots.bat and replace placeholders with your real Alpaca keys.")
        sys.exit(1)

    print(f"Connecting to Alpaca at: {u}")
    api = REST(k, s, u)
    
    SYMBOL = args.symbol # Update global symbol
    bot_key = args.bot
    model_data = MODELS[bot_key]
    
    print("="*60)
    print(f"🚀 DEPLOYING: {model_data['name']}")
    print("="*60)
    
    # Load model / models
    if model_data['type'] == "RF":
        model_instance = RandomForestModel()
        model_instance.load(model_data['path'])
    elif model_data['type'] == "ENSEMBLE":
        print("Loading Ensemble Models...")
        aggressive_bot = MODELS[model_data['aggressive']]
        model_instance = {
            'aggressive': PPO.load(aggressive_bot['path'].replace('.zip', '')),
            'defender': PPO.load(model_data['defender'].replace('.zip', '')),
            'switcher': RegimeSwitcher(model_data['low_vol_thresh'])
        }
        print("Ensemble (Aggressive + Defender) loaded.")
    else:
        model_instance = PPO.load(model_data['path'].replace('.zip', ''))
    
    print(f"Model {bot_key} loaded successfully.")
    start_equity = get_equity()
    
    schedule.every(1).minutes.do(lambda: trading_step(model_data, model_instance))
    trading_step(model_data, model_instance)
    
    print("\n[ACTIVE] Monitoring market... Press Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down. Best of luck!")
            sys.exit(0)
        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(10)
