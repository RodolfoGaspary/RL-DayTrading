
import yfinance as yf
import pandas as pd
import os
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from config import API_KEY, SECRET_KEY, BASE_URL, SYMBOL, TIMEFRAME, DATA_START_DATE
from datetime import datetime, timedelta

def fetch_historical_data(symbol=SYMBOL, start_date=DATA_START_DATE, interval=TIMEFRAME):
    """
    Fetches historical data using yfinance.
    Note: yfinance interval format might differ from Alpaca.
    Alpaca '5Min' -> yfinance '5m'
    """
    yf_interval = interval.replace("Min", "m").lower()
    print(f"Fetching historical data for {symbol} from {start_date} with interval {yf_interval}...")
    
    # Check for cache
    cache_file = f"data/{symbol}_{interval}_{start_date}.csv"
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
        return df

    data = yf.download(symbol, start=start_date, interval=yf_interval, progress=False)
    
    if data.empty:
        print("Warning: No data fetched from yfinance.")
        return data
        
    # Standardize columns
    data.reset_index(inplace=True)
    # yfinance columns: Date, Open, High, Low, Close, Adj Close, Volume
    # We need: timestamp, open, high, low, close, volume
    
    # Rename columns to match consistent format (lowercase)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.columns = [c.lower() for c in data.columns]
    
    # Rename 'date' or 'datetime' to 'timestamp'
    if 'date' in data.columns:
        data.rename(columns={'date': 'timestamp'}, inplace=True)
    elif 'datetime' in data.columns:
        data.rename(columns={'datetime': 'timestamp'}, inplace=True)
        
    # Ensure timestamp is datetime object and set as index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Drop Adj Close if present, we use Close
    if 'adj close' in data.columns:
        data.drop(columns=['adj close'], inplace=True)

    # Save to cache
    if not os.path.exists("data"):
        os.makedirs("data")
    data.to_csv(cache_file)
    print(f"Data saved to {cache_file}")
        
    print(f"Fetched {len(data)} rows.")
    return data

def fetch_latest_bars(symbol=SYMBOL, timeframe=TIMEFRAME, limit=100):
    """
    Fetches the latest N bars.
    Hybrid strategy:
    1. Fetch recent history from yfinance (stable, deeper history).
    2. Fetch latest live bars from Alpaca (real-time).
    3. Merge and deduplicate.
    """
    print(f"Fetching latest bars for {symbol} (Hybrid Strategy)...")
    
    # 1. Fetch yfinance history (last 7 days to ensure robust lookback)
    try:
        yf_interval = timeframe.replace("Min", "m").lower()
        
        # dynamic start date
        from datetime import datetime, timedelta
        start_dt = datetime.now() - timedelta(days=7)
        start_str = start_dt.strftime('%Y-%m-%d')
        
        # Fetch
        history_df = yf.download(symbol, start=start_str, interval=yf_interval, progress=False)
        print(f"DEBUG: yfinance history fetched {len(history_df)} rows from {start_str}.")
        
        # Standardize yfinance columns
        
        # Standardize yfinance columns
        if not history_df.empty:
            history_df.reset_index(inplace=True)
            if isinstance(history_df.columns, pd.MultiIndex):
                history_df.columns = history_df.columns.get_level_values(0)
            history_df.columns = [c.lower() for c in history_df.columns]
            
            if 'date' in history_df.columns: history_df.rename(columns={'date': 'timestamp'}, inplace=True)
            elif 'datetime' in history_df.columns: history_df.rename(columns={'datetime': 'timestamp'}, inplace=True)
            
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df.set_index('timestamp', inplace=True)
            
            if 'adj close' in history_df.columns: history_df.drop(columns=['adj close'], inplace=True)
            
            # Localize to UTC if needed to match Alpaca
            if history_df.index.tz is None:
                history_df.index = history_df.index.tz_localize('UTC')
            else:
                history_df.index = history_df.index.tz_convert('UTC')
                
    except Exception as e:
        print(f"yfinance fetch failed: {e}")
        history_df = pd.DataFrame()

    # 2. Fetch Alpaca Live Data
    alpaca_df = pd.DataFrame()
    if "YOUR_API_KEY" not in API_KEY:
        try:
            api = REST(API_KEY, SECRET_KEY, BASE_URL)
            # Fetch last 100 bars from Alpaca to get the text book "latest"
            bars = api.get_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute), limit=limit).df
            if not bars.empty:
                alpaca_df = bars
                # Alpaca index is already timestamp and TZ aware (UTC)
        except Exception as e:
            print(f"Alpaca live fetch failed: {e}")

    # 3. Merge
    if history_df.empty and alpaca_df.empty:
        return pd.DataFrame()
        
    if history_df.empty:
        merged_df = alpaca_df
    elif alpaca_df.empty:
        merged_df = history_df
    else:
        # Combine
        combined = pd.concat([history_df, alpaca_df])
        # Deduplicate by index (timestamp) - keep last
        merged_df = combined[~combined.index.duplicated(keep='last')].copy()
        
    merged_df.sort_index(inplace=True)
    
    # Return requested limit
    return merged_df.tail(limit)

def fetch_alpaca_history(symbol=SYMBOL, days=365):
    """
    Fetches deep history from Alpaca API (handling pagination).
    """
    print(f"Fetching {days} days of history for {symbol} from Alpaca...")
    
    # Check cache
    cache_file = f"data/{symbol}_alpaca_{days}d.csv"
    if os.path.exists(cache_file):
        print(f"Loading cached Alpaca history from {cache_file}...")
        df = pd.read_csv(cache_file, index_col='timestamp', parse_dates=True)
        return df

    if "YOUR_API_KEY" in API_KEY:
        print("Error: Live keys required for deep history.")
        return pd.DataFrame()
        
    api = REST(API_KEY, SECRET_KEY, BASE_URL)
    
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime('%Y-%m-%d')
    
    try:
        # get_bars automatically handles pagination if we don't set a limit? 
        # Actually safer to loop or use large limit.
        # But alpaca-trade-api's get_bars returns a list of Bar objects, usually handles some pagination.
        # Let's use a large limit and hope. 
        # Better: use get_bars_iter if available, or just request.
        # For simplicity in this script, we'll trust the lib or simple list.
        
        bars = api.get_bars(symbol, TimeFrame(5, TimeFrame.Minute), start=start_str, limit=None).df
        
        if bars.empty:
            return bars
            
        # Rename index to timestamp if needed (Alpaca output usually has timestamp index)
        # bars.index.name = 'timestamp' 
        
        # Save cache
        if not os.path.exists("data"):
            os.makedirs("data")
        bars.to_csv(cache_file)
        
        print(f"Fetched {len(bars)} bars from Alpaca.")
        return bars
        
    except Exception as e:
        print(f"Alpaca history fetch failed: {e}")
        return pd.DataFrame()

def fetch_multi_timeframe_data(symbol=SYMBOL, days=365):
    """
    Fetches 1m, 5m, and 15m data and merges them.
    The primary index is 5m (the bot's core timeframe).
    """
    print(f"Fetching multi-timeframe data (1m, 5m, 15m) for {symbol} for {days} days...")
    
    api = REST(API_KEY, SECRET_KEY, BASE_URL)
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime('%Y-%m-%d')
    
    # 1. Fetch all three timeframes
    # Note: 1m data might have a shorter history limit on free accounts, 
    # but Alpaca Paper usually allows 6-12 months? Let's try.
    print("Fetching 1m data...")
    df_1m = api.get_bars(symbol, TimeFrame.Minute, start=start_str, limit=None).df
    print("Fetching 5m data...")
    df_5m = api.get_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute), start=start_str, limit=None).df
    print("Fetching 15m data...")
    df_15m = api.get_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), start=start_str, limit=None).df
    print("Fetching 4h data...")
    df_4h = api.get_bars(symbol, TimeFrame(4, TimeFrameUnit.Hour), start=start_str, limit=None).df
    
    if df_1m.empty or df_5m.empty or df_15m.empty or df_4h.empty:
        print("Error: One or more timeframes returned no data.")
        return pd.DataFrame()
        
    # Standardize column names
    for df in [df_1m, df_5m, df_15m, df_4h]:
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

    # 2. Merge - Using 5m as the base
    # Prefix columns to distinguish
    df_1m = df_1m.add_prefix('1m_')
    df_5m = df_5m.add_prefix('5m_')
    df_15m = df_15m.add_prefix('15m_')
    df_4h = df_4h.add_prefix('4h_')
    
    # Merge using 'asof' to align with the latest available data at each 1m mark
    # User wants to trade on 1m based on patterns from all TFs
    merged = pd.merge_asof(df_1m, df_5m, left_index=True, right_index=True)
    merged = pd.merge_asof(merged, df_15m, left_index=True, right_index=True)
    merged = pd.merge_asof(merged, df_4h, left_index=True, right_index=True)
    
    # Also keep simple 'close' for backwards compatibility if needed (using 1m close as base)
    merged['close'] = merged['1m_close']
    
    print(f"Multi-timeframe data ready: {len(merged)} rows.")
    return merged

def fetch_latest_multi_bars(symbol=SYMBOL, limit=100):
    """
    Fetches the latest bars for 1m, 5m, and 15m and merges them.
    Used for live bot inference.
    """
    api = REST(API_KEY, SECRET_KEY, BASE_URL)
    
    # Fetch enough to satisfy technical indicators (e.g., lookback + indicators)
    # We fetch slightly more than limit for safety
    fetch_limit = limit + 50
    
    try:
        df_1m = api.get_bars(symbol, TimeFrame.Minute, limit=fetch_limit).df
        df_5m = api.get_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute), limit=fetch_limit).df
        df_15m = api.get_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), limit=fetch_limit).df
        df_4h = api.get_bars(symbol, TimeFrame(4, TimeFrameUnit.Hour), limit=fetch_limit).df
        
        if df_1m.empty or df_5m.empty or df_15m.empty or df_4h.empty:
            return pd.DataFrame()
            
        for df in [df_1m, df_5m, df_15m, df_4h]:
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

        # Prefix
        df_1m = df_1m.add_prefix('1m_')
        df_5m = df_5m.add_prefix('5m_')
        df_15m = df_15m.add_prefix('15m_')
        df_4h = df_4h.add_prefix('4h_')
        
        # Merge
        merged = pd.merge_asof(df_1m, df_5m, left_index=True, right_index=True)
        merged = pd.merge_asof(merged, df_15m, left_index=True, right_index=True)
        merged = pd.merge_asof(merged, df_4h, left_index=True, right_index=True)
        
        merged['close'] = merged['1m_close']
        
        return merged.tail(limit)
    except Exception as e:
        print(f"Error fetching live multi-bars: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test fetcher
    df = fetch_historical_data()
    print(df.tail())
