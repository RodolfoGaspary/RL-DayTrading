import os

# Alpaca API Credentials
# Replace these with your actual keys from the Alpaca Dashboard (Paper Trading)
API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY_HERE")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Use paper trading URL

API_KEY2 = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY_HERE")
SECRET_KEY2 = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
BASE_URL2 = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Use paper trading URL

API_KEY3 = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY_HERE")
SECRET_KEY3 = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
BASE_URL3 = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Use paper trading URL

# Trading Settings
SYMBOL = "QQQ"         # NASDAQ 100 Proxy
TIMEFRAME = "5Min"     # 5-minute candles
TRADE_AMOUNT = 100     # Amount in USD to trade per signal (Small account challenge)
COMPOUND_MODE = True   # Reenabled Snowball Compounding
MAX_POSITIONS = 1      # Maximum concurrent positions
DAILY_TARGET_PCT = 0.02 # Stop for the day if up > 2% from daily open

# Risk Management
TAKE_PROFIT_PCT = 0.02   # 2.0% Take Profit
STOP_LOSS_PCT = 0.01     # 1.0% Stop Loss
TRAILING_STOP_PCT = 0.005 # 0.5% Trailing Stop (activates after some profit)

# Model Settings
MODEL_TYPE = "LSTM"            # Options: "RandomForest", "LSTM", "Transformer"
MODEL_PATH = "models/champion.pth" # Tournament Winner (LSTM 730d)
LOOKBACK_WINDOW = 60           # Number of past 5-min candles to feed into the model
MIN_CONFIDENCE = 0.55          # Minimum confidence required to trade (0.5 to 1.0)
PREDICTION_HORIZON = 1 # Predict next candle (or N candles ahead)

# Data Settings
DATA_START_DATE = "2025-11-20" # Last ~60 days for 5-min data (yfinance limit)
