# This script is working as intended.

import ccxt
import pandas as pd
import time
from datetime import datetime

# -----------------------
# CONFIG
# -----------------------
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '15m'
train_file = 'data/btc_usdt_15m_2017_2023.csv'  # existing training data
test_file = 'data/btc_usdt_15m_test.csv'        # new test data

# Load existing training CSV
df_train = pd.read_csv(train_file)
last_train_timestamp = df_train['timestamp'].iloc[-1]

# Convert last timestamp to milliseconds
since = int(pd.to_datetime(last_train_timestamp).timestamp() * 1000) + 1
until = exchange.milliseconds()  # current time

all_data = []

print(f"Fetching new data from {pd.to_datetime(since, unit='ms')} to now...")

while since < until:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
        if len(ohlcv) == 0:
            break
        all_data += ohlcv
        since = ohlcv[-1][0] + 1
        last_time = pd.to_datetime(ohlcv[-1][0], unit='ms')
        print(f"Downloaded up to: {last_time}")
        time.sleep(exchange.rateLimit / 1000)
    except Exception as e:
        print("Error:", e)
        print("Waiting 5 seconds before retrying...")
        time.sleep(5)

# Save new data
if all_data:
    df_new = pd.DataFrame(all_data, columns=['timestamp','open','high','low','close','volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')

    # Append to training CSV
    df_combined = pd.concat([df_train, df_new], ignore_index=True)
    df_combined.to_csv(train_file, index=False)
    print(f"Training data updated! Total rows: {len(df_combined)}")

    # Save new data as separate test CSV
    df_new.to_csv(test_file, index=False)
    print(f"Test data saved to {test_file}, rows: {len(df_new)}")
else:
    print("No new data available.")
