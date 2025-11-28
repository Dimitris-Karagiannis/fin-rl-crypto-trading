# This script now allows manual input for train and test start dates.

import ccxt
import pandas as pd
import time

# -----------------------
# USER INPUT
# -----------------------
train_start = "2020-09-14"     #(YYYY-MM-DD HH:MM:SS)
test_start = "2023-12-31"

# Convert to milliseconds
since_train = int(pd.to_datetime(train_start).timestamp() * 1000)
since_test = int(pd.to_datetime(test_start).timestamp() * 1000)

# -----------------------
# CONFIG
# -----------------------
exchange = ccxt.binance()
symbol = 'SOL/USDT'
timeframe = '15m'
train_file = 'data/sol_usdt_15m_train.csv'
test_file = 'data/sol_usdt_15m_test.csv'

# -----------------------
# FETCH TRAINING DATA
# -----------------------
all_train = []
until_train = since_test  # stop train data at test start

print(f"\n📥 Fetching TRAINING data from {train_start} to {test_start}...\n")

while since_train < until_train:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_train, limit=500)
        if not ohlcv:
            break
        all_train += ohlcv
        since_train = ohlcv[-1][0] + 1
        print("Downloaded train up to:", pd.to_datetime(ohlcv[-1][0], unit='ms'))
        time.sleep(exchange.rateLimit / 1000)
    except Exception as e:
        print("Error:", e, "| Waiting 5 seconds...")
        time.sleep(5)

if all_train:
    df_train = pd.DataFrame(all_train, columns=['timestamp','open','high','low','close','volume'])
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], unit='ms')
    df_train.to_csv(train_file, index=False)
    print(f"\n✅ TRAINING data saved! Rows: {len(df_train)} → {train_file}")
else:
    print("\n❌ No training data downloaded.")

# -----------------------
# FETCH TEST DATA
# -----------------------
all_test = []
since_test_loop = since_test
until = exchange.milliseconds()

print(f"\n📥 Fetching TEST data from {test_start} to now...\n")

while since_test_loop < until:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_test_loop, limit=500)
        if not ohlcv:
            break
        all_test += ohlcv
        since_test_loop = ohlcv[-1][0] + 1
        print("Downloaded test up to:", pd.to_datetime(ohlcv[-1][0], unit='ms'))
        time.sleep(exchange.rateLimit / 1000)
    except Exception as e:
        print("Error:", e, "| Waiting 5 seconds...")
        time.sleep(5)

if all_test:
    df_test = pd.DataFrame(all_test, columns=['timestamp','open','high','low','close','volume'])
    df_test['timestamp'] = pd.to_datetime(df_test['timestamp'], unit='ms')
    df_test.to_csv(test_file, index=False)
    print(f"\n✅ TEST data saved! Rows: {len(df_test)} → {test_file}")
else:
    print("\n❌ No test data downloaded.")

print("\n🎯 Done!")
