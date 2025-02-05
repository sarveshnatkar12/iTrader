import os
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pandas_ta as ta
import logging
import warnings

from trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

logging.basicConfig(filename='trading_bot.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_indicators(data):
    data['RSI'] = ta.rsi(data['close'], length=14)
    data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    data.rename(columns={
        'MACD_12_26_9': 'MACD',
        'MACDh_12_26_9': 'MACD_hist',
        'MACDs_12_26_9': 'MACD_signal'
    }, inplace=True)
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['STD_20'] = data['close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (2 * data['STD_20'])
    data['Lower_Band'] = data['SMA_20'] - (2 * data['STD_20'])
    stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    if not stoch.empty:
        data['Stochastic_K'] = stoch['STOCHk_14_3_3']
        data['Stochastic_D'] = stoch['STOCHd_14_3_3']
    data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=14)
    data['OBV'] = ta.obv(data['close'], data['volume'])
    data.dropna(inplace=True)
    return data

def place_trade(api, symbol, side):
    try:
        # Check if the market is open
        clock = api.get_clock()
        if not clock.is_open:
            print("Market is closed. Waiting for the next open session.")
            logging.info("Market is closed. Skipping trade.")
            time.sleep(60)  # Wait 1 minute before checking again
            return

        # Check if there are any pending orders
        open_orders = api.list_orders(status='open')
        if open_orders:
            print("There are pending orders. Skipping trade.")
            logging.info("Skipping trade due to pending orders.")
            return

        print(f"Fetching account info for {symbol}...")
        account = api.get_account()
        cash_available = float(account.cash)
        print(f"Cash Available: {cash_available}")

        print(f"Fetching last trade price for {symbol}...")
        last_trade = api.get_latest_trade(symbol)   
        stock_price = float(last_trade.price)
        print(f"Stock price: {stock_price}")

        risk_percentage = 0.05  # 5% of available cash
        qty = int((cash_available * risk_percentage) / stock_price)

        print(f"Placing {side} order for {qty} shares of {symbol}...")

        if qty > 0:
            order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='day')
            logging.info(f"Trade placed: {side.upper()} {qty} shares of {symbol}")
            print(f"Trade successful: {order}")
        else:
            print("Not enough funds to buy MSFT.")
            logging.info("Not enough funds to buy MSFT.")
    except Exception as e:
        logging.error(f"Failed to place order for {symbol}: {e}")
        print(f"Error placing order: {e}")



def analyze_trade(api, symbol):
    try:
        position = api.get_position(symbol)
        current_price = float(api.get_last_trade(symbol).price)
        avg_entry_price = float(position.avg_entry_price)
        qty = int(position.qty)
        profit_loss = (current_price - avg_entry_price) * qty
        logging.info(f"Current Price: {current_price}, Avg Entry Price: {avg_entry_price}, P/L: {profit_loss}")
    except Exception as e:
        logging.error(f"No active position for {symbol} or error fetching position: {e}")

def main():
    load_dotenv()
    api = tradeapi.REST(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'), 'https://paper-api.alpaca.markets', api_version='v2')
    print("Loading model...")
    model = PPO.load("trading_bot_ppo")
    print("Model loaded.")

    env_norm = VecNormalize.load("trading_env_normalize.pkl", DummyVecEnv([lambda: TradingEnv(pd.DataFrame())]))
    print("Entering while loop...")
    while True:
        logging.debug("Checking conditions and placing trades...")
        place_trade(api, 'MSFT', 'buy')
        analyze_trade(api, 'MSFT')
        time.sleep(60)

if __name__ == '__main__':
    main()
