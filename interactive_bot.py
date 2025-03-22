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
import sys

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

logging.basicConfig(filename='trading_bot.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Stock mapping: full name to symbol
stock_mapping = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Nvidia": "NVDA",
    "Google": "GOOGL",
    "Microsoft": "MSFT",
    "Netflix": "NFLX"
}

def place_trade(api, stock_name, side):
    symbol = stock_mapping.get(stock_name)
    if not symbol:
        print(f"Stock name '{stock_name}' not recognized.")
        logging.error(f"Stock name '{stock_name}' not recognized.")
        return
    
    try:
        clock = api.get_clock()
        if not clock.is_open:
            print("Market is closed. Waiting for the next open session.")
            logging.info("Market is closed. Skipping trade.")
            time.sleep(60)
            return

        open_orders = api.list_orders(status='open')
        if open_orders:
            print("There are pending orders. Skipping trade.")
            logging.info("Skipping trade due to pending orders.")
            return

        print(f"Fetching account info for {stock_name} ({symbol})...")
        account = api.get_account()
        cash_available = float(account.cash)
        print(f"Cash Available: {cash_available}")

        print(f"Fetching last trade price for {stock_name} ({symbol})...")
        last_trade = api.get_latest_trade(symbol)
        stock_price = float(last_trade.price)
        print(f"Stock price: {stock_price}")

        risk_percentage = 0.05
        qty = int((cash_available * risk_percentage) / stock_price)

        print(f"Placing {side} order for {qty} shares of {stock_name} ({symbol})...")

        if qty > 0:
            order = api.submit_order(symbol=symbol, qty=qty, side=side, type='market', time_in_force='day')
            logging.info(f"Trade placed: {side.upper()} {qty} shares of {stock_name} ({symbol})")
            print(f"Trade successful: {order}")
        else:
            print(f"Not enough funds to buy {stock_name}.")
            logging.info(f"Not enough funds to buy {stock_name}.")
    except Exception as e:
        logging.error(f"Failed to place order for {stock_name} ({symbol}): {e}")
        print(f"Error placing order: {e}")

def analyze_trade(api, stock_name):
    symbol = stock_mapping.get(stock_name)
    if not symbol:
        print(f"Stock name '{stock_name}' not recognized.")
        logging.error(f"Stock name '{stock_name}' not recognized.")
        return
    
    try:
        position = api.get_position(symbol)
        current_price = float(api.get_last_trade(symbol).price)
        avg_entry_price = float(position.avg_entry_price)
        qty = int(position.qty)
        profit_loss = (current_price - avg_entry_price) * qty
        logging.info(f"{stock_name} ({symbol}) - Current Price: {current_price}, Avg Entry Price: {avg_entry_price}, P/L: {profit_loss}")
    except Exception as e:
        logging.error(f"No active position for {stock_name} ({symbol}) or error fetching position: {e}")

def force_exit(api):
    try:
        print("exiting all positions...")
        api.close_all_positions(cancel_orders=True)
    except Exception as e:
        print(f"Error exiting positions : {e}")

def main(stock_name):
    load_dotenv()
    api = tradeapi.REST(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_API_SECRET'), 'https://paper-api.alpaca.markets', api_version='v2')
    stock_name = " "
    
    print("Loading model...")
    model_path = f"models/trading_bot_ppo_{stock_name}.zip"
    env_path = f"models/trading_env_normalize_{stock_name}.pkl"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    if not os.path.exists(env_path):
        print(f"Environment file not found: {env_path}")
        return
    
    model = PPO.load(model_path)
    print("Model loaded.")
    env_norm = VecNormalize.load(env_path, DummyVecEnv([lambda: TradingEnv(pd.DataFrame())]))
    
    print("Entering while loop...")
    while True:
        logging.debug("Checking conditions and placing trades...")
        place_trade(api, stock_name, 'buy')
        analyze_trade(api, stock_name)
        time.sleep(60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No stock name provided for backtesting.")
        sys.exit(1)
    
    # New Condition to address force_exit
    if sys.argv[1] == "force_exit":
        load_dotenv()
        api = tradeapi.REST(os.getenv('ALPACA_API_KEY'), 
                          os.getenv('ALPACA_API_SECRET'), 
                          'https://paper-api.alpaca.markets',
                          api_version='v2')
        force_exit(api)
        print("Force exit completed")
    else:
        stock_name = sys.argv[1]
        main(stock_name)
