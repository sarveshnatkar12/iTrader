from datetime import datetime, timedelta
import os
import logging
import json
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from trading_env import TradingEnv
import pandas as pd
import pandas_ta as ta
import sys
from .constants import STOCK_MAPPINGS

# Load environment variables
load_dotenv()

# Alpaca API credentials from .env file
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading URL

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def backtest(stock_name):
    SYMBOL = STOCK_MAPPINGS.get(stock_name)
    if not SYMBOL:
        raise ValueError(f"Stock name '{stock_name}' not recognized. Available options: {list(STOCK_MAPPINGS.keys())}")
    
    TIMEFRAME = '1H'
    
    # Set date range for fetching data 
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching data from {start_date_str} to {end_date_str} for {stock_name}...")

    # Fetch historical data
    data = api.get_bars(SYMBOL, TIMEFRAME, start=start_date_str, end=end_date_str, feed='iex').df

    if data.empty:
        raise ValueError("No data fetched from Alpaca. Check API credentials and data availability.")
    
    # Compute technical indicators
    data['RSI'] = ta.rsi(data['close'], length=14)
    data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    data['STD_20'] = data['close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (2 * data['STD_20'])
    data['Lower_Band'] = data['SMA_20'] - (2 * data['STD_20'])
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)
    data['VWAP'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'])

    # Stochastic Oscillator
    stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    data['Stochastic_K'] = stoch['STOCHk_14_3_3']
    data['Stochastic_D'] = stoch['STOCHd_14_3_3']

    # Add CCI and OBV
    data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=14)
    data['OBV'] = ta.obv(data['close'], data['volume'])

    # Rename MACD columns
    data.rename(columns={'MACD_12_26_9': 'MACD', 
                         'MACDh_12_26_9': 'MACD_hist', 
                         'MACDs_12_26_9': 'MACD_signal'}, inplace=True)

    # Drop NaN values
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError("Data contains too many NaN values after processing. Adjust indicator calculations.")

    # Verify model and environment files exist
    model_path = f"models/trading_bot_ppo_{stock_name}.zip"
    env_path = f"models/trading_env_normalize_{stock_name}.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"{env_path} not found. Ensure the environment normalization file exists.")

    # Create trading environment with updated logic
    env = TradingEnv(data)
    env = DummyVecEnv([lambda: env])

    # Load the trained model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

    # Load normalization settings
    env = VecNormalize.load(env_path, env)

    # Set up logging
    logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting trading bot simulation...")

    # Simulation
    obs = env.reset()
    done = [False]
    portfolio_values = []
    action_logs = []

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if done[0]:
            current_value = info[0]["net_worth"]
        else:
            current_value = env.get_attr("net_worth")[0]
            
        portfolio_values.append(current_value)
        
        action_log = {
            "action": int(action[0]),
            "reward": float(reward[0]),
            "net_worth": current_value
        }
        action_logs.append(action_log)
        
        logging.info(f"Action: {action}, Reward: {reward}, Net Worth: {current_value}")

    # Prepare results
    backtest_results = {
        "stock": stock_name,
        "initial_portfolio_value": portfolio_values[0],
        "final_portfolio_value": portfolio_values[-1],
        "total_return_percentage": ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100,
        "action_logs": action_logs
    }

    # Save results to a file
    results_file = f"backtest_results_{stock_name}.json"
    with open(results_file, 'w') as f:
        json.dump(backtest_results, f, indent=2)

    # Print and log final results
    logging.info(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
    print(json.dumps(backtest_results, indent=2))

    return backtest_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No stock name provided for backtesting.")
        sys.exit(1)
    
    stock_name = sys.argv[1]  # Get stock name from command-line arguments
    backtest(stock_name)