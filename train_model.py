import pandas as pd
import pandas_ta as ta
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from trading_env import TradingEnv
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Alpaca API credentials
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Stock mapping: symbol to full name
stock_mapping = {
    
   "NFLX": "Netflix"
}

# Training parameters
total_timesteps = 1000000

for stock_symbol, stock_name in stock_mapping.items():
    print(f"\n🔄 Training on {stock_name} ({stock_symbol})...")
    
    while True:  # Keep retrying until data is fetched
        try:
            print(f"📡 Fetching data for {stock_symbol} from Alpaca...")
            data = api.get_bars(stock_symbol, '1H', start='2020-01-01', end='2025-02-01').df
            if not data.empty:
                print(f"✅ Received {len(data)} bars for {stock_symbol}")
                break
            else:
                print(f"⚠️ No data fetched for {stock_symbol}. Retrying in 10 seconds...")
                time.sleep(10)
        except tradeapi.rest.APIError as e:
            print(f"⚠️ API Error: {e}. Retrying in 10 seconds...")
            time.sleep(10)
    
    # Compute technical indicators
    print("📊 Calculating technical indicators...")
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
    if stoch is not None and not stoch.empty:
        data['Stochastic_K'] = stoch['STOCHk_14_3_3']
        data['Stochastic_D'] = stoch['STOCHd_14_3_3']
    else:
        print(f"⚠️ Stochastic indicator failed for {stock_symbol}. Skipping training.")
        continue
    
    # Rename MACD columns
    data.rename(columns={'MACD_12_26_9': 'MACD', 
                         'MACDh_12_26_9': 'MACD_hist', 
                         'MACDs_12_26_9': 'MACD_signal'}, inplace=True)
    
    # Add CCI and OBV
    data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=14)
    data['OBV'] = ta.obv(data['close'], data['volume'])
    
    # Drop NaN values after all indicators are added
    data.dropna(inplace=True)
    
    print(f"✅ Data preprocessing complete for {stock_symbol}. Starting training.")
    
    # Create environment
    env = TradingEnv(data)
    env = make_vec_env(lambda: env, n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Initialize PPO model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=f"./trading_tensorboard/{stock_name}/",
        learning_rate=1e-04,
        n_steps=4096,
        batch_size=128,
        gamma=0.98,
        gae_lambda=0.9,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5
    )
    
    # Train the model
    print(f"🚀 Training PPO model for {stock_symbol}...")
    model.learn(total_timesteps=total_timesteps)
    print(f"✅ Training complete for {stock_symbol}.")
    
    # Save trained model
    model.save(f"models/trading_bot_ppo_{stock_name}")
    env.save(f"models/trading_env_normalize_{stock_name}.pkl")
    
    print(f"💾 Model saved: models/trading_bot_ppo_{stock_name}")
    
print("\n🎉 All training complete. Models saved using company names.")
