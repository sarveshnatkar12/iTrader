import pandas as pd
import pandas_ta as ta
import alpaca_trade_api as tradeapi
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from trading_env import TradingEnv
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Alpaca API credentials from .env file
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading for testing

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Get historical data for stock
data = api.get_bars('MSFT', '1H', start='2023-01-01', end='2025-02-01').df

# Compute technical indicators
data['RSI'] = ta.rsi(data['close'], length=14)
data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
data['SMA_10'] = data['close'].rolling(window=10).mean()  # For decision making in env
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

# Rename MACD columns
data.rename(columns={'MACD_12_26_9': 'MACD', 
                     'MACDh_12_26_9': 'MACD_hist', 
                     'MACDs_12_26_9': 'MACD_signal'}, inplace=True)

# Add CCI and OBV
data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=14)
data['OBV'] = ta.obv(data['close'], data['volume'])

# Drop NaN values after all indicators are added
data.dropna(inplace=True)

# Verify required columns exist
required_columns = ['RSI', 'MACD', 'MACD_hist', 'Stochastic_K', 'Stochastic_D',
                    'Upper_Band', 'Lower_Band', 'CCI', 'OBV', 'SMA_10']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"{col} column is missing from the dataset after processing")

# Create the environment
env = TradingEnv(data)
env = make_vec_env(lambda: env, n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Initialize the PPO agent
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    tensorboard_log="./trading_tensorboard/",
    learning_rate=1e-04,
    n_steps=4096,
    batch_size=128,
    gamma=0.98,
    gae_lambda=0.9,
    clip_range=0.2,
    ent_coef=0.01,
    max_grad_norm=0.5
)

# Train the agent (increase timesteps as needed)
model.learn(total_timesteps=500000)

# Save the model and normalization statistics
model.save("trading_bot_ppo")
env.save("trading_env_normalize.pkl")

print("Training complete. Model and environment saved.")
