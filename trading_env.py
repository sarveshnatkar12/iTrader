import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(data) - 1
        self.net_worth_history = [initial_balance]
        self.transaction_fee = transaction_fee  # Transaction fee

        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: 15 features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(15,), dtype=np.float32
        )

    def seed(self, seed=None):  
        np.random.seed(seed)

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        self.net_worth_history = [self.initial_balance]
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.data.iloc[self.current_step]['open'] / 1000,
            self.data.iloc[self.current_step]['high'] / 1000,
            self.data.iloc[self.current_step]['low'] / 1000,
            self.data.iloc[self.current_step]['close'] / 1000,
            self.data.iloc[self.current_step]['RSI'] / 100,
            self.data.iloc[self.current_step]['MACD'],
            self.data.iloc[self.current_step]['MACD_hist'],
            self.data.iloc[self.current_step]['Stochastic_K'],
            self.data.iloc[self.current_step]['Stochastic_D'],
            self.data.iloc[self.current_step]['Upper_Band'] / 1000,
            self.data.iloc[self.current_step]['Lower_Band'] / 1000,
            self.shares_held / 1000,
            self.balance / self.initial_balance,
            self.data.iloc[self.current_step]['CCI'] / 1000,
            self.data.iloc[self.current_step]['OBV'] / 1000000
        ])
        return obs.astype(np.float32)

    def step(self, action):
        # Get the current price
        current_price = self.data.iloc[self.current_step]['close']

        # Increase the step counter first
        self.current_step += 1

        # Retrieve the SMA_10 from the current step
        sma_10 = self.data.iloc[self.current_step]['SMA_10'] if 'SMA_10' in self.data.columns else current_price

        # Execute the action with added risk controls:
        if action == 1:  # Buy
            if current_price < sma_10:
                invest_amount = self.balance * 0.5  # Invest 50% of available cash
                shares_to_buy = invest_amount // current_price
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                    if cost <= self.balance:
                        self.balance -= cost
                        self.shares_held += shares_to_buy
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Sell if price is above SMA_10 or if profit is above 2%
                if current_price > sma_10 or (self.net_worth / self.initial_balance - 1) >= 0.02:  # 2% profit
                    shares_to_sell = int(self.shares_held * 0.5)
                    if shares_to_sell < 1 and self.shares_held > 0:
                        shares_to_sell = self.shares_held
                    revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
                    self.balance += revenue
                    self.shares_held -= shares_to_sell
        # If action == 0 (Hold), do nothing

        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        reward = (self.net_worth - self.net_worth_history[-1]) / self.net_worth_history[-1]
        self.net_worth_history.append(self.net_worth)

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        # Include the current net worth in the info dict
        info = {"net_worth": self.net_worth}

        return self._next_observation(), reward, done, info

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')

    def get_portfolio_value(self):
        return self.net_worth
