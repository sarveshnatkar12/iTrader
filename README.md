# 📈 iTrader – Reinforcement Learning Based Stock Trading Bot

iTrader is an automated trading bot that uses **Proximal Policy Optimization (PPO)** — a reinforcement learning algorithm — to make intelligent buy/hold/sell decisions based on real stock market data and technical indicators.

It supports end-to-end automation: from data collection and feature engineering to training, backtesting, live paper trading, and visualization.

---

## 🚀 Features

- ✅ **Historical Data Fetching** from Alpaca API (1-hour candles)
- 📊 **15+ Technical Indicators**: RSI, MACD, SMA, Bollinger Bands, OBV, and more
- 🧠 **Custom OpenAI Gym Environment** for trading logic
- 🤖 **Stable-Baselines3 PPO** RL Agent
- 📈 **Backtesting Mode** to simulate on recent market data
- 💵 **Live Trading** using Alpaca Paper Trading API
- 🌐 **FastAPI Interface** to trigger `/train`, `/backtest`, `/trade`, `/exit`
- 📊 **Visualizations** for profits, drawdowns, win rates
- 🔗 **Ngrok** support for local API exposure

---

## 🧱 Project Structure

I_Trader/
├── app.py # FastAPI server for training, trading, and backtesting
├── train_model.py # Fetches data, computes indicators, trains PPO model
├── trading_env.py # Custom Gym environment for trading
├── Backtest_bot.py # Backtests trained model on unseen data
├── interactive_bot.py # Paper trading loop with Alpaca API
├── constants.py # Stock name to ticker symbol mapping
├── visualization.py # Charts and performance metrics
├── port_forward.py # Ngrok tunnel for dev access
├── models/ # Saved models (.zip) and envs (.pkl)
├── .env # Alpaca API keys
├── requirements.txt


---

## 📊 Technical Indicators Used

- RSI (Relative Strength Index)
- MACD (and Histogram)
- SMA 10/20/50
- Bollinger Bands
- Stochastic Oscillator (%K, %D)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- CCI (Commodity Channel Index)
- OBV (On Balance Volume)

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Main programming language |
| 🧠 Stable-Baselines3 (PPO) | Reinforcement Learning |
| 📉 pandas-ta | Technical indicator computation |
| 🧪 OpenAI Gym | Custom trading environment |
| 💹 Alpaca API | Live and historical stock market data |
| 🚀 FastAPI | REST API to trigger pipelines |
| 📊 Matplotlib / Seaborn | Visual performance reporting |
| 🔗 Ngrok | Public tunnel for local API |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/iTrader.git
cd I_Trader

2. Install Dependencies
pip install -r requirements.txt

3. Add Alpaca API Credentials
Create a .env file in the root directory:
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here

4. Train a Model
python train_model.py Google

5. Backtest the Model
python Backtest_bot.py Google

6. Start Live Trading (Paper)
python interactive_bot.py Google

🌐 API Endpoints (via FastAPI)
| Endpoint                 | Description                           |
| ------------------------ | ------------------------------------- |
| `/train?stock=Google`    | Train PPO model on specified stock    |
| `/backtest?stock=Google` | Run backtest and return metrics       |
| `/trade?stock=Google`    | Start live paper trading              |
| `/exit`                  | Forcefully close all positions        |
| `/stocks`                | Get list of supported stocks          |
| `/performance`           | (Stub) Return current portfolio stats |

🧠 Key Concepts
Reinforcement learning agent learns from rewards based on net worth changes.

Trades are restricted by SMA-based rules to prevent overtrading.

Normalized observations and rewards improve training convergence.

Modular scripts allow you to retrain, test, deploy, or trade independently.

🛡 Disclaimer
This project uses paper trading only and is intended for educational purposes. It does not offer financial advice or guarantee profitable trading.

👨‍💻 Author
Your Name
ML Engineer | MLOps | Reinforcement Learning
LinkedIn • Portfolio
