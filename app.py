from fastapi import FastAPI, BackgroundTasks
import subprocess
import os
import sys
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/train")
def train_model(stock: str):
    """Train the RL model on a specified stock."""
    subprocess.Popen([sys.executable, "train_model.py", [stock]], env=os.environ.copy())
    return {"message": f"Training started for {stock}"}

@app.get("/backtest")
def backtest_model(stock: str):
    """Backtest the trained model on historical data."""
    subprocess.Popen([sys.executable, "Backtest_bot.py", stock], env=os.environ.copy())
    return {"message": f"Backtesting started for {stock}"}

@app.get("/trade")
def start_trading(stock: str):
    """Start live trading using the trained model."""
    subprocess.Popen([sys.executable, "interactive_bot.py", stock], env=os.environ.copy())
    return {"message": f"Trading started for {stock}"}

@app.get("/performance")
def get_performance():
    """Fetch trading bot performance metrics."""
    # This should be replaced with actual logic to fetch live data
    return {"net_worth": 10500, "profit_loss": 500}

@app.get("/stocks")
def get_available_stocks():
    """Get the list of stocks available for training and trading."""
    stock_mapping = {"Apple": "AAPL", "Tesla": "TSLA", "Nvidia": "NVDA", "Google": "GOOGL", "Microsoft": "MSFT", "Netflix": "NFLX"}
    return {"available_stocks": list(stock_mapping.keys())}
