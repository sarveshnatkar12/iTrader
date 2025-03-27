from fastapi import FastAPI, HTTPException
import subprocess
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/train")
def train_model(stock: str):
    """Train the RL model on a specified stock."""
    subprocess.Popen([sys.executable, "train_model.py", stock], env=os.environ.copy())
    return {"message": f"Training started for {stock}"}

@app.get("/backtest")
async def backtest_model(stock: str):
    """Backtest the trained model on historical data and return the result."""
    try:
        # Run backtest and capture output
        result = subprocess.run(
            [sys.executable, "Backtest_bot.py", stock],
            capture_output=True, 
            text=True, 
            check=False  # Remove check=True to capture non-zero exit codes
        )
        
        # Check if the process exited with an error
        if result.returncode != 0:
            # If there's stderr output, use that
            error_message = result.stderr.strip() or result.stdout.strip()
            return {
                "error": "Backtesting failed", 
                "details": error_message,
                "return_code": result.returncode
            }
        
        # Try to parse the JSON output
        try:
            backtest_results = json.loads(result.stdout)
            return {
                "message": f"Backtesting completed for {stock}", 
                "data": backtest_results
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw output
            return {
                "message": f"Backtesting completed for {stock}", 
                "data": result.stdout
            }
    
    except Exception as e:
        # Catch any unexpected errors
        return {
            "error": "Backtesting failed", 
            "details": str(e)
        }

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

@app.get("/exit")
def exit():
    """Exit All Positions Forcefully"""
    subprocess.Popen(
        [sys.executable, "interactive_bot.py", "force_exit"],
        env=os.environ.copy()
    )
    return {"message": "Force exit command sent successfully!"}