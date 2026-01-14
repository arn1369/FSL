import torch
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from torch.utils.data import DataLoader
from utils import FSLPredictor, UniversalDataset

# Configuration
SAVE_PATH = "./saves/test_results.pkl"
MODEL_PATH = "./saves/fsl.pth"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "JPM", "BAC", "XOM", "CVX", "JNJ", "PFE", "PG", "KO", "MCD", "HD", "DIS"]

def run_backtest():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Download Data (Out-of-Sample)
    print("Downloading Out-of-Sample data (2019-2024)...")
    data = yf.download(TICKERS, start="2019-01-01", end="2024-12-31", auto_adjust=True)
    
    # Handling multi-index (new version of yfinance...)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            else:
                data = data.xs('Close', axis=1, level=0)
        except:
            data = data
    else:
        data = data['Close'] if 'Close' in data.columns else data

    # Cleaning data
    data = data.ffill().bfill()
    # Keep tickers order
    data = data[TICKERS]
    returns = data.pct_change().dropna()
    
    # 2. Prepare Dataset
    test_ds = UniversalDataset(returns, window_size=20, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = FSLPredictor(n_assets=len(TICKERS), window_size=20).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("Error: Model not found.")
        return

    model.eval()
    
    # 4. Inference Loop
    results = []
    test_dates = returns.index[20:] # Skip the initial 20-day window
    
    print("Running Inference...")
    with torch.no_grad():
        for i, (inputs, target) in enumerate(test_loader):
            inputs = [x.to(device).float() for x in inputs]
            preds, res = model(inputs)
            
            # Store raw results
            results.append({
                'date': test_dates[i],
                'pred': preds.cpu().numpy().flatten(),
                'actual': target.cpu().numpy().flatten(),
                'h1': res['h1_score'].item(),
                'h1_contributions': res['individual_h1'].cpu().numpy().flatten()
            })
            
            if i % 100 == 0:
                print(f"Step {i}/{len(test_loader)}")

    # 5. Save data for graphs.py
    # Also save 'returns' because graphs.py needs it to compare to the market
    package = {
        'results': results,
        'returns_df': returns,
        'tickers': TICKERS
    }
    
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(package, f)
        
    print(f"Done. Results saved to '{SAVE_PATH}'.")

if __name__ == "__main__":
    run_backtest()