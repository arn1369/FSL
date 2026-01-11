
"""
FSL Implementation
@author : Arnaud Ullens
@created :  8th dec.2025
last modification : 11th jan.2025
"""

import torch
import numpy as np
import random
import pickle
from torch.utils.data import DataLoader
from collections import Counter
import yfinance as yf

from utils import MultiRegimeHMM, RollingWindowDataset, RollingZScore, FSLPredictor

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_market_data(start_date="2019-01-01", end_date="2025-01-01"):
    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", 
        "JPM", "BAC", 
        "XOM", "CVX",
        "JNJ", "PFE",
        "PG", "KO", "MCD", "HD",
        "DIS"
    ]
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close']
    
    valid_tickers = [t for t in tickers if t in data.columns]
    
    if len(valid_tickers) < len(tickers):
        print(f"Warning: {len(tickers) - len(valid_tickers)} tickers failed to download.")
    
    data = data[valid_tickers]
    data = data.ffill()
    returns = np.log(data / data.shift(1))
    returns = returns.dropna(how='all').fillna(0.0)
    return returns, valid_tickers

def compute_metrics(pnl_series, returns_series, exposure_history):
    pnl_arr = np.array(pnl_series)
    
    sharpe = np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-6) * np.sqrt(252)
    
    cum_pnl = np.cumsum(pnl_arr)
    wealth = 1.0 + cum_pnl
    running_max = np.maximum.accumulate(wealth)
    drawdown_pct = (wealth - running_max) / running_max * 100
    max_dd = np.min(drawdown_pct)
    
    annual_return = np.mean(pnl_arr) * 252
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    
    wins = np.sum(pnl_arr > 0)
    total_trades = len(pnl_arr)
    win_rate = wins / total_trades * 100
    
    avg_win = np.mean(pnl_arr[pnl_arr > 0]) if np.any(pnl_arr > 0) else 0
    avg_loss = np.mean(pnl_arr[pnl_arr < 0]) if np.any(pnl_arr < 0) else 0
    
    gross_profit = np.sum(pnl_arr[pnl_arr > 0])
    gross_loss = abs(np.sum(pnl_arr[pnl_arr < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    avg_exposure = np.mean(exposure_history) * 100
    
    return {
        'sharpe': sharpe, 'max_dd': max_dd, 'calmar': calmar,
        'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss,
        'profit_factor': profit_factor, 'avg_exposure': avg_exposure,
        'total_return': cum_pnl[-1] * 100
    }

def run_backtest():
    set_seed(1)
    
    # Load Data
    df_returns, ticker_names = get_market_data(start_date="2019-01-01", end_date="2025-11-30")
    
    test_start = "2021-06-01" # warm-up
    test_end = "2025-06-01"
    
    # Pre-calculate Market Trend & Volatility
    market_index = df_returns.mean(axis=1) # Equal weight index
    market_cum = market_index.cumsum()
    market_trend = market_cum.rolling(window=100).mean() # 100-day MA
    market_vol = market_index.rolling(window=20).std()   # 20-day Vol
    
    test_df = df_returns[(df_returns.index >= test_start) & (df_returns.index < test_end)]
    
    # Align pre-calculated metrics with test set
    test_trend = market_trend[test_df.index]
    test_price = market_cum[test_df.index]
    test_vol = market_vol[test_df.index]
    
    print(f"Test Set: {len(test_df)} days")
    
    WINDOW_SIZE = 20
    test_ds = RollingWindowDataset(test_df, window_size=WINDOW_SIZE)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = FSLPredictor(n_assets=16, window_size=WINDOW_SIZE)
    model.to(device)
    try:
        model.load_state_dict(torch.load("./saves/final_fsl.pth", map_location=device))
    except Exception as e:
        raise Exception(f"Error loading model : {e}")
    model.eval()
        
    # Initialization HMM & Z-Score
    hmm = MultiRegimeHMM(inertia=0.90)
    hmm.means = np.array([3.0, 1.5, 0.0, -1.0]) #NOTE: BAD. Needs to be changed
    hmm.vars = np.array([1.0, 0.8, 0.5, 0.5])
    
    # Calibrating HMM on first 100 days
    
    calibration_h1 = []
    # Small pass to calibrate
    limit = min(100, len(test_loader))
    for i, (inputs, _) in enumerate(test_loader):
        if i >= limit: break
        inputs = [x.to(device).float() for x in inputs]
        _, res = model(inputs)
        calibration_h1.append(res['h1_score'].item())
    
    mean_h1 = np.mean(calibration_h1)
    std_h1 = np.std(calibration_h1)
    print(f"H1 Stats: Mean={mean_h1:.4f}, Std={std_h1:.4f}")
    
    # Reset loader for the actual test
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    z_scorer = RollingZScore(window=60)
    
    # Adaptive Params: (Threshold, Conviction, Leverage)
    ADAPTIVE_PARAMS = {
        3: (0.2, 0.002, 1.5), # BULL
        2: (0.5, 0.008, 1.0), # NORMAL
        1: (1.2, 0.015, 0.5), # VOLATIL
        0: (2.0, 0.020, 0.0)  # CRISIS
    }
    
    # Buffers
    pnl_strategy = []
    pnl_market = []
    exposure_history = []
    regime_history = []
    prob_calm_history = []
    
    print("\nStarting Backtest...")
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            # Prepare Data
            inputs = [x.to(device).float() for x in inputs]
            targets_np = targets.numpy().flatten()
            
            # Inference
            preds, res = model(inputs)
            
            # Process Results
            h1_val = res['h1_score'].item() if isinstance(res['h1_score'], torch.Tensor) else res['h1_score']
            pred_val = preds.detach().cpu().numpy().flatten()
            
            # Get external market context for this step
            current_vol = test_vol.iloc[i + WINDOW_SIZE]
            current_trend = test_trend.iloc[i + WINDOW_SIZE]
            current_lvl = test_price.iloc[i + WINDOW_SIZE]
            
            # Composite Signal: H1 (Structure) + Volatility (Risk)
            # If Volatility is high, we push the signal up (towards Crisis state)
            # We multiply H1 by (1 + Vol) to penalize high-volatility structures
            raw_signal = h1_val * 100.0 * (1.0 + (current_vol * 100))
            
            h1_zscore = z_scorer.update(raw_signal)
            state_probs = hmm.update(h1_zscore)
            hmm.smooth_prob = (0.2 * state_probs) + (0.8 * hmm.smooth_prob)
            dominant_regime = np.argmax(hmm.smooth_prob)
            
            # If market is below 100-day MA, cap regime at "Volatile" (max 0.5x leverage)
            # This prevents holding 1.0x leverage during a bear market.
            if current_lvl < current_trend:
                dominant_regime = min(dominant_regime, 1) # Force Volatile or Crisis
            
            # Get Params
            current_cluster_th, current_convic_th, current_leverage = ADAPTIVE_PARAMS[dominant_regime]
            
            # Strategy Logic (Alpha)
            alpha_strength = np.std(pred_val)
            pred_mean = np.mean(pred_val)
            z_scores_cross = (pred_val - pred_mean) / (np.std(pred_val) + 1e-6)
            
            weights = np.zeros(16)
            regime_label = ["Crisis", "Volatil", "Normal", "Bull"][dominant_regime]
            
            # Simplified Trading Logic
            if dominant_regime == 3: # BULL
                long_mask = z_scores_cross > current_cluster_th
                if np.sum(long_mask) > 0:
                    weights[long_mask] = 1.0 / np.sum(long_mask)
                current_exposure = current_leverage 

            elif dominant_regime == 2: # NORMAL
                # Only go aggressive if alpha is strong AND prediction is positive
                if alpha_strength > current_convic_th and pred_mean > 0:
                    # CONVICTION MODE 
                    # invest on best ones
                    weights = np.exp(pred_val * 2.0) / np.sum(np.exp(pred_val * 2.0))
                    current_exposure = 1.0 
                    regime_label += " (Conviction)"
                else:
                    long_mask = z_scores_cross > 0.5 # best assets
                    
                    short_mask = z_scores_cross < -0.5 # bad  assets
                    
                    if np.sum(long_mask) > 0: 
                        weights[long_mask] = 0.5 / np.sum(long_mask) # 50% on long
                    
                    # 50% in cash if no shorts (will be improved later with options)
                    
                    current_exposure = 1.0
                    regime_label += " (Neutral Long)"

            elif dominant_regime == 1: # VOLATIL
                # cash (again, will be improved later with options)
                weights = np.zeros(16)
                current_exposure = 0.0
            
            else: # CRISIS
                current_exposure = 0.0 # Cash (for now, to be improved)
            
            # Execution
            daily_ret = np.sum(weights * targets_np) * current_exposure
            
            pnl_strategy.append(daily_ret)
            pnl_market.append(np.mean(targets_np))
            exposure_history.append(current_exposure)
            regime_history.append(regime_label)
            prob_calm_history.append(hmm.smooth_prob[2] + hmm.smooth_prob[3])
    
    # Metrics and Saving
    print("Computing final metrics...")
    metrics = compute_metrics(pnl_strategy, test_df.iloc[WINDOW_SIZE:].values, exposure_history)
    
    print(f"\nPERFORMANCE:")
    print(f"  FSL+HMM Strategy: Sharpe={metrics['sharpe']:.2f}, Return={metrics['total_return']:.2f}%")
    print(f"  Max Drawdown: {metrics['max_dd']:.2f}%")
    
    regime_counts = Counter(regime_history)
    print(f"\nREGIME DISTRIBUTION:")
    for regime, count in regime_counts.most_common():
        pct = count / len(regime_history) * 100
        print(f"  • {regime:20s}  : {pct:5.1f}%")
    
    flat_pnl = np.array(pnl_strategy)
    flat_market = np.array(pnl_market)
    
    # Correlation when the model is exposed (>0.1)
    mask = np.array(exposure_history) > 0.1
    if np.sum(mask) > 10:
        active_correlation = np.corrcoef(flat_pnl[mask], flat_market[mask])[0, 1]
        print(f"\nDIAGNOSTIC:")
        print(f"  Correlation with Market when Active: {active_correlation:.2f}")
        
    results_data = {
        "dates": test_df.index[WINDOW_SIZE:],
        "pnl_strategy": pnl_strategy,
        "pnl_market": pnl_market,
        "prob_calm": prob_calm_history,
        "exposure": exposure_history,
        "regime_history": regime_history,
        "metrics": metrics
    }
    
    with open("./saves/backtest_results.pkl", "wb") as f:
        pickle.dump(results_data, f)
    
    print("Done.")

if __name__ == "__main__":
    run_backtest()