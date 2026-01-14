import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

from fsl import HierarchicalFSL, TopologicalTripletLoss
from utils import FSLPredictor, UniversalDataset

def get_universal_data():
    """
    Downloads and cleans S&P 500 data.
    Returns RAW RETURNS.
    Global normalization is avoided here to prevent Look-Ahead Bias.
    """
    
    print("Downloading data...")
    
    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", 
        "JPM", "BAC", 
        "XOM", "CVX",
        "JNJ", "PFE",
        "PG", "KO", "MCD", "HD",
        "DIS"
    ]
    
    data = yf.download(tickers, start="2010-01-01", end="2018-12-31", progress=True, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        try:
            if 'Close' in data.columns.get_level_values(0):
                df = data['Close']
            else:
                df = data.xs('Close', axis=1, level=0)
        except:
            print("This isn't supposed to happen.")
            df = data['Adj Close']
    else:
        df = data['Close'] if 'Close' in data.columns else data

    # Cleaning Data
    df = df.dropna(axis=1, thresh=int(len(df)*0.9)) # keep tickers with at least 90% data
    df = df.ffill().bfill()
    
    returns = df.pct_change().dropna()
    
    print(f"Valid tickers retrieved: {len(returns.columns)}")
    print(f"Data ready. Shape: {returns.shape}")
    
    return returns, returns.columns.tolist()



def correlation_loss(pred, target):
    """
    Maximizes Pearson correlation between prediction and target.
    Encourages the model to get the relative ranking of assets right.
    """
    pred_n = (pred - pred.mean(dim=1, keepdim=True)) / (pred.std(dim=1, keepdim=True) + 1e-8)
    target_n = (target - target.mean(dim=1, keepdim=True)) / (target.std(dim=1, keepdim=True) + 1e-8)
    corr = (pred_n * target_n).mean(dim=1)
    return 1 - corr.mean()

def sign_loss(pred, target):
    """
    Penalizes directional errors.
    Uses a soft differentiable approximation (tanh) to penalize wrong signs.
    """
    # Scale by 5.0 to make the tanh slope steeper around 0
    pred_sign = torch.tanh(pred * 5.0)
    target_sign = torch.sign(target)
    
    # We want pred_sign and target_sign to match (product close to 1)
    return 1 - torch.mean(pred_sign * target_sign)

def train_model():
    # 1. Loading data
    try:
        df_univ, tickers = get_universal_data() 
    except Exception as e:
        print(f"Error while loading data : {e}")
        return
    
    # Saves directory
    if not os.path.exists("./saves"):
        os.makedirs("./saves")
    
    # 2. Strict temporal split
    # Define boundaries before any statistical calculation
    train_split = int(len(df_univ) * 0.8)
    val_split = int(len(df_univ) * 0.9)
    
    train_df = df_univ.iloc[:train_split]
    val_df = df_univ.iloc[train_split:val_split]
    
    print(f"Train: {len(train_df)} days | Val: {len(val_df)} days")

    # 3. CALCULATING THE PRIOR (Only on the Train set)
    # We calculate the historical correlation only on the training period.
    corr_matrix = train_df.corr().values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior_tensor = torch.tensor(corr_matrix, dtype=torch.float32).to(device)
    
    # 4. Initializing Datasets and Loaders
    train_ds = UniversalDataset(train_df, augment=True)
    val_ds = UniversalDataset(val_df, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 5. Instantiation of the Model with the Prior
    model = FSLPredictor(n_assets=len(tickers), window_size=20, prior=prior_tensor).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    triplet_criterion = TopologicalTripletLoss(margin=0.5).to(device)
    mse_criterion = nn.MSELoss()
    
    # 6. Training Loop
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs = [x.to(device).float() for x in inputs]
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            
            # --- ANCHOR PASS (Real data) ---
            preds_anchor, res_anchor = model(inputs)
            
            # --- POSITIVE PASS (Light noise) ---
            inputs_pos = [x + torch.randn_like(x) * 0.005 for x in inputs]
            _, res_pos = model(inputs_pos)
            
            # --- NEGATIVE PASS (Shuffle assets to break topology) ---
            # We permute assets within the batch
            n_assets = len(inputs)
            asset_perm = torch.randperm(n_assets)
            inputs_neg = [inputs[p].clone() for p in asset_perm]
            # We add some noise to mask residual similarities
            inputs_neg = [x + torch.randn_like(x) * 0.05 for x in inputs_neg]
            _, res_neg = model(inputs_neg)
            
            # --- CALCULATION OF LOSSES ---
            # 1. MSE Prediction
            loss_mse = mse_criterion(preds_anchor, targets)
            
            # 2. Topological Triplet Loss (H1)
            loss_triplet = triplet_criterion(
                res_anchor['h1_score'], 
                res_pos['h1_score'], 
                res_neg['h1_score']
            )
            
            # 3. Orthogonality Loss (Beam Regularization)
            loss_ortho = res_anchor['ortho_loss']
            
            # Total (Weighting)
            loss = loss_mse + 0.5 * loss_triplet + 0.1 * loss_ortho
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()

        # 7. VALIDATION (Independent of the training prior)
        model.eval()
        val_mse = 0
        with torch.no_grad():
            for v_inputs, v_targets in val_loader:
                v_inputs = [x.to(device).float() for x in v_inputs]
                v_targets = v_targets.to(device).float()
                v_preds, _ = model(v_inputs)
                val_mse += mse_criterion(v_preds, v_targets).item()
        
        avg_val = val_mse / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.6f} | Val MSE: {avg_val:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "./saves/fsl.pth")

    print("Training completed.")
    
train_model()