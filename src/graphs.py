import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
VISUALS_DIR = "./visuals/"
SAVES_DIR = "./saves/"
RESULTS_FILE = f"{SAVES_DIR}test_results.pkl"

def load_data():
    if not os.path.exists(RESULTS_FILE):
        print(f"File '{RESULTS_FILE}' not found.")
        exit()
        
    print(f"Loading {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['returns_df'], data['tickers']

def analyze_structural_breaks(results, returns_df):
    h1_scores = [r['h1'] for r in results]
    dates = returns_df.index[20:]
    
    df_analysis = pd.DataFrame({'H1': h1_scores}, index=dates)
    
    # 1. Computing Mean Return (Benchmark)
    df_analysis['Market_Return'] = returns_df.mean(axis=1)
    df_analysis['Cumulative_Market'] = (1 + df_analysis['Market_Return']).cumprod()
    
    # 2. Rolling Volatility
    df_analysis['Rolling_Vol'] = df_analysis['Market_Return'].rolling(20).std()

    # --- GRAPH 1 : H1 VS PERFORMANCE ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    ax1.plot(df_analysis['Cumulative_Market'], color='black', lw=1.5, label='Market (S&P 500 Proxy)')
    ax1.set_title("Market Performance vs Topological Incoherence")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Alert Zones
    threshold = df_analysis['H1'].mean() + 2 * df_analysis['H1'].std()
    high_h1_zones = df_analysis[df_analysis['H1'] > threshold].index
    for zone in high_h1_zones:
        ax1.axvline(zone, color='red', alpha=0.1)

    ax2.plot(df_analysis['H1'], color='purple', lw=1, label='H1 Score (Incoherence)')
    ax2.axhline(y=threshold, color='red', linestyle='--', label="Alert Threshold (Mean + 2Std)")
    ax2.set_ylabel("Topological Disorder")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{VISUALS_DIR}h1_vs_market.png')
    
    # --- GRAPH 2 : SCATTER H1 VS VOL ---
    plt.figure(figsize=(10, 7))
    plt.scatter(df_analysis['Rolling_Vol'], df_analysis['H1'], alpha=0.5, c=df_analysis['H1'], cmap='viridis')
    plt.xlabel("Classic Volatility (Std Dev)")
    plt.ylabel("Topological Incoherence (H1)")
    plt.title("Does H1 capture different information than Volatility?")
    plt.colorbar(label='H1 Intensity')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{VISUALS_DIR}h1_vs_vol_scatter.png')

    # --- GRAPH 3 : OUT OF SAMPLE RAW ---
    plt.figure(figsize=(15, 6))
    plt.plot(df_analysis['H1'], label='H1 Coherence Score', color='purple', alpha=0.7)
    plt.axhline(y=threshold, color='r', linestyle='--', label='Instability Threshold')
    plt.title("Topological Instability (H1) - Out of Sample")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{VISUALS_DIR}h1_out_of_sample.png')

def plot_h1_contribution_heatmap(results, tickers):
    dates = pd.to_datetime([r['date'] for r in results])
    contributions = np.stack([r['h1_contributions'] for r in results])
    
    df_heat = pd.DataFrame(contributions, index=dates, columns=tickers)
    
    # Sectors
    sectors = {
        'Tech': ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],
        'Finance': ['JPM', 'BAC'],
        'Energy': ['XOM', 'CVX'],
        'Health': ['JNJ', 'PFE'],
        'Consumer': ['PG', 'KO', 'MCD', 'HD', 'DIS']
    }
    
    ordered_tickers = [t for s in sectors.values() for t in s if t in df_heat.columns]
    df_heat = df_heat[ordered_tickers]

    # Smoothing
    df_smooth = df_heat.rolling(window=10).mean().dropna()
    h1_scores = pd.Series([r['h1'] for r in results], index=dates)
    h1_smooth = h1_scores.rolling(window=10).mean().loc[df_smooth.index] 
    
    # Low threshold filter to keep only interesting moments
    threshold = h1_smooth.mean() + 1.0 * h1_smooth.std()
    df_final = df_smooth[h1_smooth > threshold]

    if df_final.empty:
        print("Not enough instability to generate the Heatmap.")
        return

    plt.figure(figsize=(20, 10))
    x_labels = [d.strftime('%Y-%m-%d') for d in df_final.index]
    step = max(1, len(x_labels) // 20)
    
    ax = sns.heatmap(df_final.T, cmap='YlOrRd', robust=True,
                     xticklabels=step,
                     cbar_kws={'label': 'Local Contribution to Break (Smoothed)'})
    
    ax.set_xticklabels(x_labels[::step], rotation=45, ha='right')

    # Sector lines
    current_pos = 0
    for sector, tks in sectors.items():
        count = len([t for t in tks if t in ordered_tickers])
        if count > 0:
            current_pos += count
            ax.axhline(current_pos, color='black', lw=1.5, alpha=0.5)
            ax.text(df_final.shape[0] + 0.5, current_pos - count/2, sector, 
                    va='center', fontweight='bold', fontsize=12)

    plt.title("Topological heatmap : Breaks by Sector", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{VISUALS_DIR}h1_heatmap_final.png')

if __name__ == "__main__":
    # 1. Loading
    results, returns_df, tickers = load_data()
    
    # 2. Plots
    analyze_structural_breaks(results, returns_df)
    plot_h1_contribution_heatmap(results, tickers)
    
    print("Visuals generated.")