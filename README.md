# Hierarchical Sheaf Networks (FSL)

[![Status](https://img.shields.io/badge/Status-Research%20Alpha-orange)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)]()
[![Math](https://img.shields.io/badge/Topological-Data%20Analysis-purple)]()

This project implements **Functorial Sheaf Learning (FSL)** to model the stock market as a topological space. By measuring **Cohomology ($H^1$)**, we extract a structural "stress" signal that is orthogonal to traditional volatility, allowing for the detection of some market fractures before they impact price.

---

## Table of Contents

- [Hierarchical Sheaf Networks (FSL)](#hierarchical-sheaf-networks-fsl)
  - [Table of Contents](#table-of-contents)
  - [Cohomology as Alpha](#cohomology-as-alpha)
    - [The Problem with Standard Metrics](#the-problem-with-standard-metrics)
    - [The Solution: $H^1$ Cohomology](#the-solution-h1-cohomology)
  - [Why Sheaves instead of standard DL ?](#why-sheaves-instead-of-standard-dl-)
  - [Architecture](#architecture)
    - [1. Multi-Head Factorized Sheaf Laplacian](#1-multi-head-factorized-sheaf-laplacian)
    - [2. Topological Triplet Loss](#2-topological-triplet-loss)
    - [3. Multi-Regime HMM (Hidden Markov Model)](#3-multi-regime-hmm-hidden-markov-model)
  - [Installation \& Usage](#installation--usage)
    - [Prerequisites](#prerequisites)
  - [Problems \& Future](#problems--future)
  - [Bibliography and notes](#bibliography-and-notes)

---

## Cohomology as Alpha

### The Problem with Standard Metrics

Traditional finance relies heavily on volatility ($\sigma$) and correlation ($\rho$).

- **Problem:** In a crash, correlations converge to 1. Traditional models lose their diversification benefits exactly when needed.
- **Problem:** A market can be low-volatility but structurally broken (e.g., the housing market in 2007 before the drop).

### The Solution: $H^1$ Cohomology

1. Why cohomology ?
    We don't use homology, because there isn't a cup product -> better structure.
    So why $H¹$ instead of other cohomology space $H^k$ ?
    $H⁰$ is the "Beta". When everything goes up, $H⁰$ is high. Bad for alpha.
    $H¹$ is the obstruction to global coherence. Locally, the cycles run, but globally, we get an impossible cycle -> anomaly (structural "stress")
    $H²$ is superior order incoherence. Hard to interpret, and it takes a lot of computational workload ($O(N³)$) !

    So we have our $H¹$, the cohomology score, an indicator of structural (systemic) stress.
    When :
        1) $H¹\approx 0$ -> Coherent market (cycles are stable)
        2) $H¹ >> 0$ -> Incoherent market (cycles are distorted)

    So let's use $H¹$ instead of volatility to find the right regime !
    Well... Not exactly. H¹ alone has problems : in a global crisis, the market is structured. So it falls with it. To solve this problem, we add volatility : We use a composite signal (H1 modulated by volatility) to detect regimes. High volatility increase structural stress, so the model can distinguish 'calm coherence' (Bull) and 'crash coherence' (Bear)."

2. Why HMM ?
    Unlike usual methods, we don't use Hidden Markov Models to predict future prices : we use it to analyse the structural quality with $H¹$. The regimes are the hidden states. Here, there is 4 regimes : Bull ($H¹$ low), Normal ($H¹$ medium), Volatil ($H¹$ high), Crisis ($H¹$ very high). Be aware that these states are not particularily the market states : it only indicates the coherence of the market. So Bull state is a high-correlated market. Unlike standard HMMs, we add a Trend Filter (Moving Average) and Realized Volatility. This avoids the 'Structural Trap' described earlier where the model falls with a (coherent) structural crash. The final signal is a combinaison : $f(H^1, \sigma)$

    Also, in HMM, we implemented inertia to avoid that the model changes too fast of opinion (avoid whipsaw). The HMM is also useful to change some investment parameters like leverage, long/short, etc.

3. Why Functorial Sheaf Learning ?
    Unlike other models, Sheaf Neural Network uses linear applications to explain different states. This is used to model more complex relations, and to explain them ! We can invert, rotate, unphase, linear applications ! It gives the keys to translate APPL to MSFT (sort of), instead of only showing the correlation (like GNN or smth else).

    The functorial part is useful to keep the topological structure between layers. In classical neural networks, geometrical structure is sometimes lost between the layers. So the functorial part is useful to keep the structural interpretability !

---

## Why Sheaves instead of standard DL ?

| | LSTM / RNN | Transformers | **Sheaf Networks (FSL)** |
| :--- | :--- | :--- | :--- |
| **Relationship Modeling** | Implicit (hidden state) | Attention (Correlation) | Explicit (Linear Maps) |
| **Interpretability** | Black Box | Attention Maps (Noisy) | High (Transport Matrices) |
| **Data Efficiency** | Needs huge history | Data Hungry | Structured (Geometric Priors) |
| **Concept of "Error"** | Prediction Error only | Prediction Error | Topological Inconsistency ($H^1$) |

**Why not Transformers?**
Transformers tends to hallucinate relationships in noisy financial data. They assume everything is connected to everything. FSL enforces a specific geometric structure, preventing the model from learning noise.

---

## Architecture

This is not a simple regression model. It is a hierarchical system designed to extract signal from noise.

### 1. Multi-Head Factorized Sheaf Laplacian

- **Current:** We use a factorized approach with $k$ latent heads.
- **Mechanism:** Assets are projected into independent latent spaces. The model learns a low-rank representation of the market topology, preserving global structure while being scalable to hundreds of tickers.

### 2. Topological Triplet Loss

Training is guided by a custom loss function:
$$\mathcal{L} = \mathcal{L}_{MSE} + \lambda \cdot \max(0, H^1_{anchor} - H^1_{negative} + \alpha)$$
The model learns to minimize $H^1$ on real data (Anchor) while maximizing it on shuffled data (Negative). This proves the model is learning **structure**, not just memorizing prices.

### 3. Multi-Regime HMM (Hidden Markov Model)

The raw $H^1$ signal can be noisy. We feed the topological signal into a customized HMM with 4 hidden states:

- **Bull:** High Coherence, Low Volatility.
- **Normal:** Average Coherence.
- **Volatile:** High Volatility, but Structurally Sound (Correction).
- **Crisis:** High Volatility + High Topological Incoherence (Structural Break).

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- GPU with cuda (if not, use the already trained model (fsl.pth))

```bash
git clone git@github.com:arn1369/FSL.git
cd FSL
pip install -r requirements.txt
python3 src/train.py
python3 src/test.py
python3 src/visuals.py
```

## Problems & Future

Problems :

- Complexity : $O(N^2)$
- Using pair consistency instead of 3 points (to get a face with cycles)
- Hardcoded HMM
- Survivor Bias
- Need to reset weigts and prior before each training
- Save HMM state at time $t$ with test file to plot it

Future implementations :

1) Learnable Residual Weights
2) [REMOVED] Diagonal in adjacency matrix is $-10^9$ (temporary solution with the Diffusion Matrix, try to test Normalized Laplacian ?)
3) [REMOVED] Maybe interesting to see $H²$, but at the price of interpretability -> dimensionality reduction ? what performance with $H²$ ? Are the links between assets sooo complex in real life ? Also complexity ($O(N^3)$) -> Using hypergraph to capture collective (sector) behavior -> Sheaf Hypergraph Networks (lighter than $H²$ computation). This is removed as we try to reduce complexity. The H¹ is sufficient enough to capture complex relations ($H²$ would be too slow for a not-so-much improvement).
4) Ollivier-Ricci curvature. Seen on a video, measures congestion. Need to dig that, would be interesting. Small explanation : in crisis, everyone rush on safe-assets (topological congestion). We could inject the curvature as a new feature in the HMM besides $H¹$ ?
5) Martin Hairer (Fields, 2014) : Path Signatures to FSLPredictor instead of gross return ?
6) Interesting thing I've seen too on HMM (from LSM Investment Club) : Hierarchical Dirichlet Process (HDP) ! For now, we hardcode the number of hidden states (Bull, Normal, Volatil, Crisis). But is this right ? -> HDP learns the number of necessary states !
7) Adversarial Training (ML course, LELEC2870). Interesting to apply it here ?
8) RL for leverage (for now, if("BULL"): leverage=1.5) to learn what is the best one. See paper on RL in finance (PPO agent);
9) Vectorize cycles computing (better use on GPU)
10) Scheduler on train (I have 10 epochs now but in the future).$

## Bibliography and notes

A lot of work comes from the paper [Sheaf Cohomology of Linear Predictive Coding Networks](https://arxiv.org/pdf/2511.11092) from Jeffrey Seely at Sakana AI  (14 nov. 2025). My work is the implementation and some improvements of its work.
Moreover, LLM were used to generate code. This helps to go faster in my research. I check what the LLM does, but my competences and checks are not perfect. Some errors may have slipped in there. I'm constantly improving the methods used, and fixing bugs.

English is not my primary language, so feel free to correct any mistakes !

Also, do not hesitate to make a P-R with new methods, comments, better approach, etc ! I'm open to any improvement of FSL :\) Don't hesitate to contact me : [arnullens@gmail.com](mailto:arnullens@gmail.com) to discuss or if you have any idea !

Thank you for your interest in my project !
