# Intraday Stock Movement Prediction

This repository implements a deep learning pipeline for predicting short-term stock price movement (buy/sell/hold) using a GRU classifier trained on intraday technical indicators. The model processes time steps of 5-minute interval data and generates actionable predictions for a given stock symbol.

---

## What It Does

- Fetches 5-minute intraday price data from Yahoo Finance
- Computes 20+ standard technical indicators (MACD, RSI, VWAP, etc.)
- Uses a PReLU-activated GRU to classify each window of price data into:
  - `Buy` (price expected to rise)
  - `Sell` (price expected to drop)
  - `Hold` (no significant movement)
- Normalizes data per stock and trains one model per symbol
- Outputs performance metrics (accuracy, confusion matrix)
- Plots buy/sell/hold signals overlaid on price chart

---

## How It Works

### Features
The model uses 21 widely accepted technical indicators:
- Trend & Momentum: `EMA`, `MACD`, `ADX`, `ROC`
- Volatility: `ATR`, `Bollinger Bands`
- Oscillators: `RSI`, `Stochastic Oscillator`
- Volume-based: `OBV`, `MFI`, `VWAP`

### Model Architecture
- Input: sequences of 60 time steps (5-minute candles × 60 = 5 hours)
- Backbone: 2-layer **GRU** with `128` hidden units
- Activation: `PReLU` after GRU and in classifier
- Output: `3-class softmax` predicting Buy / Hold / Sell
- Loss: `CrossEntropyLoss`, optionally class-weighted for imbalance

---

## Training Setup

- Sequence Length: 60
- Training Horizon: 3 timesteps (15 minutes ahead)
- Normalization: z-score per feature, per stock
- Epochs: 32 (adjustable)
- Evaluation: Balanced accuracy, classification report, confusion matrix

---

## Output & Visualization

For each stock:
- Console logs:
  - Training time
  - Balanced accuracy
  - Classification report (accuracy/precision/recall/F1)
  - Confusion matrix
- Plot: Price with buy/sell/hold markers
