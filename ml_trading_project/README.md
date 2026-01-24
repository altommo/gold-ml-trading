# Gold Trading ML Model

Two ML approaches for XAUUSD trading:
1. **Guided Model** - Trained on actual trading data (mean-reversion style)
2. **Pure Model** - Learns from scratch using price prediction

## Quick Start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/altommo/gold-ml-trading/blob/main/ml_trading_model.ipynb)

1. Click the Colab badge above
2. Run all cells
3. The notebook will clone this repo and load the data automatically

## Local Setup

```bash
git clone https://github.com/altommo/gold-ml-trading.git
cd gold-ml-trading
pip install -r requirements.txt
jupyter notebook ml_trading_model.ipynb
```

## Files

- `ml_trading_model.ipynb` - Main notebook with both models
- `data/trades_with_features.csv` - Actual trades matched to chart data
- `data/XAUUSD_1h.csv` - XAUUSD hourly chart data
- `src/indicators.py` - Technical indicator calculations
- `src/backtest.py` - Backtesting utilities
- `requirements.txt` - Python dependencies

## Trading Strategy

Based on analysis of actual trades:

**Buy signals** (mean reversion):
- WaveTrend < 0 (oversold)
- Wolfpack < 0 (pullback)
- RSI < 50
- Price below MA20

**Sell signals**:
- WaveTrend > 40 (overbought)
- Wolfpack > 5
- RSI > 65
- Price above MA20

## Results

From 138 actual Gold trades (Dec 2025 - Jan 2026):
- Win Rate: 93.5%
- R:R: 2.27
- Profit Factor: 32.50
- Sharpe: 10.24

## License

MIT
