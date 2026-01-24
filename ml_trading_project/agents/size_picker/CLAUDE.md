# Size Picker Agent

## Objective

Pick optimal position size based on candle patterns and equity. The model should learn to recognize patterns in price action that indicate whether a large or small position is appropriate - not memorize specific trades.

Goal: Maximize return without getting liquidated.

## Constraints

- Liquidation occurs at 30% equity
- Leverage: 20x (max position = equity * 20 / gold_price)
- No stop loss - trade runs until exit signal
- Position sizes in increments of 0.15 oz (0.15 to 4.50, 30 sizes)

## Core Insight

Position sizing is not just about the trade outcome - it's about surviving the drawdown during the trade. A winning trade can still liquidate you if the position is too large and price moves against you before moving in your favor.

The model must learn: given these candles, what's the largest size I can take without getting liquidated?

## V2 Architecture (Current)

### Two-Head Model
Instead of 30-class classification, predict for EACH of 30 sizes:
- **P(liquidation)** - probability the trade liquidates at this size
- **E[return]** - expected return percentage at this size

### Decision Rule
Pick the largest size where P(liq) < threshold (default 10%) with the best expected return.

### Input Features (V2)
5 features per candle (instead of raw OHLC):
- Log returns (close-to-close)
- Range (high-low)/close - volatility
- Body |close-open|/close - conviction
- Upper wick - rejection from highs
- Lower wick - rejection from lows

### Loss Function
Combined loss with asymmetric penalties:
1. **BCE loss** for liquidation prediction
2. **Huber loss** for return prediction
3. **Asymmetric penalty** (5x) for predicting "safe" when actually liquidates

### Training Stability
- AdamW optimizer with weight decay
- Gradient clipping (max_norm=1.0)
- OneCycleLR scheduler

### Metrics
- **Liquidation Rate** - % of chosen sizes that liquidate (lower is better)
- **Average Return** - mean return from model's choices
- **Regret** - gap between optimal and actual returns (lower is better)

## What We've Learned

### V1 Flaws (Fixed in V2)
- Cross-entropy treats all errors equally (off by 1 step = off by 20 steps)
- Ordinal structure ignored
- Label noise from near-ties caused oscillation
- Optimizing accuracy when we should optimize expected return under constraint

### Broker Simulation
- P&L calculation: `price_change * size` (not multiplied by leverage)
- Leverage affects position sizing capacity, not P&L magnitude
- Must simulate bar-by-bar to check for liquidation during trade

### Key Distinction
- **Wrong**: Memorize "Trade 3 = small size"
- **Right**: Learn "This candle pattern = risky, go small"

## Training Data (V2)

V2 datasets include `all_results` - simulation outcomes for all 30 sizes per trade.

Generate using `generate_all_datasets_v2()`:
- 100k small optimal (size <= 1.5)
- 100k mid optimal (1.5 < size <= 3.0)
- 100k big optimal (size > 3.0)

Then create:
- `balanced_v2_100k` - 33k of each category
- `reinforcement_v2_200k` - remaining trades

## Files

- `generate_datasets.ipynb` - Synthetic trade generation
- `size_picker_v1.ipynb` - V2 training notebook (naming preserved for continuity)

## Status

- [x] Broker simulation working
- [x] V1 training loop (cross-entropy)
- [x] V2 two-head architecture
- [x] Asymmetric loss function
- [x] Training stability (AdamW, grad clip, OneCycleLR)
- [x] Risk-aware metrics
- [ ] Generate V2 synthetic training data
- [ ] Train V2 on synthetic data
- [ ] Validate V2 performance vs V1
- [ ] Test on real trade data
