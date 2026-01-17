# Trading Strategy Backtesting Project

## Objective
Backtest and optimize a trading strategy for XAUUSD (Gold) that combines:
- **Potato Signal (Range Filter)** - Primary signal generator
- **Wolfpack ID** - Trend confirmation (MACD variant: EMA3 - EMA8)
- **WaveTrend** - Momentum confirmation

Goal: Understand what parameters work best, identify patterns in winning vs losing trades, and find ways to filter out likely losers.

## Current Strategy Logic
1. Range Filter generates buy/sell signals when price crosses the filter line
2. Wolfpack ID confirms trend direction (bullish when positive, bearish when negative)
3. WaveTrend confirms momentum
4. **Entry**: Signal + confirmations aligned
5. **Exit**: Stop Loss OR Take Profit only (signals don't close positions)
6. **Partial TP**: 50% of position closes at 50% of target
7. **Multiple positions**: Can have concurrent trades open

## Data Sources
- TradingView via tvDatafeed (primary)
- Cached locally in `data_cache/` per timeframe
- Each run extends the cache with new data

## Backtest Results So Far

### 15m Timeframe (78 days: Oct 2025 - Jan 2026)
**Best params:** SP=10, RM=0.5, SL=2.0%, TP=5.0%
- Return: +19.79%
- Profit Factor: 1.35
- Win Rate: 48.4%
- Trades: 572
- Max Concurrent: 143

### 30m Timeframe (155 days: Aug 2025 - Jan 2026)
**Best params:** SP=21, RM=0.5, SL=1.5%, TP=5.0%
- Return: +48.37%
- Profit Factor: 1.81
- Win Rate: 49.3%
- Trades: 584
- Max Concurrent: 63

### Common Patterns in Top Performers
| Parameter | Optimal Value |
|-----------|---------------|
| Range Multiplier | **0.5** (tight) |
| Take Profit | **5.0%** |
| Stop Loss | 1.5-2.0% |
| Sampling Period | 10-27 (flexible) |

## Key Finding: Long/Short Imbalance

During this test period (gold trending UP), there's a massive imbalance:

| Exit Type | Longs | Shorts |
|-----------|-------|--------|
| **Stop Loss** | ~25-30% | **~70-75%** |
| **Take Profit** | **~94-96%** | ~4-6% |

**Interpretation:**
- Shorts are getting stopped out in an uptrend
- Longs are hitting take profit
- The strategy works, but counter-trend trades are destroying edge

### WaveTrend as Predictor
| Trade Outcome | Avg WaveTrend at Entry |
|---------------|------------------------|
| Hit Stop Loss | -0.5 to -6.5 (bearish) |
| Hit Take Profit | +22 to +23 (bullish) |

When WaveTrend is positive at entry → trade more likely to win.

## Next Steps to Investigate

1. **Trend Filter** - Only take longs in uptrend, shorts in downtrend
   - Use higher timeframe direction (e.g., 1h/4h MA)
   - Or use WaveTrend threshold as filter

2. **ML Model** - Train classifier to predict SL vs TP
   - Features: WaveTrend, Wolfpack, volatility (ATR), time features, higher TF trend
   - Label: 1 = hit TP, 0 = hit SL

3. **Longer Timeframes** - Run 1h, 2h, 4h backtests
   - More data history available
   - Different characteristics expected

4. **Out-of-Sample Testing** - Validate on unseen data periods

## File Structure
```
backtest.py           # Main entry point
config_manager.py     # Strategy configuration
data_fetcher.py       # Data fetching & caching
flexible_backtester.py # Core backtest engine
indicators.py         # Wolfpack, WaveTrend, etc.
range_filter_strategy.py # Range Filter (Potato Signal)
results/              # Backtest results by run
data_cache/           # Cached OHLCV data
```

## Running Backtests
```bash
python backtest.py                 # Sweep all timeframes
python backtest.py --tf 1h 4h      # Specific timeframes
python backtest.py --tf 1h --single --sp 21 --sl 1.5 --tp 5.0  # Single test
```
