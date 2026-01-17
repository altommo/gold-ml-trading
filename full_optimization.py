"""
Full Multi-Timeframe Optimization

Tests all parameter combinations across multiple timeframes.
Each timeframe has its own cached data that grows over time.
Results are saved with timestamps for comparison between runs.
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

from config_manager import StrategyConfig
from flexible_backtester import FlexibleBacktester, BacktestResult
from data_fetcher import fetch_gold_data, list_cached_data, CACHE_DIR


# All timeframes to test
TIMEFRAMES = ['15m', '30m', '1h', '2h', '4h']

# Parameter ranges to test
PARAM_RANGES = {
    'range_filter.sampling_period': [7, 10, 14, 21, 27, 35, 50],
    'range_filter.range_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0],
    'risk.stop_loss_value': [1.0, 1.5, 2.0, 2.5, 3.0],
    'risk.take_profit_value': [1.5, 2.0, 3.0, 4.0, 5.0],
    'confirmation.min_bars_between_trades': [0, 2, 3, 5],
}

# Results directory
RESULTS_DIR = Path('./optimization_results')

# Master results file (accumulates all runs)
MASTER_RESULTS_FILE = RESULTS_DIR / 'all_optimization_runs.csv'


def fetch_timeframe_data(interval: str, fetch_latest: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch data for a specific timeframe (uses caching)

    Args:
        interval: Timeframe ('1m', '5m', '15m', '30m', '1h', '2h', '4h')
        fetch_latest: Whether to fetch latest candles and merge

    Returns:
        DataFrame with OHLCV data
    """
    try:
        df = fetch_gold_data(
            source='tradingview',
            interval=interval,
            n_bars=5000,
            fetch_latest=fetch_latest
        )
        return df
    except Exception as e:
        print(f"Error fetching {interval} data: {e}")
        return None


def calculate_score(result: BacktestResult, min_trades: int = 15) -> float:
    """Calculate optimization score"""
    if result.total_trades < min_trades:
        return -999

    score = (
        result.total_return_pct * 0.25 +
        (result.profit_factor - 1) * 30 * 0.25 +
        result.sharpe_ratio * 10 * 0.20 +
        (result.win_rate - 40) * 0.15 +
        result.max_drawdown_pct * 0.15  # Negative, penalizes drawdown
    )

    return score


def run_single_test(
    df: pd.DataFrame,
    params: Dict,
    timeframe: str,
    data_start: str,
    data_end: str
) -> Optional[Dict]:
    """Run a single parameter combination test"""
    try:
        # Create config
        config = StrategyConfig(name=f"test_{timeframe}")

        # Apply parameters
        for param_name, value in params.items():
            parts = param_name.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        # Run backtest
        backtester = FlexibleBacktester(config)
        result = backtester.run(df)

        score = calculate_score(result)

        # Return with clear parameter names
        return {
            'timeframe': timeframe,
            'sampling_period': params['range_filter.sampling_period'],
            'range_multiplier': params['range_filter.range_multiplier'],
            'stop_loss_pct': params['risk.stop_loss_value'],
            'take_profit_pct': params['risk.take_profit_value'],
            'min_bars_between': params['confirmation.min_bars_between_trades'],
            'return_pct': round(result.total_return_pct, 2),
            'trades': result.total_trades,
            'win_rate': round(result.win_rate, 1),
            'profit_factor': round(result.profit_factor, 2),
            'max_drawdown_pct': round(result.max_drawdown_pct, 2),
            'sharpe_ratio': round(result.sharpe_ratio, 2),
            'expectancy_r': round(result.expectancy_r, 3),
            'max_concurrent_pos': result.max_concurrent_positions,
            'avg_bars_held': round(result.avg_bars_held, 1),
            'score': round(score, 2),
            'data_start': data_start,
            'data_end': data_end,
            'data_bars': len(df)
        }
    except Exception as e:
        return None


def optimize_timeframe(
    interval: str,
    param_ranges: Dict[str, List] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run full optimization for a single timeframe

    Args:
        interval: Timeframe to optimize
        param_ranges: Parameter ranges (uses defaults if None)
        verbose: Print progress

    Returns:
        DataFrame with all results
    """
    if param_ranges is None:
        param_ranges = PARAM_RANGES

    print(f"\n{'='*60}")
    print(f"OPTIMIZING: {interval} timeframe")
    print(f"{'='*60}")

    # Fetch data
    df = fetch_timeframe_data(interval, fetch_latest=True)
    if df is None or len(df) < 100:
        print(f"Insufficient data for {interval}")
        return pd.DataFrame()

    data_start = str(df.index[0])[:10]
    data_end = str(df.index[-1])[:10]
    print(f"Data: {len(df)} bars from {data_start} to {data_end}")

    # Generate combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(product(*param_values))

    total = len(combinations)
    print(f"Testing {total} combinations...")

    results = []
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        result = run_single_test(df, params, interval, data_start, data_end)

        if result is not None:
            results.append(result)

        if verbose and (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('score', ascending=False)

    return results_df


def optimize_all_timeframes(
    timeframes: List[str] = None,
    param_ranges: Dict[str, List] = None,
    save_results: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run optimization across all timeframes

    Args:
        timeframes: List of timeframes to test
        param_ranges: Parameter ranges
        save_results: Save results to files

    Returns:
        Dict of timeframe -> results DataFrame
    """
    if timeframes is None:
        timeframes = TIMEFRAMES

    RESULTS_DIR.mkdir(exist_ok=True)

    # Generate run timestamp
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / f"run_{run_timestamp}"
    run_dir.mkdir(exist_ok=True)

    all_results = {}
    best_per_tf = []

    print("="*70)
    print("FULL MULTI-TIMEFRAME OPTIMIZATION")
    print(f"Run ID: {run_timestamp}")
    print("="*70)
    print(f"Timeframes: {timeframes}")
    print(f"Parameters: {list(PARAM_RANGES.keys())}")

    total_combos = 1
    for values in PARAM_RANGES.values():
        total_combos *= len(values)
    print(f"Combinations per timeframe: {total_combos}")
    print(f"Total tests: {total_combos * len(timeframes)}")

    for tf in timeframes:
        results_df = optimize_timeframe(tf, param_ranges)
        all_results[tf] = results_df

        if not results_df.empty:
            # Add run metadata
            results_df['run_id'] = run_timestamp

            # Save individual timeframe results
            if save_results:
                filepath = run_dir / f"optimization_{tf}.csv"
                results_df.to_csv(filepath, index=False)
                print(f"Results saved to: {filepath}")

            # Track best for this timeframe
            best = results_df.iloc[0]
            best_per_tf.append({
                'run_id': run_timestamp,
                'timeframe': tf,
                'sampling_period': best['sampling_period'],
                'range_multiplier': best['range_multiplier'],
                'stop_loss_pct': best['stop_loss_pct'],
                'take_profit_pct': best['take_profit_pct'],
                'min_bars_between': best['min_bars_between'],
                'return_pct': best['return_pct'],
                'win_rate': best['win_rate'],
                'profit_factor': best['profit_factor'],
                'sharpe_ratio': best['sharpe_ratio'],
                'max_drawdown_pct': best['max_drawdown_pct'],
                'trades': best['trades'],
                'max_concurrent_pos': best['max_concurrent_pos'],
                'score': best['score'],
                'data_start': best['data_start'],
                'data_end': best['data_end'],
                'data_bars': best['data_bars']
            })

    # Save run summary
    if save_results and best_per_tf:
        summary_df = pd.DataFrame(best_per_tf)

        # Save to run directory
        summary_path = run_dir / "best_per_timeframe.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nRun summary saved to: {summary_path}")

        # Append to master results file for comparison across runs
        if MASTER_RESULTS_FILE.exists():
            master_df = pd.read_csv(MASTER_RESULTS_FILE)
            master_df = pd.concat([master_df, summary_df], ignore_index=True)
        else:
            master_df = summary_df
        master_df.to_csv(MASTER_RESULTS_FILE, index=False)
        print(f"Master results updated: {MASTER_RESULTS_FILE}")

    print(f"\nAll results saved to: {run_dir}")

    return all_results


def print_best_results(all_results: Dict[str, pd.DataFrame], top_n: int = 5):
    """Print best results for each timeframe"""
    print("\n" + "="*110)
    print("BEST PARAMETERS PER TIMEFRAME")
    print("="*110)

    for tf, results_df in all_results.items():
        if results_df.empty:
            print(f"\n{tf}: No valid results")
            continue

        print(f"\n{'-'*70}")
        print(f"TIMEFRAME: {tf} (Top {top_n})")
        print(f"{'-'*70}")

        display_cols = [
            'sampling_period', 'range_multiplier', 'stop_loss_pct', 'take_profit_pct',
            'return_pct', 'win_rate', 'profit_factor', 'sharpe_ratio',
            'max_drawdown_pct', 'trades', 'score'
        ]

        top = results_df.head(top_n)[display_cols].copy()
        top.columns = ['Period', 'Mult', 'SL%', 'TP%',
                       'Return', 'WinRate', 'PF', 'Sharpe', 'MaxDD', 'Trades', 'Score']

        # Format
        top['Return'] = top['Return'].apply(lambda x: f"{x:+.2f}%")
        top['WinRate'] = top['WinRate'].apply(lambda x: f"{x:.1f}%")
        top['PF'] = top['PF'].apply(lambda x: f"{x:.2f}")
        top['Sharpe'] = top['Sharpe'].apply(lambda x: f"{x:.2f}")
        top['MaxDD'] = top['MaxDD'].apply(lambda x: f"{x:.2f}%")
        top['Score'] = top['Score'].apply(lambda x: f"{x:.2f}")

        print(top.to_string(index=False))


def print_overall_best(all_results: Dict[str, pd.DataFrame]):
    """Find and print the single best configuration across all timeframes"""
    all_rows = []
    for tf, results_df in all_results.items():
        if not results_df.empty:
            for _, row in results_df.iterrows():
                row_dict = row.to_dict()
                all_rows.append(row_dict)

    if not all_rows:
        print("No valid results found")
        return

    combined = pd.DataFrame(all_rows)
    combined = combined.sort_values('score', ascending=False)

    best = combined.iloc[0]

    print("\n" + "="*70)
    print("OVERALL BEST CONFIGURATION")
    print("="*70)
    print(f"\nTimeframe: {best['timeframe']}")
    print(f"Data: {best['data_start']} to {best['data_end']} ({best['data_bars']} bars)")
    print(f"\nParameters:")
    print(f"  Sampling Period:    {int(best['sampling_period'])}")
    print(f"  Range Multiplier:   {best['range_multiplier']}")
    print(f"  Stop Loss:          {best['stop_loss_pct']}%")
    print(f"  Take Profit:        {best['take_profit_pct']}%")
    print(f"  Min Bars Between:   {int(best['min_bars_between'])}")
    print(f"\nPerformance:")
    print(f"  Return:           {best['return_pct']:+.2f}%")
    print(f"  Win Rate:         {best['win_rate']:.1f}%")
    print(f"  Profit Factor:    {best['profit_factor']:.2f}")
    print(f"  Sharpe Ratio:     {best['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown:     {best['max_drawdown_pct']:.2f}%")
    print(f"  Trades:           {int(best['trades'])}")
    print(f"  Max Concurrent:   {int(best['max_concurrent_pos'])}")
    print(f"  Avg Bars Held:    {best['avg_bars_held']:.1f}")
    print(f"  Expectancy:       {best['expectancy_r']:.3f}R")
    print(f"  Score:            {best['score']:.2f}")

    # Save best config
    run_id = best.get('run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
    best_config = {
        'run_id': run_id,
        'timeframe': best['timeframe'],
        'data_range': {
            'start': best['data_start'],
            'end': best['data_end'],
            'bars': int(best['data_bars'])
        },
        'parameters': {
            'sampling_period': int(best['sampling_period']),
            'range_multiplier': float(best['range_multiplier']),
            'stop_loss_pct': float(best['stop_loss_pct']),
            'take_profit_pct': float(best['take_profit_pct']),
            'min_bars_between': int(best['min_bars_between'])
        },
        'performance': {
            'return_pct': float(best['return_pct']),
            'win_rate': float(best['win_rate']),
            'profit_factor': float(best['profit_factor']),
            'sharpe_ratio': float(best['sharpe_ratio']),
            'max_drawdown_pct': float(best['max_drawdown_pct']),
            'expectancy_r': float(best['expectancy_r']),
            'trades': int(best['trades']),
            'max_concurrent_pos': int(best['max_concurrent_pos']),
            'avg_bars_held': float(best['avg_bars_held'])
        },
        'score': float(best['score'])
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    config_path = RESULTS_DIR / f'best_config_{run_id}.json'
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    # Also save as latest
    with open(RESULTS_DIR / 'best_config_latest.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\nBest config saved to: {config_path}")


def show_cached_data():
    """Show all cached data"""
    print("\n" + "="*60)
    print("CACHED DATA")
    print("="*60)

    cached = list_cached_data()
    if cached.empty:
        print("No cached data found. Run optimization to fetch data.")
    else:
        print(cached.to_string(index=False))


def main():
    """Main optimization runner"""
    print("="*70)
    print("FULL MULTI-TIMEFRAME OPTIMIZATION")
    print("="*70)

    # Show current cache
    show_cached_data()

    # Run optimization
    all_results = optimize_all_timeframes()

    # Print results
    print_best_results(all_results)
    print_overall_best(all_results)

    return all_results


if __name__ == "__main__":
    results = main()
