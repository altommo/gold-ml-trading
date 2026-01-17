"""
Backtest Runner

Single entry point for all backtesting.
Sweeps all parameter combinations across timeframes by default.

Usage:
    python backtest.py                     # Sweep all timeframes
    python backtest.py --tf 1h 4h          # Sweep specific timeframes
    python backtest.py --tf 1h --single    # Single test with default params
"""

import argparse
import pandas as pd
import numpy as np
from itertools import product
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from config_manager import StrategyConfig
from flexible_backtester import FlexibleBacktester, BacktestResult, print_detailed_results
from data_fetcher import fetch_gold_data, list_cached_data

# Directories
RESULTS_DIR = Path('./results')
CACHE_DIR = Path('./data_cache')

# All available timeframes
ALL_TIMEFRAMES = ['15m', '30m', '1h', '2h', '4h']

# Parameter ranges for sweep
PARAM_GRID = {
    'sampling_period': [7, 10, 14, 21, 27, 35, 50],
    'range_multiplier': [0.5, 0.75, 1.0, 1.5, 2.0],
    'stop_loss_pct': [1.0, 1.5, 2.0, 2.5, 3.0],
    'take_profit_pct': [1.5, 2.0, 3.0, 4.0, 5.0],
}


def calculate_score(result: BacktestResult, min_trades: int = 15) -> float:
    """Calculate ranking score for a backtest result"""
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


def fetch_data(timeframe: str) -> Optional[pd.DataFrame]:
    """Fetch data for a timeframe (uses cache, extends with latest)"""
    try:
        df = fetch_gold_data(
            source='tradingview',
            interval=timeframe,
            n_bars=5000,
            fetch_latest=True
        )
        return df
    except Exception as e:
        print(f"Error fetching {timeframe}: {e}")
        return None


def run_single_backtest(
    df: pd.DataFrame,
    timeframe: str,
    sampling_period: int = 21,
    range_multiplier: float = 1.0,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 3.0
) -> Dict:
    """Run a single backtest with given parameters"""

    config = StrategyConfig(name=f"backtest_{timeframe}")
    config.range_filter.sampling_period = sampling_period
    config.range_filter.range_multiplier = range_multiplier
    config.risk.stop_loss_value = stop_loss_pct
    config.risk.take_profit_value = take_profit_pct

    backtester = FlexibleBacktester(config)
    result = backtester.run(df)
    score = calculate_score(result)

    return {
        'timeframe': timeframe,
        'sampling_period': sampling_period,
        'range_multiplier': range_multiplier,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
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
        'data_start': str(df.index[0])[:10],
        'data_end': str(df.index[-1])[:10],
        'data_bars': len(df)
    }, result


def sweep_timeframe(timeframe: str, df: pd.DataFrame) -> pd.DataFrame:
    """Sweep all parameter combinations for a timeframe"""

    combinations = list(product(
        PARAM_GRID['sampling_period'],
        PARAM_GRID['range_multiplier'],
        PARAM_GRID['stop_loss_pct'],
        PARAM_GRID['take_profit_pct']
    ))

    total = len(combinations)
    print(f"  Testing {total} combinations...")

    results = []
    for i, (sp, rm, sl, tp) in enumerate(combinations):
        try:
            row, _ = run_single_backtest(df, timeframe, sp, rm, sl, tp)
            results.append(row)
        except:
            pass

        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{total} ({(i+1)/total*100:.0f}%)")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('score', ascending=False)

    return results_df


def run_sweep(timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """Run parameter sweep across specified timeframes"""

    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BACKTEST PARAMETER SWEEP")
    print(f"Run ID: {run_id}")
    print("=" * 70)
    print(f"Timeframes: {timeframes}")
    print(f"Parameters: {list(PARAM_GRID.keys())}")

    total_combos = 1
    for values in PARAM_GRID.values():
        total_combos *= len(values)
    print(f"Combinations per timeframe: {total_combos}")
    print(f"Total tests: {total_combos * len(timeframes)}")

    all_results = {}
    best_per_tf = []

    for tf in timeframes:
        print(f"\n{'='*60}")
        print(f"TIMEFRAME: {tf}")
        print(f"{'='*60}")

        df = fetch_data(tf)
        if df is None or len(df) < 100:
            print(f"  Insufficient data for {tf}")
            continue

        print(f"  Data: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

        results_df = sweep_timeframe(tf, df)
        all_results[tf] = results_df

        if not results_df.empty:
            # Save timeframe results
            results_df['run_id'] = run_id
            filepath = run_dir / f"backtest_{tf}.csv"
            results_df.to_csv(filepath, index=False)
            print(f"  Results saved: {filepath}")

            # Track best
            best = results_df.iloc[0]
            best_per_tf.append(best.to_dict())

            # Print top 3
            print(f"\n  Top 3 for {tf}:")
            for j, row in results_df.head(3).iterrows():
                print(f"    SP={int(row['sampling_period']):2d} RM={row['range_multiplier']:.2f} "
                      f"SL={row['stop_loss_pct']:.1f}% TP={row['take_profit_pct']:.1f}% "
                      f"-> {row['return_pct']:+.2f}% PF={row['profit_factor']:.2f} "
                      f"WR={row['win_rate']:.1f}% Score={row['score']:.1f}")

    # Save summary
    if best_per_tf:
        summary_df = pd.DataFrame(best_per_tf)
        summary_df.to_csv(run_dir / "best_per_timeframe.csv", index=False)

        # Append to master file
        master_file = RESULTS_DIR / "all_runs.csv"
        if master_file.exists():
            master_df = pd.read_csv(master_file)
            master_df = pd.concat([master_df, summary_df], ignore_index=True)
        else:
            master_df = summary_df
        master_df.to_csv(master_file, index=False)

        # Find overall best
        overall_best = summary_df.loc[summary_df['score'].idxmax()]

        print("\n" + "=" * 70)
        print("OVERALL BEST")
        print("=" * 70)
        print(f"Timeframe:        {overall_best['timeframe']}")
        print(f"Sampling Period:  {int(overall_best['sampling_period'])}")
        print(f"Range Multiplier: {overall_best['range_multiplier']}")
        print(f"Stop Loss:        {overall_best['stop_loss_pct']}%")
        print(f"Take Profit:      {overall_best['take_profit_pct']}%")
        print(f"Return:           {overall_best['return_pct']:+.2f}%")
        print(f"Win Rate:         {overall_best['win_rate']:.1f}%")
        print(f"Profit Factor:    {overall_best['profit_factor']:.2f}")
        print(f"Sharpe:           {overall_best['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:     {overall_best['max_drawdown_pct']:.2f}%")
        print(f"Trades:           {int(overall_best['trades'])}")
        print(f"Score:            {overall_best['score']:.2f}")

        # Save best config
        best_config = {
            'run_id': run_id,
            'timeframe': overall_best['timeframe'],
            'parameters': {
                'sampling_period': int(overall_best['sampling_period']),
                'range_multiplier': float(overall_best['range_multiplier']),
                'stop_loss_pct': float(overall_best['stop_loss_pct']),
                'take_profit_pct': float(overall_best['take_profit_pct'])
            },
            'performance': {
                'return_pct': float(overall_best['return_pct']),
                'win_rate': float(overall_best['win_rate']),
                'profit_factor': float(overall_best['profit_factor']),
                'sharpe_ratio': float(overall_best['sharpe_ratio']),
                'max_drawdown_pct': float(overall_best['max_drawdown_pct']),
                'trades': int(overall_best['trades'])
            }
        }

        with open(run_dir / "best_config.json", 'w') as f:
            json.dump(best_config, f, indent=2)
        with open(RESULTS_DIR / "best_config_latest.json", 'w') as f:
            json.dump(best_config, f, indent=2)

    print(f"\nAll results saved to: {run_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Backtest trading strategy')
    parser.add_argument('--tf', nargs='+', default=ALL_TIMEFRAMES,
                        help='Timeframes to test (default: all)')
    parser.add_argument('--single', action='store_true',
                        help='Run single test instead of sweep')
    parser.add_argument('--sp', type=int, default=21,
                        help='Sampling period (for --single)')
    parser.add_argument('--rm', type=float, default=1.0,
                        help='Range multiplier (for --single)')
    parser.add_argument('--sl', type=float, default=2.0,
                        help='Stop loss %% (for --single)')
    parser.add_argument('--tp', type=float, default=3.0,
                        help='Take profit %% (for --single)')

    args = parser.parse_args()

    # Validate timeframes
    for tf in args.tf:
        if tf not in ALL_TIMEFRAMES:
            print(f"Invalid timeframe: {tf}")
            print(f"Available: {ALL_TIMEFRAMES}")
            return

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.single:
        # Single backtest
        tf = args.tf[0] if args.tf else '1h'
        print(f"Running single backtest on {tf}...")

        df = fetch_data(tf)
        if df is None:
            return

        result_dict, result_obj = run_single_backtest(
            df, tf, args.sp, args.rm, args.sl, args.tp
        )

        print_detailed_results(result_obj)
    else:
        # Parameter sweep
        run_sweep(args.tf)


if __name__ == "__main__":
    main()
