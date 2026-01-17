"""
Strategy Comparison Tool

Compare multiple strategy configurations side by side.
Run parameter sweeps and find optimal settings.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from itertools import product
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

from config_manager import (
    StrategyConfig, ConfigManager, create_preset_configs, print_config
)
from flexible_backtester import FlexibleBacktester, BacktestResult, print_detailed_results
from data_fetcher import fetch_gold_data


@dataclass
class ComparisonResult:
    """Result of comparing multiple strategies"""
    results: Dict[str, BacktestResult]
    comparison_df: pd.DataFrame
    best_by_return: str
    best_by_sharpe: str
    best_by_profit_factor: str
    best_by_win_rate: str


def compare_strategies(
    configs: List[StrategyConfig],
    df: pd.DataFrame,
    verbose: bool = True
) -> ComparisonResult:
    """
    Compare multiple strategy configurations

    Args:
        configs: List of strategy configurations to test
        df: OHLCV data
        verbose: Print progress

    Returns:
        ComparisonResult with all results
    """
    results = {}

    for i, config in enumerate(configs):
        if verbose:
            print(f"\n[{i+1}/{len(configs)}] Testing: {config.name}")

        backtester = FlexibleBacktester(config)
        result = backtester.run(df)
        results[config.name] = result

        if verbose:
            print(f"  Return: {result.total_return_pct:+.2f}%, "
                  f"Win Rate: {result.win_rate:.1f}%, "
                  f"Trades: {result.total_trades}")

    # Create comparison DataFrame
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Strategy': name,
            'Return %': result.total_return_pct,
            'Trades': result.total_trades,
            'Win Rate %': result.win_rate,
            'Profit Factor': result.profit_factor,
            'Avg Win $': result.avg_win,
            'Avg Loss $': result.avg_loss,
            'Expectancy R': result.expectancy_r,
            'Max DD %': result.max_drawdown_pct,
            'Sharpe': result.sharpe_ratio,
            'Final Capital': result.final_capital
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Find best strategies
    best_return = comparison_df.loc[comparison_df['Return %'].idxmax(), 'Strategy']
    best_sharpe = comparison_df.loc[comparison_df['Sharpe'].idxmax(), 'Strategy']
    best_pf = comparison_df.loc[comparison_df['Profit Factor'].idxmax(), 'Strategy']
    best_wr = comparison_df.loc[comparison_df['Win Rate %'].idxmax(), 'Strategy']

    return ComparisonResult(
        results=results,
        comparison_df=comparison_df,
        best_by_return=best_return,
        best_by_sharpe=best_sharpe,
        best_by_profit_factor=best_pf,
        best_by_win_rate=best_wr
    )


def print_comparison(comparison: ComparisonResult):
    """Print comparison results"""
    print("\n" + "="*100)
    print("STRATEGY COMPARISON")
    print("="*100)

    # Format the DataFrame for display
    df = comparison.comparison_df.copy()
    df['Return %'] = df['Return %'].apply(lambda x: f"{x:+.2f}%")
    df['Win Rate %'] = df['Win Rate %'].apply(lambda x: f"{x:.1f}%")
    df['Profit Factor'] = df['Profit Factor'].apply(lambda x: f"{x:.2f}")
    df['Avg Win $'] = df['Avg Win $'].apply(lambda x: f"${x:.2f}")
    df['Avg Loss $'] = df['Avg Loss $'].apply(lambda x: f"${x:.2f}")
    df['Expectancy R'] = df['Expectancy R'].apply(lambda x: f"{x:.2f}R")
    df['Max DD %'] = df['Max DD %'].apply(lambda x: f"{x:.2f}%")
    df['Sharpe'] = df['Sharpe'].apply(lambda x: f"{x:.2f}")
    df['Final Capital'] = df['Final Capital'].apply(lambda x: f"${x:,.2f}")

    print("\n" + df.to_string(index=False))

    print(f"\n{'='*50}")
    print("BEST PERFORMERS")
    print(f"{'='*50}")
    print(f"  Best Return:        {comparison.best_by_return}")
    print(f"  Best Sharpe:        {comparison.best_by_sharpe}")
    print(f"  Best Profit Factor: {comparison.best_by_profit_factor}")
    print(f"  Best Win Rate:      {comparison.best_by_win_rate}")


def parameter_sweep(
    base_config: StrategyConfig,
    df: pd.DataFrame,
    param_ranges: Dict[str, List],
    verbose: bool = True
) -> Tuple[pd.DataFrame, StrategyConfig]:
    """
    Sweep through parameter combinations to find optimal settings

    Args:
        base_config: Base configuration to modify
        df: OHLCV data
        param_ranges: Dict of parameter names to lists of values to test
                     Use dot notation for nested params: 'risk.stop_loss_value'
        verbose: Print progress

    Returns:
        Tuple of (results DataFrame, best config)

    Example:
        param_ranges = {
            'range_filter.sampling_period': [14, 21, 27, 35],
            'risk.stop_loss_value': [1.5, 2.0, 2.5],
            'risk.take_profit_value': [2.0, 3.0, 4.0],
        }
    """
    # Generate all combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(product(*param_values))

    total = len(combinations)
    if verbose:
        print(f"\nParameter Sweep: Testing {total} combinations")
        print(f"Parameters: {param_names}")

    results_data = []
    best_score = -float('inf')
    best_config = None

    for i, combo in enumerate(combinations):
        # Create config variant
        config = StrategyConfig.from_dict(base_config.to_dict())
        config.name = f"sweep_{i+1}"

        # Apply parameter values
        for param_name, value in zip(param_names, combo):
            parts = param_name.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        # Run backtest
        try:
            backtester = FlexibleBacktester(config)
            result = backtester.run(df)

            # Calculate score (weighted combination)
            score = (
                result.total_return_pct * 0.3 +
                result.profit_factor * 10 * 0.25 +
                result.sharpe_ratio * 5 * 0.25 +
                (result.win_rate - 50) * 0.1 +
                result.max_drawdown_pct * 0.1  # Negative, so penalties drawdown
            )

            # Store results
            row = {'combo_id': i+1, 'score': score}
            for param_name, value in zip(param_names, combo):
                row[param_name] = value

            row.update({
                'return_pct': result.total_return_pct,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'max_dd': result.max_drawdown_pct,
                'sharpe': result.sharpe_ratio,
                'expectancy_r': result.expectancy_r
            })

            results_data.append(row)

            if score > best_score and result.total_trades >= 10:
                best_score = score
                best_config = config

            if verbose and (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

        except Exception as e:
            if verbose:
                print(f"  Error on combo {i+1}: {e}")

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('score', ascending=False)

    if verbose:
        print(f"\nSweep complete. Best score: {best_score:.2f}")

    return results_df, best_config


def print_sweep_results(results_df: pd.DataFrame, top_n: int = 20):
    """Print parameter sweep results"""
    print(f"\n{'='*120}")
    print(f"PARAMETER SWEEP RESULTS (Top {top_n})")
    print(f"{'='*120}")

    # Get parameter columns
    param_cols = [c for c in results_df.columns if c not in
                  ['combo_id', 'score', 'return_pct', 'trades', 'win_rate',
                   'profit_factor', 'max_dd', 'sharpe', 'expectancy_r']]

    # Format display
    display_df = results_df.head(top_n).copy()
    display_df['return_pct'] = display_df['return_pct'].apply(lambda x: f"{x:+.2f}%")
    display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%")
    display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f"{x:.2f}")
    display_df['max_dd'] = display_df['max_dd'].apply(lambda x: f"{x:.2f}%")
    display_df['sharpe'] = display_df['sharpe'].apply(lambda x: f"{x:.2f}")
    display_df['expectancy_r'] = display_df['expectancy_r'].apply(lambda x: f"{x:.2f}R")
    display_df['score'] = display_df['score'].apply(lambda x: f"{x:.2f}")

    # Select columns to display
    display_cols = param_cols + ['return_pct', 'win_rate', 'profit_factor', 'sharpe', 'trades', 'score']
    print("\n" + display_df[display_cols].to_string(index=False))


def run_preset_comparison(df: pd.DataFrame) -> ComparisonResult:
    """Run comparison using preset configurations"""
    presets = create_preset_configs()
    configs = list(presets.values())

    print(f"\nComparing {len(configs)} preset configurations...")
    return compare_strategies(configs, df)


def run_quick_optimization(df: pd.DataFrame) -> Tuple[pd.DataFrame, StrategyConfig]:
    """Run a quick parameter optimization"""
    base_config = StrategyConfig(name='base')

    param_ranges = {
        'range_filter.sampling_period': [14, 21, 27, 35],
        'range_filter.range_multiplier': [0.75, 1.0, 1.5],
        'risk.stop_loss_value': [1.5, 2.0, 2.5],
        'risk.take_profit_value': [2.0, 3.0, 4.0],
        'confirmation.min_bars_between_trades': [0, 3, 5],
    }

    return parameter_sweep(base_config, df, param_ranges)


def main():
    """Main comparison runner"""
    print("="*70)
    print("STRATEGY COMPARISON & OPTIMIZATION TOOL")
    print("="*70)

    # Fetch data from TradingView (with caching)
    print("\nFetching gold data from TradingView...")
    try:
        df = fetch_gold_data(
            source='tradingview',
            interval='1h',
            n_bars=5000,
            fetch_latest=True  # Get latest candles and merge with cache
        )
    except Exception as e:
        print(f"TradingView failed: {e}")
        print("Falling back to Yahoo Finance...")
        try:
            df = fetch_gold_data(source='yahoo', interval='1h')
        except Exception as e2:
            print(f"Error fetching data: {e2}")
            return

    print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Compare presets
    print("\n" + "="*70)
    print("PHASE 1: COMPARING PRESET STRATEGIES")
    print("="*70)

    comparison = run_preset_comparison(df)
    print_comparison(comparison)

    # Run parameter sweep
    print("\n" + "="*70)
    print("PHASE 2: PARAMETER OPTIMIZATION")
    print("="*70)

    sweep_results, best_config = run_quick_optimization(df)
    print_sweep_results(sweep_results)

    if best_config:
        print(f"\n{'='*50}")
        print("BEST CONFIGURATION FOUND")
        print(f"{'='*50}")
        print_config(best_config)

        # Run detailed backtest on best config
        print(f"\n{'='*50}")
        print("DETAILED RESULTS FOR BEST CONFIG")
        print(f"{'='*50}")

        backtester = FlexibleBacktester(best_config)
        best_result = backtester.run(df)
        print_detailed_results(best_result)

    return comparison, sweep_results, best_config


if __name__ == "__main__":
    comparison, sweep_results, best_config = main()
