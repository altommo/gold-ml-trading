"""
Parameter Optimizer for Range Filter Strategy

Tests different parameter combinations to find optimal settings for gold trading.
"""

import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass
from typing import List, Dict, Tuple
import concurrent.futures
import warnings

from range_filter_strategy import StrategyConfig
from backtester import Backtester, BacktestConfig, BacktestResult, fetch_gold_data

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Result from a single parameter combination test"""
    sampling_period: int
    range_multiplier: float
    stop_loss_pct: float
    take_profit_pct: float
    use_heikin_ashi: bool
    total_return: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    score: float  # Combined optimization score


def calculate_score(result: BacktestResult, min_trades: int = 20) -> float:
    """
    Calculate optimization score combining multiple metrics

    Higher is better. Penalizes:
    - Low trade count (might be overfitting)
    - Large drawdowns
    - Low win rates
    """
    if result.total_trades < min_trades:
        return -999  # Not enough trades to be reliable

    # Weighted scoring
    # - Return is important but can be misleading
    # - Profit factor shows edge
    # - Sharpe shows risk-adjusted return
    # - Win rate affects psychology

    score = (
        result.total_return_pct * 0.3 +  # 30% weight on returns
        (result.profit_factor - 1) * 50 * 0.25 +  # 25% weight on profit factor
        result.sharpe_ratio * 10 * 0.25 +  # 25% weight on Sharpe
        (result.win_rate - 50) * 0.1 +  # 10% weight on win rate above 50%
        result.max_drawdown_pct * 0.1  # 10% penalty for drawdown (negative value)
    )

    return score


def test_parameters(
    df: pd.DataFrame,
    sampling_period: int,
    range_multiplier: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    use_heikin_ashi: bool,
    backtest_config: BacktestConfig
) -> OptimizationResult:
    """Test a single parameter combination"""

    strategy_config = StrategyConfig(
        sampling_period=sampling_period,
        range_multiplier=range_multiplier,
        use_heikin_ashi=use_heikin_ashi
    )

    # Update backtest config with SL/TP
    bc = BacktestConfig(
        initial_capital=backtest_config.initial_capital,
        position_size_pct=backtest_config.position_size_pct,
        spread_pct=backtest_config.spread_pct,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        use_stop_loss=True,
        use_take_profit=True,
        allow_shorting=backtest_config.allow_shorting
    )

    backtester = Backtester(strategy_config, bc)
    result = backtester.run(df)
    score = calculate_score(result)

    return OptimizationResult(
        sampling_period=sampling_period,
        range_multiplier=range_multiplier,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        use_heikin_ashi=use_heikin_ashi,
        total_return=result.total_return_pct,
        win_rate=result.win_rate,
        profit_factor=result.profit_factor,
        max_drawdown=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        total_trades=result.total_trades,
        score=score
    )


def optimize(
    df: pd.DataFrame,
    sampling_periods: List[int] = None,
    range_multipliers: List[float] = None,
    stop_losses: List[float] = None,
    take_profits: List[float] = None,
    test_heikin_ashi: bool = True
) -> List[OptimizationResult]:
    """
    Run parameter optimization

    Args:
        df: OHLCV data
        sampling_periods: List of periods to test
        range_multipliers: List of multipliers to test
        stop_losses: List of stop loss percentages
        take_profits: List of take profit percentages
        test_heikin_ashi: Whether to test with HA candles

    Returns:
        Sorted list of optimization results (best first)
    """
    # Default parameter ranges
    if sampling_periods is None:
        sampling_periods = [10, 14, 20, 27, 35, 50]
    if range_multipliers is None:
        range_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    if stop_losses is None:
        stop_losses = [1.0, 1.5, 2.0, 2.5, 3.0]
    if take_profits is None:
        take_profits = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    ha_options = [False, True] if test_heikin_ashi else [False]

    # Generate all combinations
    combinations = list(product(
        sampling_periods,
        range_multipliers,
        stop_losses,
        take_profits,
        ha_options
    ))

    total = len(combinations)
    print(f"Testing {total} parameter combinations...")

    backtest_config = BacktestConfig(
        initial_capital=10000,
        position_size_pct=0.1,
        spread_pct=0.03,
        allow_shorting=True
    )

    results = []
    for i, (period, mult, sl, tp, ha) in enumerate(combinations):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")

        try:
            result = test_parameters(df, period, mult, sl, tp, ha, backtest_config)
            results.append(result)
        except Exception as e:
            continue

    # Sort by score (descending)
    results.sort(key=lambda x: x.score, reverse=True)

    return results


def print_optimization_results(results: List[OptimizationResult], top_n: int = 20):
    """Print top optimization results"""
    print("\n" + "="*120)
    print("TOP PARAMETER COMBINATIONS")
    print("="*120)
    print(f"{'Period':>6} {'Mult':>6} {'SL%':>6} {'TP%':>6} {'HA':>4} {'Return':>10} {'WinRate':>8} {'PF':>6} {'MaxDD':>8} {'Sharpe':>7} {'Trades':>6} {'Score':>8}")
    print("-"*120)

    for r in results[:top_n]:
        ha_str = 'Yes' if r.use_heikin_ashi else 'No'
        print(f"{r.sampling_period:>6} {r.range_multiplier:>6.2f} {r.stop_loss_pct:>6.1f} {r.take_profit_pct:>6.1f} {ha_str:>4} "
              f"{r.total_return:>9.2f}% {r.win_rate:>7.1f}% {r.profit_factor:>6.2f} {r.max_drawdown:>7.2f}% "
              f"{r.sharpe_ratio:>7.2f} {r.total_trades:>6} {r.score:>8.2f}")


def run_quick_optimization(df: pd.DataFrame) -> List[OptimizationResult]:
    """Run a quick optimization with fewer parameters"""
    print("Running QUICK optimization (fewer parameters)...")
    return optimize(
        df,
        sampling_periods=[14, 21, 27, 35],
        range_multipliers=[0.75, 1.0, 1.5, 2.0],
        stop_losses=[1.5, 2.0, 2.5],
        take_profits=[2.0, 3.0, 4.0],
        test_heikin_ashi=True
    )


def run_full_optimization(df: pd.DataFrame) -> List[OptimizationResult]:
    """Run a comprehensive optimization"""
    print("Running FULL optimization (this may take a while)...")
    return optimize(
        df,
        sampling_periods=[7, 10, 14, 18, 21, 27, 35, 50, 70],
        range_multipliers=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
        stop_losses=[0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        take_profits=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        test_heikin_ashi=True
    )


if __name__ == "__main__":
    print("Range Filter Strategy Optimizer for Gold")
    print("="*50)

    # Fetch gold data
    df = fetch_gold_data(start_date="2020-01-01")

    # Run optimization
    results = run_quick_optimization(df)

    # Print results
    print_optimization_results(results, top_n=25)

    # Show best parameters
    if results and results[0].score > -999:
        best = results[0]
        print("\n" + "="*50)
        print("RECOMMENDED PARAMETERS FOR GOLD:")
        print("="*50)
        print(f"  Sampling Period:   {best.sampling_period}")
        print(f"  Range Multiplier:  {best.range_multiplier}")
        print(f"  Stop Loss:         {best.stop_loss_pct}%")
        print(f"  Take Profit:       {best.take_profit_pct}%")
        print(f"  Use Heikin Ashi:   {best.use_heikin_ashi}")
        print(f"\n  Expected Return:   {best.total_return:.2f}%")
        print(f"  Win Rate:          {best.win_rate:.1f}%")
        print(f"  Profit Factor:     {best.profit_factor:.2f}")
        print(f"  Max Drawdown:      {best.max_drawdown:.2f}%")
