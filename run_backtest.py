"""
Main Backtest Runner

Run backtests on the combined strategy with different configurations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')

from data_fetcher import fetch_gold_data, fetch_from_tradingview, fetch_from_yahoo
from combined_strategy import (
    calculate_combined_signals, CombinedStrategyConfig,
    get_signals_list, print_signal_summary
)


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1  # Risk 10% per trade
    spread_pct: float = 0.03  # Spread cost
    use_stop_loss: bool = True
    stop_loss_pct: float = 2.0
    use_take_profit: bool = True
    take_profit_pct: float = 3.0
    allow_shorting: bool = True


@dataclass
class Trade:
    """Trade record"""
    entry_time: datetime
    entry_price: float
    direction: str
    size: float
    signal_strength: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


class CombinedBacktester:
    """Backtester for the combined strategy"""

    def __init__(
        self,
        strategy_config: Optional[CombinedStrategyConfig] = None,
        backtest_config: Optional[BacktestConfig] = None
    ):
        self.strategy_config = strategy_config or CombinedStrategyConfig()
        self.backtest_config = backtest_config or BacktestConfig()

    def run(self, df: pd.DataFrame) -> dict:
        """
        Run backtest on data

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary with results
        """
        # Calculate all signals
        data = calculate_combined_signals(df, self.strategy_config)

        config = self.backtest_config
        capital = config.initial_capital
        position = None
        trades = []
        equity = []

        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]

            # Check for exit if in position
            if position is not None:
                exit_triggered = False
                exit_price = current_price
                exit_reason = None

                if position.direction == 'long':
                    # Stop loss
                    if config.use_stop_loss:
                        stop_price = position.entry_price * (1 - config.stop_loss_pct / 100)
                        if low_price <= stop_price:
                            exit_price = stop_price
                            exit_reason = 'stop_loss'
                            exit_triggered = True

                    # Take profit
                    if not exit_triggered and config.use_take_profit:
                        tp_price = position.entry_price * (1 + config.take_profit_pct / 100)
                        if high_price >= tp_price:
                            exit_price = tp_price
                            exit_reason = 'take_profit'
                            exit_triggered = True

                    # Signal reversal
                    if not exit_triggered and data['combined_short_signal'].iloc[i]:
                        exit_reason = 'signal'
                        exit_triggered = True

                else:  # short
                    # Stop loss
                    if config.use_stop_loss:
                        stop_price = position.entry_price * (1 + config.stop_loss_pct / 100)
                        if high_price >= stop_price:
                            exit_price = stop_price
                            exit_reason = 'stop_loss'
                            exit_triggered = True

                    # Take profit
                    if not exit_triggered and config.use_take_profit:
                        tp_price = position.entry_price * (1 - config.take_profit_pct / 100)
                        if low_price <= tp_price:
                            exit_price = tp_price
                            exit_reason = 'take_profit'
                            exit_triggered = True

                    # Signal reversal
                    if not exit_triggered and data['combined_long_signal'].iloc[i]:
                        exit_reason = 'signal'
                        exit_triggered = True

                # Close position
                if exit_triggered:
                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason

                    if position.direction == 'long':
                        gross_pnl = (exit_price - position.entry_price) * position.size
                    else:
                        gross_pnl = (position.entry_price - exit_price) * position.size

                    spread_cost = (position.entry_price + exit_price) * position.size * (config.spread_pct / 100)
                    position.pnl = gross_pnl - spread_cost
                    position.pnl_pct = (position.pnl / (position.entry_price * position.size)) * 100

                    capital += position.pnl
                    trades.append(position)
                    position = None

            # Check for entry
            if position is None:
                signal_strength = int(data['signal_strength'].iloc[i])

                if data['combined_long_signal'].iloc[i]:
                    risk_capital = capital * config.position_size_pct
                    size = risk_capital / current_price

                    position = Trade(
                        entry_time=current_time,
                        entry_price=current_price,
                        direction='long',
                        size=size,
                        signal_strength=signal_strength
                    )

                elif data['combined_short_signal'].iloc[i] and config.allow_shorting:
                    risk_capital = capital * config.position_size_pct
                    size = risk_capital / current_price

                    position = Trade(
                        entry_time=current_time,
                        entry_price=current_price,
                        direction='short',
                        size=size,
                        signal_strength=signal_strength
                    )

            # Track equity
            current_equity = capital
            if position is not None:
                if position.direction == 'long':
                    unrealized = (current_price - position.entry_price) * position.size
                else:
                    unrealized = (position.entry_price - current_price) * position.size
                current_equity = capital + unrealized

            equity.append({'time': current_time, 'equity': current_equity})

        # Close remaining position
        if position is not None:
            position.exit_time = data.index[-1]
            position.exit_price = data['close'].iloc[-1]
            position.exit_reason = 'end_of_data'

            if position.direction == 'long':
                gross_pnl = (position.exit_price - position.entry_price) * position.size
            else:
                gross_pnl = (position.entry_price - position.exit_price) * position.size

            spread_cost = (position.entry_price + position.exit_price) * position.size * (config.spread_pct / 100)
            position.pnl = gross_pnl - spread_cost
            position.pnl_pct = (position.pnl / (position.entry_price * position.size)) * 100

            capital += position.pnl
            trades.append(position)

        # Calculate metrics
        equity_curve = pd.DataFrame(equity).set_index('time')['equity'] if equity else pd.Series()

        results = self._calculate_metrics(trades, equity_curve, config.initial_capital, capital)
        results['trades'] = trades
        results['equity_curve'] = equity_curve
        results['data'] = data

        return results

    def _calculate_metrics(self, trades: List[Trade], equity_curve: pd.Series,
                          initial_capital: float, final_capital: float) -> dict:
        """Calculate performance metrics"""
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': ((final_capital - initial_capital) / initial_capital) * 100,
            'total_trades': len(trades),
        }

        if not trades:
            return results

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        results['winning_trades'] = len(winning)
        results['losing_trades'] = len(losing)
        results['win_rate'] = (len(winning) / len(trades)) * 100 if trades else 0
        results['avg_win'] = np.mean([t.pnl for t in winning]) if winning else 0
        results['avg_loss'] = np.mean([t.pnl for t in losing]) if losing else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        results['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        if len(equity_curve) > 0:
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve - peak) / peak * 100
            results['max_drawdown_pct'] = drawdown.min()

            # Sharpe ratio
            returns = equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                results['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                results['sharpe_ratio'] = 0
        else:
            results['max_drawdown_pct'] = 0
            results['sharpe_ratio'] = 0

        # By signal strength
        for strength in [1, 2, 3]:
            strength_trades = [t for t in trades if t.signal_strength == strength]
            if strength_trades:
                strength_winning = [t for t in strength_trades if t.pnl > 0]
                results[f'strength_{strength}_trades'] = len(strength_trades)
                results[f'strength_{strength}_win_rate'] = (len(strength_winning) / len(strength_trades)) * 100

        return results


def print_results(results: dict, config: BacktestConfig):
    """Print formatted results"""
    print("\n" + "="*70)
    print("BACKTEST RESULTS - COMBINED STRATEGY")
    print("="*70)

    print(f"\nCapital:")
    print(f"  Initial:     ${results['initial_capital']:,.2f}")
    print(f"  Final:       ${results['final_capital']:,.2f}")
    print(f"  Return:      {results['total_return_pct']:+.2f}%")

    print(f"\nTrades:")
    print(f"  Total:       {results['total_trades']}")
    print(f"  Winners:     {results.get('winning_trades', 0)} ({results.get('win_rate', 0):.1f}%)")
    print(f"  Losers:      {results.get('losing_trades', 0)}")

    print(f"\nPerformance:")
    print(f"  Avg Win:     ${results.get('avg_win', 0):,.2f}")
    print(f"  Avg Loss:    ${results.get('avg_loss', 0):,.2f}")
    print(f"  Profit Factor: {results.get('profit_factor', 0):.2f}")

    print(f"\nRisk:")
    print(f"  Max Drawdown:  {results.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Sharpe Ratio:  {results.get('sharpe_ratio', 0):.2f}")

    # By signal strength
    print(f"\nBy Signal Strength:")
    for strength in [1, 2, 3]:
        count = results.get(f'strength_{strength}_trades', 0)
        wr = results.get(f'strength_{strength}_win_rate', 0)
        if count > 0:
            stars = '*' * strength
            print(f"  Strength {strength} ({stars}): {count} trades, {wr:.1f}% win rate")

    print("="*70)


def print_trade_log(trades: List[Trade], limit: int = 20):
    """Print recent trades"""
    print(f"\nRecent Trades (last {min(limit, len(trades))}):")
    print("-"*110)
    print(f"{'Entry Time':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'P&L%':>8} {'Str':>4} {'Reason':<12}")
    print("-"*110)

    for trade in trades[-limit:]:
        entry_str = str(trade.entry_time)[:16]
        exit_price = trade.exit_price if trade.exit_price else 0
        pnl_sign = '+' if trade.pnl > 0 else ''
        print(f"{entry_str:<20} {trade.direction:<6} {trade.entry_price:>10.2f} {exit_price:>10.2f} "
              f"{pnl_sign}{trade.pnl:>11.2f} {trade.pnl_pct:>7.2f}% {trade.signal_strength:>4} {trade.exit_reason or '':<12}")


def main():
    """Main function to run backtest"""
    print("="*70)
    print("COMBINED STRATEGY BACKTEST")
    print("Potato Signal + Wolfpack ID + WaveTrend Confirmation")
    print("="*70)

    # Try to fetch data
    print("\nFetching gold data...")

    try:
        # Try TradingView first
        df = fetch_gold_data(source='tradingview', interval='1h', n_bars=5000)
    except Exception as e:
        print(f"TradingView failed: {e}")
        print("Falling back to Yahoo Finance...")
        try:
            df = fetch_gold_data(source='yahoo', interval='1h')
        except Exception as e2:
            print(f"Yahoo also failed: {e2}")
            print("\nPlease install tvdatafeed:")
            print("  pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git")
            return

    print(f"\nData loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Configure strategy
    strategy_config = CombinedStrategyConfig(
        # Range Filter settings
        rf_sampling_period=27,
        rf_range_multiplier=1.0,
        rf_source='close',  # Can be 'close', 'wolfpack', or 'wavetrend'

        # Confirmation settings
        require_wolfpack_confirm=True,
        require_wavetrend_confirm=True,
        wt_confirm_mode='trend',

        # WaveTrend settings (optimized for 4H per VuManChu)
        wt_channel_length=9,
        wt_average_length=12,
    )

    backtest_config = BacktestConfig(
        initial_capital=10000,
        position_size_pct=0.1,
        spread_pct=0.03,  # Plus500 typical spread
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        allow_shorting=True
    )

    # Run backtest
    print("\nRunning backtest...")
    backtester = CombinedBacktester(strategy_config, backtest_config)
    results = backtester.run(df)

    # Print signal summary
    print_signal_summary(results['data'])

    # Print results
    print_results(results, backtest_config)

    # Print trade log
    if results['trades']:
        print_trade_log(results['trades'])

    return results


if __name__ == "__main__":
    results = main()
