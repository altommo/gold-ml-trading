"""
Backtesting Engine for Range Filter Strategy

Features:
- Fetch historical data from Yahoo Finance
- Run backtests with configurable parameters
- Calculate performance metrics
- Generate trade logs and statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import yfinance as yf

from range_filter_strategy import apply_strategy, StrategyConfig


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 10000.0  # Starting capital
    position_size_pct: float = 0.1  # Risk 10% of capital per trade
    commission_pct: float = 0.0  # Commission per trade (Plus500 is spread-based)
    spread_pct: float = 0.03  # Spread cost (0.03% typical for gold)
    use_stop_loss: bool = True
    stop_loss_pct: float = 2.0  # Stop loss percentage
    use_take_profit: bool = True
    take_profit_pct: float = 3.0  # Take profit percentage
    allow_shorting: bool = True  # Allow short positions


@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'signal', 'stop_loss', 'take_profit'
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    final_capital: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration: timedelta = field(default_factory=timedelta)


def fetch_gold_data(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch gold (XAU/USD) historical data from Yahoo Finance

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date (defaults to today)
        interval: Data interval ('1d', '1h', '15m', '5m', etc.)

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # GC=F is Gold Futures, GLD is Gold ETF
    # For spot gold, we'll use GC=F as proxy
    print(f"Fetching gold data from {start_date} to {end_date}...")

    ticker = yf.Ticker("GC=F")
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError("No data fetched. Check your date range and internet connection.")

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    print(f"Fetched {len(df)} bars of data")
    return df


def fetch_data(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical data for any Yahoo Finance symbol

    Args:
        symbol: Yahoo Finance symbol (e.g., 'GC=F' for gold, 'SI=F' for silver)
        start_date: Start date
        end_date: End date
        interval: Data interval

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching {symbol} data from {start_date} to {end_date}...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    df.columns = [c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    print(f"Fetched {len(df)} bars")
    return df


class Backtester:
    """
    Backtesting engine for the Range Filter strategy
    """

    def __init__(
        self,
        strategy_config: Optional[StrategyConfig] = None,
        backtest_config: Optional[BacktestConfig] = None
    ):
        self.strategy_config = strategy_config or StrategyConfig()
        self.backtest_config = backtest_config or BacktestConfig()

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            BacktestResult with all metrics and trade log
        """
        # Apply strategy to get signals
        data = apply_strategy(df, self.strategy_config)

        result = BacktestResult()
        config = self.backtest_config

        capital = config.initial_capital
        position = None  # Current position (Trade object or None)
        equity = []
        trades = []

        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]

            # Check for exit conditions if in position
            if position is not None:
                exit_triggered = False
                exit_price = current_price
                exit_reason = None

                if position.direction == 'long':
                    # Check stop loss (hit low of bar)
                    if config.use_stop_loss:
                        stop_price = position.entry_price * (1 - config.stop_loss_pct / 100)
                        if low_price <= stop_price:
                            exit_price = stop_price
                            exit_reason = 'stop_loss'
                            exit_triggered = True

                    # Check take profit (hit high of bar)
                    if not exit_triggered and config.use_take_profit:
                        tp_price = position.entry_price * (1 + config.take_profit_pct / 100)
                        if high_price >= tp_price:
                            exit_price = tp_price
                            exit_reason = 'take_profit'
                            exit_triggered = True

                    # Check for signal reversal
                    if not exit_triggered and data['short_signal'].iloc[i]:
                        exit_reason = 'signal'
                        exit_triggered = True

                else:  # short position
                    # Check stop loss
                    if config.use_stop_loss:
                        stop_price = position.entry_price * (1 + config.stop_loss_pct / 100)
                        if high_price >= stop_price:
                            exit_price = stop_price
                            exit_reason = 'stop_loss'
                            exit_triggered = True

                    # Check take profit
                    if not exit_triggered and config.use_take_profit:
                        tp_price = position.entry_price * (1 - config.take_profit_pct / 100)
                        if low_price <= tp_price:
                            exit_price = tp_price
                            exit_reason = 'take_profit'
                            exit_triggered = True

                    # Check for signal reversal
                    if not exit_triggered and data['long_signal'].iloc[i]:
                        exit_reason = 'signal'
                        exit_triggered = True

                # Close position if exit triggered
                if exit_triggered:
                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.exit_reason = exit_reason

                    # Calculate P&L
                    if position.direction == 'long':
                        gross_pnl = (exit_price - position.entry_price) * position.size
                    else:
                        gross_pnl = (position.entry_price - exit_price) * position.size

                    # Deduct spread costs
                    spread_cost = (position.entry_price + exit_price) * position.size * (config.spread_pct / 100)
                    position.pnl = gross_pnl - spread_cost
                    position.pnl_pct = (position.pnl / (position.entry_price * position.size)) * 100

                    capital += position.pnl
                    trades.append(position)
                    position = None

            # Check for entry signals (only if not in position)
            if position is None:
                if data['long_signal'].iloc[i]:
                    # Calculate position size based on capital
                    risk_capital = capital * config.position_size_pct
                    size = risk_capital / current_price

                    position = Trade(
                        entry_time=current_time,
                        entry_price=current_price,
                        direction='long',
                        size=size
                    )

                elif data['short_signal'].iloc[i] and config.allow_shorting:
                    risk_capital = capital * config.position_size_pct
                    size = risk_capital / current_price

                    position = Trade(
                        entry_time=current_time,
                        entry_price=current_price,
                        direction='short',
                        size=size
                    )

            # Track equity
            current_equity = capital
            if position is not None:
                # Mark to market
                if position.direction == 'long':
                    unrealized = (current_price - position.entry_price) * position.size
                else:
                    unrealized = (position.entry_price - current_price) * position.size
                current_equity = capital + unrealized

            equity.append({'time': current_time, 'equity': current_equity})

        # Close any remaining position at end
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
        result.trades = trades
        result.equity_curve = pd.DataFrame(equity).set_index('time')['equity']
        result.final_capital = capital
        result.total_return_pct = ((capital - config.initial_capital) / config.initial_capital) * 100
        result.total_trades = len(trades)

        if trades:
            winning = [t for t in trades if t.pnl > 0]
            losing = [t for t in trades if t.pnl <= 0]

            result.winning_trades = len(winning)
            result.losing_trades = len(losing)
            result.win_rate = (len(winning) / len(trades)) * 100 if trades else 0

            result.avg_win = np.mean([t.pnl for t in winning]) if winning else 0
            result.avg_loss = np.mean([t.pnl for t in losing]) if losing else 0

            gross_profit = sum(t.pnl for t in winning)
            gross_loss = abs(sum(t.pnl for t in losing))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Max drawdown
            peak = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - peak) / peak * 100
            result.max_drawdown_pct = drawdown.min()

            # Sharpe ratio (assuming daily returns, annualized)
            returns = result.equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

            # Average trade duration
            durations = [(t.exit_time - t.entry_time) for t in trades if t.exit_time is not None]
            if durations:
                result.avg_trade_duration = sum(durations, timedelta()) / len(durations)

        return result


def print_results(result: BacktestResult, config: BacktestConfig):
    """Print formatted backtest results"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\nCapital:")
    print(f"  Initial:     ${config.initial_capital:,.2f}")
    print(f"  Final:       ${result.final_capital:,.2f}")
    print(f"  Return:      {result.total_return_pct:+.2f}%")

    print(f"\nTrades:")
    print(f"  Total:       {result.total_trades}")
    print(f"  Winners:     {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"  Losers:      {result.losing_trades}")

    print(f"\nPerformance:")
    print(f"  Avg Win:     ${result.avg_win:,.2f}")
    print(f"  Avg Loss:    ${result.avg_loss:,.2f}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown:  {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")

    print(f"\nTiming:")
    print(f"  Avg Duration:  {result.avg_trade_duration}")

    print("\n" + "="*60)


def print_trade_log(result: BacktestResult, limit: int = 20):
    """Print trade log"""
    print(f"\nLast {min(limit, len(result.trades))} trades:")
    print("-"*100)
    print(f"{'Entry Time':<20} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'P&L%':>8} {'Reason':<12}")
    print("-"*100)

    for trade in result.trades[-limit:]:
        entry_str = trade.entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(trade.entry_time, 'strftime') else str(trade.entry_time)[:16]
        exit_price = trade.exit_price if trade.exit_price else 0
        pnl_color = '+' if trade.pnl > 0 else ''
        print(f"{entry_str:<20} {trade.direction:<6} {trade.entry_price:>10.2f} {exit_price:>10.2f} {pnl_color}{trade.pnl:>11.2f} {trade.pnl_pct:>7.2f}% {trade.exit_reason or '':<12}")


if __name__ == "__main__":
    # Run a quick backtest on gold
    print("Range Filter Strategy Backtester")
    print("-" * 40)

    # Fetch gold data
    df = fetch_gold_data(start_date="2022-01-01")

    # Configure strategy
    strategy_config = StrategyConfig(
        sampling_period=27,
        range_multiplier=1.0,
        use_heikin_ashi=False
    )

    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=10000,
        position_size_pct=0.1,
        spread_pct=0.03,
        stop_loss_pct=2.0,
        take_profit_pct=3.0,
        allow_shorting=True
    )

    # Run backtest
    backtester = Backtester(strategy_config, backtest_config)
    result = backtester.run(df)

    # Print results
    print_results(result, backtest_config)
    print_trade_log(result)
