"""
Flexible Backtester

Properly tracks trades with their SL/TP levels.
Supports multiple configurations and comparison.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from config_manager import StrategyConfig, RiskConfig
from data_fetcher import fetch_gold_data
from indicators import (
    calculate_wolfpack_id, calculate_wavetrend, calculate_money_flow, calculate_rsi,
    WolfpackConfig, WaveTrendConfig, MoneyFlowConfig
)
from range_filter_strategy import apply_strategy, StrategyConfig as RFStrategyConfig


@dataclass
class TradeRecord:
    """Detailed trade record with all levels"""
    id: int
    entry_time: datetime
    entry_price: float
    direction: str  # 'long' or 'short'
    size: float
    capital_at_entry: float

    # Levels calculated at entry
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: Optional[float] = None

    # Signal info
    signal_strength: int = 0
    wolfpack_value: float = 0.0
    wavetrend_value: float = 0.0
    wolfpack_confirmed: bool = False
    wavetrend_confirmed: bool = False

    # Exit info
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # P&L
    gross_pnl: float = 0.0
    spread_cost: float = 0.0
    net_pnl: float = 0.0
    pnl_pct: float = 0.0
    r_multiple: float = 0.0  # P&L in terms of risk units

    # Duration
    bars_held: int = 0

    def calculate_exit(self, exit_price: float, exit_time: datetime,
                       exit_reason: str, spread_pct: float, bars: int):
        """Calculate exit metrics"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.bars_held = bars

        if self.direction == 'long':
            self.gross_pnl = (exit_price - self.entry_price) * self.size
            risk_per_unit = self.entry_price - self.stop_loss_price
        else:
            self.gross_pnl = (self.entry_price - exit_price) * self.size
            risk_per_unit = self.stop_loss_price - self.entry_price

        self.spread_cost = (self.entry_price + exit_price) * self.size * (spread_pct / 100)
        self.net_pnl = self.gross_pnl - self.spread_cost

        trade_value = self.entry_price * self.size
        self.pnl_pct = (self.net_pnl / trade_value) * 100 if trade_value > 0 else 0

        # R-multiple: how many R did we make/lose
        risk_amount = risk_per_unit * self.size
        self.r_multiple = self.net_pnl / risk_amount if risk_amount > 0 else 0


@dataclass
class BacktestResult:
    """Complete backtest results"""
    config_name: str
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # Summary stats
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return_pct: float = 0.0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    expectancy_r: float = 0.0

    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    avg_bars_held: float = 0.0
    avg_bars_winner: float = 0.0
    avg_bars_loser: float = 0.0

    # By exit reason
    exits_by_reason: Dict[str, int] = field(default_factory=dict)

    # By signal strength
    stats_by_strength: Dict[int, Dict] = field(default_factory=dict)


class FlexibleBacktester:
    """
    Flexible backtester with proper trade tracking
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.trade_counter = 0

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        result = df.copy()

        # Wolfpack ID
        wolfpack_config = WolfpackConfig(
            fast_length=self.config.indicators.wolfpack_fast,
            slow_length=self.config.indicators.wolfpack_slow
        )
        result = calculate_wolfpack_id(result, wolfpack_config)

        # WaveTrend
        wavetrend_config = WaveTrendConfig(
            channel_length=self.config.indicators.wt_channel_length,
            average_length=self.config.indicators.wt_average_length,
            ma_length=self.config.indicators.wt_ma_length,
            ob_level_1=self.config.indicators.wt_overbought,
            os_level_1=self.config.indicators.wt_oversold
        )
        result = calculate_wavetrend(result, wavetrend_config)

        # Money Flow (if needed)
        if self.config.confirmation.require_money_flow:
            mf_config = MoneyFlowConfig(
                period=self.config.indicators.mfi_period,
                multiplier=self.config.indicators.mfi_multiplier
            )
            result = calculate_money_flow(result, mf_config)

        # RSI (if needed)
        if self.config.confirmation.require_rsi:
            result = calculate_rsi(result, self.config.indicators.rsi_period)

        # Range Filter
        rf_config = RFStrategyConfig(
            sampling_period=self.config.range_filter.sampling_period,
            range_multiplier=self.config.range_filter.range_multiplier,
            source_type=self.config.range_filter.source_type,
            use_heikin_ashi=self.config.range_filter.use_heikin_ashi
        )
        result = apply_strategy(result, rf_config)

        return result

    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate combined signals with confirmations"""
        result = df.copy()
        conf = self.config.confirmation

        # Wolfpack confirmation
        result['wolfpack_long_confirm'] = result['wolfpack_bullish']
        result['wolfpack_short_confirm'] = result['wolfpack_bearish']

        # WaveTrend confirmation
        if conf.wt_mode == 'trend':
            result['wt_long_confirm'] = result['wt_bullish'] | (result['wt1'] > result['wt1'].shift(1))
            result['wt_short_confirm'] = result['wt_bearish'] | (result['wt1'] < result['wt1'].shift(1))
        elif conf.wt_mode == 'cross':
            result['wt_long_confirm'] = (
                result['wt_cross_up'] |
                result['wt_cross_up'].shift(1).fillna(False) |
                result['wt_cross_up'].shift(2).fillna(False)
            )
            result['wt_short_confirm'] = (
                result['wt_cross_down'] |
                result['wt_cross_down'].shift(1).fillna(False) |
                result['wt_cross_down'].shift(2).fillna(False)
            )
        else:  # level
            result['wt_long_confirm'] = ~result['wt_overbought']
            result['wt_short_confirm'] = ~result['wt_oversold']

        # Calculate signal strength
        result['signal_strength'] = 0

        # Long signals
        long_mask = result['long_signal'].fillna(False)
        result.loc[long_mask, 'signal_strength'] = 1
        result.loc[long_mask & result['wolfpack_long_confirm'], 'signal_strength'] += 1
        result.loc[long_mask & result['wt_long_confirm'], 'signal_strength'] += 1

        # Short signals
        short_mask = result['short_signal'].fillna(False)
        result.loc[short_mask, 'signal_strength'] = 1
        result.loc[short_mask & result['wolfpack_short_confirm'], 'signal_strength'] += 1
        result.loc[short_mask & result['wt_short_confirm'], 'signal_strength'] += 1

        # Build combined signals
        long_valid = result['long_signal'].fillna(False).copy()
        short_valid = result['short_signal'].fillna(False).copy()

        if conf.require_wolfpack:
            long_valid = long_valid & result['wolfpack_long_confirm']
            short_valid = short_valid & result['wolfpack_short_confirm']

        if conf.require_wavetrend:
            long_valid = long_valid & result['wt_long_confirm']
            short_valid = short_valid & result['wt_short_confirm']

        # Signal strength filter
        long_valid = long_valid & (result['signal_strength'] >= conf.min_signal_strength)
        short_valid = short_valid & (result['signal_strength'] >= conf.min_signal_strength)

        result['combined_long'] = long_valid
        result['combined_short'] = short_valid

        return result

    def _calculate_sl_tp(
        self,
        entry_price: float,
        direction: str,
        df: pd.DataFrame,
        idx: int
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels for a trade"""
        risk = self.config.risk

        # Stop Loss
        if risk.stop_loss_type == 'percent':
            if direction == 'long':
                sl = entry_price * (1 - risk.stop_loss_value / 100)
            else:
                sl = entry_price * (1 + risk.stop_loss_value / 100)

        elif risk.stop_loss_type == 'atr':
            # Calculate ATR
            atr = self._calculate_atr(df, idx)
            if direction == 'long':
                sl = entry_price - (atr * risk.stop_loss_value)
            else:
                sl = entry_price + (atr * risk.stop_loss_value)

        else:  # fixed
            if direction == 'long':
                sl = entry_price - risk.stop_loss_value
            else:
                sl = entry_price + risk.stop_loss_value

        # Take Profit
        if risk.take_profit_type == 'percent':
            if direction == 'long':
                tp = entry_price * (1 + risk.take_profit_value / 100)
            else:
                tp = entry_price * (1 - risk.take_profit_value / 100)

        elif risk.take_profit_type == 'rr_ratio':
            # TP based on risk/reward ratio
            risk_amount = abs(entry_price - sl)
            reward = risk_amount * risk.risk_reward_ratio
            if direction == 'long':
                tp = entry_price + reward
            else:
                tp = entry_price - reward

        elif risk.take_profit_type == 'atr':
            atr = self._calculate_atr(df, idx)
            if direction == 'long':
                tp = entry_price + (atr * risk.take_profit_value)
            else:
                tp = entry_price - (atr * risk.take_profit_value)

        else:  # fixed
            if direction == 'long':
                tp = entry_price + risk.take_profit_value
            else:
                tp = entry_price - risk.take_profit_value

        return sl, tp

    def _calculate_atr(self, df: pd.DataFrame, idx: int, period: int = 14) -> float:
        """Calculate ATR at given index"""
        if idx < period:
            return df['high'].iloc[:idx+1].max() - df['low'].iloc[:idx+1].min()

        high = df['high'].iloc[idx-period:idx]
        low = df['low'].iloc[idx-period:idx]
        close = df['close'].iloc[idx-period:idx]

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        return tr.mean()

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Run backtest"""
        # Calculate indicators and signals
        data = self._calculate_indicators(df)
        data = self._generate_signals(data)

        result = BacktestResult(config_name=self.config.name)
        result.initial_capital = self.config.initial_capital

        capital = self.config.initial_capital
        position: Optional[TradeRecord] = None
        trades: List[TradeRecord] = []
        equity = []
        last_trade_bar = -999  # For min bars between trades

        for i in range(1, len(data)):
            current_time = data.index[i]
            current_bar = i
            open_price = data['open'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]
            close_price = data['close'].iloc[i]

            # Check exits if in position
            if position is not None:
                exit_triggered = False
                exit_price = close_price
                exit_reason = None
                bars_held = current_bar - position.id  # Approximate

                if position.direction == 'long':
                    # Check stop loss (triggered at low)
                    if low_price <= position.stop_loss_price:
                        exit_price = position.stop_loss_price
                        exit_reason = 'stop_loss'
                        exit_triggered = True

                    # Check take profit (triggered at high)
                    if not exit_triggered and high_price >= position.take_profit_price:
                        exit_price = position.take_profit_price
                        exit_reason = 'take_profit'
                        exit_triggered = True

                    # Check signal reversal
                    if not exit_triggered and data['combined_short'].iloc[i]:
                        exit_reason = 'signal_reverse'
                        exit_triggered = True

                else:  # short
                    # Check stop loss (triggered at high)
                    if high_price >= position.stop_loss_price:
                        exit_price = position.stop_loss_price
                        exit_reason = 'stop_loss'
                        exit_triggered = True

                    # Check take profit (triggered at low)
                    if not exit_triggered and low_price <= position.take_profit_price:
                        exit_price = position.take_profit_price
                        exit_reason = 'take_profit'
                        exit_triggered = True

                    # Check signal reversal
                    if not exit_triggered and data['combined_long'].iloc[i]:
                        exit_reason = 'signal_reverse'
                        exit_triggered = True

                if exit_triggered:
                    position.calculate_exit(
                        exit_price, current_time, exit_reason,
                        self.config.spread_pct, bars_held
                    )
                    capital += position.net_pnl
                    trades.append(position)
                    last_trade_bar = current_bar
                    position = None

            # Check for new entry
            if position is None:
                # Respect min bars between trades
                bars_since_last = current_bar - last_trade_bar
                if bars_since_last < self.config.confirmation.min_bars_between_trades:
                    pass  # Skip

                elif data['combined_long'].iloc[i]:
                    self.trade_counter += 1
                    entry_price = close_price
                    sl, tp = self._calculate_sl_tp(entry_price, 'long', data, i)

                    risk_capital = capital * self.config.risk.position_size_pct
                    size = risk_capital / entry_price

                    position = TradeRecord(
                        id=self.trade_counter,
                        entry_time=current_time,
                        entry_price=entry_price,
                        direction='long',
                        size=size,
                        capital_at_entry=capital,
                        stop_loss_price=sl,
                        take_profit_price=tp,
                        signal_strength=int(data['signal_strength'].iloc[i]),
                        wolfpack_value=data['wolfpack'].iloc[i],
                        wavetrend_value=data['wt1'].iloc[i],
                        wolfpack_confirmed=data['wolfpack_long_confirm'].iloc[i],
                        wavetrend_confirmed=data['wt_long_confirm'].iloc[i]
                    )

                elif data['combined_short'].iloc[i] and self.config.allow_shorting:
                    self.trade_counter += 1
                    entry_price = close_price
                    sl, tp = self._calculate_sl_tp(entry_price, 'short', data, i)

                    risk_capital = capital * self.config.risk.position_size_pct
                    size = risk_capital / entry_price

                    position = TradeRecord(
                        id=self.trade_counter,
                        entry_time=current_time,
                        entry_price=entry_price,
                        direction='short',
                        size=size,
                        capital_at_entry=capital,
                        stop_loss_price=sl,
                        take_profit_price=tp,
                        signal_strength=int(data['signal_strength'].iloc[i]),
                        wolfpack_value=data['wolfpack'].iloc[i],
                        wavetrend_value=data['wt1'].iloc[i],
                        wolfpack_confirmed=data['wolfpack_short_confirm'].iloc[i],
                        wavetrend_confirmed=data['wt_short_confirm'].iloc[i]
                    )

            # Track equity
            current_equity = capital
            if position is not None:
                if position.direction == 'long':
                    unrealized = (close_price - position.entry_price) * position.size
                else:
                    unrealized = (position.entry_price - close_price) * position.size
                current_equity = capital + unrealized

            equity.append({'time': current_time, 'equity': current_equity})

        # Close any remaining position
        if position is not None:
            position.calculate_exit(
                data['close'].iloc[-1], data.index[-1], 'end_of_data',
                self.config.spread_pct, len(data) - position.id
            )
            capital += position.net_pnl
            trades.append(position)

        # Calculate results
        result.trades = trades
        result.equity_curve = pd.DataFrame(equity).set_index('time')['equity'] if equity else pd.Series()
        result.final_capital = capital
        result = self._calculate_stats(result)

        return result

    def _calculate_stats(self, result: BacktestResult) -> BacktestResult:
        """Calculate all statistics"""
        trades = result.trades
        result.total_trades = len(trades)

        if not trades:
            return result

        result.total_return_pct = ((result.final_capital - result.initial_capital) /
                                    result.initial_capital) * 100

        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]

        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.win_rate = (len(winners) / len(trades)) * 100

        result.gross_profit = sum(t.net_pnl for t in winners)
        result.gross_loss = abs(sum(t.net_pnl for t in losers))
        result.net_profit = result.gross_profit - result.gross_loss

        result.profit_factor = (result.gross_profit / result.gross_loss
                                if result.gross_loss > 0 else float('inf'))

        result.avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
        result.avg_loss = np.mean([t.net_pnl for t in losers]) if losers else 0
        result.avg_trade = np.mean([t.net_pnl for t in trades])

        result.largest_win = max([t.net_pnl for t in winners]) if winners else 0
        result.largest_loss = min([t.net_pnl for t in losers]) if losers else 0

        # R-multiples
        result.avg_win_r = np.mean([t.r_multiple for t in winners]) if winners else 0
        result.avg_loss_r = np.mean([t.r_multiple for t in losers]) if losers else 0
        result.expectancy_r = np.mean([t.r_multiple for t in trades])

        # Duration
        result.avg_bars_held = np.mean([t.bars_held for t in trades])
        result.avg_bars_winner = np.mean([t.bars_held for t in winners]) if winners else 0
        result.avg_bars_loser = np.mean([t.bars_held for t in losers]) if losers else 0

        # Drawdown
        if len(result.equity_curve) > 0:
            peak = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - peak) / peak * 100
            result.max_drawdown_pct = drawdown.min()

            # Sharpe
            returns = result.equity_curve.pct_change().dropna()
            if len(returns) > 1 and returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

                # Sortino (downside only)
                downside = returns[returns < 0]
                if len(downside) > 0 and downside.std() > 0:
                    result.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(252)

        # By exit reason
        for trade in trades:
            reason = trade.exit_reason or 'unknown'
            result.exits_by_reason[reason] = result.exits_by_reason.get(reason, 0) + 1

        # By signal strength
        for strength in [1, 2, 3]:
            strength_trades = [t for t in trades if t.signal_strength == strength]
            if strength_trades:
                strength_winners = [t for t in strength_trades if t.net_pnl > 0]
                result.stats_by_strength[strength] = {
                    'count': len(strength_trades),
                    'win_rate': (len(strength_winners) / len(strength_trades)) * 100,
                    'avg_pnl': np.mean([t.net_pnl for t in strength_trades]),
                    'avg_r': np.mean([t.r_multiple for t in strength_trades])
                }

        return result


def print_detailed_results(result: BacktestResult):
    """Print detailed backtest results"""
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS: {result.config_name}")
    print(f"{'='*70}")

    print(f"\n[CAPITAL]")
    print(f"  Initial:     ${result.initial_capital:,.2f}")
    print(f"  Final:       ${result.final_capital:,.2f}")
    print(f"  Net Profit:  ${result.net_profit:,.2f}")
    print(f"  Return:      {result.total_return_pct:+.2f}%")

    print(f"\n[TRADES]")
    print(f"  Total:       {result.total_trades}")
    print(f"  Winners:     {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"  Losers:      {result.losing_trades}")

    print(f"\n[PROFIT/LOSS]")
    print(f"  Gross Profit:  ${result.gross_profit:,.2f}")
    print(f"  Gross Loss:    ${result.gross_loss:,.2f}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Avg Win:       ${result.avg_win:,.2f}")
    print(f"  Avg Loss:      ${result.avg_loss:,.2f}")
    print(f"  Largest Win:   ${result.largest_win:,.2f}")
    print(f"  Largest Loss:  ${result.largest_loss:,.2f}")

    print(f"\n[R-MULTIPLES]")
    print(f"  Avg Win R:    {result.avg_win_r:.2f}R")
    print(f"  Avg Loss R:   {result.avg_loss_r:.2f}R")
    print(f"  Expectancy:   {result.expectancy_r:.2f}R per trade")

    print(f"\n[RISK]")
    print(f"  Max Drawdown:  {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:  {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")

    print(f"\n[DURATION]")
    print(f"  Avg Bars Held:    {result.avg_bars_held:.1f}")
    print(f"  Avg Winner Bars:  {result.avg_bars_winner:.1f}")
    print(f"  Avg Loser Bars:   {result.avg_bars_loser:.1f}")

    print(f"\n[EXIT REASONS]")
    for reason, count in sorted(result.exits_by_reason.items()):
        pct = (count / result.total_trades) * 100 if result.total_trades > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")

    if result.stats_by_strength:
        print(f"\n[BY SIGNAL STRENGTH]")
        for strength, stats in sorted(result.stats_by_strength.items()):
            stars = '*' * strength
            print(f"  Strength {strength} ({stars}):")
            print(f"    Trades: {stats['count']}, Win Rate: {stats['win_rate']:.1f}%, "
                  f"Avg R: {stats['avg_r']:.2f}")


def print_trade_details(trades: List[TradeRecord], limit: int = 20):
    """Print detailed trade log"""
    print(f"\n{'='*120}")
    print(f"TRADE LOG (Last {min(limit, len(trades))} trades)")
    print(f"{'='*120}")
    print(f"{'#':>4} {'Entry Time':<16} {'Dir':<5} {'Entry':>9} {'SL':>9} {'TP':>9} "
          f"{'Exit':>9} {'P&L':>10} {'R':>6} {'Reason':<12}")
    print("-"*120)

    for trade in trades[-limit:]:
        entry_str = str(trade.entry_time)[:16]
        exit_price = trade.exit_price or 0
        r_str = f"{trade.r_multiple:+.2f}R"
        pnl_str = f"${trade.net_pnl:+.2f}"

        print(f"{trade.id:>4} {entry_str:<16} {trade.direction:<5} "
              f"{trade.entry_price:>9.2f} {trade.stop_loss_price:>9.2f} "
              f"{trade.take_profit_price:>9.2f} {exit_price:>9.2f} "
              f"{pnl_str:>10} {r_str:>6} {trade.exit_reason or '':<12}")


if __name__ == "__main__":
    from config_manager import StrategyConfig, print_config

    # Create a test config
    config = StrategyConfig(name='test_run')

    print_config(config)

    # Fetch data
    print("\nFetching data...")
    try:
        df = fetch_gold_data(source='yahoo', interval='1h')
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    # Run backtest
    print("\nRunning backtest...")
    backtester = FlexibleBacktester(config)
    result = backtester.run(df)

    # Print results
    print_detailed_results(result)
    print_trade_details(result.trades)
