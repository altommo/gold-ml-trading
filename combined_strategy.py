"""
Combined Trading Strategy

Signal Flow:
1. Potato Signal (Range Filter) = PRIMARY signal generator
2. Wolfpack ID = Confirmation #1 (trend direction)
3. WaveTrend (Cipher) = Confirmation #2 (momentum + overbought/oversold)

Trade Logic:
- LONG: Potato gives buy signal + Wolfpack bullish (green) + WaveTrend confirms
- SHORT: Potato gives sell signal + Wolfpack bearish (red) + WaveTrend confirms
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from datetime import datetime, timedelta

from indicators import (
    calculate_wolfpack_id, calculate_wavetrend, calculate_money_flow,
    calculate_rsi, calculate_all_indicators,
    WolfpackConfig, WaveTrendConfig, MoneyFlowConfig
)
from range_filter_strategy import apply_strategy, StrategyConfig


@dataclass
class CombinedStrategyConfig:
    """Configuration for the combined strategy"""

    # Range Filter (Potato Signal) settings
    rf_sampling_period: int = 27
    rf_range_multiplier: float = 1.0
    rf_source: str = 'close'  # 'close', 'wolfpack', 'wavetrend'

    # Wolfpack ID settings
    wolfpack_fast: int = 3
    wolfpack_slow: int = 8

    # WaveTrend settings
    wt_channel_length: int = 9
    wt_average_length: int = 12
    wt_ma_length: int = 3

    # Confirmation requirements
    require_wolfpack_confirm: bool = True
    require_wavetrend_confirm: bool = True
    require_wt_not_extreme: bool = True  # Don't buy overbought, don't sell oversold

    # WaveTrend confirmation modes
    wt_confirm_mode: str = 'trend'  # 'trend', 'cross', 'level'
    # 'trend': WT must be bullish/bearish
    # 'cross': WT must have recent cross in direction
    # 'level': WT must not be at extreme levels

    # Additional filters
    use_money_flow_filter: bool = False
    use_rsi_filter: bool = False
    rsi_oversold: int = 30
    rsi_overbought: int = 70


@dataclass
class Signal:
    """Represents a trading signal"""
    time: datetime
    direction: str  # 'long' or 'short'
    price: float
    potato_signal: bool
    wolfpack_confirm: bool
    wavetrend_confirm: bool
    wt_value: float
    wolfpack_value: float
    strength: int  # Number of confirmations (1-3)


def calculate_combined_signals(
    df: pd.DataFrame,
    config: Optional[CombinedStrategyConfig] = None
) -> pd.DataFrame:
    """
    Calculate all indicators and generate combined signals

    Args:
        df: DataFrame with OHLC data
        config: Strategy configuration

    Returns:
        DataFrame with all indicators and signals
    """
    if config is None:
        config = CombinedStrategyConfig()

    result = df.copy()

    # Step 1: Calculate Wolfpack ID
    wolfpack_config = WolfpackConfig(
        fast_length=config.wolfpack_fast,
        slow_length=config.wolfpack_slow
    )
    result = calculate_wolfpack_id(result, wolfpack_config)

    # Step 2: Calculate WaveTrend
    wavetrend_config = WaveTrendConfig(
        channel_length=config.wt_channel_length,
        average_length=config.wt_average_length,
        ma_length=config.wt_ma_length
    )
    result = calculate_wavetrend(result, wavetrend_config)

    # Step 3: Calculate Money Flow (optional)
    if config.use_money_flow_filter:
        result = calculate_money_flow(result)

    # Step 4: Calculate RSI (optional)
    if config.use_rsi_filter:
        result = calculate_rsi(result)

    # Step 5: Apply Range Filter (Potato Signal)
    rf_config = StrategyConfig(
        sampling_period=config.rf_sampling_period,
        range_multiplier=config.rf_range_multiplier,
        source_type=config.rf_source
    )
    result = apply_strategy(result, rf_config)

    # Step 6: Generate confirmation signals

    # Wolfpack confirmation
    result['wolfpack_long_confirm'] = result['wolfpack_bullish']  # Green = bullish
    result['wolfpack_short_confirm'] = result['wolfpack_bearish']  # Red = bearish

    # WaveTrend confirmation based on mode
    if config.wt_confirm_mode == 'trend':
        result['wt_long_confirm'] = result['wt_bullish'] | (result['wt1'] > result['wt1'].shift(1))
        result['wt_short_confirm'] = result['wt_bearish'] | (result['wt1'] < result['wt1'].shift(1))
    elif config.wt_confirm_mode == 'cross':
        # Recent cross (within last 3 bars)
        result['wt_long_confirm'] = result['wt_cross_up'] | result['wt_cross_up'].shift(1) | result['wt_cross_up'].shift(2)
        result['wt_short_confirm'] = result['wt_cross_down'] | result['wt_cross_down'].shift(1) | result['wt_cross_down'].shift(2)
    else:  # 'level'
        result['wt_long_confirm'] = ~result['wt_overbought']  # Not overbought
        result['wt_short_confirm'] = ~result['wt_oversold']  # Not oversold

    # Extreme level filter
    if config.require_wt_not_extreme:
        result['wt_long_ok'] = ~result['wt_overbought_extreme']
        result['wt_short_ok'] = ~result['wt_oversold_extreme']
    else:
        result['wt_long_ok'] = True
        result['wt_short_ok'] = True

    # Step 7: Combine all signals

    # Base signals from Potato
    potato_long = result['long_signal']
    potato_short = result['short_signal']

    # Build confirmation requirements
    long_confirms = potato_long.copy()
    short_confirms = potato_short.copy()

    if config.require_wolfpack_confirm:
        long_confirms = long_confirms & result['wolfpack_long_confirm']
        short_confirms = short_confirms & result['wolfpack_short_confirm']

    if config.require_wavetrend_confirm:
        long_confirms = long_confirms & result['wt_long_confirm'] & result['wt_long_ok']
        short_confirms = short_confirms & result['wt_short_confirm'] & result['wt_short_ok']

    if config.use_money_flow_filter:
        long_confirms = long_confirms & result['mf_bullish']
        short_confirms = short_confirms & result['mf_bearish']

    if config.use_rsi_filter:
        long_confirms = long_confirms & (result['rsi'] < config.rsi_overbought)
        short_confirms = short_confirms & (result['rsi'] > config.rsi_oversold)

    # Final combined signals
    result['combined_long_signal'] = long_confirms
    result['combined_short_signal'] = short_confirms

    # Calculate signal strength (number of confirmations)
    result['signal_strength'] = 0

    # Add strength for each confirmation present on long signals
    long_mask = result['long_signal']
    result.loc[long_mask, 'signal_strength'] += 1  # Potato signal
    result.loc[long_mask & result['wolfpack_long_confirm'], 'signal_strength'] += 1
    result.loc[long_mask & result['wt_long_confirm'], 'signal_strength'] += 1

    # Add strength for each confirmation present on short signals
    short_mask = result['short_signal']
    result.loc[short_mask, 'signal_strength'] += 1  # Potato signal
    result.loc[short_mask & result['wolfpack_short_confirm'], 'signal_strength'] += 1
    result.loc[short_mask & result['wt_short_confirm'], 'signal_strength'] += 1

    return result


def get_signals_list(df: pd.DataFrame) -> List[Signal]:
    """
    Extract list of Signal objects from processed DataFrame

    Args:
        df: DataFrame with combined signals calculated

    Returns:
        List of Signal objects
    """
    signals = []

    for idx in df.index:
        row = df.loc[idx]

        if row.get('combined_long_signal', False):
            signals.append(Signal(
                time=idx,
                direction='long',
                price=row['close'],
                potato_signal=True,
                wolfpack_confirm=row.get('wolfpack_long_confirm', False),
                wavetrend_confirm=row.get('wt_long_confirm', False),
                wt_value=row.get('wt1', 0),
                wolfpack_value=row.get('wolfpack', 0),
                strength=int(row.get('signal_strength', 1))
            ))

        if row.get('combined_short_signal', False):
            signals.append(Signal(
                time=idx,
                direction='short',
                price=row['close'],
                potato_signal=True,
                wolfpack_confirm=row.get('wolfpack_short_confirm', False),
                wavetrend_confirm=row.get('wt_short_confirm', False),
                wt_value=row.get('wt1', 0),
                wolfpack_value=row.get('wolfpack', 0),
                strength=int(row.get('signal_strength', 1))
            ))

    return signals


def print_signal_summary(df: pd.DataFrame):
    """Print summary of signals generated"""
    total_potato_long = df['long_signal'].sum()
    total_potato_short = df['short_signal'].sum()
    total_combined_long = df['combined_long_signal'].sum()
    total_combined_short = df['combined_short_signal'].sum()

    print("\n" + "="*60)
    print("SIGNAL SUMMARY")
    print("="*60)
    print(f"\nPotato Signal (Raw):")
    print(f"  Long signals:  {total_potato_long}")
    print(f"  Short signals: {total_potato_short}")
    print(f"  Total:         {total_potato_long + total_potato_short}")

    print(f"\nCombined (With Confirmations):")
    print(f"  Long signals:  {total_combined_long}")
    print(f"  Short signals: {total_combined_short}")
    print(f"  Total:         {total_combined_long + total_combined_short}")

    if total_potato_long + total_potato_short > 0:
        filter_rate = (1 - (total_combined_long + total_combined_short) /
                      (total_potato_long + total_potato_short)) * 100
        print(f"\n  Filtered out:  {filter_rate:.1f}% of signals")

    # Signal strength distribution
    strength_counts = df[df['signal_strength'] > 0]['signal_strength'].value_counts().sort_index()
    if not strength_counts.empty:
        print(f"\nSignal Strength Distribution:")
        for strength, count in strength_counts.items():
            stars = '*' * int(strength)
            print(f"  Strength {int(strength)} ({stars}): {count} signals")


if __name__ == "__main__":
    print("Combined Strategy Module")
    print("="*50)
    print("\nThis module combines:")
    print("  1. Potato Signal (Range Filter) - Primary signals")
    print("  2. Wolfpack ID - Trend confirmation")
    print("  3. WaveTrend (Cipher) - Momentum confirmation")
    print("\nUsage:")
    print("  from combined_strategy import calculate_combined_signals")
    print("  df = calculate_combined_signals(ohlc_data, config)")
