"""
Range Filter Strategy (Potato Signal)
Converted from PineScript - Market Liberator / Wolfpack ID style indicator

This strategy uses a smoothed range filter to identify trend direction
and generate buy/sell signals on direction changes.

IMPORTANT: This strategy can use different sources:
- 'close': Raw close price (default)
- 'wolfpack': Wolfpack ID indicator output
- 'wavetrend': WaveTrend (wt1) oscillator output
- 'custom': Any custom series passed in
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Literal
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuration for the Range Filter strategy"""
    sampling_period: int = 27  # Smoothing period for range calculation
    range_multiplier: float = 1.0  # Multiplier for the range bands
    use_heikin_ashi: bool = False  # Use Heikin Ashi candles

    # Source type: 'close', 'wolfpack', 'wavetrend', 'custom'
    source_type: str = 'close'

    # EMA settings (optional)
    ema_enabled: bool = False
    ema_1: int = 21
    ema_2: int = 50
    ema_3: int = 200

    # Bollinger Bands settings (optional)
    bb_enabled: bool = False
    bb_length: int = 20
    bb_std_dev: float = 2.0


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLC data to Heikin Ashi candles

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        DataFrame with Heikin Ashi OHLC values
    """
    ha_df = df.copy()

    # HA Close = (Open + High + Low + Close) / 4
    ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # HA Open = (previous HA Open + previous HA Close) / 2
    ha_df['ha_open'] = 0.0
    ha_df.iloc[0, ha_df.columns.get_loc('ha_open')] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2

    for i in range(1, len(ha_df)):
        ha_df.iloc[i, ha_df.columns.get_loc('ha_open')] = (
            ha_df['ha_open'].iloc[i-1] + ha_df['ha_close'].iloc[i-1]
        ) / 2

    # HA High = max(High, HA Open, HA Close)
    ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)

    # HA Low = min(Low, HA Open, HA Close)
    ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

    return ha_df


def smooth_range(source: pd.Series, period: int, multiplier: float) -> pd.Series:
    """
    Calculate the smoothed range

    This is the core of the Range Filter - it calculates a smoothed
    average of price movement to determine volatility/range.

    Args:
        source: Price series (typically close)
        period: Sampling period
        multiplier: Range multiplier

    Returns:
        Smoothed range series
    """
    # wper = period * 2 - 1
    wper = period * 2 - 1

    # Average range: EMA of absolute price changes
    avrng = source.diff().abs().ewm(span=period, adjust=False).mean()

    # Smooth the average range with wider EMA, then multiply
    smoothrng = avrng.ewm(span=wper, adjust=False).mean() * multiplier

    return smoothrng


def range_filter(source: pd.Series, smooth_rng: pd.Series) -> pd.Series:
    """
    Calculate the range filter line

    The filter follows price but with a lag determined by the smoothed range.
    It only moves when price moves beyond the range threshold.

    Args:
        source: Price series
        smooth_rng: Smoothed range series

    Returns:
        Range filter series
    """
    filt = pd.Series(index=source.index, dtype=float)
    filt.iloc[0] = source.iloc[0]

    for i in range(1, len(source)):
        prev_filt = filt.iloc[i-1]
        curr_src = source.iloc[i]
        curr_rng = smooth_rng.iloc[i]

        if curr_src > prev_filt:
            # Price above filter - filter can only move up
            if curr_src - curr_rng < prev_filt:
                filt.iloc[i] = prev_filt
            else:
                filt.iloc[i] = curr_src - curr_rng
        else:
            # Price below filter - filter can only move down
            if curr_src + curr_rng > prev_filt:
                filt.iloc[i] = prev_filt
            else:
                filt.iloc[i] = curr_src + curr_rng

    return filt


def calculate_direction(filt: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate upward and downward direction counters

    Args:
        filt: Range filter series

    Returns:
        Tuple of (upward, downward) series
    """
    upward = pd.Series(index=filt.index, dtype=float)
    downward = pd.Series(index=filt.index, dtype=float)

    upward.iloc[0] = 0
    downward.iloc[0] = 0

    for i in range(1, len(filt)):
        if filt.iloc[i] > filt.iloc[i-1]:
            upward.iloc[i] = upward.iloc[i-1] + 1
            downward.iloc[i] = 0
        elif filt.iloc[i] < filt.iloc[i-1]:
            downward.iloc[i] = downward.iloc[i-1] + 1
            upward.iloc[i] = 0
        else:
            upward.iloc[i] = upward.iloc[i-1]
            downward.iloc[i] = downward.iloc[i-1]

    return upward, downward


def generate_signals(
    source: pd.Series,
    filt: pd.Series,
    upward: pd.Series,
    downward: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate buy and sell signals based on conditions

    Signals are generated when:
    - Long: price > filter AND upward trending, on direction change from short
    - Short: price < filter AND downward trending, on direction change from long

    Args:
        source: Price series
        filt: Range filter series
        upward: Upward counter series
        downward: Downward counter series

    Returns:
        Tuple of (long_signal, short_signal) boolean series
    """
    # Condition for being in long/short mode
    long_cond = ((source > filt) & (source > source.shift(1)) & (upward > 0)) | \
                ((source > filt) & (source < source.shift(1)) & (upward > 0))

    short_cond = ((source < filt) & (source < source.shift(1)) & (downward > 0)) | \
                 ((source < filt) & (source > source.shift(1)) & (downward > 0))

    # Track condition state (1 = long, -1 = short)
    cond_ini = pd.Series(index=source.index, dtype=float)
    cond_ini.iloc[0] = 0

    for i in range(1, len(source)):
        if long_cond.iloc[i]:
            cond_ini.iloc[i] = 1
        elif short_cond.iloc[i]:
            cond_ini.iloc[i] = -1
        else:
            cond_ini.iloc[i] = cond_ini.iloc[i-1]

    # Signals only on direction changes
    long_signal = long_cond & (cond_ini.shift(1) == -1)
    short_signal = short_cond & (cond_ini.shift(1) == 1)

    return long_signal, short_signal


def apply_strategy(
    df: pd.DataFrame,
    config: Optional[StrategyConfig] = None,
    custom_source: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Apply the Range Filter strategy to OHLC data

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
              Can also contain 'wolfpack' or 'wt1' columns if using those sources
        config: Strategy configuration (uses defaults if None)
        custom_source: Custom series to use as source (when source_type='custom')

    Returns:
        DataFrame with all strategy indicators and signals added
    """
    if config is None:
        config = StrategyConfig()

    result = df.copy()

    # Determine source based on config
    if config.source_type == 'wolfpack':
        if 'wolfpack' not in df.columns:
            raise ValueError("DataFrame must contain 'wolfpack' column. Run calculate_wolfpack_id first.")
        source = df['wolfpack']
    elif config.source_type == 'wavetrend':
        if 'wt1' not in df.columns:
            raise ValueError("DataFrame must contain 'wt1' column. Run calculate_wavetrend first.")
        source = df['wt1']
    elif config.source_type == 'custom':
        if custom_source is None:
            raise ValueError("custom_source must be provided when source_type='custom'")
        source = custom_source
    elif config.use_heikin_ashi:
        ha_df = calculate_heikin_ashi(df)
        source = ha_df['ha_close']
    else:
        source = df['close']

    # Store which source was used
    result['rf_source_type'] = config.source_type

    # Calculate smoothed range
    result['smooth_range'] = smooth_range(source, config.sampling_period, config.range_multiplier)

    # Calculate range filter
    result['range_filter'] = range_filter(source, result['smooth_range'])

    # Calculate direction
    result['upward'], result['downward'] = calculate_direction(result['range_filter'])

    # Calculate target bands
    result['upper_band'] = result['range_filter'] + result['smooth_range']
    result['lower_band'] = result['range_filter'] - result['smooth_range']

    # Generate signals
    result['long_signal'], result['short_signal'] = generate_signals(
        source, result['range_filter'], result['upward'], result['downward']
    )

    # Add optional EMAs
    if config.ema_enabled:
        result['ema_1'] = df['close'].ewm(span=config.ema_1, adjust=False).mean()
        result['ema_2'] = df['close'].ewm(span=config.ema_2, adjust=False).mean()
        result['ema_3'] = df['close'].ewm(span=config.ema_3, adjust=False).mean()

    # Add optional Bollinger Bands
    if config.bb_enabled:
        result['bb_basis'] = df['close'].rolling(window=config.bb_length).mean()
        result['bb_std'] = df['close'].rolling(window=config.bb_length).std()
        result['bb_upper'] = result['bb_basis'] + (config.bb_std_dev * result['bb_std'])
        result['bb_lower'] = result['bb_basis'] - (config.bb_std_dev * result['bb_std'])

    # Add trend indicator for easy reference
    result['trend'] = np.where(result['upward'] > 0, 1, np.where(result['downward'] > 0, -1, 0))

    return result


if __name__ == "__main__":
    # Quick test with sample data
    print("Range Filter Strategy module loaded successfully")
    print("Use apply_strategy(df, config) to apply the strategy to OHLC data")
