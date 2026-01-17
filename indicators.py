"""
Trading Indicators Module

Contains:
- Wolfpack ID (MACD 3,8 based trend identifier)
- WaveTrend Oscillator (Market Cipher / VuManChu style)
- Money Flow Index
- Supporting functions
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# WOLFPACK ID INDICATOR
# Source: Decoded from original, essentially MACD with EMA(3) and EMA(8)
# =============================================================================

@dataclass
class WolfpackConfig:
    """Configuration for Wolfpack ID"""
    fast_length: int = 3
    slow_length: int = 8
    multiplier: float = 1.001  # Original uses 1.001


def calculate_wolfpack_id(
    df: pd.DataFrame,
    config: Optional[WolfpackConfig] = None
) -> pd.DataFrame:
    """
    Calculate Wolfpack ID indicator

    Wolfpack ID = (EMA(close, 3) - EMA(close, 8)) * 1.001

    - Positive (green): Bullish trend
    - Negative (red): Bearish trend

    Args:
        df: DataFrame with 'close' column
        config: Wolfpack configuration

    Returns:
        DataFrame with wolfpack columns added
    """
    if config is None:
        config = WolfpackConfig()

    result = df.copy()

    # Calculate EMAs
    fast_ema = df['close'].ewm(span=config.fast_length, adjust=False).mean()
    slow_ema = df['close'].ewm(span=config.slow_length, adjust=False).mean()

    # Wolfpack line (bspread in original)
    result['wolfpack'] = (fast_ema - slow_ema) * config.multiplier

    # Trend direction
    result['wolfpack_bullish'] = result['wolfpack'] > 0
    result['wolfpack_bearish'] = result['wolfpack'] < 0

    # Crossover signals
    result['wolfpack_cross_up'] = (result['wolfpack'] > 0) & (result['wolfpack'].shift(1) <= 0)
    result['wolfpack_cross_down'] = (result['wolfpack'] < 0) & (result['wolfpack'].shift(1) >= 0)

    return result


# =============================================================================
# WAVETREND OSCILLATOR (Market Cipher / VuManChu style)
# =============================================================================

@dataclass
class WaveTrendConfig:
    """Configuration for WaveTrend Oscillator"""
    channel_length: int = 9      # WT Channel Length
    average_length: int = 12     # WT Average Length
    ma_length: int = 3           # WT MA Length (for wt2)

    # Overbought/Oversold levels
    ob_level_1: int = 53
    ob_level_2: int = 60
    ob_level_3: int = 100
    os_level_1: int = -53
    os_level_2: int = -60
    os_level_3: int = -75


def calculate_wavetrend(
    df: pd.DataFrame,
    config: Optional[WaveTrendConfig] = None,
    source: str = 'hlc3'
) -> pd.DataFrame:
    """
    Calculate WaveTrend Oscillator (Market Cipher B / VuManChu style)

    The WaveTrend is calculated as:
    1. ap = hlc3 (or other source)
    2. esa = EMA(ap, channel_length)
    3. d = EMA(abs(ap - esa), channel_length)
    4. ci = (ap - esa) / (0.015 * d)
    5. wt1 = EMA(ci, average_length)
    6. wt2 = SMA(wt1, ma_length)

    Args:
        df: DataFrame with OHLC data
        config: WaveTrend configuration
        source: Price source ('hlc3', 'close', 'ohlc4')

    Returns:
        DataFrame with wavetrend columns added
    """
    if config is None:
        config = WaveTrendConfig()

    result = df.copy()

    # Calculate source
    if source == 'hlc3':
        src = (df['high'] + df['low'] + df['close']) / 3
    elif source == 'ohlc4':
        src = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    else:
        src = df['close']

    # ESA = EMA of source
    esa = src.ewm(span=config.channel_length, adjust=False).mean()

    # D = EMA of absolute deviation
    d = (src - esa).abs().ewm(span=config.channel_length, adjust=False).mean()

    # CI = Commodity Index style calculation
    # Avoid division by zero
    ci = pd.Series(index=df.index, dtype=float)
    ci = np.where(d != 0, (src - esa) / (0.015 * d), 0)
    ci = pd.Series(ci, index=df.index)

    # WT1 = EMA of CI
    wt1 = ci.ewm(span=config.average_length, adjust=False).mean()

    # WT2 = SMA of WT1
    wt2 = wt1.rolling(window=config.ma_length).mean()

    # VWAP style difference
    wt_vwap = wt1 - wt2

    result['wt1'] = wt1
    result['wt2'] = wt2
    result['wt_vwap'] = wt_vwap

    # Overbought/Oversold conditions
    result['wt_overbought'] = wt2 >= config.ob_level_1
    result['wt_oversold'] = wt2 <= config.os_level_1
    result['wt_overbought_extreme'] = wt2 >= config.ob_level_2
    result['wt_oversold_extreme'] = wt2 <= config.os_level_2

    # Crossover signals
    result['wt_cross'] = ((wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))) | \
                         ((wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1)))
    result['wt_cross_up'] = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    result['wt_cross_down'] = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))

    # Buy/Sell signals (like green/red circles in Cipher)
    result['wt_buy_signal'] = result['wt_cross_up'] & result['wt_oversold']
    result['wt_sell_signal'] = result['wt_cross_down'] & result['wt_overbought']

    # Trend based on WT position
    result['wt_bullish'] = wt2 > 0
    result['wt_bearish'] = wt2 < 0

    return result


# =============================================================================
# MONEY FLOW INDEX (RSI + MFI style from VuManChu)
# =============================================================================

@dataclass
class MoneyFlowConfig:
    """Configuration for Money Flow indicator"""
    period: int = 60
    multiplier: float = 150.0


def calculate_money_flow(
    df: pd.DataFrame,
    config: Optional[MoneyFlowConfig] = None
) -> pd.DataFrame:
    """
    Calculate Money Flow indicator (VuManChu / Market Cipher style)

    MFI = SMA(((close - open) / (high - low)) * multiplier, period)

    Args:
        df: DataFrame with OHLC data
        config: Money Flow configuration

    Returns:
        DataFrame with money flow columns added
    """
    if config is None:
        config = MoneyFlowConfig()

    result = df.copy()

    # Calculate candle body ratio
    hl_range = df['high'] - df['low']
    body = df['close'] - df['open']

    # Avoid division by zero
    ratio = np.where(hl_range != 0, body / hl_range, 0)

    # Apply multiplier and smooth
    mfi_raw = ratio * config.multiplier
    result['money_flow'] = pd.Series(mfi_raw, index=df.index).rolling(window=config.period).mean()

    # Bullish/Bearish
    result['mf_bullish'] = result['money_flow'] > 0
    result['mf_bearish'] = result['money_flow'] < 0

    return result


# =============================================================================
# RSI (Relative Strength Index)
# =============================================================================

def calculate_rsi(
    df: pd.DataFrame,
    period: int = 14,
    source: str = 'close'
) -> pd.DataFrame:
    """
    Calculate RSI indicator

    Args:
        df: DataFrame with price data
        period: RSI period
        source: Price source column

    Returns:
        DataFrame with RSI columns added
    """
    result = df.copy()

    src = df[source]
    delta = src.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result['rsi'] = 100 - (100 / (1 + rs))

    result['rsi_overbought'] = result['rsi'] >= 70
    result['rsi_oversold'] = result['rsi'] <= 30

    return result


# =============================================================================
# COMBINED INDICATOR CALCULATION
# =============================================================================

def calculate_all_indicators(
    df: pd.DataFrame,
    wolfpack_config: Optional[WolfpackConfig] = None,
    wavetrend_config: Optional[WaveTrendConfig] = None,
    money_flow_config: Optional[MoneyFlowConfig] = None,
    rsi_period: int = 14
) -> pd.DataFrame:
    """
    Calculate all indicators at once

    Args:
        df: DataFrame with OHLC data
        wolfpack_config: Wolfpack ID configuration
        wavetrend_config: WaveTrend configuration
        money_flow_config: Money Flow configuration
        rsi_period: RSI period

    Returns:
        DataFrame with all indicator columns
    """
    result = df.copy()

    # Wolfpack ID
    result = calculate_wolfpack_id(result, wolfpack_config)

    # WaveTrend
    result = calculate_wavetrend(result, wavetrend_config)

    # Money Flow
    result = calculate_money_flow(result, money_flow_config)

    # RSI
    result = calculate_rsi(result, rsi_period)

    return result


if __name__ == "__main__":
    print("Indicators module loaded successfully")
    print("Available indicators:")
    print("  - Wolfpack ID (calculate_wolfpack_id)")
    print("  - WaveTrend Oscillator (calculate_wavetrend)")
    print("  - Money Flow (calculate_money_flow)")
    print("  - RSI (calculate_rsi)")
    print("  - All combined (calculate_all_indicators)")
