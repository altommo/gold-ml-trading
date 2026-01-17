"""
Data Fetcher Module

Supports multiple data sources:
- TradingView via tvdatafeed (primary)
- Yahoo Finance (backup)
- CSV files (for Kaggle data)

Features:
- Automatic caching to avoid re-fetching
- Combine multiple fetches for larger datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Literal
from pathlib import Path
import warnings
import hashlib

warnings.filterwarnings('ignore')

# Default cache directory
CACHE_DIR = Path('./data_cache')


def get_cache_path(symbol: str, exchange: str, interval: str) -> Path:
    """Get cache file path for given parameters"""
    CACHE_DIR.mkdir(exist_ok=True)
    filename = f"{symbol}_{exchange}_{interval}.csv"
    return CACHE_DIR / filename


def save_to_cache(df: pd.DataFrame, symbol: str, exchange: str, interval: str) -> Path:
    """
    Save DataFrame to cache

    Args:
        df: OHLCV DataFrame
        symbol: Trading symbol
        exchange: Exchange name
        interval: Timeframe

    Returns:
        Path to saved file
    """
    filepath = get_cache_path(symbol, exchange, interval)
    df.to_csv(filepath)
    print(f"Data cached to: {filepath}")
    return filepath


def load_from_cache(
    symbol: str,
    exchange: str,
    interval: str,
    max_age_hours: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Load data from cache if available

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        interval: Timeframe
        max_age_hours: Maximum age of cache in hours (None = no limit)

    Returns:
        DataFrame if cache exists and is valid, None otherwise
    """
    filepath = get_cache_path(symbol, exchange, interval)

    if not filepath.exists():
        return None

    # Check age if specified
    if max_age_hours is not None:
        file_age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        if file_age > timedelta(hours=max_age_hours):
            print(f"Cache expired ({file_age.total_seconds()/3600:.1f}h old)")
            return None

    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} bars from cache: {filepath}")
        return df
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def merge_dataframes(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames, removing duplicates and sorting by time

    Args:
        old_df: Existing data
        new_df: New data to merge

    Returns:
        Combined DataFrame
    """
    combined = pd.concat([old_df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    return combined


def list_cached_data() -> pd.DataFrame:
    """List all cached data files with their info"""
    if not CACHE_DIR.exists():
        return pd.DataFrame()

    files = list(CACHE_DIR.glob('*.csv'))
    data = []

    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True, nrows=1)
            full_df = pd.read_csv(f, index_col=0, parse_dates=True)

            parts = f.stem.split('_')
            symbol = parts[0] if len(parts) > 0 else 'unknown'
            exchange = parts[1] if len(parts) > 1 else 'unknown'
            interval = parts[2] if len(parts) > 2 else 'unknown'

            data.append({
                'file': f.name,
                'symbol': symbol,
                'exchange': exchange,
                'interval': interval,
                'bars': len(full_df),
                'start': full_df.index[0],
                'end': full_df.index[-1],
                'size_kb': f.stat().st_size / 1024
            })
        except:
            continue

    return pd.DataFrame(data)


# Interval mappings
INTERVALS = {
    '1m': '1 minute',
    '5m': '5 minutes',
    '15m': '15 minutes',
    '30m': '30 minutes',
    '1h': '1 hour',
    '2h': '2 hours',
    '4h': '4 hours',
    '1d': '1 day',
    '1w': '1 week',
}


def fetch_from_tradingview(
    symbol: str = 'XAUUSD',
    exchange: str = 'OANDA',
    interval: str = '1h',
    n_bars: int = 5000,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch data from TradingView using tvdatafeed

    Args:
        symbol: Trading symbol (e.g., 'XAUUSD', 'BTCUSD')
        exchange: Exchange name (e.g., 'OANDA', 'FOREXCOM', 'BINANCE')
        interval: Timeframe ('1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w')
        n_bars: Number of bars to fetch (max ~5000)
        username: TradingView username (optional, for more access)
        password: TradingView password (optional)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        from tvDatafeed import TvDatafeed, Interval  # Note: capital D
    except ImportError:
        raise ImportError(
            "tvdatafeed not installed. Install with:\n"
            "pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git"
        )

    # Map interval string to tvdatafeed Interval
    interval_map = {
        '1m': Interval.in_1_minute,
        '5m': Interval.in_5_minute,
        '15m': Interval.in_15_minute,
        '30m': Interval.in_30_minute,
        '1h': Interval.in_1_hour,
        '2h': Interval.in_2_hour,
        '4h': Interval.in_4_hour,
        '1d': Interval.in_daily,
        '1w': Interval.in_weekly,
    }

    if interval not in interval_map:
        raise ValueError(f"Invalid interval: {interval}. Must be one of {list(interval_map.keys())}")

    tv_interval = interval_map[interval]

    print(f"Fetching {symbol} from {exchange} ({interval}, {n_bars} bars)...")

    # Initialize TvDatafeed
    if username and password:
        tv = TvDatafeed(username=username, password=password)
    else:
        tv = TvDatafeed()

    # Fetch data
    df = tv.get_hist(
        symbol=symbol,
        exchange=exchange,
        interval=tv_interval,
        n_bars=n_bars
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol} on {exchange}")

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 0

    print(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_from_yahoo(
    symbol: str = 'GC=F',
    interval: str = '1h',
    period: str = '60d'
) -> pd.DataFrame:
    """
    Fetch data from Yahoo Finance (backup option)

    Note: Yahoo has limited intraday history (~60 days for hourly)

    Args:
        symbol: Yahoo symbol ('GC=F' for gold futures)
        interval: Timeframe ('1h', '1d', etc.)
        period: Lookback period ('60d', '1y', etc.)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Install with: pip install yfinance")

    print(f"Fetching {symbol} from Yahoo Finance ({interval}, {period})...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df.columns = [c.lower() for c in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    print(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def load_from_csv(
    filepath: str,
    datetime_col: str = 'datetime',
    datetime_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load data from CSV file (for Kaggle datasets)

    Args:
        filepath: Path to CSV file
        datetime_col: Name of datetime column
        datetime_format: Datetime format string (auto-detected if None)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading data from {filepath}...")

    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]

    # Find datetime column
    dt_col = None
    for col in ['datetime', 'date', 'time', 'timestamp', 'gmt time']:
        if col in df.columns:
            dt_col = col
            break

    if dt_col is None:
        raise ValueError(f"Could not find datetime column. Available: {list(df.columns)}")

    # Parse datetime
    if datetime_format:
        df['datetime'] = pd.to_datetime(df[dt_col], format=datetime_format)
    else:
        df['datetime'] = pd.to_datetime(df[dt_col])

    df.set_index('datetime', inplace=True)

    # Map column names
    col_mapping = {
        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume',
        'op': 'open', 'hi': 'high', 'lo': 'low', 'cl': 'close', 'vol': 'volume'
    }

    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available: {list(df.columns)}")

    if 'volume' not in df.columns:
        df['volume'] = 0

    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Sort by datetime
    df.sort_index(inplace=True)

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def fetch_gold_data(
    source: Literal['tradingview', 'yahoo', 'csv'] = 'tradingview',
    interval: str = '1h',
    n_bars: int = 5000,
    csv_path: Optional[str] = None,
    use_cache: bool = True,
    fetch_latest: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to fetch gold (XAUUSD) data with caching

    Historical data is always preserved. New fetches only ADD recent candles.
    Tries multiple exchanges if one fails.

    Args:
        source: Data source ('tradingview', 'yahoo', 'csv')
        interval: Timeframe
        n_bars: Number of bars (for tradingview)
        csv_path: Path to CSV file (for csv source)
        use_cache: Load from cache if available
        fetch_latest: Also fetch latest data and merge with cache
        **kwargs: Additional arguments passed to underlying function

    Returns:
        DataFrame with OHLCV data
    """
    symbol = 'XAUUSD'

    # Exchanges to try in order of preference
    exchanges = ['OANDA', 'FOREXCOM', 'FX', 'FXCM']

    # Try to load from any existing cache first
    cached = None
    used_exchange = None
    if use_cache and source != 'csv':
        for exchange in exchanges:
            cached = load_from_cache(symbol, exchange, interval, max_age_hours=None)
            if cached is not None:
                used_exchange = exchange
                break

    # If we have cache and don't need latest, return it
    if cached is not None and not fetch_latest:
        return cached

    # Fetch new data
    if source == 'tradingview':
        df = None
        fetch_exchange = None

        # Try each exchange until one works
        for exchange in exchanges:
            try:
                df = fetch_from_tradingview(
                    symbol='XAUUSD',
                    exchange=exchange,
                    interval=interval,
                    n_bars=n_bars,
                    **kwargs
                )
                if df is not None and not df.empty:
                    fetch_exchange = exchange
                    break
            except Exception as e:
                print(f"  {exchange} failed: {e}")
                continue

        if df is None or df.empty:
            raise ValueError(f"Could not fetch {symbol} data from any exchange")

        # Use the exchange that worked for caching
        used_exchange = fetch_exchange

    elif source == 'yahoo':
        used_exchange = 'YAHOO'
        period = '60d' if interval in ['1h', '1m', '5m', '15m', '30m'] else '5y'
        df = fetch_from_yahoo(
            symbol='GC=F',
            interval=interval,
            period=period
        )
    elif source == 'csv':
        if csv_path is None:
            raise ValueError("csv_path required for csv source")
        return load_from_csv(csv_path, **kwargs)
    else:
        raise ValueError(f"Unknown source: {source}")

    # Merge with existing cache (preserves all historical + adds new)
    if cached is not None:
        df = merge_dataframes(cached, df)
        print(f"Merged with cache: now {len(df)} total bars")

    # Save combined data
    save_to_cache(df, symbol, used_exchange, interval)

    return df


def fetch_and_extend(
    symbol: str = 'XAUUSD',
    exchange: str = 'OANDA',
    interval: str = '1h',
    n_bars: int = 5000,
    times: int = 1
) -> pd.DataFrame:
    """
    Fetch data multiple times to build larger dataset

    Note: TradingView limits to ~5000 bars per request, but you can
    fetch multiple times and merge to get more historical data.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        interval: Timeframe
        n_bars: Bars per request
        times: Number of times to fetch (spaced out)

    Returns:
        Combined DataFrame
    """
    print(f"Fetching {symbol} data {times} time(s)...")

    # Load existing cache
    df = load_from_cache(symbol, exchange, interval) or pd.DataFrame()

    for i in range(times):
        print(f"\nFetch {i+1}/{times}...")
        try:
            new_data = fetch_from_tradingview(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                n_bars=n_bars
            )

            if df.empty:
                df = new_data
            else:
                df = merge_dataframes(df, new_data)

            print(f"Total bars: {len(df)}")

        except Exception as e:
            print(f"Error on fetch {i+1}: {e}")
            break

    # Save combined data
    if not df.empty:
        save_to_cache(df, symbol, exchange, interval)

    return df


def resample_data(
    df: pd.DataFrame,
    target_interval: str
) -> pd.DataFrame:
    """
    Resample data to a different timeframe

    Args:
        df: Source DataFrame with OHLCV data
        target_interval: Target timeframe ('4h', '1d', etc.)

    Returns:
        Resampled DataFrame
    """
    # Map interval to pandas resample string
    resample_map = {
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '2h': '2H',
        '4h': '4H',
        '1d': '1D',
        '1w': '1W',
    }

    if target_interval not in resample_map:
        raise ValueError(f"Invalid target interval: {target_interval}")

    rule = resample_map[target_interval]

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    print(f"Resampled from {len(df)} to {len(resampled)} bars ({target_interval})")

    return resampled


if __name__ == "__main__":
    print("Data Fetcher Module")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - fetch_gold_data(source='tradingview', interval='1h') - Main function with caching")
    print("  - fetch_from_tradingview(symbol, exchange, interval)")
    print("  - fetch_from_yahoo(symbol, interval)")
    print("  - load_from_csv(filepath)")
    print("  - resample_data(df, target_interval)")
    print("\nCaching functions:")
    print("  - list_cached_data() - Show all cached datasets")
    print("  - load_from_cache(symbol, exchange, interval)")
    print("  - save_to_cache(df, symbol, exchange, interval)")
    print("  - fetch_and_extend(symbol, exchange, interval, times=N) - Fetch multiple times")

    print(f"\nCache directory: {CACHE_DIR.absolute()}")

    # List cached data
    print("\n" + "="*50)
    print("CACHED DATA:")
    print("="*50)
    cached = list_cached_data()
    if cached.empty:
        print("  No cached data found")
    else:
        for _, row in cached.iterrows():
            print(f"  {row['file']}: {row['bars']} bars, {row['start']} to {row['end']}")
