"""
Technical indicators for Gold trading
"""
import pandas as pd
import numpy as np


def calculate_wavetrend(df, n1=10, n2=21):
    """Calculate WaveTrend indicator"""
    df = df.copy()
    ap = (df['high'] + df['low'] + df['close']) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    df['wt1'] = ci.ewm(span=n2, adjust=False).mean()
    df['wt2'] = df['wt1'].rolling(4).mean()
    return df


def calculate_wolfpack(df):
    """Calculate Wolfpack indicator (EMA3 - EMA8)"""
    df = df.copy()
    df['wolfpack'] = df['close'].ewm(span=3, adjust=False).mean() - df['close'].ewm(span=8, adjust=False).mean()
    return df


def calculate_rsi(df, period=14):
    """Calculate RSI"""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    return df


def calculate_atr(df, period=14):
    """Calculate ATR and ATR%"""
    df = df.copy()
    df['atr'] = (df['high'] - df['low']).rolling(period).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100
    return df


def calculate_moving_averages(df):
    """Calculate common moving averages"""
    df = df.copy()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    df['price_vs_ma20'] = (df['close'] - df['ma20']) / df['ma20'] * 100
    df['price_vs_ma50'] = (df['close'] - df['ma50']) / df['ma50'] * 100
    return df


def calculate_returns(df):
    """Calculate various return periods"""
    df = df.copy()
    df['ret_1h'] = df['close'].pct_change() * 100
    df['ret_4h'] = df['close'].pct_change(4) * 100
    df['ret_24h'] = df['close'].pct_change(24) * 100
    return df


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands and %B"""
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(period).mean()
    df['bb_std'] = df['close'].rolling(period).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * std_dev)
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])  # 0-1 range
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100  # volatility measure
    return df


def calculate_momentum(df):
    """Calculate momentum indicators"""
    df = df.copy()
    # Rate of change
    df['roc_5'] = (df['close'] / df['close'].shift(5) - 1) * 100
    df['roc_10'] = (df['close'] / df['close'].shift(10) - 1) * 100

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Stochastic RSI
    rsi = df['rsi'] if 'rsi' in df.columns else calculate_rsi(df)['rsi']
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    df['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

    return df


def calculate_time_features(df):
    """Add time-based features"""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        # Trading session (0=Asia, 1=London, 2=NY)
        df['session'] = pd.cut(df['hour'], bins=[-1, 8, 14, 21, 24],
                               labels=[0, 1, 2, 0]).astype(float)
        # Is London/NY overlap (most volatile)
        df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] <= 17)).astype(int)
    return df


def calculate_higher_tf_trend(df, periods=[24, 48, 96]):
    """Calculate higher timeframe trend signals"""
    df = df.copy()
    for p in periods:
        # Higher TF moving average
        df[f'ma_{p}h'] = df['close'].rolling(p).mean()
        df[f'price_vs_ma_{p}h'] = (df['close'] - df[f'ma_{p}h']) / df[f'ma_{p}h'] * 100

        # Higher TF trend direction
        df[f'trend_{p}h'] = np.where(df['close'] > df[f'ma_{p}h'], 1, -1)

    # Overall trend score (-3 to +3)
    df['trend_score'] = sum(df[f'trend_{p}h'] for p in periods)

    return df


def calculate_volatility_regime(df):
    """Identify volatility regime"""
    df = df.copy()
    # ATR percentile (is current volatility high or low vs recent history)
    df['atr_percentile'] = df['atr'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    # Volatility regime: 0=low, 1=normal, 2=high
    df['vol_regime'] = pd.cut(df['atr_percentile'], bins=[0, 0.33, 0.67, 1],
                              labels=[0, 1, 2]).astype(float)
    return df


def calculate_wt_signals(df):
    """WaveTrend cross and divergence signals"""
    df = df.copy()
    # WT cross
    df['wt_cross_up'] = ((df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))).astype(int)
    df['wt_cross_down'] = ((df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))).astype(int)

    # WT zones
    df['wt_oversold'] = (df['wt1'] < -40).astype(int)
    df['wt_overbought'] = (df['wt1'] > 40).astype(int)

    # Distance from zero
    df['wt_distance'] = abs(df['wt1'])

    return df


def add_all_indicators(df):
    """Add all indicators to dataframe"""
    df = calculate_wavetrend(df)
    df = calculate_wolfpack(df)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = calculate_moving_averages(df)
    df = calculate_returns(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = calculate_time_features(df)
    df = calculate_higher_tf_trend(df)
    df = calculate_volatility_regime(df)
    df = calculate_wt_signals(df)

    df['volatility'] = df['ret_1h'].rolling(24).std()
    df['trend'] = np.where(df['ma20'] > df['ma50'], 1, -1)

    return df
