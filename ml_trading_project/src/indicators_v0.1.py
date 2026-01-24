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


def add_all_indicators(df):
    """Add all indicators to dataframe"""
    df = calculate_wavetrend(df)
    df = calculate_wolfpack(df)
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = calculate_moving_averages(df)
    df = calculate_returns(df)
    df['volatility'] = df['ret_1h'].rolling(24).std()
    df['trend'] = np.where(df['ma20'] > df['ma50'], 1, -1)
    return df
