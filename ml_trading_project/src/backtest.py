"""
Backtesting utilities for ML trading models
"""
import pandas as pd
import numpy as np


def backtest_model(df, model, scaler, features, threshold=0.5, hold_hours=24):
    """
    Backtest a model on historical data

    Args:
        df: DataFrame with OHLCV and features
        model: Trained classifier
        scaler: Fitted scaler for features
        features: List of feature column names
        threshold: Probability threshold for signals
        hold_hours: How long to hold each trade

    Returns:
        results dict, trades DataFrame
    """
    df = df.copy()
    df = df.dropna(subset=features)

    X = scaler.transform(df[features])
    df['prob'] = model.predict_proba(X)[:, 1]
    df['signal'] = (df['prob'] > threshold).astype(int)

    # Calculate future returns
    df['future_ret'] = (df['close'].shift(-hold_hours) / df['close'] - 1) * 100

    # Only trade on signals
    trades = df[df['signal'] == 1].copy()

    if len(trades) == 0:
        return {
            'total_trades': 0,
            'avg_return': 0,
            'total_return': 0,
            'win_rate': 0,
            'sharpe': 0,
            'max_drawdown': 0
        }, pd.DataFrame()

    # Calculate metrics
    results = {
        'total_trades': len(trades),
        'avg_return': trades['future_ret'].mean(),
        'total_return': trades['future_ret'].sum(),
        'win_rate': (trades['future_ret'] > 0).mean() * 100,
        'sharpe': trades['future_ret'].mean() / trades['future_ret'].std() * np.sqrt(252) if trades['future_ret'].std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(trades['future_ret'])
    }

    return results, trades


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown from a series of returns"""
    cumulative = (1 + returns / 100).cumprod()
    running_max = cumulative.cummax()
    drawdown = (running_max - cumulative) / running_max * 100
    return drawdown.max()


def walk_forward_backtest(df, model_class, features, train_size=1000, test_size=100, **model_params):
    """
    Walk-forward backtest with rolling training window

    Args:
        df: Full dataset with features
        model_class: Sklearn-compatible classifier class
        features: List of feature names
        train_size: Number of bars for training
        test_size: Number of bars for testing
        **model_params: Parameters for the model

    Returns:
        List of test results
    """
    from sklearn.preprocessing import StandardScaler

    results = []
    df = df.dropna(subset=features)

    for i in range(train_size, len(df) - test_size, test_size):
        # Train window
        train_df = df.iloc[i-train_size:i]
        test_df = df.iloc[i:i+test_size]

        # Prepare data
        X_train = train_df[features]
        y_train = (train_df['close'].shift(-24) > train_df['close']).astype(int)

        X_test = test_df[features]
        y_test = (test_df['close'].shift(-24) > test_df['close']).astype(int)

        # Remove NaN labels
        valid_train = ~y_train.isna()
        valid_test = ~y_test.isna()

        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()

        results.append({
            'start': test_df.index[0],
            'end': test_df.index[-1],
            'accuracy': accuracy,
            'trades': len(y_test)
        })

    return pd.DataFrame(results)


def compare_models(df, models, scalers, features, test_size=500):
    """
    Compare multiple models on the same test set

    Args:
        df: DataFrame with features
        models: Dict of {name: model}
        scalers: Dict of {name: scaler}
        features: List of feature names
        test_size: Number of bars for testing

    Returns:
        Comparison DataFrame
    """
    test_data = df.tail(test_size)

    results = []
    for name, model in models.items():
        scaler = scalers.get(name)
        if scaler is None:
            continue

        res, _ = backtest_model(test_data, model, scaler, features)
        res['model'] = name
        results.append(res)

    return pd.DataFrame(results)
