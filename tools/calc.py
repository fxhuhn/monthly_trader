"""Toolset for Indicators and Upsampling for Week"""

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, intervall: int = 14, smoothing: str = "sma") -> pd.Series:
    # Ref: https://stackoverflow.com/a/74282809/

    high, low, prev_close = df["High"], df["Low"], df["Close"].shift()
    tr_all = [high - low, high - prev_close, low - prev_close]
    tr_all = [tr.abs() for tr in tr_all]
    tr = pd.concat(tr_all, axis=1).max(axis=1)

    if smoothing == "rma":
        return rma(tr, intervall)
    if smoothing == "ema":
        return ema(tr, intervall)
    if smoothing == "sma":
        return sma(tr, intervall)
    raise ValueError(f"unknown smothing type {smoothing}")


def resample_week(df: pd.DataFrame) -> pd.DataFrame:
    # expected columns Date, Open, High, Low, Close

    df["Date"] = df.index
    df["week"] = df["Date"].dt.strftime("%y-%W")

    df = df.groupby("week").agg(
        Date=("Date", "last"),
        Low=("Low", "min"),
        High=("High", "max"),
        Open=("Open", "first"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    return df.reset_index().set_index("Date")


def sma(close: pd.Series, period: int = 200) -> pd.Series:
    return round(close.rolling(period).mean(), 2)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return round((close - close.shift(period)) / close.shift(period) * 100, 2)


def ema(close: pd.Series, period: int = 200) -> pd.Series:
    return (
        close.ewm(span=period, min_periods=period, adjust=False, ignore_na=False)
        .mean()
        .round(2)
    )


def rma(close: pd.Series, intervall: int) -> pd.Series:
    return (
        close.ewm(
            alpha=1 / intervall, min_periods=intervall, adjust=False, ignore_na=False
        )
        .mean()
        .round(2)
    )


def rsi(close: pd.Series, period: int = 7) -> pd.Series:
    """
    The relative strength index (RSI) is intended to chart the current and
    historical strength or weakness of a stock or market based on the
    closing prices of a recent trading period.

    Args:
        close (pd.Series): _description_
        period (int, optional): _description_. Defaults to 7.

    Returns:
        pd.Series: _description_
    """

    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = close.diff(1)

    # Make the positive gains (up) and negative gains (down) Series
    up = delta.where(delta > 0, 0.0)
    down = -1 * delta.where(delta < 0, 0.0)

    # Calculate the EWMA
    roll_up = up.ewm(min_periods=period, adjust=False, alpha=(1 / period)).mean()
    roll_down = down.ewm(min_periods=period, adjust=False, alpha=1 / period).mean()

    # Calculate the RSI based on EWMA
    rs = roll_up / roll_down
    rs_index = 100.0 - (100.0 / (1.0 + rs))

    return round(rs_index, 0)


def adx(stock, n=14):
    """Calculate the Average Directional Index (ADX)

    Parameters:
    df (pd.DataFrame): DataFrame which contain the asset price
    n (int): The period. default is 14

    Returns:
    df (pd.DataFrame): Updated DataFrame with the ADX values
    """
    df = stock.copy()
    df["H-L"] = abs(df["High"] - df["Low"])
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["+DM"] = np.where(
        (df["High"] > df["High"].shift(1)), df["High"] - df["High"].shift(1), 0
    )
    df["-DM"] = np.where(
        (df["Low"] < df["Low"].shift(1)), df["Low"].shift(1) - df["Low"], 0
    )
    df["+DM"] = np.where(df["+DM"] < df["-DM"], 0, df["+DM"])
    df["-DM"] = np.where(df["-DM"] < df["+DM"], 0, df["-DM"])
    TR_n = df["TR"].rolling(n).sum()
    DM_plus_n = df["+DM"].rolling(n).sum()
    DM_minus_n = df["-DM"].rolling(n).sum()
    df["+DI"] = 100 * (DM_plus_n / TR_n)
    df["-DI"] = 100 * (DM_minus_n / TR_n)
    df["DX"] = 100 * abs((df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]))
    return df["DX"].rolling(n).mean()


def macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
):
    fast = df["Close"].ewm(span=fast_period, min_periods=fast_period).mean()
    slow = df["Close"].ewm(span=slow_period, min_periods=slow_period).mean()
    signal = (fast - slow).ewm(span=signal_period, min_periods=signal_period).mean()
    return fast, slow, signal
