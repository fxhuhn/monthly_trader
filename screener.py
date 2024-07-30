import operator
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from nasdaq_100_ticker_history import tickers_as_of

from tools import atr, roc, sma

# def
MAX_STOCKS = 10


def get_monthly_index():
    sp_500 = yf.download("^GSPC")
    sp_500["sma"] = sma(sp_500.Close, 300)
    sp_500["Date"] = sp_500.index
    sp_500["month"] = sp_500["Date"].dt.strftime("%y-%m")

    sp_500 = (
        sp_500.groupby("month")
        .agg(
            Date=("Date", "last"),
            Close=("Close", "last"),
            sma=("sma", "last"),
        )
        .reset_index()
        .set_index("Date")
        .sort_index()
    )
    return sp_500


def get_nasdaq_symbols_monthly(year: int, month: int) -> List:
    return list(tickers_as_of(year, month, 1))


def get_nasdaq_symbols() -> List:
    nasdaq_tickers = dict()
    for year in range(2016, 2025, 1):
        for month in range(1, 13, 1):
            nasdaq_tickers[f"{year - 2000}-{month:02}"] = list(
                tickers_as_of(year, month, 1)
            )

    all = []

    for value in nasdaq_tickers.values():
        all = all + value
    nasdaq_tickers["all"] = list(set(all))

    return nasdaq_tickers["all"]


def get_stocks(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """_summary_

    Args:
        symbols (List[str]): _description_

    Returns:
        Dict[str, pd.DataFrame]: _description_
    """

    dfs = {}
    stock_data = yf.download(
        symbols,
        rounding=2,
        progress=False,
        group_by="ticker",
    )

    # perform some pre preparation
    for symbol in stock_data.columns.get_level_values(0).unique():
        # drop unclear items
        df = stock_data[symbol]
        df = df[~(df.High == df.Low)]
        df = df.dropna()
        df.index = pd.to_datetime(df.index)

        if len(df):
            dfs[symbol.lower()] = df

    return dfs


def resample_stocks_to_month(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = df.index
    df["month"] = df["Date"].dt.strftime("%y-%m")

    df = df.groupby("month").agg(
        Date=("Date", "last"),
        Open=("Open", "first"),
        Close=("Close", "last"),
        score=("score", "last"),
    )
    return df.reset_index().set_index("Date").sort_index()


def get_score(data: pd.DataFrame) -> pd.Series:
    roc_intervall = [intervall for intervall in range(20, 260, 20)]

    for intervall in roc_intervall:
        data[f"roc_{intervall}"] = (roc(data.Close, 20)).shift(intervall)

    data["score"] = np.where(
        (data.Close > sma(data.Close, 100)) & (atr(data, 100) > atr(data, 20)),
        data[[f"roc_{intervall}" for intervall in roc_intervall]].mean(axis=1),
        np.nan,
    )

    return data["score"]


def ndx100_list():
    table = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100#Components")[4]
    return list(table.Ticker)


def prepare_stocks(index: pd.DataFrame) -> pd.DataFrame:
    tracker = index.copy()
    # stocks = get_stocks(ndx100_list())
    stocks = get_stocks(get_nasdaq_symbols())

    for symbol, df in stocks.items():
        df["score"] = get_score(df)
        df = resample_stocks_to_month(df)
        df[symbol] = df["score"]

        tracker = pd.merge(
            tracker, df[[symbol]], left_index=True, right_index=True, how="left"
        )
    return tracker


def get_top_stocks(stocks: dict) -> dict:
    year = 2000 + int(stocks["month"][:2])
    month = int(stocks["month"][-2:])

    if month == 12:
        year = year + 1
        month = 1
    else:
        month = month + 1

    nasdaq_symbols = get_nasdaq_symbols_monthly(year, month)
    nasdaq_symbols = [symbol.lower() for symbol in nasdaq_symbols]

    stocks.pop("month")
    stocks.pop("Close")
    stocks.pop("sma")

    stocks.pop("googl")

    stocks = {k: v for k, v in stocks.items() if v > 0 and k in nasdaq_symbols}

    return sorted(stocks.items(), key=operator.itemgetter(1))[-MAX_STOCKS:]


if __name__ == "__main__":
    sp_500 = get_monthly_index()
    ndx_stocks = prepare_stocks(index=sp_500)

    if sp_500.iloc[-1].Close > sp_500.iloc[-1].sma:
        top_stocks = get_top_stocks(ndx_stocks.iloc[-2].dropna().to_dict())
        current_month = [ticker for (ticker, _) in top_stocks]

        top_stocks = get_top_stocks(ndx_stocks.iloc[-1].to_dict())
        next_month = [ticker for (ticker, _) in top_stocks]

        unchanged_stocks = set(current_month).intersection(next_month)
        removed_stocks = set(current_month) - set(next_month)
        added_stocks = set(next_month) - set(current_month)

        changes_txt = "# Planned transactions for next month\n"
        changes_txt = (
            changes_txt
            + "\n## New\n"
            + "\n".join([f"+ {stocks}" for stocks in added_stocks])
        )
        changes_txt = (
            changes_txt
            + "\n## Unchanged\n"
            + "\n".join([f"* {stocks}" for stocks in unchanged_stocks])
        )
        changes_txt = (
            changes_txt
            + "\n## Leave\n"
            + "\n".join([f"- {stocks}" for stocks in removed_stocks])
        )

        with open("CHANGES.md", "w") as text_file:
            text_file.write(changes_txt)
