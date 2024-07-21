import operator
from datetime import timedelta
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
    sp_500["sma"] = sma(sp_500.Close, 275)
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
    roc_intervall = [intervall for intervall in range(20, 200, 20)]

    for intervall in roc_intervall:
        data[f"roc_{intervall}"] = (roc(data.Close, 20)).shift(intervall)

    data["score"] = np.where(
        (data.Close > sma(data.Close, 100)) & (atr(data, 60) > atr(data, 20)),
        data[[f"roc_{intervall}" for intervall in roc_intervall]].mean(axis=1),
        np.nan,
    )

    return data["score"]


def ndx100_list():
    table = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100#Components")[4]
    return list(table.Ticker)


def prepare_stocks(index: pd.DataFrame) -> pd.DataFrame:
    tracker = index.copy()
    stocks = get_stocks(ndx100_list())

    for symbol, df in stocks.items():
        df["score"] = get_score(df)
        df = resample_stocks_to_month(df)
        df[symbol] = df["score"]

        tracker = pd.merge(
            tracker, df[[symbol]], left_index=True, right_index=True, how="left"
        )
    return tracker


def get_top_stocks(stocks: dict) -> dict:
    nasdaq_symbols = get_nasdaq_symbols_monthly(
        2000 + int(stocks["month"][:2]), int(stocks["month"][-2:])
    )
    nasdaq_symbols = [symbol.lower() for symbol in nasdaq_symbols]

    stocks.pop("month")
    stocks.pop("Close")
    stocks.pop("sma")
    stocks = {k: v for k, v in stocks.items() if v > 0 and k in nasdaq_symbols}

    return sorted(stocks.items(), key=operator.itemgetter(1))[-MAX_STOCKS:]


if __name__ == "__main__":
    sp_500 = get_monthly_index()
    ndx_stocks = prepare_stocks(index=sp_500)
    ndx_stocks = ndx_stocks["2019-12-01":]

    portfolio = []
    for month in range(len(ndx_stocks)):
        if (ndx_stocks.iloc[month].Close > ndx_stocks.iloc[month].sma) and (
            ndx_stocks.iloc[month].month != ndx_stocks.month.max()
        ):
            top_stocks = get_top_stocks(ndx_stocks.iloc[month].dropna().to_dict())

            for ticker in [ticker for (ticker, _) in top_stocks]:
                portfolio.append(
                    {
                        "month": f"{(ndx_stocks.iloc[month].name+timedelta(days=10)).year}-{(ndx_stocks.iloc[month].name+timedelta(days=10)).month:0>2}",
                        "symbol": ticker,
                    }
                )
    portfolio = pd.DataFrame(portfolio)

    stocks = get_stocks(list(portfolio.symbol.unique()))
    for pos, position in portfolio.iterrows():
        df = stocks[position.symbol]
        df["Date"] = df.index
        df["month"] = df["Date"].dt.strftime("%Y-%m")

        df_month = df.groupby("month").agg(
            End=("Date", "last"),
            Start=("Date", "first"),
            Open=("Open", "first"),
            Close=("Close", "last"),
        )
        try:
            portfolio.loc[pos, "start"] = df_month.loc[position.month].Start
            portfolio.loc[pos, "end"] = df_month.loc[position.month].End
            portfolio.loc[pos, "buy"] = df_month.loc[position.month].Open
            portfolio.loc[pos, "sell"] = df_month.loc[position.month].Close
            portfolio.loc[pos, "profit"] = (
                (
                    (
                        df_month.loc[position.month].Close
                        / df_month.loc[position.month].Open
                    )
                    - 1
                )
                * 100
            ).round(1)
        except:
            pass

    trade_journal = portfolio.set_index("month").astype(str).to_markdown(floatfmt=".2f")

    portfolio["invest"] = (10_000 / portfolio["buy"]).astype(int) * portfolio["buy"]
    portfolio["profit"] = (10_000 / portfolio["buy"]).astype(int) * portfolio["sell"]

    monthly = (
        portfolio.groupby("month")
        .agg(
            Positions=("symbol", "count"),
            Invest=("invest", "sum"),
            Profit=("profit", "sum"),
        )
        .dropna()
    )
    monthly["earning"] = (
        (monthly.Profit - monthly.Invest) / monthly.Invest * 100
    ).round(1)

    readme_txt = f"# NASDAQ 100 Trader\nStock Trading and Screening only end of month. With an average monthly return of {monthly.earning.mean():.2f}%. Every month!\n\n"
    readme_txt = (
        readme_txt
        + f'## Average Monthly Return\n{monthly.groupby(monthly.index.str[-2:]).agg(profit=("earning", "mean")).to_markdown(floatfmt=".2f")}\n\n'
    )

    readme_txt = readme_txt + f"## Tradehistory\n{trade_journal}\n\n"

    with open("README.md", "w") as text_file:
        text_file.write(readme_txt)
