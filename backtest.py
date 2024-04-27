import operator
from datetime import timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf

from tools import roc, sma

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
    roc_intervall = [intervall for intervall in range(20, 240, 20)]

    for intervall in roc_intervall:
        data[f"roc_{intervall}"] = roc(data.Close, intervall)

    return data[[f"roc_{intervall}" for intervall in roc_intervall]].sum(axis=1)


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
    stocks.pop("month")
    stocks.pop("Close")
    stocks.pop("sma")

    return sorted(stocks.items(), key=operator.itemgetter(1))[-MAX_STOCKS:]


if __name__ == "__main__":
    sp_500 = get_monthly_index()
    ndx_stocks = prepare_stocks(index=sp_500)
    ndx_stocks = ndx_stocks["2022-12-01":]

    portfolio = []
    for month in range(len(ndx_stocks)):
        if ndx_stocks.iloc[month].Close > ndx_stocks.iloc[month].sma:
            top_stocks = get_top_stocks(ndx_stocks.iloc[month].to_dict())

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

    trade_journal = portfolio.set_index("month").to_markdown()

    monthly = (
        portfolio.groupby("month")
        .agg(
            Positions=("symbol", "count"),
            Profit=("profit", "sum"),
        )
        .dropna()
    )

    readme_txt = f"# NASDAQ 100 Trader\nStock Trading and Screening only end of month. With a monthly return of {monthly.Profit.mean():.2f} %. Every month!\n\n"
    readme_txt = (
        readme_txt
        + f'## Monthly Return\n{portfolio.groupby(portfolio.month.str[-2:]).agg(profit=("profit", "sum")).to_markdown()}\n\n'
    )
    readme_txt = (
        readme_txt
        + f'## Tradehistory\n{portfolio.set_index("month").to_markdown()}\n\n'
    )
    # print(monthly)

    with open("README.md", "w") as text_file:
        text_file.write(readme_txt)