"""
data_loader.py
--------------
Functions for fetching raw data from FRED and yfinance and persisting
to a local SQLite database.  Notebooks import from here to keep
analysis cells clean.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "raw" / "fed_vol.db"


def get_db_connection() -> sqlite3.Connection:
    """Return a connection to the project SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


# ── FRED ─────────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """
    Fetch a single FRED series by ID.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. 'FEDFUNDS' or 'DFF'.
    start, end : str
        Date strings in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.Series with a DatetimeIndex.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Copy .env.example to .env and add your key."
        )
    fred = Fred(api_key=api_key)
    series = fred.get_series(series_id, observation_start=start, observation_end=end)
    series.name = series_id
    return series


def fetch_fomc_dates(start: str, end: str) -> pd.DataFrame:
    """
    Fetch the effective federal funds rate (daily, DFF) and derive FOMC
    decision dates as days where the rate changed.

    This is an approximation — for production use you would scrape the
    official FOMC calendar from federalreserve.gov.  The approximation
    is valid for event-study purposes because rate changes and meeting
    dates coincide almost perfectly since 1994.

    Returns
    -------
    DataFrame with columns: date, rate, rate_change, is_fomc_date
    """
    dff = fetch_fred_series("DFF", start, end)
    df = dff.reset_index()
    df.columns = ["date", "rate"]
    df["rate_change"] = df["rate"].diff()
    df["is_fomc_date"] = df["rate_change"].abs() > 0.001
    return df


def fetch_fed_futures_surprise(start: str, end: str) -> pd.DataFrame:
    """
    Approximate the 'Fed surprise' component using the method of
    Kuttner (2001): the change in the current-month fed funds futures
    contract on FOMC days.  We proxy this with the 30-day Fed Funds
    futures (ZQ) available via yfinance for recent history.

    Note: For a full replication going back to the 1990s, you would
    need the CME futures data.  This function fetches what is available
    and notes the limitation clearly — documenting data constraints is
    itself a sign of good analytical practice.

    Returns
    -------
    DataFrame with columns: date, futures_rate, surprise
    """
    # ZQ=F is the front-month 30-day Fed Funds futures contract
    ticker = yf.Ticker("ZQ=F")
    hist = ticker.history(start=start, end=end)
    if hist.empty:
        raise ValueError(
            "Could not fetch Fed Funds futures (ZQ=F) from yfinance. "
            "This ticker may require a paid data subscription for historical depth. "
            "See notebooks/01_data_ingestion.ipynb for the manual download fallback."
        )
    hist = hist[["Close"]].reset_index()
    hist.columns = ["date", "futures_rate"]
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None).dt.date
    hist["surprise"] = hist["futures_rate"].diff()
    return hist


# ── Equities / Volatility ────────────────────────────────────────────────────

def fetch_equity_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : list of str
        e.g. ['^GSPC', '^VIX']
    start, end : str
        Date strings in 'YYYY-MM-DD' format.

    Returns
    -------
    DataFrame with MultiIndex columns (field, ticker) or flat columns
    if only one ticker is passed.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return raw


def compute_realised_vol(prices: pd.Series, window: int = 21) -> pd.Series:
    """
    Compute annualised realised volatility from a price series using a
    rolling window of log returns.

    Parameters
    ----------
    prices : pd.Series
        Daily close prices.
    window : int
        Rolling window in trading days (default 21 ≈ 1 month).

    Returns
    -------
    pd.Series of annualised volatility (252-day convention).
    """
    log_returns = prices.pct_change().apply(lambda x: pd.Series.pipe(x, lambda s: s))
    log_returns = prices.pct_change()
    realised_vol = log_returns.rolling(window).std() * (252 ** 0.5)
    realised_vol.name = f"realised_vol_{window}d"
    return realised_vol


# ── Persistence ──────────────────────────────────────────────────────────────

def save_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
    """
    Persist a DataFrame to the project SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
    table_name : str
        Target table name.
    if_exists : str
        'replace' (default) or 'append'.
    """
    with get_db_connection() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=True)
    print(f"Saved {len(df):,} rows to table '{table_name}' in {DB_PATH.name}")


def load_from_db(table_name: str) -> pd.DataFrame:
    """Load a table from the project SQLite database into a DataFrame."""
    with get_db_connection() as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    return df
