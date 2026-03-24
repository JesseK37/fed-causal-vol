# %% [markdown]
# # Notebook 01 — Data ingestion
#
# **Goal:** Pull all raw data we need, do minimal cleaning, and store
# everything in a local SQLite database so subsequent notebooks can
# load data instantly without hitting the APIs again.
#
# **Data sources:**
# | Source | What we fetch | Why |
# |--------|--------------|-----|
# | FRED API | Daily Fed Funds rate (DFF), 10yr Treasury (DGS10) | Official Fed policy data |
# | yfinance | S&P 500 (^GSPC), VIX (^VIX) daily OHLCV | Equity + implied vol |
#
# **Prerequisites:**
# 1. `pip install -r requirements.txt`
# 2. Copy `.env.example` → `.env` and paste your FRED API key
#    (free at https://fred.stlouisfed.org/docs/api/api_key.html)

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path

# Allow imports from src/ regardless of where Jupyter is launched from
sys.path.insert(0, str(Path("..").resolve()))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.data_loader import (
    fetch_fred_series,
    fetch_fomc_dates,
    fetch_equity_data,
    compute_realised_vol,
    save_to_db,
)

# Date range for the whole project
# We use 2000–present to capture multiple rate cycles and volatility regimes
START = "2000-01-01"
END   = "2024-12-31"

print(f"Project date range: {START} → {END}")

# %% [markdown]
# ## 1. Fetch Fed policy data from FRED
#
# We pull two series:
# - **DFF** — Daily Effective Federal Funds Rate (the policy rate itself)
# - **DGS10** — 10-Year Treasury Constant Maturity Rate (a control variable)
#
# The FRED API is rate-limited but generous for personal use.

# %%
print("Fetching DFF (daily fed funds rate)...")
dff = fetch_fred_series("DFF", START, END)
print(f"  DFF: {len(dff):,} observations, {dff.index.min().date()} to {dff.index.max().date()}")

print("Fetching DGS10 (10yr Treasury)...")
dgs10 = fetch_fred_series("DGS10", START, END)
print(f"  DGS10: {len(dgs10):,} observations")

# %%
# Quick sanity check — plot the rate history
fig, ax = plt.subplots(figsize=(12, 4))
dff.plot(ax=ax, color="#185FA5", linewidth=1.2, label="Fed Funds Rate (DFF)")
ax.set_title("Federal Funds Rate — daily, 2000–2024", fontsize=13)
ax.set_ylabel("Rate (%)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/01_fed_funds_rate.png", dpi=150)
plt.show()
print("Plot saved to outputs/figures/")

# %% [markdown]
# ## 2. Derive FOMC decision dates
#
# We identify FOMC meeting dates as days when the daily fed funds rate
# changed.  This is the Kuttner (2001) convention and is accurate for
# the post-1994 period when the Fed began explicitly announcing targets.
#
# **Why this matters for causal inference:** The FOMC meets on a
# pre-announced schedule.  This gives us a clean "event" around which
# to study market reactions — the timing is not endogenous to daily
# market moves, which is a key identification assumption we will
# formalise in Notebook 03.

# %%
fomc_df = fetch_fomc_dates(START, END)
fomc_events = fomc_df[fomc_df["is_fomc_date"]].copy()

print(f"Total FOMC rate-change events identified: {len(fomc_events)}")
print("\nSample of FOMC events:")
print(fomc_events[["date", "rate", "rate_change"]].head(10).to_string(index=False))

# %%
# Distribution of rate changes
fig, ax = plt.subplots(figsize=(8, 4))
fomc_events["rate_change"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color="#0F6E56", edgecolor="white", linewidth=0.5
)
ax.set_title("Distribution of FOMC rate changes (2000–2024)", fontsize=13)
ax.set_xlabel("Rate change (percentage points)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../outputs/figures/01_fomc_rate_changes.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Fetch equity and volatility data from yfinance
#
# - **^GSPC** — S&P 500 index (our equity market proxy)
# - **^VIX** — CBOE Volatility Index (our implied volatility proxy)
#
# We also compute *realised* volatility from S&P 500 returns using a
# 21-day rolling window (≈ 1 trading month).  This gives us two
# volatility measures to study: forward-looking (VIX) and backward-
# looking (realised vol).

# %%
print("Fetching equity data (S&P 500 + VIX)...")
equity_raw = fetch_equity_data(["^GSPC", "^VIX"], START, END)
print(f"Shape: {equity_raw.shape}")
print(equity_raw.head())

# %%
# Extract clean close prices
spx_close = equity_raw["Close"]["^GSPC"].rename("spx_close")
vix_close = equity_raw["Close"]["^VIX"].rename("vix_close")

# Daily log returns on S&P 500
spx_returns = spx_close.pct_change().rename("spx_return")

# Realised vol — 21-day and 5-day windows (we'll use both in analysis)
realised_vol_21d = compute_realised_vol(spx_close, window=21)
realised_vol_5d  = compute_realised_vol(spx_close, window=5)

print("S&P 500 daily returns summary:")
print(spx_returns.describe().round(4))

# %%
# Visual check: VIX vs realised vol
fig, ax = plt.subplots(figsize=(12, 4))
vix_close.plot(ax=ax, color="#D85A30", linewidth=0.8, alpha=0.8, label="VIX (implied vol)")
(realised_vol_21d * 100).plot(ax=ax, color="#185FA5", linewidth=0.8, alpha=0.8,
                               label="Realised vol 21d (annualised %)")
ax.set_title("Implied vs realised volatility — S&P 500, 2000–2024", fontsize=13)
ax.set_ylabel("Volatility (%)")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/01_vol_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 4. Build the master dataset and store to SQLite
#
# We merge everything into a single daily panel and write to SQLite.
# The database lives at `data/raw/fed_vol.db` and is gitignored
# (too large to commit, and reproducible from this notebook).

# %%
master = pd.DataFrame({
    "date":           spx_close.index,
    "spx_close":      spx_close.values,
    "spx_return":     spx_returns.values,
    "vix_close":      vix_close.reindex(spx_close.index).values,
    "realised_vol_21d": realised_vol_21d.values,
    "realised_vol_5d":  realised_vol_5d.values,
    "dff":            dff.reindex(spx_close.index, method="ffill").values,
    "dgs10":          dgs10.reindex(spx_close.index, method="ffill").values,
})

# Merge FOMC flags
fomc_flags = fomc_df.set_index("date")[["rate_change", "is_fomc_date"]]
fomc_flags.index = pd.to_datetime(fomc_flags.index)
master = master.set_index("date")
master = master.join(fomc_flags, how="left")
master["is_fomc_date"] = master["is_fomc_date"].fillna(False)
master["rate_change"]  = master["rate_change"].fillna(0.0)

print(f"Master dataset: {master.shape[0]:,} rows × {master.shape[1]} columns")
print(f"FOMC event days: {master['is_fomc_date'].sum()}")
print("\nMissing values per column:")
print(master.isna().sum())

# %%
save_to_db(master.reset_index(), "master_daily")
save_to_db(fomc_events, "fomc_events")

print("\nAll data saved. Proceed to 02_eda.py")

# %% [markdown]
# ## 5. What we have — a summary
#
# | Table | Rows | Key columns |
# |-------|------|-------------|
# | `master_daily` | ~6,200 | date, spx_return, vix_close, realised_vol_21d, dff, is_fomc_date, rate_change |
# | `fomc_events` | ~60 | date, rate, rate_change |
#
# **Next:** `02_eda.py` — exploratory analysis, regime identification,
# and preliminary event plots around FOMC dates.
