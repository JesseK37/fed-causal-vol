# %% [markdown]
# # Notebook 02 — Exploratory data analysis
#
# **Goal:** Understand the data before touching any causal machinery.
# Good EDA in a causal project has a specific purpose: it helps us spot
# whether our key identifying assumptions are plausible, and it reveals
# any data quirks that could confound our estimates.
#
# **Questions we want to answer here:**
# 1. What does volatility look like around FOMC dates — is there a
#    visible pattern even naively?
# 2. Are there distinct market *regimes* (low-vol vs high-vol) that
#    we should condition on?
# 3. Is the rate-change variable (our "treatment") correlated with
#    other variables in a way that would confound a naive OLS?

# %% [markdown]
# ## 0. Setup

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from src.data_loader import load_from_db

sns.set_theme(style="whitegrid", font_scale=1.05)

df = load_from_db("master_daily")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

fomc = load_from_db("fomc_events")
fomc["date"] = pd.to_datetime(fomc["date"])

print(f"Loaded master_daily: {df.shape}")
print(df.head())

# %% [markdown]
# ## 1. Volatility distribution and summary stats
#
# Before any event analysis, we want to know the unconditional
# distribution of our outcome variable (VIX and realised vol).

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

df["vix_close"].hist(bins=60, ax=axes[0], color="#185FA5", edgecolor="white", linewidth=0.3)
axes[0].set_title("VIX distribution (2000–2024)")
axes[0].set_xlabel("VIX level")
axes[0].axvline(df["vix_close"].mean(), color="#D85A30", linestyle="--", label=f"Mean = {df['vix_close'].mean():.1f}")
axes[0].legend()

df["realised_vol_21d"].dropna().hist(bins=60, ax=axes[1], color="#0F6E56", edgecolor="white", linewidth=0.3)
axes[1].set_title("Realised vol 21d (annualised, 2000–2024)")
axes[1].set_xlabel("Realised volatility")

plt.tight_layout()
plt.savefig("../outputs/figures/02_vol_distributions.png", dpi=150)
plt.show()

print("\nVIX summary:")
print(df["vix_close"].describe().round(2))

# %% [markdown]
# ## 2. Volatility regimes
#
# Volatility is well-known to be *regime-switching* — there are
# extended periods of calm punctuated by crisis spikes.  We define
# regimes using VIX quantiles:
# - **Low vol:** VIX < 15th percentile
# - **Normal:** 15th–75th percentile
# - **High vol / stress:** VIX > 75th percentile
#
# This is important for our causal analysis because the *effect* of Fed
# policy on volatility may differ substantially across regimes.  We will
# use regime as a conditioning variable later.

# %%
low_thresh  = df["vix_close"].quantile(0.15)
high_thresh = df["vix_close"].quantile(0.75)

df["vol_regime"] = pd.cut(
    df["vix_close"],
    bins=[-np.inf, low_thresh, high_thresh, np.inf],
    labels=["low", "normal", "high"]
)

print(f"Regime thresholds:  Low < {low_thresh:.1f}  |  Normal  |  High > {high_thresh:.1f}")
print("\nRegime counts:")
print(df["vol_regime"].value_counts())

# %%
fig, ax = plt.subplots(figsize=(13, 4))
colors = {"low": "#1D9E75", "normal": "#185FA5", "high": "#D85A30"}
for regime, color in colors.items():
    mask = df["vol_regime"] == regime
    ax.fill_between(df.index, df["vix_close"].where(mask), alpha=0.4, color=color, label=regime)
ax.plot(df.index, df["vix_close"], color="#d62728", linewidth=0.5, alpha=0.6)
ax.set_title("VIX coloured by volatility regime")
ax.set_ylabel("VIX")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(title="Regime")
plt.tight_layout()
plt.savefig("../outputs/figures/02_vol_regimes.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Naive event study — volatility around FOMC dates
#
# An event study plots the average value of an outcome variable in a
# window around a set of events.  This is a *descriptive* tool —
# it does NOT establish causation.  But it tells us whether there is
# something interesting to explain.
#
# We look at the [-10, +10] trading day window around each FOMC
# rate-change event.

# %%
WINDOW = 10  # trading days before and after

event_returns = []
event_vix     = []

for _, row in fomc.iterrows():
    event_date = row["date"]
    try:
        loc = df.index.get_loc(event_date)
    except KeyError:
        continue
    start_loc = max(0, loc - WINDOW)
    end_loc   = min(len(df) - 1, loc + WINDOW)
    window_df = df.iloc[start_loc : end_loc + 1].copy()
    window_df["t"] = range(-(loc - start_loc), end_loc - loc + 1)
    window_df["rate_change"] = row["rate_change"]
    event_vix.append(window_df[["t", "vix_close", "rate_change"]])

event_panel = pd.concat(event_vix, ignore_index=True)

# Average VIX level relative to event day (day 0 = FOMC date)
avg_by_t = event_panel.groupby("t")["vix_close"].agg(["mean", "sem"])
avg_by_t["ci_upper"] = avg_by_t["mean"] + 1.96 * avg_by_t["sem"]
avg_by_t["ci_lower"] = avg_by_t["mean"] - 1.96 * avg_by_t["sem"]

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_by_t.index, avg_by_t["mean"], color="#185FA5", linewidth=2, label="Mean VIX")
ax.fill_between(avg_by_t.index, avg_by_t["ci_lower"], avg_by_t["ci_upper"],
                alpha=0.2, color="#185FA5", label="95% CI")
ax.axvline(0, color="#D85A30", linestyle="--", linewidth=1.2, label="FOMC decision date")
ax.axhline(df["vix_close"].mean(), color="gray", linestyle=":", linewidth=0.8,
           label=f"Unconditional mean VIX ({df['vix_close'].mean():.1f})")
ax.set_xlabel("Trading days relative to FOMC decision")
ax.set_ylabel("VIX level")
ax.set_title("Average VIX in ±10-day window around FOMC rate-change events")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/02_event_study_naive.png", dpi=150)
plt.show()

# %% [markdown]
# **Reading this plot carefully:**
# We can see [describe what you observe after running].
# However, this is a *raw* average — it mixes hikes and cuts, large and
# small moves, and different market regimes.  It also does not
# distinguish *anticipated* from *surprise* policy moves.  That is
# exactly the identification problem we will solve in Notebook 03.

# %% [markdown]
# ## 4. Correlation check — the confounding problem
#
# This is the key EDA step for causal inference.  We want to understand
# which variables are correlated with *both* the treatment (rate_change)
# and the outcome (VIX change) — these are potential confounders that
# our identification strategy must handle.

# %%
# We test the correlations between our variables.
analysis_vars = ["vix_close", "realised_vol_21d", "spx_return",
                 "dff", "dgs10", "rate_change"]
corr = df[analysis_vars].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Pairwise correlations — key variables")
plt.tight_layout()
plt.savefig("../outputs/figures/02_correlation_heatmap2.png", dpi=150)
plt.show()

# The rate_change variable has almost no correlation with any of the other variables.
# We have large positive correlation (to be expected) between DFF and DGS10
# and between VIX and the realised volatility. There is a mild negative correlation between
# VIX and S&P return/DFF
# With this in mind, we will take a look at the correlations on the days when the
# rate change is large enough (>0.24% i.e. >24 bps).

df_moves = df[df["rate_change"].abs() >= 0.24].copy()

analysis_vars = ["vix_close", "realised_vol_21d", "spx_return",
                 "dff", "dgs10", "rate_change"]
corr = df_moves[analysis_vars].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title(f"Pairwise correlations — FOMC move days only (n={len(df_moves)})")
plt.tight_layout()
plt.savefig("../outputs/figures/02_correlation_heatmap.png", dpi=150)
plt.show()


# %% [markdown]
# ## 5. Pre-FOMC drift — a classic confounder
#
# Literature (Lucca & Moench 2015) documents a 'pre-FOMC announcement
# drift': equity markets rise in the 24h *before* decisions.  This is
# evidence that FOMC dates are not just random events — markets
# *anticipate* them.  We need to account for this.

# %%
pre_fomc_returns = []
for _, row in fomc.iterrows():
    event_date = row["date"]
    try:
        loc = df.index.get_loc(event_date)
    except KeyError:
        continue
    if loc < 2:
        continue
    pre_return = df["spx_return"].iloc[loc - 1]
    pre_fomc_returns.append({
        "date": event_date,
        "pre_return": pre_return,
        "rate_change": row["rate_change"]
    })

pre_df = pd.DataFrame(pre_fomc_returns)
print("Mean S&P 500 return on day BEFORE FOMC decision:", pre_df["pre_return"].mean().round(4))
print("Mean S&P 500 return on ALL days:", df["spx_return"].mean().round(4))
print("\nThis difference will be visible in a t-test:")

from scipy import stats
t_stat, p_val = stats.ttest_1samp(pre_df["pre_return"].dropna(), 0)
print(f"  t = {t_stat:.3f},  p = {p_val:.4f}")

# %% [markdown]
# ## Summary — what EDA tells us
#
# 1. VIX is right-skewed and regime-dependent — we should model
#    log(VIX) or use regime-stratified estimates.
# 2. There is a *visible* pattern around FOMC dates in the raw data,
#    but it mixes anticipated and surprise moves.
# 3. The rate level (DFF) is strongly correlated with VIX and correlated with the
#    rate change — a confounder we must include as a control.
# 4. There is a pre-FOMC drift effect, suggesting markets partially
#    anticipate decisions.  This motivates the surprise instrument.
#
# **Next:** `03_causal_model.py` — formalising these insights into a
# DAG and implementing the IV identification strategy.
