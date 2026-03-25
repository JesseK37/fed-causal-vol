# %% [markdown]
# # Notebook 04 — Robustness checks
#
# **Goal:** Stress-test the main result from Notebook 03.  A result
# that only holds under one specific set of choices is not credible.
# Robustness checks are what separate a rigorous analysis from a
# cherry-picked one.
#
# **Tests we run:**
# 1. **Placebo test** — run the same IV on random non-FOMC dates.
#    If we find a similar effect, our result is spurious.
# 2. **Event window sensitivity** — does the result hold for [0,+1],
#    [0,+3], [0,+5] day windows?
# 3. **Regime heterogeneity** — does the effect differ in low vs high
#    volatility regimes?
# 4. **Subsample stability** — is the result driven by a particular
#    crisis period (e.g. 2008, 2020)?

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from src.data_loader import load_from_db

df = load_from_db("master_daily")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

fomc_analysis = pd.read_csv("../data/processed/fomc_analysis.csv", 
                             index_col="date", parse_dates=True)
print(f"Loaded fomc_analysis: {fomc_analysis.shape}")
print(fomc_analysis.head())

fomc_events = load_from_db("fomc_events")
fomc_events["date"] = pd.to_datetime(fomc_events["date"])
fomc_set = set(fomc_events["date"])

MAIN_ESTIMATE = 0.3049 #IV coefficient

# %% [markdown]
# ## 1. Placebo test
#
# **Logic:** If our IV picks up a genuine causal effect of FOMC
# decisions on VIX, the same IV estimated on *randomly chosen non-FOMC
# dates* should produce coefficients near zero.
#
# We draw 1,000 random samples of pseudo-event dates (same number as
# real FOMC events, drawn from non-FOMC trading days) and run the full
# IV on each.  We then compare the distribution of placebo coefficients
# to our main estimate.

# %%
np.random.seed(42)

non_fomc_dates = df.index[~df.index.isin(fomc_set)].tolist()
n_fomc = len(fomc_events)
N_PLACEBO = 1000

placebo_coefs = []

for _ in range(N_PLACEBO):
    # Shift each FOMC date by a random offset of 20-60 trading days
    # Far enough from any real FOMC event to be pure noise
    offsets = np.random.randint(20, 60, size=len(fomc_analysis))
    placebo_dates = [
        df.index[min(df.index.get_loc(d) + o, len(df.index) - 1)]
        for d, o in zip(fomc_analysis.index, offsets)
    ]
    placebo_dates = pd.DatetimeIndex(placebo_dates)

    pseudo = pd.DataFrame({
        "vix_change_1d": (
            df["vix_close"].reindex(placebo_dates).values
            - df["vix_close"].shift(1).reindex(placebo_dates).values
        ),
        "rate_change": df["rate_change"].reindex(placebo_dates).fillna(0).values,
        "surprise": df["rate_change"].reindex(placebo_dates).fillna(0).values
                    - df["rate_change"].rolling(6, min_periods=3).mean().shift(1)
                    .reindex(placebo_dates).fillna(0).values,
        "dff_level": df["dff"].reindex(placebo_dates).values,
        "spx_pre": df["spx_return"].shift(1).reindex(placebo_dates).values,
    }, index=placebo_dates).dropna()

    if len(pseudo) < 10:
        continue

    try:
        iv_pl = IV2SLS(
            dependent=pseudo["vix_change_1d"],
            exog=sm.add_constant(pseudo[["dff_level", "spx_pre"]]),
            endog=pseudo[["rate_change"]],
            instruments=pseudo[["surprise"]]
        ).fit(cov_type="robust")

        coef = iv_pl.params["rate_change"]
        if np.isfinite(coef) and abs(coef) < 50:
            placebo_coefs.append(coef)
    except Exception:
        continue

print(f"Placebo simulations completed: {len(placebo_coefs)}")
print(f"Placebo mean: {np.mean(placebo_coefs):.4f}")
print(f"Placebo std:  {np.std(placebo_coefs):.4f}")
print(f"Real IV estimate: {MAIN_ESTIMATE:.4f}")
print(f"Percentile of real estimate: {np.mean(np.array(placebo_coefs) < MAIN_ESTIMATE)*100:.1f}%")


fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(placebo_coefs, bins=50, color="#B5D4F4", edgecolor="white", linewidth=0.3,
        label="Placebo distribution (1,000 random samples)")
if MAIN_ESTIMATE is not None:
    ax.axvline(MAIN_ESTIMATE, color="#D85A30", linewidth=2, linestyle="--",
               label=f"Main IV estimate ({MAIN_ESTIMATE:.3f})")
ax.axvline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax.set_xlabel("IV coefficient on rate_change")
ax.set_ylabel("Count")
ax.set_title("Placebo test: IV coefficient on random non-FOMC dates")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/04_placebo_test.png", dpi=150)
plt.show()

# %% [markdown]
# **Interpretation:** If the main estimate lies in the far tail of the
# placebo distribution, this confirms it is not a statistical artefact
# of our method.  If the placebo distribution is centred near zero and
# the main estimate is far from it, this is strong evidence of a
# genuine effect.

# %% [markdown]
# ## 2. Event window sensitivity
#
# Does the effect fade over time, or does it persist?
# We re-estimate the IV for windows of [0,+1] through [0,+10] days.

# %%
# [This section requires the fomc_analysis DataFrame from notebook 03.
#  In a real workflow you would save it to the database at the end of
#  notebook 03 and load it here.  For now, we sketch the structure.]

# window_coefs = {}
# for window in range(1, 11):
#     # Construct vix_change for this window
#     # Re-run IV
#     # Store coefficient and CI
#     pass


window_results = []

for w in range(1,11):
    col = f"vix_change_{w}d"
    if col not in fomc_analysis.columns:
        continue
    iv = IV2SLS(
        dependent=fomc_analysis[col],
        exog=sm.add_constant(fomc_analysis[["dff_level", "spx_pre"]]),
        endog=fomc_analysis[["rate_change"]],
        instruments=fomc_analysis[["surprise"]]
    ).fit(cov_type="robust")
    window_results.append({
        "window": w,
        "coef": iv.params["rate_change"],
        "ci_low": iv.conf_int().loc["rate_change", "lower"],
        "ci_high": iv.conf_int().loc["rate_change", "upper"],
        "pval": iv.pvalues["rate_change"]
    })

window_df = pd.DataFrame(window_results)
print(window_df)


fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(window_df["window"], window_df["coef"], 
        color="#1f77b4", linewidth=2, marker="o", label="IV estimate")
ax.fill_between(window_df["window"], 
                window_df["ci_low"], 
                window_df["ci_high"], 
                alpha=0.2, color="#1f77b4", label="95% CI")

ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_xlabel("Event window (trading days after FOMC decision)")
ax.set_ylabel("IV coefficient on rate_change")
ax.set_title("Effect of rate change on VIX — sensitivity to event window definition")
ax.legend()
ax.set_xticks(range(1, 11))
plt.tight_layout()
plt.savefig("../outputs/figures/04_window_sensitivity.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Regime heterogeneity
#
# We split the FOMC events into low-vol and high-vol regimes and
# re-estimate the IV in each subsample.
# A larger effect in high-vol regimes would be economically meaningful
# — it suggests Fed communication is most market-moving when
# uncertainty is already elevated.

# %%
# [Requires fomc_analysis with vol_regime column merged in]
# Split by regime, re-run IV, compare coefficients
print("Regime heterogeneity analysis:")
print("Expected structure:")
print("  - Subset fomc_analysis to regime == 'low' → run IV → store coef")
print("  - Subset to regime == 'high' → run IV → store coef")
print("  - Bar chart comparing regime-specific estimates with CIs")

for regime in ["low_vol", "high_vol"]:
    subset = fomc_analysis[fomc_analysis["vol_regime"] == regime]
    print(f"\nRegime: {regime}, n={len(subset)}")
    if len(subset) < 10:
        print("  Too few observations, skipping.")
        continue
    try:
        iv = IV2SLS(
            dependent=subset["vix_change_1d"],
            exog=sm.add_constant(subset[["dff_level", "spx_pre"]]),
            endog=subset[["rate_change"]],
            instruments=subset[["surprise"]]
        ).fit(cov_type="robust")
        print(f"  IV coef: {iv.params['rate_change']:.4f}")
        print(f"  95% CI: [{iv.conf_int().loc['rate_change', 'lower']:.4f}, "
              f"{iv.conf_int().loc['rate_change', 'upper']:.4f}]")
        print(f"  p-value: {iv.pvalues['rate_change']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")


# %% [markdown]
# ## 4. Subsample stability — leave-one-crisis-out
#
# We re-estimate the IV three times, each time dropping one major
# crisis period:
# - Drop 2008–2009 (Global Financial Crisis)
# - Drop 2020 Q1–Q2 (COVID crash)
# - Drop 2022 (rate hike cycle)
#
# If the main result collapses when any one period is removed, the
# finding is not general.

# %%
crisis_windows = {
    "ex-GFC":    ("2008-09-01", "2009-06-30"),
    "ex-COVID":  ("2020-01-01", "2020-06-30"),
    "ex-hikes22":("2022-01-01", "2022-12-31"),
}

print("Subsample stability:")
print("For each window, drop FOMC events that fall in that period,")
print("re-run IV on the remaining events, and compare to full-sample estimate.")

subsamples = {
    "2000-2007": ("2000-01-01", "2007-12-31"),
    "2008-2015": ("2008-01-01", "2015-12-31"),
    "2016-2024": ("2016-01-01", "2024-12-31"),
}

for label, (start, end) in subsamples.items():
    subset = fomc_analysis.loc[start:end]
    if len(subset) < 10:
        print(f"{label}: too few observations ({len(subset)})")
        continue
    print(f"\nSubsample: {label}, n={len(subset)}")
    try:
        iv = IV2SLS(
            dependent=subset["vix_change_1d"],
            exog=sm.add_constant(subset[["dff_level", "spx_pre"]]),
            endog=subset[["rate_change"]],
            instruments=subset[["surprise"]]
        ).fit(cov_type="robust")
        print(f"  IV coef: {iv.params['rate_change']:.4f}")
        print(f"  95% CI: [{iv.conf_int().loc['rate_change', 'lower']:.4f}, "
              f"{iv.conf_int().loc['rate_change', 'upper']:.4f}]")
        print(f"  p-value: {iv.pvalues['rate_change']:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")



# %% [markdown]
# ## Summary — robustness checklist
#
# | Check | Status | Interpretation |
# |-------|--------|----------------|
# | Placebo test | [ ] | Placebo dist centred at 0, main estimate in tail? |
# | Window sensitivity | [ ] | Effect stable / fades as expected? |
# | Regime heterogeneity | [ ] | Larger effect in high-vol regime? |
# | Subsample stability | [ ] | Result survives dropping each crisis? |
#
# **Next:** `05_write_up.py` — producing the final figures and summary
# statistics for the README.
