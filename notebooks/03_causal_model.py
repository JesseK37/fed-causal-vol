# %% [markdown]
# # Notebook 03 — Causal model and identification
#
# **Goal:** Formalise our causal question as a Directed Acyclic Graph
# (DAG), state our identification assumptions explicitly, and implement
# the Instrumental Variable (IV) estimator.
#
# This notebook is the methodological core of the project.

# %% [markdown]
# ## 0. The identification problem — why OLS fails
#
# We want to estimate the causal effect of a Fed rate change on VIX.
# A naive OLS regression of VIX on rate_change would be biased because:
#
# 1. **Reverse causality:** The Fed *responds* to market stress.  When
#    VIX spikes (e.g. 2008, 2020), the Fed cuts rates.  So high VIX
#    *causes* rate cuts, not just the reverse.
#
# 2. **Common causes (confounders):** Economic conditions (GDP growth,
#    inflation expectations) drive *both* Fed decisions *and* market
#    volatility.  Any regression that omits these will produce a biased
#    coefficient.
#
# **The solution:** We need variation in the rate change that is driven
# by something *other than* current economic or market conditions.
# That is the definition of an instrument.

# %% [markdown]
# ## 1. The DAG — formalising assumptions
#
# Below we define the causal graph using DoWhy.
# Nodes = variables.  Directed edges = assumed causal relationships.
# The absence of an edge is itself an assumption.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import dowhy
from dowhy import CausalModel
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

from src.data_loader import load_from_db

df = load_from_db("master_daily")
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date").sort_index()

# %% [markdown]
# ### DAG specification
#
# Variables in the graph:
# | Variable | Role | Description |
# |----------|------|-------------|
# | `surprise` | Instrument (Z) | Unexpected component of rate change |
# | `rate_change` | Treatment (T) | Actual Fed Funds rate change |
# | `vix_change` | Outcome (Y) | Change in VIX on/after FOMC date |
# | `dff_level` | Confounder | Level of rates (economic cycle proxy) |
# | `spx_pre` | Confounder | Pre-FOMC equity return (market anticipation) |
# | `macro_stress` | Latent confounder | Unobserved economic stress |
#
# **Key exclusion restriction (the IV assumption):**
# The surprise component affects VIX *only through* the actual rate
# change — not through any other channel.  This is defensible because
# the surprise is, by construction, the part of the decision that
# markets did not predict, so it should not correlate with pre-existing
# market conditions.

# %%
# We will draw the DAG manually using networkx for clarity

G = nx.DiGraph()

nodes = {
    "surprise":     "Instrument\n(Fed Funds\n surprise)",
    "rate_change":  "Treatment\n(rate change)",
    "vix_change":   "Outcome\n(ΔVIX)",
    "dff_level":    "Confounder\n(rate level)",
    "spx_pre":      "Confounder\n(pre-FOMC\n return)",
    "macro_stress": "Latent\n(macro stress)",
}

edges = [
    ("surprise",     "rate_change"),   # IV → Treatment
    ("rate_change",  "vix_change"),    # Treatment → Outcome  [CAUSAL EFFECT WE WANT]
    ("dff_level",    "rate_change"),   # Fed reacts to rate level
    ("dff_level",    "vix_change"),    # Rate level also drives vol
    ("spx_pre",      "vix_change"),    # Pre-FOMC drift affects vol
    ("macro_stress", "rate_change"),   # Fed reacts to macro (latent)
    ("macro_stress", "vix_change"),    # Macro also drives vol (latent)
]

G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

pos = {
    "surprise":     (-2,  0),
    "rate_change":  ( 0,  0),
    "vix_change":   ( 2,  0),
    "dff_level":    ( 0,  1.5),
    "spx_pre":      ( 1,  1.5),
    "macro_stress": ( 0, -1.5),
}

node_colors = {
    "surprise":     "#1D9E75",
    "rate_change":  "#185FA5",
    "vix_change":   "#D85A30",
    "dff_level":    "#888780",
    "spx_pre":      "#888780",
    "macro_stress": "#B4B2A9",
}

fig, ax = plt.subplots(figsize=(11, 7))
nx.draw_networkx(
    G, pos=pos, ax=ax,
    labels=nodes,
    node_color=[node_colors[n] for n in G.nodes()],
    node_size=3500,
    font_size=8,
    font_color="white",
    font_weight="bold",
    edge_color="#444441",
    arrows=True,
    arrowsize=18,
    connectionstyle="arc3,rad=0.1",
)
ax.set_title("Causal DAG — Fed policy and equity volatility", fontsize=13, pad=20)
ax.axis("off")
plt.tight_layout()
plt.savefig("../outputs/figures/03_causal_dag.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Constructing the surprise instrument
#
# The standard Kuttner (2001) surprise measure uses the change in
# the front-month Fed Funds futures contract on FOMC days.
#
# Since full historical futures data requires a paid feed, we
# implement another approximation: the *unexpected* component
# is proxied by the residual of a simple AR(1) model for the rate
# change, estimated on non-FOMC days.  This captures the idea that
# the "surprise" is what a simple forecast would not have predicted.
#
# **Important methodological note:**
# A better instrument would be the actual futures-implied expectation.
# We document this limitation and discuss its likely direction of bias.

# %%
fomc_dates_df = load_from_db("fomc_events")
fomc_dates_df["date"] = pd.to_datetime(fomc_dates_df["date"])
fomc_set = set(fomc_dates_df["date"])

print(fomc_dates_df.shape)
print(fomc_dates_df.head(10))
print(fomc_dates_df.dtypes)


# Expected rate = previous FOMC meeting's rate (carry-forward)
# On FOMC days, the baseline expectation is zero change
# We model the surprise as a deviation of the rate change from a moving average
# of the past 6 meetings.

# Use days where rate actually changed as FOMC events
# AR(1) on flat target rate is degenerate — surprise = rate change directly; not a good model.

fomc_days = df[df["rate_change"].abs() >= 0.24].copy()

# Use lagged DFF level as a proxy for expected rate path
# Instrument: deviation of rate_change from its rolling mean over prior 6 meetings
fomc_days = fomc_days.sort_index()
fomc_days["surprise"] = (fomc_days["rate_change"] - 
                         fomc_days["rate_change"].rolling(6, min_periods=3).mean().shift(1))

print(fomc_days[["rate_change", "surprise"]].describe())
print(f"\nCorrelation: {fomc_days['rate_change'].corr(fomc_days['surprise']):.3f}")

print(f"\nFOMC event days with surprise instrument: {len(fomc_days)}")
print(fomc_days[["rate_change", "surprise"]].head(10))



# %% [markdown]
# ## 3. Constructing the outcome variable
#
# We use the *change* in VIX over a short window after the decision
# as our outcome.  We try two windows:
# - **Same-day:** VIX change on the FOMC date itself
# - **3-day:** Cumulative VIX change over days [0, +3]
#
# Both are informative; the 3-day window is less noisy.

# %%
vix_changes = []
for event_date in fomc_days.index:
    try:
        loc = df.index.get_loc(event_date)
    except KeyError:
        continue
    vix_t0 = df["vix_close"].iloc[loc]
    # same-day change vs prior close
    vix_prev = df["vix_close"].iloc[loc - 1] if loc > 0 else np.nan
    vix_3d   = df["vix_close"].iloc[min(loc + 3, len(df) - 1)]
    vix_changes.append({
        "date":         event_date,
        "vix_change_1d": vix_t0 - vix_prev,
        "vix_change_3d": vix_3d - vix_prev,
        "log_vix_t0":   np.log(vix_t0) if vix_t0 > 0 else np.nan,
    })

vix_change_df = pd.DataFrame(vix_changes).set_index("date")
fomc_days = fomc_days.join(vix_change_df)

# Also add control variables
fomc_days["dff_level"] = df["dff"].reindex(fomc_days.index)
fomc_days["spx_pre"]   = df["spx_return"].shift(1).reindex(fomc_days.index)

if "vol_regime" in fomc_days.columns:
    fomc_days = fomc_days.drop(columns=["vol_regime"])

fomc_analysis = fomc_days.dropna(subset=["surprise", "vix_change_1d", "dff_level"])
print(f"\nClean FOMC analysis sample: {len(fomc_analysis)} events")
fomc_analysis = fomc_analysis.dropna(subset=["rate_change", "surprise", "dff_level", "spx_pre"])
print(f"After dropping NaNs: {len(fomc_analysis)} events")

# Compute regime directly from fomc_analysis VIX values
vix_median = fomc_analysis["vix_close"].median()
fomc_analysis["vol_regime"] = np.where(
    fomc_analysis["vix_close"] > vix_median, "high_vol", "low_vol"
)
print(fomc_analysis["vol_regime"].value_counts())

# %% [markdown]
# ## 4. First stage — instrument relevance
#
# A valid IV must be *relevant* (correlated with the treatment).
# We verify this with an F-test in the first stage regression.
# The rule of thumb is F > 10 for a "strong" instrument.

# %%
first_stage = sm.OLS(
    fomc_analysis["rate_change"],
    sm.add_constant(fomc_analysis[["surprise", "dff_level", "spx_pre"]])
).fit()

print("First stage: rate_change ~ surprise + controls")
print(first_stage.summary().tables[1])
print(f"\nFirst-stage F-statistic (instrument relevance): {first_stage.fvalue:.2f}")
print("Rule of thumb: F > 10 indicates a strong instrument")

# %% [markdown]
# ## 5. IV estimation (2SLS) — the main result
#
# We use Two-Stage Least Squares (2SLS) via the `linearmodels` package.
# 2SLS is the standard IV estimator:
# - Stage 1: Regress treatment on instrument (+ controls)
# - Stage 2: Regress outcome on *fitted* treatment from stage 1

# %%
# Outcome: 1-day VIX change
endog  = fomc_analysis["vix_change_1d"]
exog   = sm.add_constant(fomc_analysis[["dff_level", "spx_pre"]])
instrs = sm.add_constant(fomc_analysis[["surprise", "dff_level", "spx_pre"]])

iv_model_1d = IV2SLS(
    dependent=endog,
    exog=exog[["const", "dff_level", "spx_pre"]],
    endog=fomc_analysis[["rate_change"]],
    instruments=fomc_analysis[["surprise"]]
).fit(cov_type="robust")

print("=== IV (2SLS) result — 1-day VIX change ===")
print(iv_model_1d.summary.tables[1])

# %%
# Outcome: 3-day VIX change
endog_3d = fomc_analysis["vix_change_3d"]

iv_model_3d = IV2SLS(
    dependent=endog_3d,
    exog=exog[["const", "dff_level", "spx_pre"]],
    endog=fomc_analysis[["rate_change"]],
    instruments=fomc_analysis[["surprise"]]
).fit(cov_type="robust")

print("\n=== IV (2SLS) result — 3-day VIX change ===")
print(iv_model_3d.summary.tables[1])

# %% [markdown]
# ## 6. OLS comparison — illustrating the bias
#
# We also run a naive OLS for comparison.  The difference between OLS
# and IV estimates is the estimated bias from the confounders.

# %%
ols_model = sm.OLS(
    fomc_analysis["vix_change_1d"],
    sm.add_constant(fomc_analysis[["rate_change", "dff_level", "spx_pre"]])
).fit(cov_type="HC3")

print("=== OLS result (biased — for comparison) ===")
print(ols_model.summary().tables[1])

coef_ols = ols_model.params["rate_change"]
coef_iv  = iv_model_1d.params["rate_change"]
print(f"\nOLS coefficient on rate_change: {coef_ols:.4f}")
print(f"IV  coefficient on rate_change: {coef_iv:.4f}")
print(f"Estimated bias (OLS - IV):       {coef_ols - coef_iv:.4f}")

# %% [markdown]
# ## 7. Visualising the main result
#
# A coefficient plot showing the IV estimate with confidence intervals,
# plus the OLS for comparison.

# %%
fig, ax = plt.subplots(figsize=(7, 4))

estimates = {
    "OLS (biased)": (coef_ols,
                     ols_model.conf_int().loc["rate_change", 0],
                     ols_model.conf_int().loc["rate_change", 1]),
    "IV / 2SLS\n(causal estimate)": (
        coef_iv,
        iv_model_1d.conf_int().loc["rate_change", "lower"],
        iv_model_1d.conf_int().loc["rate_change", "upper"],
    ),
}

colors = {"OLS (biased)": "#888780", "IV / 2SLS\n(causal estimate)": "#185FA5"}
y_pos = list(range(len(estimates)))

for i, (label, (coef, lo, hi)) in enumerate(estimates.items()):
    ax.barh(i, coef, color=colors[label], alpha=0.85, height=0.4)
    ax.plot([lo, hi], [i, i], color=colors[label], linewidth=2.5, solid_capstyle="round")
    ax.plot([lo, lo], [i - 0.1, i + 0.1], color=colors[label], linewidth=2)
    ax.plot([hi, hi], [i - 0.1, i + 0.1], color=colors[label], linewidth=2)

ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_yticks(y_pos)
ax.set_yticklabels(list(estimates.keys()))
ax.set_xlabel("Effect of 1pp rate increase on same-day VIX change (VIX points)")
ax.set_title("Causal vs associational estimates\n(with 95% confidence intervals)")
plt.tight_layout()
plt.savefig("../outputs/figures/03_main_result.png", dpi=150, bbox_inches="tight")
plt.show()

#For notebook 04, run the windows for analysis frame.
for w in range(1, 11):
    fomc_analysis[f"vix_change_{w}d"] = [
        df["vix_close"].iloc[min(df.index.get_loc(d) + w, len(df)-1)] 
        - df["vix_close"].iloc[df.index.get_loc(d) - 1]
        for d in fomc_analysis.index
    ]

#Used in notebook 04
fomc_analysis.to_csv("../data/processed/fomc_analysis.csv")
print(f"Saved fomc_analysis: {fomc_analysis.shape}")

# %% [markdown]
# ## Summary — what we found and why it is credible
#
# - OLS estimate: 0.3467 VIX points per 1pp rate hike
# - IV estimate: 0.3049 VIX points per 1pp rate hike
# - First-stage F: 237.64 (strong instrument)
# - Statistical significance: The IV coefficient on rate_change is 0.305 (1-day) and 0.186 (3-day),
#   both statistically insignificant (p = 0.72 and p = 0.88). 
#   OLS gives 0.347, also insignificant. 
#   The estimated bias is small at 0.042.
# - Interpretation: Fed rate changes do not causally drive short-term VIX movements in a 
#   statistically detectable way, at least not through the rate change magnitude alone.
#
# **Next:** `04_robustness.py` — placebo tests and sensitivity analysis.
