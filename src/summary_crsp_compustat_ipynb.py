# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: finm-32900-venv-p31211 (3.12.11.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fama-French (1993) Factor Replication

# %%
import sys
sys.path.insert(0, "./src")

import chartbook
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from calc_Fama_French_1993 import compare_with_actual_ff_factors

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

# %% [markdown]
# ## Data Overview
#
# This pipeline produces CRSP stock returns and Fama-French factor data:
# - Individual stock monthly returns (with and without dividends)
# - Fama-French SMB and HML factors (replicated from CRSP/Compustat)
#
# Data sources:
#
# - CRSP Monthly Stock File (CIZ format)
# - Compustat Fundamentals Annual
# - CRSP-Compustat Link Table

# %% [markdown]
# ### CRSP Monthly Stock Returns
#
# | Variable | Description |
# |----------|-------------|
# | unique_id | CRSP PERMNO (permanent security identifier) |
# | ds | Month-end date |
# | y | Monthly return (including dividends) |

# %%
df_ret = pd.read_parquet(DATA_DIR / "ftsfr_CRSP_monthly_stock_ret.parquet")
print(f"Shape: {df_ret.shape}")
print(f"Columns: {df_ret.columns.tolist()}")
print(f"\nDate range: {df_ret['ds'].min()} to {df_ret['ds'].max()}")
print(f"Number of unique stocks: {df_ret['unique_id'].nunique()}")
display(df_ret)

# %%
df_ret.describe()

# %% [markdown]
# ### Compustat Fundamentals

# %%
df_fund = pd.read_parquet(DATA_DIR / "Compustat.parquet")
print(f"Shape: {df_fund.shape}")
print(f"Columns: {df_fund.columns.tolist()}")
display(df_fund)

# %%
df_fund.describe()

# %% [markdown]
# ### CRSP-Compustat Link

# %%
df_link = pd.read_parquet(DATA_DIR / "CRSP_Comp_Link_Table.parquet")
print(f"Shape: {df_link.shape}")
print(f"Columns: {df_link.columns.tolist()}")
display(df_link)

# %%
df_link.describe()

# %% [markdown]
# ### Fama-French Factors (Replicated)
#
# | Factor | Description |
# |--------|-------------|
# | SMB | Small Minus Big - Size factor |
# | HML | High Minus Low - Value factor |
#
# Factors are constructed following Fama-French (1993) methodology using NYSE breakpoints.

# %%
df_ff = pd.read_parquet(DATA_DIR / "FF_1993_factors.parquet")
print(f"Shape: {df_ff.shape}")
print(f"Columns: {df_ff.columns.tolist()}")
print(f"\nDate range: {df_ff['date'].min()} to {df_ff['date'].max()}")

# Summary statistics
print("\nSummary Statistics:")
print(df_ff[["SMB", "HML"]].describe())

# Time series plot
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
axes[0].plot(df_ff["date"], df_ff["SMB"], label="SMB", alpha=0.7)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Monthly Return")
axes[0].set_title("Small Minus Big (SMB) Factor")
axes[0].grid(True, alpha=0.3)
axes[1].plot(df_ff["date"], df_ff["HML"], label="HML", color="orange", alpha=0.7)
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Monthly Return")
axes[1].set_title("High Minus Low (HML) Factor")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Return Distribution Over Time

# %%
# Monthly average return
monthly_avg = df_ret.groupby("ds")["y"].mean().reset_index()

# Time series plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(monthly_avg["ds"], monthly_avg["y"], alpha=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Average Monthly Return")
ax.set_title("Cross-Sectional Average Monthly Stock Return")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Number of Stocks Over Time

# %%
stock_count = df_ret.groupby("ds")["unique_id"].nunique().reset_index()
stock_count.columns = ["date", "n_stocks"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(stock_count["date"], stock_count["n_stocks"], alpha=0.7)
ax.set_xlabel("Date")
ax.set_ylabel("Number of Stocks")
ax.set_title("Number of Stocks in Universe Over Time")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare Manually Created Factors vs Ken French Data Library

# %% [markdown]
# ### Read Parquet Files

# %%
## Read Parquet Files
ff_factors = pd.read_parquet(DATA_DIR / "FF_1993_factors.parquet")
ff_nfirms = pd.read_parquet(DATA_DIR / "FF_1993_nfirms.parquet")

display(ff_factors.tail())
display(ff_nfirms.tail())

# %% [markdown]
# ### Calculate Comparisons

# %%
# Calculate Comparisons
ff_compare, ff_compare_post_1970 = compare_with_actual_ff_factors(
    ff_factors, data_dir=DATA_DIR
)

print("Comparison with actual Fama-French factors:")
display(ff_compare)

# %% [markdown]
# ### Plot Comparisons

# %%
## Plot Comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
axes[0].plot(ff_compare["smb_actual"], "r--", ff_compare["smb_manual"], "b-", alpha=0.7)
axes[0].set_xlim([datetime.datetime(1961, 1, 1), datetime.datetime(2022, 6, 30)])
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Monthly Return")
axes[0].set_title("Small Minus Big (SMB) Factor")
axes[0].grid(True, alpha=0.3)
axes[0].legend(("smb_actual", "smb_manual"), loc="upper right")
axes[1].plot(ff_compare["hml_actual"], "r--", ff_compare["hml_manual"], "b-", alpha=0.7)
axes[1].set_xlim([datetime.datetime(1961, 1, 1), datetime.datetime(2022, 6, 30)])
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Monthly Return")
axes[1].set_title("High Minus Low (HML) Factor")
axes[1].grid(True, alpha=0.3)
axes[1].legend(("hml_actual", "hml_manual"), loc="upper right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Calculate Correlations Between Manually Created Factors and Ken French Data Library

# %%
# Calc correlations
corr_smb = ff_compare["smb_actual"].corr(ff_compare["smb_manual"])
print(f"Correlation between actual and manual SMB: {corr_smb:.4f}")
corr_hml = ff_compare["hml_actual"].corr(ff_compare["hml_manual"])
print(f"Correlation between actual and manual HML: {corr_hml:.4f}")


# %%
# Calc rolling correlations
rolling_window = 12  # 5 years of monthly data
rolling_corr_smb = ff_compare["smb_actual"].rolling(window=rolling_window).corr(ff_compare["smb_manual"])
rolling_corr_hml = ff_compare["hml_actual"].rolling(window=rolling_window).corr(ff_compare["hml_manual"])

# %%
# Plot rolling correlations
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
axes[0].plot(rolling_corr_smb, label="SMB", alpha=0.7)
axes[0].set_xlim([datetime.datetime(1961, 1, 1), datetime.datetime(2022, 6, 30)])
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Rolling 1 Year Correlation")
axes[0].set_title("Small Minus Big (SMB) Factor")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper left")
axes[1].plot(rolling_corr_hml, label="HML", alpha=0.7)
axes[1].set_xlim([datetime.datetime(1961, 1, 1), datetime.datetime(2022, 6, 30)])
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Rolling 1 Year Correlation")
axes[1].set_title("High Minus Low (HML) Factor")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="upper left")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Table 1

# %% [markdown]
# For reference, table_1.png:
#
# ![Table 1](../assets/table_1.png)

# %% [markdown]
# ### Published Fama-French (1993) Table 1 values
#
# The numbers below are transcribed directly from the original Table 1 (1963-1991,
# 25 portfolios sorted by size down and BE/ME across) for side-by-side comparison
# with the panels we compute from CRSP/Compustat further down.
#
# **Panel 1: Average of annual averages of firm size ($ millions)**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 20.6 | 20.8 | 20.2 | 19.4 | 15.1 |
# | 2 | 89.7 | 89.3 | 89.3 | 89.9 | 88.5 |
# | 3 | 209.3 | 211.9 | 210.8 | 214.8 | 210.7 |
# | 4 | 535.1 | 537.4 | 545.4 | 551.6 | 538.7 |
# | Big | 3583.7 | 2885.8 | 2819.5 | 2700.5 | 2337.9 |
#
# **Panel 2: Average of annual BE/ME ratios for portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 0.30 | 0.62 | 0.84 | 1.09 | 1.80 |
# | 2 | 0.31 | 0.60 | 0.83 | 1.09 | 1.71 |
# | 3 | 0.31 | 0.60 | 0.84 | 1.08 | 1.66 |
# | 4 | 0.31 | 0.61 | 0.84 | 1.09 | 1.67 |
# | Big | 0.29 | 0.59 | 0.83 | 1.08 | 1.56 |
#
# **Panel 3: Average of annual percent of market value in portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 0.69 | 0.49 | 0.46 | 0.48 | 0.64 |
# | 2 | 0.92 | 0.71 | 0.65 | 0.61 | 0.55 |
# | 3 | 1.78 | 1.36 | 1.26 | 1.14 | 0.82 |
# | 4 | 3.95 | 3.01 | 2.71 | 2.41 | 1.50 |
# | Big | 30.13 | 15.87 | 12.85 | 10.44 | 4.61 |
#
# **Panel 4: Average of annual number of firms in portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 428.0 | 276.6 | 263.8 | 291.5 | 512.7 |
# | 2 | 121.6 | 94.0 | 86.7 | 79.8 | 71.3 |
# | 3 | 102.7 | 78.3 | 73.0 | 64.5 | 45.9 |
# | 4 | 90.1 | 68.9 | 60.7 | 53.1 | 33.4 |
# | Big | 93.6 | 63.7 | 52.7 | 44.0 | 23.6 |
#
# **Panel 5: Average of annual E/P ratios (in percent) for portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 2.42 | 7.24 | 8.26 | 9.06 | 2.66 |
# | 2 | 5.20 | 8.61 | 10.16 | 10.95 | 9.28 |
# | 3 | 5.91 | 8.72 | 10.43 | 11.62 | 10.78 |
# | 4 | 5.85 | 8.94 | 10.45 | 11.64 | 11.39 |
# | Big | 6.00 | 9.07 | 10.90 | 12.45 | 13.92 |
#
# **Panel 6: Average of annual D/P ratios (in percent) for portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 1.00 | 1.94 | 2.60 | 3.13 | 2.82 |
# | 2 | 1.59 | 2.45 | 3.45 | 4.25 | 4.53 |
# | 3 | 1.56 | 3.03 | 4.04 | 4.68 | 4.64 |
# | 4 | 1.80 | 3.09 | 4.22 | 5.01 | 4.94 |
# | Big | 2.34 | 3.69 | 4.68 | 5.49 | 5.90 |

# %% [markdown]
# ### Build the 5x5 sorted firm-June panel
#
# Descriptive statistics for the 25 size x book-to-market portfolios, following Fama-French (1993) Table 1. Stocks are sorted independently each June into NYSE size quintiles and NYSE BE/ME quintiles, forming a 5x5 grid (size rows 1=Small..5=Big, BE/ME columns 1=Low..5=High).
#
# Rebuilt from the raw CRSP/Compustat data via the pipeline functions; the resulting `ff5` frame (one row per firm per June, tagged with `sz5`/`bm5`) is reused by every panel below.

# %%
import numpy as np

import pull_CRSP_Compustat as pcc
import calc_Fama_French_1993 as ff

# Rebuild the per-firm June panel (one row per permno per June)
_comp = ff.calc_book_equity_and_years_in_compustat(pcc.load_compustat(data_dir=DATA_DIR))
_crsp = ff.subset_CRSP_to_common_stock_and_exchanges(pcc.load_CRSP_stock_ciz(data_dir=DATA_DIR))
_ccm = pcc.load_CRSP_Comp_Link_Table(data_dir=DATA_DIR)
_crsp2 = ff.calculate_market_equity(_crsp)
_crsp3, _crsp_jun = ff.use_dec_market_equity(_crsp2)
ccm_jun = ff.merge_CRSP_and_Compustat(_crsp_jun, _comp, _ccm)

# Universe filter: positive BE/ME, positive June ME, positive Dec ME, >= 1 year in
# Compustat. The dec_me > 0 condition drops a handful of firms whose December market
# equity is zero, which would otherwise make beme = be*1000/dec_me infinite.
ff5 = ccm_jun[
    (ccm_jun["beme"] > 0)
    & (ccm_jun["me"] > 0)
    & (ccm_jun["dec_me"] > 0)
    & (ccm_jun["count"] >= 1)
].copy()

# NYSE quintile breakpoints per June, computed separately for size and BE/ME
_nyse = ff5[ff5["primaryexch"] == "N"]


def _nyse_quintiles(col):
    brk = (
        _nyse.groupby("jdate")[col]
        .quantile([0.2, 0.4, 0.6, 0.8])
        .unstack()
        .rename(columns={0.2: f"{col}_20", 0.4: f"{col}_40", 0.6: f"{col}_60", 0.8: f"{col}_80"})
        .reset_index()
    )
    return brk


ff5 = ff5.merge(_nyse_quintiles("me"), on="jdate").merge(_nyse_quintiles("beme"), on="jdate")


def _quintile(v, b20, b40, b60, b80):
    if v <= b20:
        return 1
    if v <= b40:
        return 2
    if v <= b60:
        return 3
    if v <= b80:
        return 4
    return 5


ff5["sz5"] = ff5.apply(lambda r: _quintile(r["me"], r["me_20"], r["me_40"], r["me_60"], r["me_80"]), axis=1)
ff5["bm5"] = ff5.apply(lambda r: _quintile(r["beme"], r["beme_20"], r["beme_40"], r["beme_60"], r["beme_80"]), axis=1)

# Express June market equity in $millions (me = price x shrout-in-thousands -> $thousands)
ff5["me_mm"] = ff5["me"] / 1000.0

print(f"Firm-June observations in 5x5 universe: {len(ff5):,}")
print(f"June dates: {ff5['jdate'].min():%Y-%m} to {ff5['jdate'].max():%Y-%m}")

# %% [markdown]
# ### Panel 1: Average of annual averages of firm size ($millions)
#
# For each June, average firm size (June ME) is computed within each of the 25
# portfolios; those annual averages are then averaged across all years.

# %%
# Calc "Average of annual averages of firm size"
_annual_size = ff5.groupby(["jdate", "sz5", "bm5"])["me_mm"].mean().reset_index()
panel_size = (
    _annual_size.groupby(["sz5", "bm5"])["me_mm"]
    .mean()
    .unstack()
    .rename_axis(index="Size", columns="BE/ME")
)
panel_size.index = ["Small", "2", "3", "4", "Big"]
panel_size.columns = ["Low", "2", "3", "4", "High"]
display(panel_size.round(1))

# %% [markdown]
# **Panel 1: Average of annual averages of firm size ($ millions)**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 20.6 | 20.8 | 20.2 | 19.4 | 15.1 |
# | 2 | 89.7 | 89.3 | 89.3 | 89.9 | 88.5 |
# | 3 | 209.3 | 211.9 | 210.8 | 214.8 | 210.7 |
# | 4 | 535.1 | 537.4 | 545.4 | 551.6 | 538.7 |
# | Big | 3583.7 | 2885.8 | 2819.5 | 2700.5 | 2337.9 |

# %% [markdown]
# ### Panel 2: Average of annual BE/ME ratios for portfolio
#
# For each June, the average book-to-market ratio (`beme`) is computed within each
# of the 25 portfolios; those annual averages are then averaged across all years.

# %%
# Calc "Average of annual BE/ME ratios for portfolio"
_annual_beme = ff5.groupby(["jdate", "sz5", "bm5"])["beme"].mean().reset_index()
panel_beme = (
    _annual_beme.groupby(["sz5", "bm5"])["beme"]
    .mean()
    .unstack()
    .rename_axis(index="Size", columns="BE/ME")
)
panel_beme.index = ["Small", "2", "3", "4", "Big"]
panel_beme.columns = ["Low", "2", "3", "4", "High"]
display(panel_beme.round(2))

# %% [markdown]
# **Panel 2: Average of annual BE/ME ratios for portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 0.30 | 0.62 | 0.84 | 1.09 | 1.80 |
# | 2 | 0.31 | 0.60 | 0.83 | 1.09 | 1.71 |
# | 3 | 0.31 | 0.60 | 0.84 | 1.08 | 1.66 |
# | 4 | 0.31 | 0.61 | 0.84 | 1.09 | 1.67 |
# | Big | 0.29 | 0.59 | 0.83 | 1.08 | 1.56 |

# %% [markdown]
# ### Panel 3: Average of annual percent of market value in portfolio
#
# For each June, each portfolio's share of total market value is computed as the sum
# of June ME in the cell divided by the total June ME across all 25 cells; those
# annual percentages are then averaged across all years. The full 25-cell grid sums
# to ~100% (small deviations come from cells that are empty in early years).

# %%
# Calc "Average of annual percent of market value in portfolio"
_cell_value = ff5.groupby(["jdate", "sz5", "bm5"])["me_mm"].sum().reset_index()
_total_value = ff5.groupby("jdate")["me_mm"].sum().rename("total_me").reset_index()
_cell_value = _cell_value.merge(_total_value, on="jdate")
_cell_value["pct"] = 100 * _cell_value["me_mm"] / _cell_value["total_me"]
panel_pct = (
    _cell_value.groupby(["sz5", "bm5"])["pct"]
    .mean()
    .unstack()
    .rename_axis(index="Size", columns="BE/ME")
)
panel_pct.index = ["Small", "2", "3", "4", "Big"]
panel_pct.columns = ["Low", "2", "3", "4", "High"]
display(panel_pct.round(2))

# %% [markdown]
# **Panel 3: Average of annual percent of market value in portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 0.69 | 0.49 | 0.46 | 0.48 | 0.64 |
# | 2 | 0.92 | 0.71 | 0.65 | 0.61 | 0.55 |
# | 3 | 1.78 | 1.36 | 1.26 | 1.14 | 0.82 |
# | 4 | 3.95 | 3.01 | 2.71 | 2.41 | 1.50 |
# | Big | 30.13 | 15.87 | 12.85 | 10.44 | 4.61 |

# %% [markdown]
# ### Panel 4: Average of annual number of firms in portfolio
#
# For each June, the number of firms in each of the 25 portfolios is counted; those
# annual counts are then averaged across all years.

# %%
# Calc "Average of annual number of firms in portfolio"
_annual_nfirms = ff5.groupby(["jdate", "sz5", "bm5"]).size().reset_index(name="n")
panel_nfirms = (
    _annual_nfirms.groupby(["sz5", "bm5"])["n"]
    .mean()
    .unstack()
    .rename_axis(index="Size", columns="BE/ME")
)
panel_nfirms.index = ["Small", "2", "3", "4", "Big"]
panel_nfirms.columns = ["Low", "2", "3", "4", "High"]
display(panel_nfirms.round(0))

# %% [markdown]
# **Panel 4: Average of annual number of firms in portfolio**
#
# | Size | Low | 2 | 3 | 4 | High |
# |------|----:|----:|----:|----:|----:|
# | Small | 428.0 | 276.6 | 263.8 | 291.5 | 512.7 |
# | 2 | 121.6 | 94.0 | 86.7 | 79.8 | 71.3 |
# | 3 | 102.7 | 78.3 | 73.0 | 64.5 | 45.9 |
# | 4 | 90.1 | 68.9 | 60.7 | 53.1 | 33.4 |
# | Big | 93.6 | 63.7 | 52.7 | 44.0 | 23.6 |

# %% [markdown]
#
