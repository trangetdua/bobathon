# %% [markdown]
# # EDA: Parkinson MDS-UPDRS true-OFF motor score prediction
#
# Exploratory analysis of the hackathon dataset (4 CSV files in `data/`).
# Each row is a patient visit; target is the "true" OFF MDS-UPDRS motor score
# (continuous, regression task).
#
# - **Raw data** is read-only — this file never modifies the CSV files.
# - **Outputs** go to `EDA_DIR` (`data/`): one `eda_<table>.html` per table.

# %%
import json
import pathlib

import pandas as pd
import skrub

# No src package: resolve from cwd (run from project root) or this file.
# The runner executes cells in an IPython shell where __file__ is unavailable,
# so we locate the data folder relative to the script's path on disk instead.
_this_dir = pathlib.Path("data")  # relative to project root (cwd when run)
EDA_DIR = _this_dir
DATA_DIR = _this_dir
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1 – Load raw data
#
# Four tables: X_train, y_train, X_test, sample_submission.

# %%
X_train = pd.read_csv(DATA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(DATA_DIR / "y_train.csv", index_col="Index")
X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col="Index")
submission = pd.read_csv(DATA_DIR / "sample_submission.csv", index_col="Index")

{
    "X_train": X_train.shape,
    "y_train": y_train.shape,
    "X_test": X_test.shape,
    "sample_submission": submission.shape,
}

# %% [markdown]
# ## 2 – Table overviews (skrub TableReport)
#
# One HTML report per table saved to `data/eda_<table>.html`.
# Per-column summary extracted from `.json()` (library-agnostic).

# %%
# --- X_train ---
report_xtrain = skrub.TableReport(
    X_train.reset_index(), title="X_train", verbose=0
)
report_xtrain.write_html(EDA_DIR / "eda_X_train.html")
s_xtrain = json.loads(report_xtrain.json())
n_rows_train = s_xtrain.get("n_rows")
cols_xtrain = [
    {
        "column": c.get("name"),
        "dtype": c.get("dtype"),
        "null_pct": round(c.get("null_proportion", 0) * 100, 1),
        "n_unique": c.get("n_unique"),
        "mean": c.get("mean"),
        "std": c.get("standard_deviation"),
        "q25": (c.get("quantiles") or {}).get("0.25"),
        "q50": (c.get("quantiles") or {}).get("0.5"),
        "q75": (c.get("quantiles") or {}).get("0.75"),
        "q0": (c.get("quantiles") or {}).get("0.0"),
        "q100": (c.get("quantiles") or {}).get("1.0"),
    }
    for c in s_xtrain.get("columns", [])
]
{"n_rows": n_rows_train, "n_columns": len(cols_xtrain), "columns": cols_xtrain}

# %%
# --- y_train ---
report_ytrain = skrub.TableReport(
    y_train.reset_index(), title="y_train", verbose=0
)
report_ytrain.write_html(EDA_DIR / "eda_y_train.html")
s_ytrain = json.loads(report_ytrain.json())
cols_ytrain = [
    {
        "column": c.get("name"),
        "null_pct": round(c.get("null_proportion", 0) * 100, 1),
        "mean": c.get("mean"),
        "std": c.get("standard_deviation"),
        "q0": (c.get("quantiles") or {}).get("0.0"),
        "q25": (c.get("quantiles") or {}).get("0.25"),
        "q50": (c.get("quantiles") or {}).get("0.5"),
        "q75": (c.get("quantiles") or {}).get("0.75"),
        "q100": (c.get("quantiles") or {}).get("1.0"),
    }
    for c in s_ytrain.get("columns", [])
]
{"n_rows": s_ytrain.get("n_rows"), "columns": cols_ytrain}

# %%
# --- X_test ---
report_xtest = skrub.TableReport(
    X_test.reset_index(), title="X_test", verbose=0
)
report_xtest.write_html(EDA_DIR / "eda_X_test.html")
s_xtest = json.loads(report_xtest.json())
n_rows_test = s_xtest.get("n_rows")
cols_xtest = [
    {
        "column": c.get("name"),
        "dtype": c.get("dtype"),
        "null_pct": round(c.get("null_proportion", 0) * 100, 1),
        "n_unique": c.get("n_unique"),
    }
    for c in s_xtest.get("columns", [])
]
{"n_rows": n_rows_test, "n_columns": len(cols_xtest), "columns": cols_xtest}

# %%
# --- sample_submission ---
report_sub = skrub.TableReport(
    submission.reset_index(), title="sample_submission", verbose=0
)
report_sub.write_html(EDA_DIR / "eda_sample_submission.html")
s_sub = json.loads(report_sub.json())
{"n_rows": s_sub.get("n_rows"), "columns": [c.get("name") for c in s_sub.get("columns", [])]}

# %% [markdown]
# ## 3 – Schema integrity checks
#
# - Same columns and types in train / test
# - y_train aligns with X_train on Index (no rows lost or duplicated)

# %%
# Column names comparison
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)
cols_only_train = train_cols - test_cols
cols_only_test = test_cols - train_cols
# dtype comparison
dtype_matches = {
    col: (str(X_train[col].dtype) == str(X_test[col].dtype))
    for col in train_cols & test_cols
}
dtype_mismatches = {k: v for k, v in dtype_matches.items() if not v}
{
    "cols_only_in_train": list(cols_only_train),
    "cols_only_in_test": list(cols_only_test),
    "dtype_mismatches": dtype_mismatches,
}

# %%
# Index alignment X_train ↔ y_train
n_index_train = len(X_train.index)
n_index_y = len(y_train.index)
joined = X_train.join(y_train, how="inner")
n_joined = len(joined)
{
    "n_X_train": n_index_train,
    "n_y_train": n_index_y,
    "n_inner_join": n_joined,
    "rows_lost": n_index_train - n_joined,
    "index_unique_X": X_train.index.is_unique,
    "index_unique_y": y_train.index.is_unique,
    "index_unique_submission": submission.index.is_unique,
}

# %% [markdown]
# ## 4 – Missing values: counts and co-missingness patterns

# %%
# Per-column missingness (train)
miss_train = (
    X_train.isnull().sum()
    .rename("n_missing")
    .to_frame()
    .assign(pct=lambda df: (df["n_missing"] / len(X_train) * 100).round(1))
)
miss_train_dict = miss_train.to_dict(orient="index")

miss_test = (
    X_test.isnull().sum()
    .rename("n_missing")
    .to_frame()
    .assign(pct=lambda df: (df["n_missing"] / len(X_test) * 100).round(1))
)
{
    "missing_train": miss_train_dict,
    "missing_test": miss_test.to_dict(orient="index"),
}

# %%
# Co-missingness: on and off together
on_miss = X_train["on"].isnull()
off_miss = X_train["off"].isnull()
co_miss = {
    "both_present": int((~on_miss & ~off_miss).sum()),
    "only_on_present": int((~on_miss & off_miss).sum()),
    "only_off_present": int((on_miss & ~off_miss).sum()),
    "both_missing": int((on_miss & off_miss).sum()),
}
# ledd and gene missing in same rows?
ledd_miss = X_train["ledd"].isnull()
gene_miss = X_train["gene"].isnull()
{
    "on_off_comissing": co_miss,
    "ledd_missing_n": int(ledd_miss.sum()),
    "gene_missing_n": int(gene_miss.sum()),
    "ledd_and_gene_both_missing": int((ledd_miss & gene_miss).sum()),
}

# %% [markdown]
# ## 5 – Variable-by-variable analysis

# %%
# Categorical / binary columns: frequencies
cat_cols = ["cohort", "sexM", "gene"]
cat_freq = {}
for col in cat_cols:
    vc = X_train[col].value_counts(dropna=False)
    cat_freq[col] = {
        str(k): {"count": int(v), "pct": round(v / len(X_train) * 100, 1)}
        for k, v in vc.items()
    }
cat_freq

# %%
# Numeric columns: descriptive stats
num_cols = [
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
num_stats = {}
for col in num_cols:
    s = X_train[col].dropna()
    num_stats[col] = {
        "n": len(s),
        "n_missing": int(X_train[col].isnull().sum()),
        "min": round(float(s.min()), 2),
        "q25": round(float(s.quantile(0.25)), 2),
        "median": round(float(s.median()), 2),
        "mean": round(float(s.mean()), 2),
        "q75": round(float(s.quantile(0.75)), 2),
        "max": round(float(s.max()), 2),
        "std": round(float(s.std()), 2),
        "n_outliers_iqr3": int(
            (
                (s < s.quantile(0.25) - 3 * (s.quantile(0.75) - s.quantile(0.25)))
                | (s > s.quantile(0.75) + 3 * (s.quantile(0.75) - s.quantile(0.25)))
            ).sum()
        ),
    }
num_stats

# %%
# Sanity check: age >= age_at_diagnosis on all rows
age_ok = X_train["age"] >= X_train["age_at_diagnosis"]
age_ok_test = X_test["age"] >= X_test["age_at_diagnosis"]
{
    "train_age_ge_age_at_diag": bool(age_ok.all()),
    "train_violations": int((~age_ok).sum()),
    "test_age_ge_age_at_diag": bool(age_ok_test.all()),
    "test_violations": int((~age_ok_test).sum()),
}

# %% [markdown]
# ## 6 – Patient / visit structure

# %%
# Patients in train and test
train_patients = X_train["patient_id"].unique()
test_patients = X_test["patient_id"].unique()
overlap = set(train_patients) & set(test_patients)

visits_per_patient_train = X_train.groupby("patient_id").size()
visits_per_patient_test = X_test.groupby("patient_id").size()

{
    "n_patients_train": len(train_patients),
    "n_patients_test": len(test_patients),
    "patients_in_both": len(overlap),
    "visits_per_patient_train": {
        "mean": round(float(visits_per_patient_train.mean()), 2),
        "median": float(visits_per_patient_train.median()),
        "min": int(visits_per_patient_train.min()),
        "q25": float(visits_per_patient_train.quantile(0.25)),
        "q75": float(visits_per_patient_train.quantile(0.75)),
        "max": int(visits_per_patient_train.max()),
    },
    "visits_per_patient_test": {
        "mean": round(float(visits_per_patient_test.mean()), 2),
        "median": float(visits_per_patient_test.median()),
        "min": int(visits_per_patient_test.min()),
        "max": int(visits_per_patient_test.max()),
    },
}

# %%
# One cohort per patient?
cohort_per_patient_train = X_train.groupby("patient_id")["cohort"].nunique()
cohort_per_patient_test = X_test.groupby("patient_id")["cohort"].nunique()
{
    "max_cohorts_per_patient_train": int(cohort_per_patient_train.max()),
    "patients_multi_cohort_train": int((cohort_per_patient_train > 1).sum()),
    "max_cohorts_per_patient_test": int(cohort_per_patient_test.max()),
}

# %% [markdown]
# ## 7 – Target analysis (y_train)

# %%
target = joined["target"]
target_stats = {
    "n": len(target),
    "min": round(float(target.min()), 2),
    "q25": round(float(target.quantile(0.25)), 2),
    "median": round(float(target.median()), 2),
    "mean": round(float(target.mean()), 2),
    "q75": round(float(target.quantile(0.75)), 2),
    "max": round(float(target.max()), 2),
    "std": round(float(target.std()), 2),
    "skewness": round(float(target.skew()), 3),
}
# By cohort
target_by_cohort = (
    joined.groupby("cohort")["target"]
    .agg(["mean", "median", "std", "count"])
    .round(2)
    .to_dict(orient="index")
)
# Within-patient variability
within_pat = joined.groupby("patient_id")["target"].std()
between_pat = joined.groupby("patient_id")["target"].mean()
{
    "target_stats": target_stats,
    "target_by_cohort": target_by_cohort,
    "within_patient_std": {
        "mean": round(float(within_pat.mean()), 2),
        "median": round(float(within_pat.median()), 2),
        "max": round(float(within_pat.max()), 2),
    },
    "between_patient_std_of_means": round(float(between_pat.std()), 2),
}

# %% [markdown]
# ## 8 – Relationships between features and target

# %%
# 8a. Target mean by missingness of key variables
df = joined.copy()
df["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]

miss_target = {}
for col in ["off", "on", "ledd", "gene"]:
    is_miss = df[col].isnull()
    miss_target[col] = {
        "target_mean_when_present": (
            round(float(df.loc[~is_miss, "target"].mean()), 2)
            if (~is_miss).sum() > 0
            else None
        ),
        "target_mean_when_missing": (
            round(float(df.loc[is_miss, "target"].mean()), 2)
            if is_miss.sum() > 0
            else None
        ),
        "n_present": int((~is_miss).sum()),
        "n_missing": int(is_miss.sum()),
    }
miss_target

# %%
# 8b. Measurement bias: (off - target) by time_since_intake_off tranches
df_off = df[df["off"].notna()].copy()
df_off["bias"] = df_off["off"] - df_off["target"]

# Tranches: missing, 0-4h, 4-8h, 8-12h, 12-24h, >24h
def intake_tranche(t):
    if pd.isna(t):
        return "missing"
    elif t <= 4:
        return "0-4h"
    elif t <= 8:
        return "4-8h"
    elif t <= 12:
        return "8-12h"
    elif t <= 24:
        return "12-24h"
    else:
        return ">24h"


df_off["tranche"] = df_off["time_since_intake_off"].apply(intake_tranche)
bias_by_tranche = (
    df_off.groupby("tranche")["bias"]
    .agg(["mean", "median", "std", "count"])
    .round(3)
    .to_dict(orient="index")
)
{
    "overall_bias_mean": round(float(df_off["bias"].mean()), 3),
    "overall_bias_std": round(float(df_off["bias"].std()), 3),
    "bias_by_time_since_intake_off": bias_by_tranche,
}

# %%
# 8c. Target progression: mean target vs time_since_diagnosis (binned)
df_diag = df[df["time_since_diagnosis"].notna()].copy()
df_diag["tsd_bin"] = pd.cut(df_diag["time_since_diagnosis"], bins=10)
progression = (
    df_diag.groupby("tsd_bin", observed=True)["target"]
    .agg(["mean", "count"])
    .round(2)
    .to_dict(orient="index")
)
{
    "target_vs_time_since_diagnosis": {
        str(k): v for k, v in progression.items()
    }
}

# %%
# 8d. 4 random patients: target, off, on vs age
import random

random.seed(0)
patients_with_multi = (
    joined.groupby("patient_id").filter(lambda g: len(g) >= 2)["patient_id"]
    .unique()
    .tolist()
)
sample_patients = random.sample(patients_with_multi, min(4, len(patients_with_multi)))

patient_trajectories = {}
for pid in sample_patients:
    sub = joined[joined["patient_id"] == pid].sort_values("age")
    patient_trajectories[pid] = sub[["age", "target", "on", "off"]].to_dict(orient="records")
{"sample_patient_trajectories": patient_trajectories}

# %%
# 8e. Correlations of target with numerical features
num_feats = [
    "age",
    "age_at_diagnosis",
    "time_since_diagnosis",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
corr_with_target = {}
for feat in num_feats:
    sub = df[[feat, "target"]].dropna()
    if len(sub) > 5:
        corr_with_target[feat] = {
            "pearson": round(float(sub.corr(method="pearson").iloc[0, 1]), 3),
            "n": len(sub),
        }
corr_with_target

# %%
# 8f. Cohort comparison: target, time_since_diagnosis, ledd, bias
cohort_compare = {}
for cohort in df["cohort"].dropna().unique():
    sub = df[df["cohort"] == cohort]
    sub_off = sub[sub["off"].notna()]
    cohort_compare[cohort] = {
        "n_visits": len(sub),
        "target_mean": round(float(sub["target"].mean()), 2),
        "target_std": round(float(sub["target"].std()), 2),
        "tsd_mean": round(float(sub["time_since_diagnosis"].dropna().mean()), 2),
        "ledd_mean": round(
            float(sub["ledd"].dropna().mean()), 2
        ) if sub["ledd"].notna().any() else None,
        "off_bias_mean": round(
            float((sub_off["off"] - sub_off["target"]).mean()), 2
        ) if len(sub_off) > 0 else None,
    }
cohort_compare

# %% [markdown]
# ## 9 – Data leakage risks

# %%
# Duplicate rows
n_dup_xtrain = int(X_train.duplicated().sum())
n_dup_xtest = int(X_test.duplicated().sum())
n_dup_joined = int(joined.duplicated().sum())

# Patient-level overlap already checked in cell 6.
# Columns to exclude from predictors (identifiers):
id_cols = ["patient_id"]  # Index already used as row id

{
    "duplicate_rows_X_train": n_dup_xtrain,
    "duplicate_rows_X_test": n_dup_xtest,
    "duplicate_rows_joined": n_dup_joined,
    "id_cols_to_exclude_from_features": id_cols,
    "on_off_are_legit_inputs": True,  # present in X_test, not leaks
    "random_rowwise_split_creates_patient_leakage": True,
}

# %% [markdown]
# ## 10 – Train / test distribution comparison

# %%
# Categorical and cohort proportions
compare_cat = {}
for col in ["cohort", "sexM", "gene"]:
    train_pct = X_train[col].value_counts(normalize=True, dropna=False).round(3)
    test_pct = X_test[col].value_counts(normalize=True, dropna=False).round(3)
    all_vals = set(train_pct.index) | set(test_pct.index)
    compare_cat[col] = {
        str(v): {
            "train_pct": float(train_pct.get(v, 0.0)),
            "test_pct": float(test_pct.get(v, 0.0)),
        }
        for v in all_vals
    }
compare_cat

# %%
# Numeric distribution comparison: mean, std, min, max
compare_num = {}
for col in ["age_at_diagnosis", "age", "ledd", "time_since_intake_on",
             "time_since_intake_off", "on", "off"]:
    tr = X_train[col].dropna()
    te = X_test[col].dropna()
    compare_num[col] = {
        "train_mean": round(float(tr.mean()), 2) if len(tr) else None,
        "test_mean": round(float(te.mean()), 2) if len(te) else None,
        "train_std": round(float(tr.std()), 2) if len(tr) else None,
        "test_std": round(float(te.std()), 2) if len(te) else None,
        "train_min": round(float(tr.min()), 2) if len(tr) else None,
        "test_min": round(float(te.min()), 2) if len(te) else None,
        "train_max": round(float(tr.max()), 2) if len(tr) else None,
        "test_max": round(float(te.max()), 2) if len(te) else None,
        "train_null_pct": round(X_train[col].isnull().mean() * 100, 1),
        "test_null_pct": round(X_test[col].isnull().mean() * 100, 1),
    }
compare_num

# %% [markdown]
# ## 11 – Column associations (skrub)
#
# Strongest pairwise associations in the joined train set (features + target).

# %%
assoc_df = skrub.column_associations(
    joined[
        [
            "cohort", "sexM", "gene",
            "age_at_diagnosis", "age", "ledd",
            "time_since_intake_on", "time_since_intake_off",
            "on", "off", "target",
        ]
    ]
)
assoc_df.head(20)

# %% [markdown]
# ## Summary
#
# Findings and modelling implications are written up in `data/eda.md`.
