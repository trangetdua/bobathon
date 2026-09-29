"""Step 11+ v2 — Extended patient features + tuned HistGradientBoosting.

make_features(df) adds per-patient aggregates, lag/lead values, linear trends,
interpolation and gap features to the 8 base columns.
Applied independently to train and test — no target leakage.

Pushes:
  - '11_hgb_v2_cv'  : CrossValidationReport comparing hgb_v1 vs hgb_v2
  - '12_hgb_v2'     : EstimatorReport for hgb_v2 (default splitter)
Writes submissions/12_hgb_v2.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login
from skore_cli.agent._skore_file import SkoreConfig

# --- Credentials from .skore -------------------------------------------------
_repo_root = Path(__file__).resolve().parents[1]
_cfg = SkoreConfig.load(_repo_root)
if _cfg is None:
    raise RuntimeError(".skore not found. Run `python scripts/skore-agent` first.")
cfg = {"workspace": _cfg.workspace, "api_key": _cfg.api_key, "hub_url": _cfg.hub_url}

# --- Feature definitions -----------------------------------------------------
BASE_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]

# V1 feature set (step 08) — used to reconstruct hgb_v1 baseline
V1_EXTRA = [
    "time_since_diagnosis",
    "p_off_mean", "p_on_mean", "p_ledd_mean", "p_off_count",
    "p_nvis", "p_age_min", "years_since_first", "visit_rank",
]
V1_COLS = BASE_COLS + V1_EXTRA


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from a raw visits table.

    Applied independently to train and test.
    Never touches 'target'. NaNs preserved; infs replaced with NaN.
    Returns columns in a fixed order: BASE_COLS + derived features.
    Index, patient_id, gene, cohort are not included.
    """
    # Work on a sorted copy; track original index to restore order at the end
    orig_index = df.index
    df_s = df.sort_values(["patient_id", "age"]).copy()

    out = pd.DataFrame(index=df_s.index)

    # 8 base columns
    for c in BASE_COLS:
        out[c] = df_s[c]

    # time_since_diagnosis
    out["time_since_diagnosis"] = df_s["age"] - df_s["age_at_diagnosis"]

    grp = df_s.groupby("patient_id")

    # Basic per-patient aggregates
    out["p_off_mean"]  = grp["off"].transform("mean")
    out["p_on_mean"]   = grp["on"].transform("mean")
    out["p_ledd_mean"] = grp["ledd"].transform("mean")
    out["p_off_count"] = grp["off"].transform("count")
    out["p_nvis"]      = grp["patient_id"].transform("count")
    out["p_age_min"]   = grp["age"].transform("min")
    out["years_since_first"] = df_s["age"] - out["p_age_min"]
    out["visit_rank"]  = grp["age"].rank(method="first").astype(float)

    # Linear trend: polyfit(age, off|on) evaluated at each visit's age
    def _trend(series_col: str) -> pd.Series:
        result = pd.Series(np.nan, index=df_s.index)
        for pid, g in df_s.groupby("patient_id"):
            valid = g[["age", series_col]].dropna()
            if len(valid) == 0:
                continue
            elif len(valid) == 1 or valid["age"].nunique() == 1:
                # Only one distinct age: use the single value for all rows
                result.loc[g.index] = valid[series_col].iloc[0]
            else:
                slope, intercept = np.polyfit(valid["age"], valid[series_col], 1)
                result.loc[g.index] = slope * g["age"] + intercept
        return result

    out["off_trend"] = _trend("off")
    out["on_trend"]  = _trend("on")

    # Lag / lead shifts (within patient, sorted by age)
    out["off_prev"]  = grp["off"].shift(1)
    out["off_next"]  = grp["off"].shift(-1)
    out["on_prev"]   = grp["on"].shift(1)
    out["on_next"]   = grp["on"].shift(-1)
    out["off_prev2"] = grp["off"].shift(2)
    out["off_next2"] = grp["off"].shift(-2)

    # Interpolation within patient
    def _interp(series_col: str) -> pd.Series:
        result = pd.Series(np.nan, index=df_s.index)
        for pid, g in df_s.groupby("patient_id"):
            interped = g[series_col].interpolate(
                method="linear", limit_direction="both"
            )
            result.loc[g.index] = interped.values
        return result

    out["off_interp"] = _interp("off")
    out["on_interp"]  = _interp("on")

    # Additional per-patient stats
    out["p_off_med"]      = grp["off"].transform("median")
    out["p_off_std"]      = grp["off"].transform("std")
    out["p_ratio"]        = out["p_off_mean"] / out["p_on_mean"]
    out["p_tsi_off_mean"] = grp["time_since_intake_off"].transform("mean")

    # Age gaps to previous / next visit
    out["gap_prev"] = df_s["age"] - grp["age"].shift(1)
    out["gap_next"] = grp["age"].shift(-1) - df_s["age"]

    # Combined interpolated gap
    out["on_interp_off_gap"] = out["off_interp"] - out["on_interp"]

    # Replace any infinities with NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Restore original row order
    return out.reindex(orig_index)


# --- 1. Load and join ---------------------------------------------------------
X_train_raw = pd.read_csv(_repo_root / "data" / "X_train.csv")
y_train_raw = pd.read_csv(_repo_root / "data" / "y_train.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")
assert len(visits) == 44_590, f"Expected 44 590 rows, got {len(visits)}."
print(f"visits shape: {visits.shape}")

y      = visits["target"]
groups = visits["patient_id"]

# Build feature matrices
X_v1 = make_features(visits)[V1_COLS]        # 17 cols — same as step 08
X_v2 = make_features(visits)                  # all new cols

print(f"\nX_v1 shape : {X_v1.shape}  (17 features, step-08 baseline)")
print(f"X_v2 shape : {X_v2.shape}  (all new features)")
print(f"New features in v2: {[c for c in X_v2.columns if c not in V1_COLS]}")

# Sanity: no target leakage, no identifier columns
assert "target" not in X_v2.columns
assert "patient_id" not in X_v2.columns
assert "Index" not in X_v2.columns

# --- 2. GroupKFold splits -----------------------------------------------------
cv_splits = list(GroupKFold(n_splits=5).split(X_v2, y, groups=groups))
print(f"\ncv_splits: {len(cv_splits)} folds")

# --- 3. Helpers for RMSE extraction ------------------------------------------
def _rmse_mean_std(report) -> tuple[float, float]:
    """Extract mean and std RMSE from CrossValidationReport."""
    df = report.metrics.rmse()
    row = df.loc["RMSE"]
    last = row.index.get_level_values(-1)
    return float(row.iloc[(last == "mean").argmax()]), \
           float(row.iloc[(last == "std").argmax()])


# --- 4. Evaluate hgb_v1 (step-08 features) -----------------------------------
hgb_v1 = HistGradientBoostingRegressor(random_state=0)
print("\nEvaluating hgb_v1 (17 features, step-08 baseline)…")
rep_v1 = evaluate(hgb_v1, X_v1, y, splitter=cv_splits)
rmse_v1_mean, rmse_v1_std = _rmse_mean_std(rep_v1)
print(f"  hgb_v1  RMSE = {rmse_v1_mean:.4f} ± {rmse_v1_std:.4f}")

# --- 5. Evaluate hgb_v2 (all new features) -----------------------------------
hgb_v2 = HistGradientBoostingRegressor(
    random_state=0, learning_rate=0.05, max_iter=1000, early_stopping=False
)
print("Evaluating hgb_v2 (all features, tuned)…")
rep_v2_cv = evaluate(hgb_v2, X_v2, y, splitter=cv_splits)
rmse_v2_mean, rmse_v2_std = _rmse_mean_std(rep_v2_cv)
print(f"  hgb_v2  RMSE = {rmse_v2_mean:.4f} ± {rmse_v2_std:.4f}")

delta = rmse_v1_mean - rmse_v2_mean
print(f"\nImprovement v1 → v2: {delta:+.4f} RMSE points "
      f"({'✓' if delta > 0 else '✗ — no improvement'})")

# --- 6. Push '11_hgb_v2_cv' --------------------------------------------------
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("11_hgb_v2_cv", rep_v2_cv)

# --- 7. EstimatorReport '12_hgb_v2' (default splitter) ----------------------
rep_v2_est = evaluate(hgb_v2, X_v2, y)
project.put("12_hgb_v2", rep_v2_est)

# --- 8. Kaggle submission -----------------------------------------------------
X_test_raw = pd.read_csv(_repo_root / "data" / "X_test.csv")
sample_sub  = pd.read_csv(_repo_root / "data" / "sample_submission.csv")

X_test_feat = make_features(X_test_raw)

# Ensure identical column order
assert list(X_test_feat.columns) == list(X_v2.columns), (
    "Column mismatch between train and test features:\n"
    f"  train: {list(X_v2.columns)}\n"
    f"  test : {list(X_test_feat.columns)}"
)

final = clone(hgb_v2).fit(X_v2, y)
preds = final.predict(X_test_feat)

submission = pd.DataFrame({
    "Index":  X_test_raw["Index"].values,
    "target": preds,
})

# Integrity checks
assert list(submission.columns) == ["Index", "target"]
assert len(submission) == 11_013, f"Expected 11 013 rows, got {len(submission)}"
assert submission["target"].isna().sum() == 0, "NaN predictions"
# Same Index and same order as sample_submission
assert list(submission["Index"]) == list(sample_sub["Index"]), \
    "Index order mismatch with sample_submission"

out_dir  = _repo_root / "submissions"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "12_hgb_v2.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  ({len(submission)} rows)")

pred_min   = preds.min()
pred_mean  = preds.mean()
pred_max   = preds.max()
n_neg      = int((preds < 0).sum())
n_above132 = int((preds > 132).sum())
print(f"Predictions  min={pred_min:.3f}  mean={pred_mean:.3f}  max={pred_max:.3f}")
if n_neg:
    print(f"WARNING: {n_neg} prediction(s) below 0 — not corrected.")
else:
    print("No predictions below 0.")
if n_above132:
    print(f"WARNING: {n_above132} prediction(s) above 132 — not corrected.")
else:
    print("No predictions above 132.")
