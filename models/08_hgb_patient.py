"""Step 11+ — Patient-level aggregate features + HistGradientBoosting.

make_features(df) adds per-patient aggregates (computed only from within df,
no target leakage) to the 8 base features from step 11.

Pushes:
  - '07_hgb_patient_cv' : CrossValidationReport comparing hgb_base vs hgb_patient
  - '08_hgb_patient'    : EstimatorReport for hgb_patient (default splitter)
Writes submissions/08_hgb_patient.csv.
"""

from pathlib import Path

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
    raise RuntimeError(
        ".skore not found. Run `python scripts/skore-agent` first to sign in."
    )
cfg = {"workspace": _cfg.workspace, "api_key": _cfg.api_key, "hub_url": _cfg.hub_url}

# --- Feature engineering -----------------------------------------------------
BASE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from a raw visits table (no target, no leakage).

    Applied independently to train and test — each table uses only its own rows.
    NaNs are preserved; target is never touched.
    Returns a DataFrame with BASE_COLS plus derived patient-level features.
    No Index, patient_id, gene, or cohort columns included.
    """
    out = df[BASE_COLS].copy()

    # Derived from individual row
    out["time_since_diagnosis"] = df["age"] - df["age_at_diagnosis"]

    # Per-patient aggregates via groupby + transform (no cross-table leakage)
    grp = df.groupby("patient_id")
    out["p_off_mean"]  = grp["off"].transform("mean")
    out["p_on_mean"]   = grp["on"].transform("mean")
    out["p_ledd_mean"] = grp["ledd"].transform("mean")
    out["p_off_count"] = grp["off"].transform("count")   # non-NaN count
    out["p_nvis"]      = grp["patient_id"].transform("count")
    out["p_age_min"]   = grp["age"].transform("min")

    # Derived from patient aggregates
    out["years_since_first"] = df["age"] - out["p_age_min"]

    # Visit rank within patient by age (1 = earliest)
    out["visit_rank"] = grp["age"].rank(method="first").astype(float)

    return out


# --- 1. Load and join ---------------------------------------------------------
X_train_raw = pd.read_csv(_repo_root / "data" / "X_train.csv")
y_train_raw = pd.read_csv(_repo_root / "data" / "y_train.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")
assert len(visits) == 44_590, f"Expected 44 590 rows, got {len(visits)}."
print(f"visits shape: {visits.shape}")

y      = visits["target"]
groups = visits["patient_id"]

# Build feature matrices
X_base    = visits[BASE_COLS]
X_patient = make_features(visits)

print(f"\nX_base    shape: {X_base.shape}")
print(f"X_patient shape: {X_patient.shape}")
print(f"New features: {[c for c in X_patient.columns if c not in BASE_COLS]}")

# --- 2. GroupKFold splits -----------------------------------------------------
cv_splits = list(GroupKFold(n_splits=5).split(X_patient, y, groups=groups))
print(f"\ncv_splits: {len(cv_splits)} folds")

# --- 3. Compare hgb_base vs hgb_patient with GroupKFold ----------------------
hgb_base    = HistGradientBoostingRegressor(random_state=0)
hgb_patient = HistGradientBoostingRegressor(random_state=0)

# Evaluate separately (different X) then print together
def _extract_mean_std(rmse_df: pd.DataFrame) -> tuple[float, float]:
    """Extract RMSE mean and std from a CrossValidationReport.rmse() DataFrame.

    Columns are MultiIndex (EstimatorName, "mean"|"std").
    Locate the columns whose last-level label is "mean" and "std".
    """
    row = rmse_df.loc["RMSE"]   # Series with MultiIndex columns as index
    # Find positions of "mean" and "std" in the last index level
    last_level = row.index.get_level_values(-1)
    mean_val = float(row.iloc[(last_level == "mean").argmax()])
    std_val  = float(row.iloc[(last_level == "std").argmax()])
    return mean_val, std_val


print("\nEvaluating hgb_base (8 features) with GroupKFold…")
rep_base      = evaluate(hgb_base, X_base, y, splitter=cv_splits)
rmse_base_df  = rep_base.metrics.rmse()
rmse_base_mean, rmse_base_std = _extract_mean_std(rmse_base_df)

print("Evaluating hgb_patient (all features) with GroupKFold…")
rep_patient_cv = evaluate(hgb_patient, X_patient, y, splitter=cv_splits)

rmse_cv_df = rep_patient_cv.metrics.rmse()
print(f"\nRMSE table (hgb_patient, GroupKFold):\n{rmse_cv_df}")
rmse_patient_mean, rmse_patient_std = _extract_mean_std(rmse_cv_df)

print(f"\nGroupKFold RMSE summary (n_splits=5):")
print(f"  hgb_base    RMSE = {rmse_base_mean:.4f} ± {rmse_base_std:.4f}")
print(f"  hgb_patient RMSE = {rmse_patient_mean:.4f} ± {rmse_patient_std:.4f}")
delta = rmse_base_mean - rmse_patient_mean
if delta > 0:
    print(f"  patient features improve base by {delta:+.4f} RMSE points ✓")
else:
    print(f"  patient features do NOT improve base (Δ = {delta:+.4f})")

# --- 4. Push '07_hgb_patient_cv' (CrossValidationReport for hgb_patient) ----
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("07_hgb_patient_cv", rep_patient_cv)

# --- 5. EstimatorReport for '08_hgb_patient' (default splitter) --------------
rep_patient_est = evaluate(hgb_patient, X_patient, y)
project.put("08_hgb_patient", rep_patient_est)

# --- 6. Kaggle submission -----------------------------------------------------
X_test_raw = pd.read_csv(_repo_root / "data" / "X_test.csv")
sample_sub  = pd.read_csv(_repo_root / "data" / "sample_submission.csv")

X_test_feat = make_features(X_test_raw)

# Ensure same column order as training
assert list(X_test_feat.columns) == list(X_patient.columns), (
    f"Column mismatch between train and test features:\n"
    f"  train: {list(X_patient.columns)}\n"
    f"  test : {list(X_test_feat.columns)}"
)

final = clone(hgb_patient).fit(X_patient, y)
preds = final.predict(X_test_feat)

submission = pd.DataFrame({"Index": X_test_raw["Index"], "target": preds})

assert list(submission.columns) == ["Index", "target"]
assert len(submission) == 11_013, f"Expected 11 013 rows, got {len(submission)}"
assert submission["target"].isna().sum() == 0, "NaN predictions found"
assert set(submission["Index"]) == set(sample_sub["Index"]), \
    "Index mismatch with sample_submission"

out_dir  = _repo_root / "submissions"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "08_hgb_patient.csv"
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
