"""Step 11 — HistGradientBoostingRegressor: missingness as signal.

Compares Dummy / Ridge / HGBR with GroupKFold(n_splits=5) on patient_id.
HGBR receives raw NaNs — no imputation. Ridge keeps its median imputer.
Pushes:
  - '05_hgb_cv'  : CrossValidationReport for HGBR (grouped CV)
  - '06_hgb'     : EstimatorReport for HGBR (default splitter, for Kaggle URL)
Writes submissions/06_hgb.csv.
"""

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
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

# --- 1. Load and join ---------------------------------------------------------
X_train_raw = pd.read_csv(_repo_root / "data" / "X_train.csv")
y_train_raw = pd.read_csv(_repo_root / "data" / "y_train.csv")

visits = X_train_raw.merge(y_train_raw, on="Index")
assert len(visits) == 44_590, f"Expected 44 590 rows after join, got {len(visits)}."
print(f"visits shape: {visits.shape}")

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]
missing_cols = [c for c in FEATURE_COLS if c not in visits.columns]
if missing_cols:
    raise ValueError(f"Missing columns in visits: {missing_cols}")

X = visits[FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]

# Confirm NaNs are present (HGBR will handle them natively)
nan_counts = X.isna().sum()
print("\nNaN counts per feature (passed as-is to HGBR):")
print(nan_counts[nan_counts > 0].to_string())
print()

# --- 2. GroupKFold splits (same as step 10) -----------------------------------
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"cv_splits: {len(cv_splits)} folds, "
      f"~{len(cv_splits[0][0])} train / ~{len(cv_splits[0][1])} val rows per fold\n")

# --- 3. Models ----------------------------------------------------------------
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
hgbr  = HistGradientBoostingRegressor(random_state=0)  # no imputation

# --- 4. Three-way comparison with GroupKFold ----------------------------------
comparison = evaluate(
    {"dummy": dummy, "ridge": ridge, "hgbr": hgbr},
    X, y,
    splitter=cv_splits,
)

rmse_df = comparison.metrics.rmse()
print("RMSE (GroupKFold n_splits=5)  mean ± std:")
print(rmse_df.to_string())

# rmse_df columns: MultiIndex with levels (stat="mean"/"std", Estimator)
# e.g. rmse_df[("mean", "dummy")] works; use xs to select by stat level
row = rmse_df.loc["RMSE"]   # Series with MultiIndex (stat, Estimator)
means = row.xs("mean", level=0)   # Series indexed by estimator name
stds  = row.xs("std",  level=0)

print()
for est in ["dummy", "ridge", "hgbr"]:
    m = float(means[est])
    s = float(stds[est])
    print(f"  {est:<6}  RMSE = {m:.4f} ± {s:.4f}")

rmse_ridge = float(means["ridge"])
rmse_hgbr  = float(means["hgbr"])
delta = rmse_ridge - rmse_hgbr
if delta > 0:
    print(f"\nHGBR improves Ridge by {delta:+.4f} RMSE points ✓")
else:
    print(f"\nHGBR does NOT improve Ridge (Δ = {delta:+.4f})")

# --- 5. Push '05_hgb_cv' (CrossValidationReport for HGBR) --------------------
report_hgbr_cv = evaluate(hgbr, X, y, splitter=cv_splits)

login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("05_hgb_cv", report_hgbr_cv)
# → prints URL

# --- 6. EstimatorReport for '06_hgb' (default splitter=0.2) ------------------
report_hgbr_est = evaluate(hgbr, X, y)
project.put("06_hgb", report_hgbr_est)
# → prints URL

# --- 7. Kaggle submission -----------------------------------------------------
X_test    = pd.read_csv(_repo_root / "data" / "X_test.csv")
sample_sub = pd.read_csv(_repo_root / "data" / "sample_submission.csv")

final = clone(hgbr).fit(X, y)
preds = final.predict(X_test[FEATURE_COLS])

submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})

assert list(submission.columns) == ["Index", "target"], "Wrong columns"
assert len(submission) == 11_013, f"Expected 11 013 rows, got {len(submission)}"
assert submission["target"].isna().sum() == 0, "NaN predictions found"
assert set(submission["Index"]) == set(sample_sub["Index"]), \
    "Index mismatch with sample_submission"

out_dir = _repo_root / "submissions"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "06_hgb.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written : {out_path}  ({len(submission)} rows)")

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

print("\nNaN handling confirmed: HGBR received raw NaN values, no imputation applied.")
