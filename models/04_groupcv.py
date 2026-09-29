"""Step 10 — Patient-grouped cross-validation (GroupKFold, n_splits=5).

Compares DummyRegressor and Ridge(alpha=1.0) with GroupKFold on patient_id.
Verifies no patient leaks across folds.
Pushes ComparisonReport to Hub under key '04_groupcv'.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
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
missing = [c for c in FEATURE_COLS if c not in visits.columns]
if missing:
    raise ValueError(f"Missing columns in visits: {missing}")

X = visits[FEATURE_COLS]
y = visits["target"]
groups = visits["patient_id"]

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Unique patients: {groups.nunique()}")

# --- 2. Build cv_splits and verify no patient leakage ------------------------
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))

print("\nFold-level patient leak check:")
all_ok = True
for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    train_patients = set(groups.iloc[train_idx])
    val_patients   = set(groups.iloc[val_idx])
    overlap        = train_patients & val_patients
    status = "OK" if not overlap else f"LEAK — {len(overlap)} patients in both"
    print(f"  Fold {fold_i + 1}: train={len(train_idx):>6} rows / {len(train_patients):>5} patients | "
          f"val={len(val_idx):>5} rows / {len(val_patients):>5} patients | {status}")
    if overlap:
        all_ok = False

if not all_ok:
    raise RuntimeError("Patient leakage detected across folds — aborting.")
print("No patient leakage across any fold. ✓\n")

# --- 3. Evaluate dummy and ridge with GroupKFold splits ----------------------
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))

comparison = evaluate({"dummy": dummy, "ridge": ridge}, X, y, splitter=cv_splits)

print("Metrics summary (GroupKFold, n_splits=5):")
summary = comparison.metrics.summarize()
print(summary)

rmse_df = comparison.metrics.rmse()
print(f"\nRMSE table (mean ± std over 5 folds):\n{rmse_df}")

# Individual CrossValidationReport for Ridge (required by project.put)
report_ridge = evaluate(ridge, X, y, splitter=cv_splits)

# --- 4. Push to Hub ----------------------------------------------------------
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("04_groupcv", report_ridge)
# Console prints: Consult your report at https://skore.probabl.ai/…
