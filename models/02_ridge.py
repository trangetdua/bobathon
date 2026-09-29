"""Step 8 — Ridge vs Dummy comparison.

SimpleImputer(median) + Ridge(alpha=1.0) compared to DummyRegressor(mean).
Default splitter (0.2 random row holdout), no GroupKFold yet (step 10).
Report pushed to Hub under key '02_ridge'.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
X_train = pd.read_csv(_repo_root / "data" / "X_train.csv")
y_train = pd.read_csv(_repo_root / "data" / "y_train.csv")

visits = X_train.merge(y_train, on="Index")
assert len(visits) == 44_590, (
    f"Expected 44 590 rows after join, got {len(visits)}."
)
print(f"visits shape: {visits.shape}")

# --- 2. Features and target ---------------------------------------------------
feature_cols = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

# Guard: fail loudly if any expected column is absent
missing_cols = [c for c in feature_cols if c not in visits.columns]
if missing_cols:
    raise ValueError(f"Missing columns in visits: {missing_cols}")

X = visits[feature_cols]
y = visits["target"]
print(f"X shape: {X.shape}, y shape: {y.shape}")

# --- 3. Models ----------------------------------------------------------------
dummy = DummyRegressor(strategy="mean")
ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0),
)

# --- 4. Evaluate both for comparison, push Ridge report ----------------------
# ComparisonReport is for display only; project.put() requires EstimatorReport.
comparison = evaluate({"dummy": dummy, "ridge": ridge}, X, y)
rmse_df = comparison.metrics.rmse()
print(f"\nRMSE comparison (splitter=0.2, random holdout):\n{rmse_df}")

rmse_dummy = float(rmse_df.loc["RMSE", "dummy"])
rmse_ridge = float(rmse_df.loc["RMSE", "ridge"])
improvement = rmse_dummy - rmse_ridge
print(f"\ndummy RMSE : {rmse_dummy:.4f}")
print(f"ridge RMSE : {rmse_ridge:.4f}")
print(f"improvement: {improvement:+.4f} ({'Ridge beats Dummy ✓' if improvement > 0 else 'Ridge does NOT beat Dummy'})")

# Separate Ridge EstimatorReport for Hub upload
report = evaluate(ridge, X, y)

# --- 5. Push to Hub -----------------------------------------------------------
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("02_ridge", report)
# Console prints: Consult your report at https://skore.probabl.ai/…
