"""Step 9 — Ridge alpha search + first Kaggle submission.

Tests alpha in {0.1, 1.0, 10.0, 100.0}, selects best by RMSE (ties → alpha=1.0),
pushes EstimatorReport under key '03_ridge', writes submissions/03_ridge.csv.
No GroupKFold (step 10). No labels from test set.
"""

from pathlib import Path

import pandas as pd
from sklearn.base import clone
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

# --- 2. Alpha search ----------------------------------------------------------
ALPHAS = [0.1, 1.0, 10.0, 100.0]

dummy = DummyRegressor(strategy="mean")
dummy_report = evaluate(dummy, X, y)
rmse_dummy = float(dummy_report.metrics.rmse())

results = {}
for alpha in ALPHAS:
    ridge_a = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha))
    rep = evaluate(ridge_a, X, y)
    rmse_val = float(rep.metrics.rmse())
    results[alpha] = rmse_val
    print(f"  alpha={alpha:>7.1f}  RMSE={rmse_val:.6f}")

print(f"\n{'Alpha':>10}  {'RMSE':>12}")
print("-" * 25)
print(f"{'dummy':>10}  {rmse_dummy:>12.6f}")
for alpha, rmse_val in results.items():
    print(f"{alpha:>10.1f}  {rmse_val:>12.6f}")

# Select best alpha (ties within 0.01 → keep 1.0)
best_alpha = min(results, key=results.__getitem__)
best_rmse = results[best_alpha]
for alpha, rmse_val in results.items():
    if alpha != best_alpha and abs(rmse_val - best_rmse) < 0.01:
        best_alpha = 1.0
        best_rmse = results[1.0]
        break

print(f"\nSelected alpha : {best_alpha}")
print(f"Ridge RMSE     : {best_rmse:.6f}")
print(f"Dummy RMSE     : {rmse_dummy:.6f}")
print(f"Improvement    : {rmse_dummy - best_rmse:+.6f}")

# --- 3. Final Ridge report for Hub -------------------------------------------
ridge = make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=best_alpha))
report = evaluate(ridge, X, y)

# --- 4. Push to Hub ----------------------------------------------------------
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("03_ridge", report)
# Console prints: Consult your report at https://skore.probabl.ai/…

# --- 5. Kaggle submission -----------------------------------------------------
X_test = pd.read_csv(_repo_root / "data" / "X_test.csv")
sample_sub = pd.read_csv(_repo_root / "data" / "sample_submission.csv")

final = clone(ridge).fit(X, y)
preds = final.predict(X_test[FEATURE_COLS])

submission = pd.DataFrame({"Index": X_test["Index"], "target": preds})

# Integrity checks
assert list(submission.columns) == ["Index", "target"], "Wrong columns"
assert len(submission) == len(X_test), f"Row count mismatch: {len(submission)} vs {len(X_test)}"
assert submission["target"].isna().sum() == 0, "NaN predictions found"
assert set(submission["Index"]) == set(sample_sub["Index"]), "Index mismatch with sample_submission"

out_dir = _repo_root / "submissions"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "03_ridge.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  ({len(submission)} rows)")

pred_min = preds.min()
pred_mean = preds.mean()
pred_max = preds.max()
n_negative = (preds < 0).sum()
n_above_132 = (preds > 132).sum()
print(f"Predictions  min={pred_min:.3f}  mean={pred_mean:.3f}  max={pred_max:.3f}")
if n_negative:
    print(f"WARNING: {n_negative} prediction(s) below 0 (MDS-UPDRS min is 0) — not corrected.")
else:
    print("No predictions below 0.")
if n_above_132:
    print(f"WARNING: {n_above_132} prediction(s) above 132 (MDS-UPDRS max is 132) — not corrected.")
else:
    print("No predictions above 132.")
