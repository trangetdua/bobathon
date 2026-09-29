"""Step 12 — skrub tabular_pipeline: mixed types without hand-encoding.

tabular_pipeline("regressor") = TableVectorizer + HistGradientBoostingRegressor.
TableVectorizer auto-encodes categoricals (gene, cohort) and passes numerics
with NaNs intact to HGBR. No manual imputation.

Evaluated with GroupKFold(n_splits=5) on patient_id.
Pushes CrossValidationReport under key '09_skrub_cv'.
No submission file — evaluation only (step 12 of guide).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login
from skore_cli.agent._skore_file import SkoreConfig
from skrub import tabular_pipeline

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
assert len(visits) == 44_590, f"Expected 44 590 rows, got {len(visits)}."
print(f"visits shape: {visits.shape}")

y      = visits["target"]
groups = visits["patient_id"]

# X_full: drop identifiers and target — keep everything else incl. gene, cohort
X_full = visits.drop(columns=["Index", "patient_id", "target"])
print(f"X_full shape : {X_full.shape}")
print(f"X_full columns ({len(X_full.columns)}): {X_full.columns.tolist()}")

# NaN counts (passed as-is)
nan_counts = X_full.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
print(f"\nNaN counts in X_full:\n{nan_counts.to_string()}\n")

# --- 2. GroupKFold splits -----------------------------------------------------
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
print(f"cv_splits: {len(cv_splits)} folds")

# --- 3. tabular_pipeline model -----------------------------------------------
model = tabular_pipeline("regressor")
print(f"\nModel pipeline:\n{model}\n")

# Inspect TableVectorizer to understand how it encodes columns
# (fit on a small sample just for reporting — not used for evaluation)
from skrub import TableVectorizer  # noqa: E402

tv = TableVectorizer()
tv.fit(X_full.head(1000))
print("TableVectorizer column encoders (fitted on 1 000-row sample):")
try:
    # named_transformers_ maps role-name → fitted transformer
    for role, fitted_t in tv.named_transformers_.items():
        print(f"  {role:<30} → {type(fitted_t).__name__}")
except Exception:
    pass
# column_to_encoder maps each column to its transformer
try:
    for col, fitted_t in tv.column_to_encoder_.items():
        print(f"  {col:<35} → {type(fitted_t).__name__}")
except Exception:
    pass
# Fallback: show feature_names_in_ grouped by dtype
print("\nColumn dtypes in X_full:")
for col in X_full.columns:
    print(f"  {col:<35} dtype={X_full[col].dtype}  n_unique={X_full[col].nunique()}  "
          f"null%={X_full[col].isna().mean()*100:.1f}")
print()

# --- 4. Evaluate with GroupKFold ---------------------------------------------
print("Evaluating tabular_pipeline with GroupKFold(n_splits=5)…")
report = evaluate(model, X_full, y, splitter=cv_splits)

rmse_df = report.metrics.rmse()
print(f"\nRMSE table:\n{rmse_df}")

# Extract mean and std (columns are MultiIndex: (EstimatorName, "mean"|"std"))
row        = rmse_df.loc["RMSE"]
last_level = row.index.get_level_values(-1)
rmse_mean  = float(row.iloc[(last_level == "mean").argmax()])
rmse_std   = float(row.iloc[(last_level == "std").argmax()])

print(f"\ntabular_pipeline RMSE = {rmse_mean:.4f} ± {rmse_std:.4f}")

# Comparison table
print("\nGroupKFold RMSE comparison (n_splits=5):")
print(f"  {'Dummy':<35} 16.4992 ± 0.3429")
print(f"  {'Ridge (median impute)':<35} 10.4365 ± 0.1968")
print(f"  {'HGBR base (8 features)':<35}  7.4362 ± 0.1353")
print(f"  {'HGBR + patient features (17)':<35}  4.5826 ± 0.0853")
print(f"  {'tabular_pipeline (skrub)':<35} {rmse_mean:.4f} ± {rmse_std:.4f}  ← this run")

if rmse_mean < 7.4362:
    print(f"\n→ tabular_pipeline improves HGBR base by {7.4362 - rmse_mean:+.4f} RMSE points ✓")
else:
    print(f"\n→ tabular_pipeline does NOT improve HGBR base (Δ = {7.4362 - rmse_mean:+.4f})")

# --- 5. Push to Hub ----------------------------------------------------------
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("09_skrub_cv", report)
# Console prints: Consult your report at https://skore.probabl.ai/…
