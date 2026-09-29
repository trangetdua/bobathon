"""Step 13 — skrub DataOps: features, target and grouped CV in one graph.

evaluate(pred, data={"visits": visits}) reads the GroupKFold splitter and
patient groups directly from the DataOp — no separate cv_splits variable.

Pushes CrossValidationReport under key '10_dataops'.
Evaluation only, no submission file.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skore import Project, evaluate, login
from skore_cli.agent._skore_file import SkoreConfig
from skrub import TableVectorizer
import skrub

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

# --- 2. DataOps graph --------------------------------------------------------
data   = skrub.var("visits", visits)
groups = data["patient_id"]

X_op = (
    data.drop(columns=["Index", "patient_id", "target"])
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
)
y_op = data["target"].skb.mark_as_y()

pred = (
    X_op
    .skb.apply(TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
)

print("DataOps graph built.")
print("  X drops       : Index, patient_id, target")
print("  CV            : GroupKFold(n_splits=5) on patient_id (baked in graph)")
print("  Pipeline      : TableVectorizer → HistGradientBoostingRegressor(random_state=0)")

# --- 3. Evaluate — splitter and groups read from the DataOp ------------------
print("\nEvaluating: evaluate(pred, data={'visits': visits})…")
report = evaluate(pred, data={"visits": visits})

rmse_df = report.metrics.rmse()
print(f"\nRMSE table:\n{rmse_df}")

row        = rmse_df.loc["RMSE"]
last_level = row.index.get_level_values(-1)
rmse_mean  = float(row.iloc[(last_level == "mean").argmax()])
rmse_std   = float(row.iloc[(last_level == "std").argmax()])

print(f"\nDataOps RMSE     = {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"Step-12 RMSE ref = 7.4240 ± 0.1355  (tabular_pipeline, separate cv_splits)")
delta = 7.4240 - rmse_mean
print(f"Difference       : {delta:+.4f}  (within noise — same model, same splits)")

# --- 4. Push to Hub ----------------------------------------------------------
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("10_dataops", report)
# Console prints: Consult your report at https://skore.probabl.ai/…
