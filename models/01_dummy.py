# %% [markdown]
# # Step 7 — Dummy mean baseline
#
# Evaluates a DummyRegressor(strategy="mean") as a floor metric.
# Push the report to the Hub under key "01_dummy".

# %%
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from skore import Project, evaluate, login
from skore_cli.agent._skore_file import SkoreConfig

# --- Load credentials from .skore -------------------------------------------
_repo_root = Path(__file__).resolve().parents[1]
_cfg = SkoreConfig.load(_repo_root)
if _cfg is None:
    raise RuntimeError(
        ".skore not found. Run `python scripts/skore-agent` first to sign in."
    )
# Matches the parkinson.hub.load_skore_credentials() contract from the guide.
cfg = {"workspace": _cfg.workspace, "api_key": _cfg.api_key, "hub_url": _cfg.hub_url}

# %%
# --- 1. Load and join --------------------------------------------------------
X_train = pd.read_csv(_repo_root / "data" / "X_train.csv")
y_train = pd.read_csv(_repo_root / "data" / "y_train.csv")

visits = X_train.merge(y_train, on="Index")
assert len(visits) == len(X_train) == len(y_train), (
    f"Row count mismatch after join: X_train={len(X_train)}, "
    f"y_train={len(y_train)}, joined={len(visits)}"
)
{"visits_shape": visits.shape}

# %%
# --- 2. Features and target --------------------------------------------------
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
X = visits[feature_cols]
y = visits["target"]

{"X_shape": X.shape, "y_shape": y.shape}

# %%
# --- 3. Evaluate DummyRegressor ----------------------------------------------
dummy = DummyRegressor(strategy="mean")
report = evaluate(dummy, X, y)  # default splitter=0.2 (random row holdout)

rmse = report.metrics.rmse()
print(f"RMSE (dummy mean, splitter=0.2): {rmse}")
{"rmse": rmse}

# %%
# --- 4 & 5. Push report to Hub -----------------------------------------------
login(mode="hub")
project = Project(name="ibm-hackathon", mode="hub", workspace=cfg["workspace"])
project.put("01_dummy", report)
# Console prints: Consult your report at https://skore.probabl.ai/…
