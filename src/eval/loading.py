"""Load a saved baseline output aligned with labels and client for one split."""
from pathlib import Path

import numpy as np

FEATURES = Path("data/processed/baseline_features.npz")
RESULTS = Path("results/baselines")

# model -> (file, score-key prefix, threshold per D27 / D43)
MODELS = {
    "lr": ("lr_probs.npz", "P", 0.5),
    "svm": ("svm_scores.npz", "S", 0.0),
    "rf": ("rf_probs.npz", "P", 0.5),
}


def load_split(model, split):
    """Return Y (n,5), S (n,5), client (n,) for `model` on `split` ('val' or 'test')."""
    fname, prefix, _ = MODELS[model]
    f = np.load(FEATURES, allow_pickle=True)
    r = np.load(RESULTS / fname)
    m = f["split"].astype(str) == split
    assert (r[f"ecg_id_{split}"] == f["ecg_id"][m]).all(), "row misalignment"
    return f["Y"][m], r[f"{prefix}_{split}"], f["client"][m].astype(str)