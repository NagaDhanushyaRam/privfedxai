"""Record-level bootstrap CIs for per-client and pooled metrics."""
import numpy as np
import pandas as pd

from src.eval.evaluate import CLIENTS, LABELS, score_slice
from src.eval.loading import MODELS, load_split


def bootstrap(model, split, n_boot=1000, seed=0):
    """Long table of replicates: client, rep, metric, label, value.

    Resampling indices depend only on (seed, slice size), so every model
    evaluated on the same rows gets the same resamples (paired across models).
    """
    Y, S, client = load_split(model, split)
    thr = MODELS[model][2]
    rows = []
    for name in ["pooled"] + CLIENTS:
        idx = np.arange(len(Y)) if name == "pooled" else np.where(client == name)[0]
        rng = np.random.default_rng(seed)
        for rep in range(n_boot):
            b = idx[rng.integers(0, len(idx), len(idx))]
            for metric, vals in score_slice(Y[b], S[b], thr).items():
                for label, v in zip(LABELS, list(vals) + [np.nanmean(vals)]):
                    rows.append((name, rep, metric, label, v))
    return pd.DataFrame(rows, columns=["client", "rep", "metric", "label", "value"])


def summarize(boot):
    """95% percentile CI and number of valid (non-NaN) replicates per cell."""
    g = boot.groupby(["client", "metric", "label"])["value"]
    return pd.DataFrame({"lo": g.quantile(0.025), "hi": g.quantile(0.975),
                         "n_valid": g.count()}).reset_index()