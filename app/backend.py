"""App backend: frozen hash-checked models on CPU, fold-9 demo records, fold-10 aggregate table."""
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from src.eval.frozen_models import registry, load_frozen

DEMO = "app/demo_records.npz"
TABLE = "results/fold10/master_table_test.csv"
DP_RUNS = {e: tuple(f"dp_eps{e}_r{r}" for r in (1, 2, 3)) for e in (10, 5, 1)}  # r98/r99 never used


@lru_cache(maxsize=None)
def _registry():
    return registry()                                   # checks the DP manifest hash once


@lru_cache(maxsize=None)
def get_model(name):
    return load_frozen(_registry(), name, "cpu")        # weights hash checked once per process


@lru_cache(maxsize=None)
def load_demo():
    d = np.load(DEMO)
    classes = [str(n) for n in d["label_names"]]
    assert classes == ["NORM", "MI", "STTC", "CD", "HYP"], classes
    return {"ecg_id": d["ecg_id"], "x": torch.from_numpy(d["x"]), "y": d["y"], "classes": classes}


@torch.no_grad()
def score(x, name):
    """x: (12, 1000) float32 tensor -> model scores (5,): sigmoid of the logits, not calibrated."""
    return torch.sigmoid(get_model(name)(x[None]))[0].numpy()


def scores(x, names=("cnn", "attn", "fl")):
    return {n: score(x, n) for n in names}


def dp_scores(x, eps):
    """eps in {10, 5, 1} -> {run: scores} for all three runs (none picked as best); inf -> {'fl': scores}."""
    runs = ("fl",) if eps == float("inf") else DP_RUNS[eps]
    return {n: score(x, n) for n in runs}


def attention(x, name="attn"):
    """(125, 5): cell j is centred at 0.08*j s, one time axis for all leads."""
    assert name != "cnn", "the plain CNN has no attention"
    return get_model(name).attn_maps(x[None])[0].numpy()


def fold10_metrics(model):
    """Pooled macro rows of the frozen fold-10 table. model: fl, dp_eps10, dp_eps5, dp_eps1, ...
    DP rows carry vmin/vmax (min-max of the 3 runs); lo/hi are the bootstrap interval."""
    t = pd.read_csv(TABLE)
    t = t[(t.model == model) & (t.client == "pooled") & (t.label == "macro")]
    return t.set_index("metric")[["value", "lo", "hi", "vmin", "vmax"]]