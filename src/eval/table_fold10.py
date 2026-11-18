import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.bootstrap import bootstrap, summarize
from src.eval.evaluate import CLIENTS, LABELS, evaluate, score_slice
from src.eval.loading import FEATURES, MODELS, load_split
from src.eval.metrics import predicted_positive_rate

NEW = Path("results/fold10")
BASE_CSV = Path("results/baselines/baseline_table_test.csv")
KEYS = ["client", "metric", "label"]


def load_new(name):
    """Fold-10 probabilities of one frozen model, aligned with Y and client."""
    f = np.load(FEATURES, allow_pickle=True)
    r = np.load(NEW / f"{name}_test_probs.npz")
    m = f["split"].astype(str) == "test"
    assert (r["ecg_id_test"] == f["ecg_id"][m]).all(), "row misalignment"
    return f["Y"][m], r["P_test"], f["client"][m].astype(str)


def boot_runs(Y, runs, client, thr=0.5, n_boot=1000, seed=0):
    """Same resampling as bootstrap.bootstrap: fresh rng per slice, one integers() call
    per replicate. runs = list of score arrays; the metric is computed per run on the
    same resample, then averaged over the runs."""
    rows = []
    for name in ["pooled"] + CLIENTS:
        idx = np.arange(len(Y)) if name == "pooled" else np.where(client == name)[0]
        rng = np.random.default_rng(seed)
        for rep in range(n_boot):
            b = idx[rng.integers(0, len(idx), len(idx))]
            per_run = [score_slice(Y[b], S[b], thr) for S in runs]
            for metric in per_run[0]:
                vals = np.mean([pr[metric] for pr in per_run], axis=0)
                for label, v in zip(LABELS, list(vals) + [np.nanmean(vals)]):
                    rows.append((name, rep, metric, label, v))
    return pd.DataFrame(rows, columns=["client", "rep", "metric", "label", "value"])


def point_rows(model, Y, S, client, thr):
    """evaluate.evaluate() logic on arrays, plus predicted-positive rate (point only)."""
    rows = []
    for name in ["pooled"] + CLIENTS:
        m = np.ones(len(Y), bool) if name == "pooled" else client == name
        res = score_slice(Y[m], S[m], thr)
        res["ppr"] = predicted_positive_rate(S[m], thr)
        for metric, vals in res.items():
            for label, v in zip(LABELS, list(vals) + [np.nanmean(vals)]):
                rows.append((model, name, int(m.sum()), metric, label, v))
    return pd.DataFrame(rows, columns=["model", "client", "n", "metric", "label", "value"])


def aggregate_runs(per_run, model):
    """Mean over runs of each point estimate, with the min and max of the runs."""
    g = per_run.groupby(["client", "n", "metric", "label"], sort=False)["value"]
    out = g.agg(value="mean", vmin="min", vmax="max").reset_index()
    out.insert(0, "model", model)
    return out


def dp_group(eps):
    names = [f"dp_eps{eps}_r{r}" for r in (1, 2, 3)]
    loaded = [load_new(n) for n in names]
    Y, _, client = loaded[0]
    assert all(np.array_equal(l[0], Y) for l in loaded)
    runs = [l[1] for l in loaded]
    per_run = pd.concat([point_rows(n, Y, S, client, 0.5) for n, S in zip(names, runs)],
                        ignore_index=True)
    ci = summarize(boot_runs(Y, runs, client, 0.5))
    return aggregate_runs(per_run, f"dp_eps{eps}").merge(ci, on=KEYS, how="left"), per_run

def selftest(n_boot=100):
    """Fold 9 only: must reproduce bootstrap('lr', 'val') exactly (one run, then three identical runs)."""
    Y, S, client = load_split("lr", "val")
    thr = MODELS["lr"][2]
    ref = bootstrap("lr", "val", n_boot=n_boot, seed=0)
    pd.testing.assert_frame_equal(ref, boot_runs(Y, [S], client, thr, n_boot))
    pd.testing.assert_frame_equal(ref, boot_runs(Y, [S, S, S], client, thr, n_boot))
    pt = point_rows("lr", Y, S, client, thr)                      # point estimates vs evaluate()
    pd.testing.assert_frame_equal(evaluate("lr", "val"),
                                  pt[pt.metric != "ppr"].reset_index(drop=True))
    agg = aggregate_runs(pd.concat([pt, pt, pt], ignore_index=True), "lr")   # 3 identical runs
    np.testing.assert_allclose(agg["value"], pt["value"])
    np.testing.assert_allclose(agg["vmin"], agg["vmax"])
    print(f"selftest OK (fold 9, lr, n_boot={n_boot}): resamples pair with bootstrap.py")


def build():
    selftest()
    parts, runs = [], []
    for name in ("cnn", "attn", "fl"):
        Y, S, client = load_new(name)
        point = point_rows(name, Y, S, client, 0.5)
        parts.append(point.merge(summarize(boot_runs(Y, [S], client, 0.5)), on=KEYS, how="left"))
        print(name, "done")
    for eps in (10, 5, 1):
        t, r = dp_group(eps)
        parts.append(t)
        runs.append(r)
        print(f"dp_eps{eps} done")
    new = pd.concat(parts, ignore_index=True)
    base = pd.read_csv(BASE_CSV)
    base_models = list(base["model"].unique())
    assert len(base_models) == 6, base_models
    ppr = []
    for m in base_models:
        Y, S, client = load_split(m, "test")
        p = point_rows(m, Y, S, client, MODELS[m][2])
        ppr.append(p[p.metric == "ppr"])
    master = pd.concat([base] + ppr + [new], ignore_index=True)
    new.to_csv(NEW / "fold10_new_models_table.csv", index=False)
    pd.concat(runs, ignore_index=True).to_csv(NEW / "fold10_dp_runs_point.csv", index=False)
    master.to_csv(NEW / "master_table_test.csv", index=False)
    print("saved", master.shape)


if __name__ == "__main__":
    build() if "--build" in sys.argv else selftest()