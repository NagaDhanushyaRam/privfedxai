"""One-shot SHAP pass (D68): frozen attention model, fold-9 records from shap_select.
Run from the repo root:  python -m src.eval.shap_run
Rehearsal:               python -m src.eval.shap_run --dry-run   (2 records, nsamples 50)"""
import os
import sys
import time

import numpy as np
import pandas as pd
import shap
import torch

from src.data.waveform_data import load_split
from src.eval.frozen_models import load_frozen, registry, sha256

DRY = "--dry-run" in sys.argv
RECORDS = "results/shap/shap_records_fold9.csv"
OUT = "results/shap/_dryrun_attn_shap.npz" if DRY else "results/shap/attn_shap_fold9.npz"
NSAMPLES, BATCH = (50, 2) if DRY else (1000, 10)


def main():
    if os.path.exists(OUT):
        sys.exit(f"REFUSED: {OUT} exists")
    torch.manual_seed(0)
    model = load_frozen(registry(), "attn", "cpu")          # weights hash checked, eval mode

    rec = pd.read_csv(RECORDS)
    ids = sorted(rec["ecg_id"].unique())                    # ascending ecg_id, as in D68
    if DRY:
        ids = ids[:2]

    tr = load_split("train")
    bg = tr.x[np.random.default_rng(0).permutation(len(tr))[:50]]   # background, as in D68
    del tr
    va = load_split("val")                                  # fold 9 only
    row = {int(e): i for i, e in enumerate(va.ecg_id)}
    idx = [row[int(e)] for e in ids]                        # KeyError here = ID not in fold 9
    x, y = va.x[idx], va.y[idx].numpy()
    del va

    explainer = shap.GradientExplainer(model, bg)
    parts, t0 = [], time.time()
    for b in range(0, len(ids), BATCH):
        phi = np.asarray(explainer.shap_values(x[b:b + BATCH], nsamples=NSAMPLES, rseed=0))
        assert phi.shape == (len(x[b:b + BATCH]), 12, 1000, 5), phi.shape
        assert np.isfinite(phi).all(), f"non-finite values in batch starting at {b}"
        parts.append(phi.astype(np.float32))
        print(f"batch {b // BATCH + 1}/{-(-len(ids) // BATCH)}  {time.time() - t0:.0f} s", flush=True)
    phi = np.concatenate(parts)

    with torch.no_grad():                                   # additivity diagnostic, logits
        gap = (model(x) - model(bg).mean(0)).numpy()
    add_err = np.abs(phi.sum(axis=(1, 2)) - gap)

    tmp = OUT + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, ecg_id=np.array(ids), phi=phi, gap=gap, y=y,
                 nsamples=NSAMPLES, rseed=0, batch=BATCH)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(ids)} records, phi {phi.shape}, finite {np.isfinite(phi).all()}")
    print(f"additivity |sum(phi) - gap| (logits): median {np.median(add_err):.3f}, max {add_err.max():.3f}")
    print(sha256(OUT), OUT)


if __name__ == "__main__":
    main()