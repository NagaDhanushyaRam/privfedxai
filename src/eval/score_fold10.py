"""One-shot fold-10 scoring (D67). Probabilities only; prints no metric.
Run from the repo root:   python -m src.eval.score_fold10
Rehearsal on fold 9:      python -m src.eval.score_fold10 --dry-run
"""
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.waveform_data import FEATURES, load_split
from src.eval.frozen_models import load_frozen, registry, sha256
from src.models.train import predict

DRY = "--dry-run" in sys.argv
SPLIT, N_EXPECT, OUT = (("val", 2146, "results/_dryrun_fold9") if DRY
                        else ("test", 2158, "results/fold10"))
MANIFEST = f"{OUT}/FOLD10_PROBS.sha256.txt"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    reg = registry()
    names = list(reg)
    assert len(names) == 12, names
    finals = [f"{OUT}/{n}_test_probs.npz" for n in names] + [MANIFEST]

    present = [p for p in finals if os.path.exists(p)]
    if present:
        sys.exit(f"REFUSED: outputs already exist, nothing was scored: {present}")

    models = {n: load_frozen(reg, n, dev) for n in names}   # weights SHA-256 checked here
    print(f"{len(models)} models loaded, hashes OK, device {dev}, split {SPLIT}")

    ds = load_split(SPLIT)
    f = np.load(FEATURES)
    expected = f["ecg_id"][f["split"].astype(str) == SPLIT]
    assert len(ds) == N_EXPECT, len(ds)
    assert np.array_equal(ds.ecg_id, expected), "ecg_id order"
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    loss_fn = nn.BCEWithLogitsLoss()                         # loss from predict() is unused

    probs = {}
    for n, model in models.items():
        _, P, y = predict(model, loader, loss_fn, dev)
        assert P.shape == (N_EXPECT, 5), (n, P.shape)
        assert np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1, n
        assert np.array_equal(y, ds.y.numpy()), f"{n}: label order"
        probs[n] = P
    print(f"scored {len(probs)} models in memory; checks passed; no metric computed")

    # Stage everything under .tmp names, then rename. A crash before the renames
    # leaves no final file, so D67 allows a rerun.
    os.makedirs(OUT, exist_ok=True)
    tmps = {}
    for n, P in probs.items():
        tmps[n] = f"{OUT}/{n}_test_probs.npz.tmp"
        with open(tmps[n], "wb") as fh:
            np.savez(fh, ecg_id_test=ds.ecg_id, P_test=P)
    lines = [f"{sha256(tmps[n])}  {n}_test_probs.npz" for n in names]
    with open(MANIFEST + ".tmp", "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    for n in names:
        os.replace(tmps[n], f"{OUT}/{n}_test_probs.npz")
    os.replace(MANIFEST + ".tmp", MANIFEST)

    print(f"wrote {len(names)} files + manifest to {OUT}, n = {N_EXPECT} each")
    print("\n".join(lines))


if __name__ == "__main__":
    main()