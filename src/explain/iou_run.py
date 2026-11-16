"""D66 run: IoU and peak-hit of the frozen attention model on fold 9. Run ONCE (D66 rules)."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.ptbxl_meta import SUPERCLASSES
from src.data.waveform_data import load_split
from src.explain.iou_regions import record_regions
from src.explain.iou_score import score_record
from src.models.cnn_attn import ECGCNNAttn

WEIGHTS = Path("results/attn/attn_last_FROZEN.pt")
SHA = "d52ff38c9df568e001b052f128f974e18cd947e8cb164929072bc0a5ebcafcdd"
CLASSES = list(SUPERCLASSES)       # Y column order, build_features.py:26
FEATURES = "data/processed/baseline_features.npz"
SCORED = ("MI", "STTC", "CD", "HYP")               # NORM skipped (D66)
OUT = Path("results/attn")


def check_labels(ds):
    """Label order = SUPERCLASSES; the waveform set's labels equal the npz labels (D66 gate)."""
    f = np.load(FEATURES)
    assert [str(n) for n in f["label_names"]] == CLASSES
    pos = {int(e): i for i, e in enumerate(f["ecg_id"])}
    rows = [pos[int(e)] for e in ds.ecg_id]
    assert (f["Y"][rows] == ds.y.numpy()).all()


def load_maps(ds):
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == SHA, "not the frozen weights"
    m = ECGCNNAttn()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(ds.x), 256):
            out.append(m.attn_maps(ds.x[s:s + 256]).numpy())
    a = np.concatenate(out)                                    # (n, 125, 5)
    assert a.shape[1:] == (125, 5) and np.allclose(a.sum(1), 1, atol=1e-4)
    return a


def collect(ds, A):
    Y, ids = ds.y.numpy(), ds.ecg_id
    rows = []
    for c in SCORED:
        k = CLASSES.index(c)
        for i in np.flatnonzero(Y[:, k] == 1):                 # true classes only (D66)
            eid = int(ids[i])
            r = score_record(A[i, :, k], record_regions(eid, c))
            rows.append({"ecg_id": eid, "cls": c, **(r or {"n_beats": 0})})
    return pd.DataFrame(rows)


def boot_ci(v, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    return np.percentile([rng.choice(v, len(v)).mean() for _ in range(B)], [2.5, 97.5])


def summarize(df):
    rows = []
    for c in SCORED:
        d = df[df.cls == c]
        ok = d[d.n_beats > 0]
        row = {"cls": c, "n_true": len(d), "n_scored": len(ok), "n_empty_regions": int(ok.n_empty.sum())}
        for v in ("peak_beat", "all_beat"):
            iou = ok[f"{v}_iou"].to_numpy(float)
            lo, hi = boot_ci(iou)
            row[f"{v}_iou_mean"], row[f"{v}_iou_median"] = iou.mean(), np.median(iou)
            row[f"{v}_iou_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
            for t in (0, 1):
                hit = ok[f"{v}_hit{t}"].to_numpy(float)
                lo, hi = boot_ci(hit)
                row[f"{v}_hit{t}"] = f"{hit.mean():.3f} [{lo:.3f}, {hi:.3f}]"
                row[f"{v}_chance{t}"] = round(float(ok[f"{v}_chance{t}"].mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    if (OUT / "iou_fold9_long.csv").exists():
        raise SystemExit("D66: scored once. A rerun after seeing numbers counts as tuning; "
                         "delete the outputs only after a crash.")
    ds = load_split("val")
    check_labels(ds)
    df = collect(ds, load_maps(ds))
    df.to_csv(OUT / "iou_fold9_long.csv", index=False)
    s = summarize(df)
    s.to_csv(OUT / "iou_fold9_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", None)
    print(s.set_index("cls").T)


if __name__ == "__main__":
    main()