"""Per-client and per-class fold-9 AUC of a saved FL model vs the centralized attention model."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loading import FEATURES, load_split
from src.eval.metrics import per_class_auc

DEFAULT = Path("results/fl/FROZEN_fl_val_probs.npz")
NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
GROUPS = ["pooled", "site0", "site1", "site2", "site3", "rest"]


def load_fl_val(path):
    f = np.load(FEATURES, allow_pickle=True)
    ids = f["ecg_id"][f["split"].astype(str) == "val"]
    r = np.load(path)
    assert (r["ecg_id_val"] == ids).all(), "row misalignment"
    return r["P_val"]


def main(path):
    Y, P_c, client = load_split("attn", "val")
    P_f = load_fl_val(path)
    assert P_f.shape == P_c.shape
    rows = []
    for g in GROUPS:
        m = np.ones(len(Y), bool) if g == "pooled" else client == g
        a_c, a_f = per_class_auc(Y[m], P_c[m]), per_class_auc(Y[m], P_f[m])
        rows.append({"group": g, "n": int(m.sum()),
                     "central": np.nanmean(a_c), "fl": np.nanmean(a_f),
                     **{f"d_{n}": x - c for n, c, x in zip(NAMES, a_c, a_f)}})
    out = pd.DataFrame(rows)
    out["d_macro"] = out["fl"] - out["central"]
    print(out.round(3).to_string(index=False))
    p = Path(path)
    out.to_csv(p.with_name(p.stem + "_vs_central.csv"), index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)