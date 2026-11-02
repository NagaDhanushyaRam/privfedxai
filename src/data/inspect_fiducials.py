import collections
from pathlib import Path

import numpy as np
import wfdb

ROOT = Path("data/raw/ptbxl_plus/fiducial_points/ecgdeli")
FEATURES = "data/processed/baseline_features.npz"


def global_base(ecg_id):                       # folder = thousands group, 5 digits
    return ROOT / f"{(ecg_id // 1000) * 1000:05d}" / f"{ecg_id:05d}_points_global"


def show(ecg_id, n=24):
    a = wfdb.rdann(str(global_base(ecg_id)), "atr")
    print(f"\necg_id {ecg_id}: {len(a.sample)} annotations, fs={getattr(a, 'fs', None)}")
    print("symbols:", dict(collections.Counter(a.symbol)))
    print("max sample:", a.sample.max(), " subtype uniq:", np.unique(a.subtype)[:10],
          " aux uniq:", sorted(set(a.aux_note))[:10])
    print("first:", list(zip(a.sample[:n].tolist(), a.symbol[:n])))


def coverage():
    f = np.load(FEATURES)
    ids, split = f["ecg_id"], f["split"].astype(str)
    have = np.array([global_base(int(i)).with_suffix(".atr").exists() for i in ids])
    print("\nlabeled records:", len(ids), " with global .atr:", int(have.sum()))
    for s in ("train", "val", "test"):
        m = split == s
        print(f"  {s}: {int(have[m].sum())}/{int(m.sum())}")


if __name__ == "__main__":
    for i in (1, 2, 3):
        show(i)
    coverage()