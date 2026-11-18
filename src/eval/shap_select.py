"""SHAP record selection (D68): fold 9, labels only. Writes the ID list; nothing else."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

from src.data.ptbxl_meta import SUPERCLASSES
from src.data.waveform_data import load_split

OUT = "results/shap/shap_records_fold9.csv"
N_PER_CLASS = 20


def main():
    if os.path.exists(OUT):
        sys.exit(f"REFUSED: {OUT} exists")
    va = load_split("val")                      # fold 9 only
    y = va.y.numpy()
    rng = np.random.default_rng(0)
    rows = []
    for k, cls in enumerate(SUPERCLASSES):      # same column order as evaluate.py uses
        pos = np.where(y[:, k] == 1)[0]
        assert len(pos) >= N_PER_CLASS, (cls, len(pos))
        pick = np.sort(rng.choice(pos, N_PER_CLASS, replace=False))
        rows += [(cls, int(va.ecg_id[i])) for i in pick]
    df = pd.DataFrame(rows, columns=["class", "ecg_id"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.groupby("class").size().to_dict(), "| unique records:", df.ecg_id.nunique())
    print(hashlib.sha256(open(OUT, "rb").read()).hexdigest(), OUT)


if __name__ == "__main__":
    main()