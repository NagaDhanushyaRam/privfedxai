"""One-off: bundle the fold-9 SHAP records for the app (labels only, D68)."""
import os, sys
import numpy as np
import pandas as pd
from src.data.waveform_data import load_split

OUT = "app/demo_records.npz"

def main():
    if os.path.exists(OUT):
        sys.exit(f"REFUSED: {OUT} exists")
    ids = sorted(pd.read_csv("results/shap/shap_records_fold9.csv")["ecg_id"].unique())
    va = load_split("val")                                   # fold 9 only
    row = {int(e): i for i, e in enumerate(va.ecg_id)}
    idx = [row[int(e)] for e in ids]                         # KeyError = id not in fold 9
    f = np.load("data/processed/baseline_features.npz", allow_pickle=True)
    names = [str(n) for n in f["label_names"]]
    np.savez_compressed(OUT, ecg_id=np.array(ids), x=va.x[idx].numpy(),
                        y=va.y[idx].numpy(), label_names=np.array(names))
    print(len(ids), "records", va.x[idx].shape, names)

if __name__ == "__main__":
    main()