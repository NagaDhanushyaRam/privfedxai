"""Exploratory amplitude check (D69): does input amplitude explain the SHAP lead profile?
Uses the D68 records, the D68 background rows and the D68 CSV. No model, no SHAP."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.ptbxl_meta import SUPERCLASSES
from src.data.waveform_data import load_split
from src.eval.shap_table import LEADS

RECORDS = "results/shap/shap_records_fold9.csv"
SHAP_CSV = "results/shap/attn_lead_importance_fold9.csv"
OUT = "results/shap/attn_amplitude_check_fold9.csv"


def main():
    rec, shap_df = pd.read_csv(RECORDS), pd.read_csv(SHAP_CSV)
    tr = load_split("train")
    bg = tr.x[np.random.default_rng(0).permutation(len(tr))[:50]].numpy()   # (50, 12, 1000), as in shap_run
    del tr
    va = load_split("val")                                                  # fold 9 only
    row = {int(e): i for i, e in enumerate(va.ecg_id)}
    rows = []
    for cls in SUPERCLASSES:
        sel = [row[int(e)] for e in rec.loc[rec["class"] == cls, "ecg_id"]]
        x = va.x[sel].numpy()                                               # (20, 12, 1000)
        amp = np.abs(x[:, None] - bg[None]).mean(axis=(0, 1, 3))            # (12,): mean |x - background|
        d = shap_df[shap_df["class"] == cls].reset_index(drop=True)
        assert d["lead"].tolist() == LEADS and len(sel) == 20
        rho, _ = spearmanr(d["mean_abs_shap"], amp)
        ratio = d["mean_abs_shap"].to_numpy() / amp
        top = [LEADS[j] for j in np.argsort(-ratio)[:3]]
        print(f"{cls:5s} rho {rho:5.2f} | top 3 by SHAP/amplitude: {top}")
        rows += [(cls, LEADS[j], amp[j], d["mean_abs_shap"][j], ratio[j]) for j in range(12)]
    pd.DataFrame(rows, columns=["class", "lead", "amplitude", "mean_abs_shap", "shap_per_amp"]).to_csv(OUT, index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()