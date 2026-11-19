"""SHAP lead-importance table and chart (D68). Re-runnable; reads the saved SHAP file only.
python -m src.eval.shap_table             -> self-test only (no data)
python -m src.eval.shap_table --build     -> self-test, then CSV + PNG"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.ptbxl_meta import SUPERCLASSES

SHAP_FILE = "results/shap/attn_shap_fold9.npz"
RECORDS = "results/shap/shap_records_fold9.csv"
OUT_CSV = Path("results/shap/attn_lead_importance_fold9.csv")
OUT_PNG = Path("results/shap/attn_lead_importance_fold9.png")
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def lead_importance(phi, ids, rec):
    """phi (n, 12, 1000, 5); ids (n,); rec DataFrame(class, ecg_id). Class k uses output k, its own records."""
    row = {int(e): i for i, e in enumerate(ids)}
    out = []
    for k, cls in enumerate(SUPERCLASSES):
        sel = [row[int(e)] for e in rec.loc[rec["class"] == cls, "ecg_id"]]
        per_rec = np.abs(phi[sel, :, :, k]).mean(axis=2)             # (records, 12): mean |phi| over time
        mean = per_rec.mean(axis=0)
        se = per_rec.std(axis=0, ddof=1) / np.sqrt(len(sel))
        share = mean / mean.sum()
        for j in range(12):
            out.append((cls, LEADS[j], len(sel), mean[j], se[j], share[j]))
    return pd.DataFrame(out, columns=["class", "lead", "n_records", "mean_abs_shap", "se", "share"])


def selftest():
    """Known answers on synthetic phi: sign ignored, only the class's own records and output used."""
    phi = np.zeros((10, 12, 1000, 5), np.float32)
    ids = np.arange(100, 110)
    rec = pd.DataFrame({"class": np.repeat(SUPERCLASSES, 2), "ecg_id": ids})
    for k in range(5):
        v = k + 1
        phi[2 * k, k + 1, :, k] = -v                  # record A: |phi| = v at lead k+1, output k
        phi[2 * k + 1, k + 1, :, k] = 3 * v           # record B: |phi| = 3v
        phi[2 * k, 0, :, (k + 1) % 5] = 99            # must not leak into another class
    df = lead_importance(phi, ids, rec)
    for k, cls in enumerate(SUPERCLASSES):
        d = df[df["class"] == cls].set_index("lead")
        v = k + 1
        np.testing.assert_allclose(d.loc[LEADS[k + 1], ["mean_abs_shap", "se", "share"]], [2 * v, v, 1.0])
        np.testing.assert_allclose(d.loc[LEADS[0], "mean_abs_shap"], 0.0)
    print("selftest OK (synthetic phi)")


def plot(df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=True)
    for ax, cls in zip(axes, SUPERCLASSES):
        d = df[df["class"] == cls]
        ax.bar(range(12), d["mean_abs_shap"], yerr=d["se"], capsize=2)
        ax.set_xticks(range(12))
        ax.set_xticklabels(d["lead"], rotation=90, fontsize=7)
        ax.set_title(f"{cls} (n={d['n_records'].iloc[0]})")
    axes[0].set_ylabel("mean |SHAP| (logit units)")
    fig.suptitle("Attention model, fold 9: SHAP importance per lead (error bars: SE over records)")
    fig.tight_layout()
    return fig


def build():
    selftest()
    z = np.load(SHAP_FILE)
    rec = pd.read_csv(RECORDS)
    assert z["phi"].shape[1:] == (12, 1000, 5), z["phi"].shape
    assert int(z["nsamples"]) == 1000 and int(z["rseed"]) == 0, "not the D68 settings"
    assert set(rec["ecg_id"].tolist()) == set(z["ecg_id"].tolist()), "records differ from the CSV"
    df = lead_importance(z["phi"], z["ecg_id"], rec)
    df.to_csv(OUT_CSV, index=False)
    plot(df).savefig(OUT_PNG, dpi=150)
    print("saved", OUT_CSV, OUT_PNG, df.shape)


if __name__ == "__main__":
    build() if "--build" in sys.argv else selftest()