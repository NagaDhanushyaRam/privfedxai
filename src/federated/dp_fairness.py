"""Fold-9 per-class AUC / F1 / MCC / predicted-positive rate: non-DP FL vs DP at each epsilon (claim 3)."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.loading import load_split
from src.eval.metrics import per_class_auc, per_class_f1_mcc, predicted_positive_rate
from src.federated.fl_per_client import load_fl_val

FL = Path("results/fl")
NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
THR = 0.5
RUNS = {"inf": [FL / "FROZEN_fl_val_probs.npz"]}
for eps in (10, 5, 1):
    RUNS[str(eps)] = [FL / f"fl_val_probs_eps{eps}_r{r}.npz" for r in (1, 2, 3)]


def metrics(Y, P):
    f1, mcc = per_class_f1_mcc(Y, P, THR)
    return {"auc": per_class_auc(Y, P), "f1": f1, "mcc": mcc,
            "ppr": predicted_positive_rate(P, THR)}


def main():
    Y, _, _ = load_split("attn", "val")
    rows = []
    for eps, paths in RUNS.items():
        found = [p for p in paths if p.exists()]
        print(f"eps={eps}: {len(found)}/{len(paths)} runs found")
        for rep, p in enumerate(found):
            for metric, vals in metrics(Y, load_fl_val(p)).items():
                for name, v in zip(NAMES + ["macro"], list(vals) + [np.nanmean(vals)]):
                    rows.append((eps, rep, metric, name, v))
    df = pd.DataFrame(rows, columns=["eps", "rep", "metric", "label", "value"])
    df.to_csv(FL / "dp_fairness_long.csv", index=False)
    order = [e for e in RUNS if e in set(df["eps"])]
    mean = df.groupby(["metric", "label", "eps"])["value"].mean()
    for metric in ["auc", "f1", "mcc", "ppr"]:
        t = mean.loc[metric].unstack("eps")[order].loc[NAMES + ["macro"]]
        print(f"\n{metric} (mean over runs)\n{t.round(3).to_string()}")
    sel = df[(df["metric"] == "auc") & (df["label"] == "macro")]
    print("\nmacro AUC sd across runs:", sel.groupby("eps")["value"].std().round(4).to_dict())


if __name__ == "__main__":
    main()