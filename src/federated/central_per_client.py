import numpy as np

from src.eval.loading import load_split
from src.eval.metrics import per_class_auc

Y, P, client = load_split("attn", "val")
names = ["NORM", "MI", "STTC", "CD", "HYP"]
for c in ["pooled", "site0", "site1", "site2", "site3", "rest"]:
    m = np.ones(len(Y), bool) if c == "pooled" else client == c
    a = per_class_auc(Y[m], P[m])
    print(f"{c:6s} n={m.sum():4d} macro {np.nanmean(a):.3f} ",
          " ".join(f"{n} {v:.3f}" for n, v in zip(names, a)))