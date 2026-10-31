"""Append pooled fold-9 bootstrap replicates for the attention model (same seed and resamples, D47)."""
from pathlib import Path

import pandas as pd

from src.eval.bootstrap import bootstrap

OUT = Path("results/cnn/boot_val_pooled.csv")


def main():
    old = pd.read_csv(OUT)
    assert "attn" not in set(old["model"]), "attn already in the file"
    b = bootstrap("attn", "val", n_boot=1000, seed=0)
    new = b[b["client"] == "pooled"].assign(model="attn")
    pd.concat([old, new], ignore_index=True).to_csv(OUT, index=False)
    print("saved", OUT, len(old), "+", len(new), "rows")


if __name__ == "__main__":
    main()