from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.waveform_data import load_split
from src.models.cnn_attn import ECGCNNAttn

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]      # verify against the Y column order
WEIGHTS = "results/attn/attn_last.pt"
OUT = Path("results/attn/heatmaps")


def load_model():
    m = ECGCNNAttn()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return m.eval()


def main(per_class=3):
    model, ds = load_model(), load_split("val")          # fold 9 only
    y_all = ds.y.numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(CLASSES):
        for i in np.flatnonzero(y_all[:, k] == 1)[:per_class]:
            x, ecg_id = ds.x[i], int(ds.ecg_id[i])
            a = model.attn_maps(x[None])[0].numpy()      # (125, 5)
            assert a.shape == (125, 5) and np.allclose(a.sum(0), 1, atol=1e-4)
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                           gridspec_kw={"height_ratios": [1, 1.4]})
            ax0.plot(np.arange(1000) / 100, x[1].numpy(), lw=0.8)
            ax0.set_ylabel("lead index 1 (display only)")
            ax1.imshow(a.T, aspect="auto", extent=(-0.04, 9.96, 4.5, -0.5),
                       vmin=0, vmax=a.max(), cmap="viridis")
            ax1.set_yticks(range(5)); ax1.set_yticklabels(CLASSES)
            ax1.set_xlim(0, 10)
            ax1.set_xlabel("time (s); one cell = one 80 ms position centred at 0.08*j s")
            true = [CLASSES[j] for j in np.flatnonzero(y_all[i])]
            ax0.set_title(f"ecg_id {ecg_id} (fold 9), true {true}, "
                          f"max alpha {a.max():.3f} (uniform = 0.008)")
            fig.tight_layout()
            fig.savefig(OUT / f"{name}_{ecg_id}.png", dpi=110)
            plt.close(fig)
            print(name, ecg_id, "max alpha", round(float(a.max()), 3))


if __name__ == "__main__":
    main()