import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.waveform_data import load_split
from src.models.cnn_attn import ECGCNNAttn
from src.models.train import EPOCHS, BATCH, LR, SEED, pos_weight, run_epoch, predict

OUT = "results/attn"


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = load_split("train"), load_split("val")
    pw = pos_weight(tr.y)
    print("pos_weight", pw.numpy().round(2))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw.to(dev))
    model = ECGCNNAttn().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tl = DataLoader(tr, batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)

    losses, prev_med = [], None
    for ep in range(1, EPOCHS + 1):
        t, start = time.time(), len(losses)
        run_epoch(model, tl, loss_fn, opt, dev, losses)
        cur = np.array(losses[start:])
        if prev_med is not None and cur.max() > 10 * prev_med:
            print(f"  SPIKE (trigger b): max batch loss {cur.max():.2f} > 10 x {prev_med:.3f}")
        prev_med = float(np.median(cur))
        vloss, p, y = predict(model, vl, loss_fn, dev)
        print(f"ep {ep:2d} train {cur.mean():.4f} val {vloss:.4f} "
              f"val macro AUC {roc_auc_score(y, p, average='macro'):.4f} ({time.time() - t:.0f}s)", flush=True)

    torch.save(model.state_dict(), f"{OUT}/attn_last.pt")
    np.save(f"{OUT}/batch_loss.npy", np.array(losses))
    np.savez(f"{OUT}/attn_val_probs.npz", ecg_id_val=va.ecg_id, P_val=p)


if __name__ == "__main__":
    main()