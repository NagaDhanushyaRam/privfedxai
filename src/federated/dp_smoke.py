"""DP-SGD smoke test: one epoch of ECGCNNAttn on site0 through Opacus (not a protocol run)."""
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from src.federated.data import load_client
from src.models.cnn_attn import ECGCNNAttn
from src.models.train import pos_weight, run_epoch

ds = load_client("site0", "train")
model = ECGCNNAttn()
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(ds.y))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

engine = PrivacyEngine()
model, opt, loader = engine.make_private(module=model, optimizer=opt, data_loader=loader,
                                         noise_multiplier=1.0, max_grad_norm=1.0)
losses, t = [], time.time()
run_epoch(model, loader, loss_fn, opt, "cpu", losses)
print(f"steps={len(losses)} mean_loss={sum(losses)/len(losses):.4f} "
      f"time={time.time()-t:.1f}s epsilon={engine.get_epsilon(1e-5):.2f}")