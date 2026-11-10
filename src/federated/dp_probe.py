"""DP-SGD probe v2: optimizer, learning rate and batch size at the sigma for a target epsilon.
Site0, 3 epochs, training loss only; fold 9 is not touched."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

from src.federated.data import load_client
from src.models.cnn_attn import ECGCNNAttn
from src.models.train import pos_weight, run_epoch

ds = load_client("site0", "train")
DELTA, ROUNDS, EPS = 1e-5, 30, 5          # sigma is calibrated as if the client trains 30 epochs

def probe(opt_name, lr, batch, epochs=3):
    torch.manual_seed(0)
    sigma = get_noise_multiplier(target_epsilon=EPS, target_delta=DELTA,
                                 sample_rate=batch / len(ds), epochs=ROUNDS)
    model = ECGCNNAttn()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight(ds.y))
    opt = (torch.optim.Adam if opt_name == "adam" else torch.optim.SGD)(model.parameters(), lr=lr)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model, opt, loader = PrivacyEngine().make_private(module=model, optimizer=opt, data_loader=loader,
                                                     noise_multiplier=sigma, max_grad_norm=1.0)
    means = []
    for _ in range(epochs):
        losses = []
        run_epoch(model, loader, loss_fn, opt, "cpu", losses)
        means.append(np.mean(losses))
    print(f"{opt_name} lr={lr:<6} B={batch:<4} sigma={sigma:.2f}  epoch mean loss: "
          + "  ".join(f"{m:.3f}" for m in means), flush=True)

for opt_name, lr in [("adam", 1e-3), ("adam", 1e-4), ("sgd", 1.0)]:
    for batch in (64, 256):
        probe(opt_name, lr, batch)