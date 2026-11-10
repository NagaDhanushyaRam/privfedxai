"""Local training for one Flower client (D53 settings, D59 per-client pos_weight)."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.train import pos_weight, run_epoch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier


def train_local(model, ds, epochs, lr, batch, dev):
    pw = pos_weight(ds.y).to(dev)                       # this client's own counts (D59)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # fresh optimizer state every call
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model.to(dev)
    losses = []
    for _ in range(epochs):
        run_epoch(model, loader, loss_fn, opt, dev, losses)
    return float(np.mean(losses))

def train_local_dp(model, ds, epochs, lr, batch, dev, target_eps, delta, dp_rounds, clip=1.0):
    """Record-level DP-SGD local training. sigma is calibrated so that dp_rounds calls of `epochs`
    epochs each stay within (target_eps, delta) for this client. pos_weight is outside the accounting."""
    pw = pos_weight(ds.y).to(dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # fresh optimizer state every call
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    sigma = get_noise_multiplier(target_epsilon=target_eps, target_delta=delta,
                                 sample_rate=1 / len(loader), epochs=dp_rounds * epochs)
    model.to(dev)
    private, opt, loader = PrivacyEngine().make_private(module=model, optimizer=opt, data_loader=loader,
                                                       noise_multiplier=sigma, max_grad_norm=clip)
    losses = []
    for _ in range(epochs):
        run_epoch(private, loader, loss_fn, opt, dev, losses)
    return float(np.mean(losses)), sigma      