"""Local training for one Flower client (D53 settings, D59 per-client pos_weight)."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.train import pos_weight, run_epoch


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