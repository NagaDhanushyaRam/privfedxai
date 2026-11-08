"""Per-client rows of the waveform cache for the Flower clients (D12, D59)."""
import numpy as np
import torch
from pathlib import Path
from src.data.waveform_data import CACHE, CACHE_IDS, FEATURES, WaveformSet

ROOT = Path(__file__).resolve().parents[2]
CLIENTS = ["site0", "site1", "site2", "site3", "rest"]


def load_client(client, split):                 # client: a name, or None for all; split: 'train' | 'val'
    assert split in ("train", "val"), "fold 10 is not loaded in the FL phase"
    f = np.load(ROOT / FEATURES)
    ids = np.load(ROOT / CACHE_IDS)
    assert (ids == f["ecg_id"]).all(), "cache order != baseline_features order"
    m = f["split"].astype(str) == split
    if client is not None:
        m &= f["client"].astype(str) == client
    W = np.load(ROOT / CACHE, mmap_mode="r")
    assert W.shape == (len(ids), 12, 1000)
    x = torch.from_numpy(W[m])                  # copies only this client's rows into RAM
    y = torch.from_numpy(f["Y"][m].astype(np.float32))
    del W
    return WaveformSet(x, y, f["ecg_id"][m])