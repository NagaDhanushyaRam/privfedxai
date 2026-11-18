"""Hash-checked loader for the 12 frozen models (plain CNN, attention, FL, nine DP runs).
Reused unchanged by score_fold10.py. Running this file as a module is the fold-9 gate."""
import hashlib
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.waveform_data import load_split
from src.models.cnn import ECGCNN
from src.models.cnn_attn import ECGCNNAttn
from src.models.train import predict

MANIFEST = "results/fl/DP_MANIFEST.sha256.txt"
MANIFEST_SHA = "f7cd55dce582dd9e04a7f88667da49b282cff86d92351ff7b508c091ee089979"
LOGGED_AUC = {"cnn": 0.9266, "attn": 0.9129, "fl": 0.9212}          # fold 9, from the log
LOGGED_DP = {10: (0.8613, 0.8605, 0.8620), 5: (0.8528, 0.8540, 0.8558),
             1: (0.8154, 0.8210, 0.8055)}                            # run labels r1, r2, r3 presumed


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hashes():
    assert sha256(MANIFEST) == MANIFEST_SHA, "DP manifest does not match the logged hash"
    out = {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for line in f.read().splitlines():
            if line.strip():
                h, name = line.split(None, 1)
                out[name.strip()] = h.lower()
    assert len(out) == 18, f"manifest has {len(out)} entries, expected 18"
    return out


def registry():
    """name -> (class, weights path, weights sha256, saved fold-9 probs path, probs sha256 or None)"""
    reg = {
        "cnn": (ECGCNN, "results/cnn/cnn_last_v0_FROZEN.pt",
                "e9e7fe064fbf6f2169b7a8c28113087ca2d16a92a2c65c9f8e30c0e4f2ca685c",
                "results/cnn/cnn_val_probs.npz", None),
        "attn": (ECGCNNAttn, "results/attn/attn_last_FROZEN.pt",
                 "d52ff38c9df568e001b052f128f974e18cd947e8cb164929072bc0a5ebcafcdd",
                 "results/attn/attn_val_probs.npz", None),
        "fl": (ECGCNNAttn, "results/fl/FROZEN_fl_last.pt",
               "c36076d3780b2bd32c3ff0f29b748e0d1eafb92bf0daa7a51f0ed8c34729f000",
               "results/fl/FROZEN_fl_val_probs.npz",
               "b7fec8a0e624a60d668536bb8872e6f63e6314e11792168dcc71ff777d3c7edc"),
    }
    m = manifest_hashes()
    for e in (10, 5, 1):
        for r in (1, 2, 3):
            w, p = f"fl_eps{e}_r{r}_last.pt", f"fl_val_probs_eps{e}_r{r}.npz"
            reg[f"dp_eps{e}_r{r}"] = (ECGCNNAttn, f"results/fl/{w}", m[w], f"results/fl/{p}", m[p])
    assert len(reg) == 12
    return reg


def load_frozen(reg, name, dev):
    cls, wpath, whash, _, _ = reg[name]
    got = sha256(wpath)
    if got != whash:
        sys.exit(f"HASH MISMATCH {name}: {wpath}\n  got      {got}\n  expected {whash}")
    model = cls()
    model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))  # strict
    return model.to(dev).eval()


def gate():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    va = load_split("val")                          # fold 9 only; "test" must never appear here
    loader = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)
    loss_fn = nn.BCEWithLogitsLoss()                # the loss returned by predict() is unused
    reg = registry()
    print(f"device {dev}, torch {torch.__version__}, fold 9 n = {len(va)}")
    print(f"{'model':<13}{'probs_hash':<11}{'ids':<5}{'max|dP|':<10}{'AUC_new':<9}{'AUC_saved':<10}{'AUC_log':<9}ok")
    all_ok = True
    for name, (_, _, _, ppath, phash) in reg.items():
        hash_ok = phash is None or sha256(ppath) == phash
        ref = np.load(ppath)
        assert {"ecg_id_val", "P_val"} <= set(ref.files), f"{ppath}: keys {ref.files}"
        model = load_frozen(reg, name, dev)
        _, P, y = predict(model, loader, loss_fn, dev)
        ids_ok = np.array_equal(ref["ecg_id_val"], va.ecg_id)
        dmax = float(np.abs(P - ref["P_val"]).max())
        auc_new = roc_auc_score(y, P, average="macro")
        auc_ref = roc_auc_score(y, ref["P_val"], average="macro")
        if name.startswith("dp_"):                  # log labels are presumed: informational only
            e, r = int(name.split("_")[1][3:]), int(name[-1])
            auc_log, log_ok = LOGGED_DP[e][r - 1], True
        else:
            auc_log = LOGGED_AUC[name]
            log_ok = abs(auc_ref - auc_log) < 5e-4
        ok = hash_ok and ids_ok and dmax < 1e-3 and abs(auc_new - auc_ref) < 5e-4 and log_ok
        all_ok &= ok
        print(f"{name:<13}{'n/a' if phash is None else str(hash_ok):<11}{str(ids_ok):<5}"
              f"{dmax:<10.2e}{auc_new:<9.4f}{auc_ref:<10.4f}{auc_log:<9.4f}{'OK' if ok else 'FAIL'}")
    print("GATE PASSED" if all_ok else "GATE FAILED")
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if gate() else 1)