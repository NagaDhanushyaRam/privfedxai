import torch

from src.federated.data import CLIENTS, load_client
from src.models.train import pos_weight

EXPECTED = {"site0": (7691, 497), "site1": (5038, 606), "site2": (3318, 834),
            "site3": (332, 115), "rest": (705, 94)}      # (train, val), D12

tot_tr = tot_va = 0
for c in CLIENTS:
    tr, va = load_client(c, "train"), load_client(c, "val")
    pw = pos_weight(tr.y)
    print(c, "train", tuple(tr.x.shape), "val", tuple(va.x.shape), "pos_weight", pw.numpy().round(2))
    assert (len(tr), len(va)) == EXPECTED[c]
    assert tr.x.dtype == torch.float32 and tr.y.shape[1] == 5
    assert torch.isfinite(pw).all()
    tot_tr += len(tr); tot_va += len(va)
assert (tot_tr, tot_va) == (17084, 2146)
print("OK")