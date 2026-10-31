import torch
from torch import nn
from src.models.cnn import ECGCNN
from src.models.cnn_attn import ECGCNNAttn

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); v0 = ECGCNN()
torch.manual_seed(0); m = ECGCNNAttn()
same = all(torch.equal(a, b) for a, b in
           zip(v0.features.state_dict().values(), m.features.state_dict().values()))
print("backbone init identical to v0:", same)

m = m.to(dev)
x = torch.randn(8, 12, 1000, device=dev)
print("logits", tuple(m(x).shape), "alpha", tuple(m.attn_maps(x).shape))
assert m(x).shape == (8, 5)
print("params:", sum(p.numel() for p in m.parameters()))

y = torch.randint(0, 2, (8, 5), device=dev).float()
nn.BCEWithLogitsLoss(pos_weight=torch.ones(5, device=dev))(m(x), y).backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters())

from opacus.validators import ModuleValidator
print("opacus errors:", ModuleValidator.validate(m, strict=False))