import torch
from src.models.cnn_attn import ClassAttention

torch.manual_seed(0)
m = ClassAttention()
logits, alpha = m(torch.randn(4, 128, 125))
print("logits", tuple(logits.shape), "alpha", tuple(alpha.shape))
assert logits.shape == (4, 5) and alpha.shape == (4, 125, 5)
assert torch.allclose(alpha.sum(dim=1), torch.ones(4, 5), atol=1e-5)    # sums to 1 over time
assert not torch.allclose(alpha.sum(dim=2), torch.ones(4, 125))         # not over classes
print("alpha min/max at init:", alpha.min().item(), alpha.max().item())
print("params:", sum(p.numel() for p in m.parameters()))