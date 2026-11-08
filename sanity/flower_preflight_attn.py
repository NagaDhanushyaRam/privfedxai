import torch
from src.models.cnn_attn import ECGCNNAttn

model = ECGCNNAttn()
x = torch.randn(4, 12, 1000)
print("logits:", model(x).shape)

sd = model.state_dict()
n_tensors = len(sd)
n_elements = sum(v.numel() for v in sd.values())
print("tensors:", n_tensors, "elements:", n_elements)
print("dtypes:", {str(v.dtype) for v in sd.values()})
print("non-averageable keys:", [k for k in sd if "running" in k or "num_batches" in k])

assert n_tensors == 29 and n_elements == 173_733
assert all(v.dtype == torch.float32 for v in sd.values())

model2 = ECGCNNAttn()
model2.load_state_dict(sd, strict=True)
print("strict load_state_dict round trip: ok")