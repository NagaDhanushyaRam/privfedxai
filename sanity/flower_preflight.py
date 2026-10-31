import torch
from src.models.cnn import ECGCNN

model = ECGCNN()
x = torch.randn(4, 12, 1000)
print("logits:", model(x).shape)

sd = model.state_dict()
print("tensors:", len(sd), "elements:", sum(v.numel() for v in sd.values()))
print("dtypes:", {str(v.dtype) for v in sd.values()})