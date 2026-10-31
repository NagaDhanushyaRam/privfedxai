import torch
from torch import nn
from src.models.cnn import ECGCNN


class ClassAttention(nn.Module):
    """Per-class additive (Bahdanau) attention over time; replaces mean pooling (D58)."""

    def __init__(self, c_in=128, d_a=64, n_classes=5):
        super().__init__()
        self.proj = nn.Linear(c_in, d_a)                     # shared, followed by tanh
        self.score = nn.Linear(d_a, n_classes, bias=False)   # one scoring vector per class
        self.heads = nn.ModuleList([nn.Linear(c_in, 1) for _ in range(n_classes)])

    def forward(self, f):                                    # f: (B, C, T)
        h = f.transpose(1, 2)                                # (B, T, C)
        alpha = torch.softmax(self.score(torch.tanh(self.proj(h))), dim=1)  # (B, T, K), sums to 1 over T
        ctx = torch.einsum("btk,btc->bkc", alpha, h)         # (B, K, C)
        logits = torch.cat([head(ctx[:, k]) for k, head in enumerate(self.heads)], dim=1)  # (B, K)
        return logits, alpha
    
class ECGCNNAttn(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, d_a=64):
        super().__init__()
        base = ECGCNN(n_leads, n_classes)   # built first, so the backbone init matches v0 under the same seed
        self.features = base.features       # base.head is discarded, not registered
        self.attn = ClassAttention(128, d_a, n_classes)

    def forward(self, x):                   # logits only: Flower and Opacus need this
        logits, _ = self.attn(self.features(x))
        return logits

    @torch.no_grad()
    def attn_maps(self, x):                 # (B, T, K), for the heatmaps; call model.eval() first
        return self.attn(self.features(x))[1]