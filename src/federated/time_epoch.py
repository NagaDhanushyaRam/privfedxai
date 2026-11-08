import time
import torch

from src.federated.data import load_client
from src.federated.local_train import train_local
from src.models.cnn_attn import ECGCNNAttn

ds = load_client("site0", "train")              # the largest client, 7,691 records
devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
for dev in devs:
    for rep in ((1, 2) if dev == "cuda" else (1,)):   # cuda run 1 includes warm-up
        torch.manual_seed(0)
        model = ECGCNNAttn()
        t = time.time()
        loss = train_local(model, ds, epochs=1, lr=1e-3, batch=64, dev=dev)
        print(dev, "run", rep, f"{time.time() - t:.1f} s", "mean loss", round(loss, 4))