"""fedecg server: FedAvg of ECGCNNAttn; pooled fold-9 evaluation is monitoring only (D53)."""
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.federated.data import ROOT, load_client
from src.models.cnn_attn import ECGCNNAttn
from src.models.train import predict

app = ServerApp()
OUT = ROOT / "results" / "fl"


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"eps{context.run_config['dp-epsilon']:g}_r{context.run_config['dp-rep']}"

    val = load_client(None, "val")              # pooled fold 9, original row order
    val_loader = DataLoader(val, batch_size=256, shuffle=False, num_workers=0)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = ECGCNNAttn()
        model.load_state_dict(arrays.to_torch_state_dict())
        _, p, y = predict(model, val_loader, nn.BCEWithLogitsLoss(), "cpu")
        auc = float(roc_auc_score(y, p, average="macro"))
        print(f"round {server_round}: pooled fold-9 macro AUC {auc:.4f}", flush=True)
        if server_round == num_rounds:
            np.savez(OUT / f"fl_val_probs_{tag}.npz", ecg_id_val=val.ecg_id, P_val=p)
        return MetricRecord({"macro_auc": auc})

    torch.manual_seed(0)                        # same seed as the centralized runs
    arrays = ArrayRecord(ECGCNNAttn().state_dict())
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    result = strategy.start(grid=grid, initial_arrays=arrays,
                            train_config=ConfigRecord({"lr": lr}),
                            num_rounds=num_rounds, evaluate_fn=global_evaluate)
    torch.save(result.arrays.to_torch_state_dict(), OUT / f"fl_{tag}_last.pt")