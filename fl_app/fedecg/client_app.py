"""fedecg client: local training of ECGCNNAttn on one PTB-XL client (D12, D59)."""
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.federated.data import CLIENTS, load_client
from src.federated.local_train import train_local, train_local_dp
from src.models.cnn_attn import ECGCNNAttn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    assert context.node_config["num-partitions"] == len(CLIENTS), "need 5 supernodes"
    client = CLIENTS[context.node_config["partition-id"]]

    model = ECGCNNAttn()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    ds = load_client(client, "train")
    rc = context.run_config
    lr = msg.content["config"]["lr"]
    if rc["dp-epsilon"] > 0:
        loss, sigma = train_local_dp(model, ds, rc["local-epochs"], lr, rc["batch-size"], "cpu",
                                    rc["dp-epsilon"], rc["dp-delta"], rc["dp-rounds"], rc["dp-clip"])
        print(f"[{client}] n={len(ds)} train_loss={loss:.4f} sigma={sigma:.2f}", flush=True)
    else:
        loss = train_local(model, ds, rc["local-epochs"], lr, rc["batch-size"], "cpu")
        print(f"[{client}] n={len(ds)} train_loss={loss:.4f}", flush=True)

    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss, "num-examples": len(ds)}),
    })
    return Message(content=content, reply_to=msg)