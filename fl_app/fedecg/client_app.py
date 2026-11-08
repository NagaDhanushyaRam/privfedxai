"""fedecg client: local training of ECGCNNAttn on one PTB-XL client (D12, D59)."""
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.federated.data import CLIENTS, load_client
from src.federated.local_train import train_local
from src.models.cnn_attn import ECGCNNAttn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    assert context.node_config["num-partitions"] == len(CLIENTS), "need 5 supernodes"
    client = CLIENTS[context.node_config["partition-id"]]

    model = ECGCNNAttn()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    ds = load_client(client, "train")
    loss = train_local(model, ds,
                       context.run_config["local-epochs"],
                       msg.content["config"]["lr"],
                       context.run_config["batch-size"], "cpu")
    print(f"[{client}] n={len(ds)} train_loss={loss:.4f}", flush=True)

    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss, "num-examples": len(ds)}),
    })
    return Message(content=content, reply_to=msg)