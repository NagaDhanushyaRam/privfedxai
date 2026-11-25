import numpy as np
from app import backend as B

REF = {"cnn": "results/cnn/cnn_val_probs.npz", "attn": "results/attn/attn_val_probs.npz",
       "fl": "results/fl/FROZEN_fl_val_probs.npz",
       **{f"dp_eps{e}_r{r}": f"results/fl/fl_val_probs_eps{e}_r{r}.npz"
          for e in (10, 5, 1) for r in (1, 2, 3)}}


def test_scores_match_saved_fold9_probs():
    d = B.load_demo()
    for name, path in REF.items():
        ref = np.load(path)
        row = {int(e): i for i, e in enumerate(ref["ecg_id_val"])}
        for k in range(5):
            got = B.score(d["x"][k], name)
            want = ref["P_val"][row[int(d["ecg_id"][k])]]
            assert np.abs(got - want).max() < 1e-3, (name, int(d["ecg_id"][k]))


def test_attention_shape_and_sums():
    a = B.attention(B.load_demo()["x"][0])
    assert a.shape == (125, 5) and np.allclose(a.sum(0), 1, atol=1e-4)


def test_table_matches_log():
    for model, auc in {"fl": 0.9195, "dp_eps10": 0.8578, "dp_eps5": 0.8477, "dp_eps1": 0.8119}.items():
        m = B.fold10_metrics(model)
        assert set(m.index) == {"auc", "f1", "mcc", "ppr"}, (model, list(m.index))
        assert abs(m.loc["auc", "value"] - auc) < 5e-4, model