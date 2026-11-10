"""Noise multiplier per client and target epsilon (record-level DP-SGD; 30 rounds x 1 epoch; batch 64)."""
import inspect
from opacus.accountants.utils import get_noise_multiplier

print(inspect.signature(get_noise_multiplier))

N_TRAIN = {"site0": 7691, "site1": 5038, "site2": 3318, "site3": 332, "rest": 705}   # D12
BATCH, ROUNDS, DELTA = 64, 30, 1e-5

print("client   q      " + "  ".join(f"eps={e:<5}" for e in (10, 5, 1)))
for c, n in N_TRAIN.items():
    q = BATCH / n
    sig = [get_noise_multiplier(target_epsilon=e, target_delta=DELTA, sample_rate=q, epochs=ROUNDS)
           for e in (10, 5, 1)]
    print(f"{c:6s} {q:.4f}  " + "  ".join(f"{s:9.2f}" for s in sig))