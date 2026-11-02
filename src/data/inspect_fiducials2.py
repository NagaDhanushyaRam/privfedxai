import collections

import numpy as np
import wfdb

from src.data.inspect_fiducials import FEATURES, global_base

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]      # verify against the Y column order
NEED = ("QRS onset", "QRS offset", "t-wave offset")


def read(ecg_id):
    a = wfdb.rdann(str(global_base(ecg_id)), "atr")
    return np.asarray(a.sample), np.asarray(a.aux_note)


def pairs(ecg_id, n=30):
    s, t = read(ecg_id)
    print(f"\necg_id {ecg_id}: sorted by sample = {bool(np.all(np.diff(s) >= 0))}")
    for x, y in list(zip(s.tolist(), t.tolist()))[:n]:
        print(x, repr(y))


if __name__ == "__main__":
    pairs(1)
    pairs(3)
    f = np.load(FEATURES)
    ids, Y = f["ecg_id"], f["Y"]
    types, edge = collections.Counter(), collections.Counter()
    need = np.zeros((len(ids), len(NEED)), int)
    unsorted = 0
    for k, i in enumerate(ids):
        s, t = read(int(i))
        unsorted += int(np.any(np.diff(s) < 0))
        types.update(t.tolist())
        for x, y in zip(s.tolist(), t.tolist()):
            if x <= 5 or x >= 4995:
                edge[y] += 1
        for j, name in enumerate(NEED):
            need[k, j] = int((t == name).sum())
        if k % 5000 == 0:
            print("progress", k, flush=True)
    print("\nall aux_note types:", dict(types))
    print("records with unsorted samples:", unsorted)
    print("annotations within 5 samples of either end, by type:", dict(edge))
    ok = need.min(1) >= 1
    print("records with >=1 of each of", NEED, ":", int(ok.sum()), "/", len(ids))
    print("min-count quantiles (0,5,50 pct):", np.percentile(need.min(1), [0, 5, 50]))
    for c, name in enumerate(CLASSES):
        m = Y[:, c] == 1
        print(f"{name}: positives {int(m.sum())}, complete {int((ok & m).sum())}")