"""D66 pre-scoring gate: do 500 Hz R-peak fiducials, divided by 5, land on the
100 Hz waveform spikes? Fold 9 only; no model involved."""
import numpy as np

from src.data.inspect_fiducials2 import read
from src.data.waveform_data import load_split

N_REC, WIN = 10, 5            # records checked; half-window in 100 Hz samples (50 ms)


def r_peaks_100hz(ecg_id):
    s, t = read(ecg_id)
    keep = s[t == "R peak"]
    if len(keep) == 0:
        raise SystemExit(f"no 'R peak' in ecg_id {ecg_id}; types: {sorted(set(t.tolist()))}")
    r = np.rint(keep / 5).astype(int)
    return r[(r > WIN) & (r < 1000 - WIN)]


def offsets(x, r):
    """x: (12, 1000); returns (12, n_beats): argmax(|x|) in +-WIN window minus the R sample."""
    out = np.empty((x.shape[0], len(r)), int)
    for b, c in enumerate(r):
        out[:, b] = np.abs(x[:, c - WIN : c + WIN + 1]).argmax(axis=1) - WIN
    return out


def main():
    ws = load_split("val")
    pick = np.random.default_rng(0).choice(len(ws.ecg_id), N_REC, replace=False)
    best_offsets = []
    for k in pick:
        eid, x = int(ws.ecg_id[k]), ws.x[k]
        r = r_peaks_100hz(eid)
        off = offsets(x, r)
        best = int(np.abs(x[:, r]).mean(axis=1).argmax())     # lead with the largest R amplitude
        best_offsets.append(off[best])
        print(f"ecg_id {eid}: beats {len(r)}, best lead {best}, "
              f"median offset {np.median(off[best]):+.1f}, "
              f"within +-2: {np.mean(np.abs(off[best]) <= 2):.2f}, "
              f"per-lead medians {np.median(off, axis=1).astype(int).tolist()}")
    allb = np.concatenate(best_offsets)
    print(f"\nall beats, best lead: n={len(allb)}, median {np.median(allb):+.1f}, "
          f"within +-2: {np.mean(np.abs(allb) <= 2):.2f}, at window edge (|off|=5): {np.mean(np.abs(allb) == WIN):.2f}")


if __name__ == "__main__":
    main()