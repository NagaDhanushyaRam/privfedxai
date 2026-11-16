"""D66 scoring: attention map + beat regions -> IoU, peak-hit, chance rate."""
import numpy as np

from src.explain.iou_regions import N_CELLS, cells_in_region

THRESH = 0.5      # D66: predicted cells = attention >= THRESH * the map's maximum


def pred_cells(a):
    return np.flatnonzero(a >= THRESH * a.max())


def peak_cell(a):
    return int(a.argmax())                 # ties: lowest index


def iou(pred, ref):
    inter = len(np.intersect1d(pred, ref))
    return inter / len(np.union1d(pred, ref))


def dist_to(cells, j):
    return int(np.abs(cells - j).min())


def peak_hit(peak, ref, t):
    return dist_to(ref, peak) <= t


def chance_hit(ref, t):
    """Share of the 125 cells within t cells of the reference (a random peak's hit rate)."""
    j = np.arange(N_CELLS)
    return float(np.mean(np.abs(j[:, None] - ref[None, :]).min(axis=1) <= t))


def score_record(a, regions):
    """a: (125,) attention of one true class; regions: valid (start, end) per beat.
    None if no beat gives a non-empty cell set; else a dict for both reference variants."""
    cells = [c for c in (cells_in_region(s, e) for s, e in regions) if len(c)]
    if not cells:
        return None
    pred, peak = pred_cells(a), peak_cell(a)
    d = [dist_to(c, peak) for c in cells]
    refs = {"peak_beat": cells[int(np.argmin(d))],      # first minimum = earlier beat on a tie
            "all_beat": np.unique(np.concatenate(cells))}
    out = {"n_beats": len(cells), "n_empty": len(regions) - len(cells)}
    for name, ref in refs.items():
        out[f"{name}_iou"] = iou(pred, ref)
        for t in (0, 1):
            out[f"{name}_hit{t}"] = peak_hit(peak, ref, t)
            out[f"{name}_chance{t}"] = chance_hit(ref, t)
    return out