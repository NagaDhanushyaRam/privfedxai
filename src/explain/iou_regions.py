"""D66 reference regions: clinical region per beat and class, as attention-cell indices."""
import numpy as np

from src.data.inspect_fiducials2 import read

N_CELLS, CELL_SAMPLES = 125, 40      # cell j is centred at 500 Hz sample 40*j
EDGE_LO, EDGE_HI = 5, 4994           # D66 edge rule: a landmark is valid if EDGE_LO < s < EDGE_HI
USES = {                             # the two landmarks each class's region runs between
    "CD": ("QRS onset", "QRS offset"),
    "HYP": ("QRS onset", "QRS offset"),
    "STTC": ("QRS offset", "t-wave offset"),
    "MI": ("QRS onset", "t-wave offset"),
}


def cells_in_region(start, end):
    """Cells whose centre lies inside [start, end] (500 Hz samples)."""
    j = np.arange(N_CELLS)
    return j[(CELL_SAMPLES * j >= start) & (CELL_SAMPLES * j <= end)]


def beat_landmarks(samples, notes):
    """One dict per beat: for each QRS onset, the first QRS offset after it and the first
    T offset after that, both before the next onset. A missing landmark is None."""
    onsets = samples[notes == "QRS onset"]
    beats = []
    for i, on in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else np.inf
        beat, lo = {"QRS onset": on}, on
        for name in ("QRS offset", "t-wave offset"):
            s = samples[(notes == name) & (samples > lo) & (samples < nxt)]
            beat[name] = s[0] if len(s) else None       # samples are sorted (D62)
            if beat[name] is not None:
                lo = beat[name]
        beats.append(beat)
    return beats


def beat_regions(samples, notes, cls):
    """Valid (start, end) per beat for class cls, after the D66 edge rule."""
    a, b = USES[cls]
    out = []
    for beat in beat_landmarks(samples, notes):
        s0, s1 = beat[a], beat[b]
        if s0 is None or s1 is None:
            continue
        if all(EDGE_LO < s < EDGE_HI for s in (s0, s1)):
            out.append((int(s0), int(s1)))
    return out


def record_regions(ecg_id, cls):
    return beat_regions(*read(ecg_id), cls)