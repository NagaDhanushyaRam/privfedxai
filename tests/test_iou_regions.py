import numpy as np

from src.explain.iou_regions import beat_regions, cells_in_region


def _rec():
    s = [100, 130, 200, 300, 330, 400, 4900, 4930, 4998]   # last T offset is a placeholder
    n = ["QRS onset", "QRS offset", "t-wave offset"] * 3
    return np.array(s), np.array(n)


def test_cells_in_region():
    assert cells_in_region(1000, 1061).tolist() == [25, 26]    # centres 1000 and 1040
    assert cells_in_region(1010, 1065).tolist() == [26]


def test_class_regions_and_edge_rule():
    s, n = _rec()
    assert beat_regions(s, n, "CD") == [(100, 130), (300, 330), (4900, 4930)]
    assert beat_regions(s, n, "HYP") == [(100, 130), (300, 330), (4900, 4930)]
    assert beat_regions(s, n, "STTC") == [(130, 200), (330, 400)]   # last beat dropped
    assert beat_regions(s, n, "MI") == [(100, 200), (300, 400)]


def test_missing_landmark_is_skipped_not_borrowed():
    s, n = _rec()
    keep = np.arange(len(s)) != 4          # drop beat 2's QRS offset (330)
    assert beat_regions(s[keep], n[keep], "CD") == [(100, 130), (4900, 4930)]
    assert beat_regions(s[keep], n[keep], "MI") == [(100, 200), (300, 400)]