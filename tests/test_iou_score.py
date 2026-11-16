import numpy as np
import pytest

from src.explain.iou_score import score_record

A = np.zeros(125)
A[5], A[6], A[7] = 0.08, 0.31, 0.19      # threshold 0.155 keeps cells 6 and 7; peak is 6


def test_picture_example():
    r = score_record(A, [(200, 261)])           # reference cells {5, 6}
    assert r["peak_beat_iou"] == pytest.approx(1 / 3)
    assert r["all_beat_iou"] == pytest.approx(1 / 3)
    assert r["peak_beat_hit0"] and r["peak_beat_hit1"]
    assert r["peak_beat_chance0"] == pytest.approx(2 / 125)
    assert r["peak_beat_chance1"] == pytest.approx(4 / 125)


def test_peak_beat_vs_all_beat():
    r = score_record(A, [(200, 261), (2000, 2061)])     # second beat is cells 50, 51
    assert r["n_beats"] == 2
    assert r["peak_beat_iou"] == pytest.approx(1 / 3)
    assert r["all_beat_iou"] == pytest.approx(1 / 5)
    assert r["all_beat_chance0"] == pytest.approx(4 / 125)


def test_tie_picks_earlier_beat():
    a = np.zeros(125)
    a[10], a[9] = 1.0, 0.6                              # predicted {9, 10}, peak 10
    r = score_record(a, [(360, 370), (440, 450)])       # reference cells [9] then [11]
    assert r["peak_beat_iou"] == pytest.approx(0.5)     # the earlier beat, cell 9


def test_empty_region_is_counted():
    assert score_record(A, [(1, 30)]) is None           # no cell centre inside
    r = score_record(A, [(1, 30), (200, 261)])
    assert r["n_beats"] == 1 and r["n_empty"] == 1