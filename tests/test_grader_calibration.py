"""Contract tests for the grader calibration harness (Phase A v2).

Run from the repo root:  python -m pytest tests/test_grader_calibration.py -q
Covers: softmax normalization, temperature fitting (including the
hit-the-edge no-signal signature), ECE, PR math, and the gate's
degenerate-pass rejection (the n=1 "pass" seen in exp1/exp2/exp3).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools" / "docs-rag"))

from calibrate import ece, fit_temperature, gate_verdict, pr_curve, sigmoid, softmax


def test_softmax_rows_sum_to_one():
    p = softmax([[2.0, 1.0, 0.1], [-1.0, 0.0, 1.0]])
    assert p.shape == (2, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all() and (p <= 1).all()


def test_sigmoid_midpoint():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(100.0) == pytest.approx(1.0)
    assert sigmoid(-100.0) == pytest.approx(0.0)


def test_temperature_improves_nll_on_separable_logits():
    rng = np.random.default_rng(7)
    # separable: class-1 logits clearly above class-0
    logits = np.vstack(
        [
            np.column_stack([rng.normal(0, 1, 40), rng.normal(3, 1, 40)]),
            np.column_stack([rng.normal(3, 1, 40), rng.normal(0, 1, 40)]),
        ]
    )
    labels = np.array([1] * 40 + [0] * 40)
    t, edge = fit_temperature(logits, labels)
    assert edge in ("low", None), "separable logits want sharpening, not flattening"
    p_t = softmax(logits / t)[np.arange(80), labels]
    p_1 = softmax(logits)[np.arange(80), labels]
    nll_t = -np.mean(np.log(p_t))
    nll_1 = -np.mean(np.log(p_1))
    assert nll_t <= nll_1 + 1e-9


def test_temperature_high_edge_means_no_signal():
    rng = np.random.default_rng(11)
    logits = rng.normal(0, 1, (200, 3))
    labels = rng.integers(0, 3, 200)
    _, edge = fit_temperature(logits, labels)
    assert edge == "high", "pure noise must flatten: the no-signal signature"


def test_ece_near_zero_for_perfectly_calibrated():
    probs = np.array([0.9] * 90 + [0.1] * 10)
    labels = np.array([1] * 81 + [0] * 9 + [1] * 1 + [0] * 9)
    assert ece(probs, labels) < 0.05


def test_ece_positive_for_overconfident():
    probs = np.array([0.99] * 100)
    labels = np.array([1] * 60 + [0] * 40)
    assert ece(probs, labels) > 0.2


def test_pr_curve_hand_worked_example():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    labels = np.array([1, 0, 1, 0])
    rows = pr_curve(scores, labels, [0.5])
    assert len(rows) == 1
    r = rows[0]
    # flagged: 0.9 (tp), 0.8 (fp); missed: 0.4 (fn)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["n_flagged"] == 2


def test_gate_passes_on_strong_row():
    rows = [{"threshold": 0.6, "precision": 0.95, "recall": 0.5, "n_flagged": 25}]
    passed, best, reason = gate_verdict(rows)
    assert passed and best["threshold"] == 0.6 and reason == "PASS"


def test_gate_rejects_degenerate_single_pair_pass():
    # the exact exp1/exp2/exp3 failure mode: p=1.0 on n=1
    rows = [{"threshold": 0.75, "precision": 1.0, "recall": 0.02, "n_flagged": 1}]
    passed, best, reason = gate_verdict(rows)
    assert not passed and best is None
    assert "degenerate" in reason


def test_gate_rejects_low_precision():
    rows = [{"threshold": 0.5, "precision": 0.68, "recall": 0.5, "n_flagged": 37}]
    passed, _, _ = gate_verdict(rows)
    assert not passed


def test_gate_picks_max_recall_among_passing():
    rows = [
        {"threshold": 0.8, "precision": 0.95, "recall": 0.3, "n_flagged": 15},
        {"threshold": 0.6, "precision": 0.92, "recall": 0.6, "n_flagged": 30},
    ]
    passed, best, _ = gate_verdict(rows)
    assert passed and best["threshold"] == 0.6
