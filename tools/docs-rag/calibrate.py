#!/usr/bin/env python3
"""Calibration harness for the corrective-RAG relevance grader.

Olga (Zion Data), Phase A v2 (2026-10-14). Extracted from the three failed
grader experiments (NLI templated, GLiClass, OpenJev-0.8B) so the next
grader attempt (MS MARCO cross-encoder, trained classifier) reuses the
same gate instead of reinventing it.

Lessons encoded here:
- A precision gate is meaningless on uncalibrated scores: temperature-scale
  first, report ECE alongside precision (from Heman10x-NGU/openJev-verdict-2.0).
- A "pass" on a single flagged pair is degenerate, not a pass: the gate
  requires minimum recall and a minimum flagged count (learned the hard way
  in exp1/exp2/exp3, where n=1 produced p=1.000/r=0.020 "passes").
- Temperature hitting the grid edge is itself a signal: it means flattening
  everything was optimal, i.e. the logits carry no discriminative information.

All functions are pure (numpy in, numpy/floats out); no model calls here.
"""

from __future__ import annotations

import numpy as np

MIN_PRECISION = 0.9
MIN_RECALL = 0.25
MIN_FLAGGED = 10
TEMP_GRID = (0.1, 5.0, 0.05)


def softmax(logits):
    """Row-wise softmax. logits: (n, k) array-like -> (n, k) probs."""
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits, labels):
    """Grid-search the temperature T minimizing NLL.

    Returns (T, edge) where edge is "low" (optimum wants T < grid: strong
    signal, sharpen), "high" (optimum wants T > grid: flatten everything,
    the logits carry no discriminative information), or None (interior
    optimum). The "high" edge is the no-signal signature seen across all
    Phase A v2 experiments.
    """
    z = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    lo, hi, step = TEMP_GRID
    grid = np.arange(lo, hi + step / 2, step)
    best_t, best_nll = grid[0], float("inf")
    for t in grid:
        p = softmax(z / t)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(p)), labels], 1e-12, 1.0)))
        if nll < best_nll:
            best_t, best_nll = t, nll
    if best_t == grid[0]:
        edge = "low"
    elif best_t == grid[-1]:
        edge = "high"
    else:
        edge = None
    return float(best_t), edge


def ece(probs, labels, n_bins=10):
    """Expected calibration error: mean |confidence - accuracy| per bin."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1])
        if m.sum():
            err += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(err)


def pr_curve(scores, labels, thresholds):
    """Precision/recall at each threshold. Returns list of dicts."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    rows = []
    for t in thresholds:
        pred = scores >= t
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        rows.append(
            {
                "threshold": float(t),
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "n_flagged": int(pred.sum()),
            }
        )
    return rows


def gate_verdict(
    rows, min_precision=MIN_PRECISION, min_recall=MIN_RECALL, min_flagged=MIN_FLAGGED
):
    """Decide PASS/FAIL on calibrated PR rows.

    PASS requires a threshold with precision >= min_precision AND
    recall >= min_recall AND n_flagged >= min_flagged. Among passing
    thresholds, picks the one with max recall.
    Returns (passed: bool, best_row_or_None, reason: str).
    """
    passing = [
        r
        for r in rows
        if r["precision"] >= min_precision
        and r["recall"] >= min_recall
        and r["n_flagged"] >= min_flagged
    ]
    if not passing:
        near = [r for r in rows if r["precision"] >= min_precision]
        if near and all(r["n_flagged"] < min_flagged for r in near):
            msg = (
                f"degenerate: precision >= {min_precision:.2f} only at "
                f"n_flagged < {min_flagged} (single-pair 'passes' are noise, not signal)"
            )
            return False, None, msg
        msg = (
            f"no threshold reaches precision >= {min_precision:.2f} "
            f"with recall >= {min_recall:.2f}"
        )
        return False, None, msg
    best = max(passing, key=lambda r: r["recall"])
    return True, best, "PASS"
