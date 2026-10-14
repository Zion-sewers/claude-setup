#!/usr/bin/env python3
"""Corrective RAG layer for the docs index (claude-setup#422).

Olga (Zion Data). Phase A: the relevance grader alone.
The retry loop (retrieve_corrective) is NOT wired until the grader
clears its precision gate on the dev set.

Grader: NLI cross-encoder, premise=chunk, hypothesis=question.
Verdict RELEVANT iff P(entailment) >= threshold. The threshold is
calibrated on tools/docs-rag/dev_set.json; see calibrate().

Usage (Phase A eval):
  python tools/docs-rag/correct.py --repo . --model cross-encoder/nli-deberta-v3-small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RELEVANT = "RELEVANT"
IRRELEVANT = "IRRELEVANT"

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
# DeBERTa-v3 context budget; premise gets the lion's share.
_MAX_CHUNK_CHARS = 1200
_MAX_Q_CHARS = 300


def load_grader(model_name=DEFAULT_MODEL):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def entailment_scores(grader, pairs):
    """pairs: list of (question, chunk_text). Returns P(entailment) list."""
    import numpy as np

    if not pairs:
        return []
    inputs = [(c[:_MAX_CHUNK_CHARS], q[:_MAX_Q_CHARS]) for q, c in pairs]
    logits = grader.predict(inputs)
    logits = np.asarray(logits, dtype=np.float64)
    # CrossEncoder NLI head: [contradiction, entailment, neutral]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[:, 1].tolist()


def grade(grader, question, chunk_texts, threshold):
    """Grade each chunk. Returns list of RELEVANT/IRRELEVANT."""
    scores = entailment_scores(grader, [(question, t) for t in chunk_texts])
    return [RELEVANT if s >= threshold else IRRELEVANT for s in scores]


def precision_recall(scores, labels, threshold):
    tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == "relevant")
    fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == "irrelevant")
    fn = sum(1 for s, l in zip(scores, labels) if s < threshold and l == "relevant")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec


def calibrate(grader, dev_pairs, thresholds):
    scores = entailment_scores(grader, [(p["question"], p["text"]) for p in dev_pairs])
    labels = [p["label"] for p in dev_pairs]
    rows = []
    for t in thresholds:
        prec, rec = precision_recall(scores, labels, t)
        rows.append(
            {
                "threshold": t,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "n_flagged": sum(1 for s in scores if s >= t),
            }
        )
    return rows, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    tool_dir = Path(args.repo).resolve() / "tools" / "docs-rag"
    dev = json.loads((tool_dir / "dev_set.json").read_text())
    pairs = dev["pairs"]
    print(f"loading grader {args.model} ...", flush=True)
    grader = load_grader(args.model)
    thresholds = [round(x / 100, 2) for x in range(50, 100, 5)]
    rows, _ = calibrate(grader, pairs, thresholds)
    print(f"n={len(pairs)} ({dev['n_relevant']} relevant / {dev['n_irrelevant']} irrelevant)")
    print(f"{'thr':>6} {'prec':>6} {'rec':>6} {'flagged':>8}")
    for r in rows:
        print(
            f"{r['threshold']:>6.2f} {r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} {r['n_flagged']:>8d}"
        )
    best = max([r for r in rows if r["precision"] >= 0.9], key=lambda r: r["recall"], default=None)
    if best:
        print(
            f"GATE PASS: threshold {best['threshold']:.2f} -> "
            f"precision {best['precision']:.3f}, recall {best['recall']:.3f}"
        )
    else:
        print("GATE FAIL: no threshold reaches precision >= 0.9")


if __name__ == "__main__":
    main()
