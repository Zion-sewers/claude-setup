"""Tests for the docs RAG harness (claude-setup#422).

Run from the repo root:  python -m pytest tests/test_docs_rag.py -q
Covers: deterministic chunking, manifest contract, RAG.md link coverage,
eval metric math, and the CI gate's fail-closed behavior.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools" / "docs-rag"))

from check_index import check, corpus_docs  # noqa: E402
from ingest import CHUNK_MAX_CHARS, build_manifest, chunk_doc  # noqa: E402

# ---------- chunking ----------

SAMPLE = """# Sample Report

**Status:** done

Intro paragraph here.

## Findings

First finding paragraph with some content.

Second paragraph in the same section.

## Method

A short method note.
"""


def test_chunking_deterministic():
    a = chunk_doc("doc-x", SAMPLE)
    b = chunk_doc("doc-x", SAMPLE)
    assert a == b


def test_chunk_ids_stable_and_ordered():
    chunks = chunk_doc("doc-x", SAMPLE)
    assert [c["chunk_id"] for c in chunks] == [f"doc-x#c{i:03d}" for i in range(len(chunks))]
    assert all(c["doc_id"] == "doc-x" for c in chunks)


def test_chunk_size_bound_and_context():
    chunks = chunk_doc("doc-x", SAMPLE)
    assert chunks, "sample must produce at least one chunk"
    for c in chunks:
        assert len(c["text"]) <= CHUNK_MAX_CHARS
        assert c["text"].startswith("[Sample Report")


def test_long_paragraph_hard_cut():
    text = "# T\n\n**Status:** done\n\n" + "x" * (CHUNK_MAX_CHARS * 3)
    chunks = chunk_doc("doc-y", text)
    assert len(chunks) >= 3
    assert all(len(c["text"]) <= CHUNK_MAX_CHARS for c in chunks)


def test_empty_sections_dropped():
    chunks = chunk_doc("doc-z", "# T\n\n**Status:** done\n\n## Empty\n\n")
    # title-only content may still yield a chunk, but never an empty one
    assert all(c["text"].strip() for c in chunks)


# ---------- manifest ----------


def test_manifest_contract():
    chunks = chunk_doc("doc-a", SAMPLE) + chunk_doc("doc-b", SAMPLE)
    m = build_manifest(chunks, "docs_rag_v1")
    assert m["collection"] == "docs_rag_v1"
    assert m["embedder"] == "nvidia/nemotron-3-embed-1b"
    assert m["dim"] == 2048
    assert m["n_docs"] == 2
    assert m["n_chunks"] == len(chunks)
    assert m["docs"]["doc-a"]["n_chunks"] == len(chunk_doc("doc-a", SAMPLE))


# ---------- RAG.md coverage (against the real repo tree) ----------


def test_rag_md_links_every_corpus_doc():
    docs = corpus_docs(REPO)
    assert len(docs) >= 200, f"expected the full corpus, got {len(docs)}"
    rag = (REPO / "docs" / "RAG.md").read_text(encoding="utf-8")
    missing = [p.name for p in docs if not (p.name in rag or p.stem in rag)]
    assert not missing, f"{len(missing)} corpus docs unlinked: {missing[:5]}"


def test_manifest_covers_every_corpus_doc():
    docs = corpus_docs(REPO)
    manifest = json.loads((REPO / "tools" / "docs-rag" / "manifest.json").read_text())
    missing = [p.stem for p in docs if p.stem not in manifest["docs"]]
    assert not missing, f"{len(missing)} docs missing from manifest"


def test_check_index_green_on_repo():
    assert check(REPO) == []


# ---------- eval metric math ----------


def recall_at_k(retrieved, truth, k=5):
    truth = set(truth)
    return len(set(retrieved[:k]) & truth) / len(truth) if truth else 0.0


def test_recall_math():
    assert recall_at_k(["a", "b", "c"], ["b"], 5) == 1.0
    assert recall_at_k(["a", "b"], ["c"], 5) == 0.0
    assert recall_at_k(["a", "b", "c", "d"], ["b", "d"], 2) == 0.5
    assert recall_at_k([], ["a"], 5) == 0.0


def test_percentile_math():
    from eval import percentile  # noqa: E402

    xs = [10, 20, 30, 40, 50]
    assert percentile(xs, 50) == 30
    assert percentile(xs, 0) == 10
    assert percentile(xs, 100) == 50
    assert percentile([], 95) == 0.0


# ---------- check_index fail-closed ----------


def _fixture(tmp: Path, rag_links=True, manifest_ok=True):
    (tmp / "docs" / "specs").mkdir(parents=True)
    (tmp / "tools" / "docs-rag").mkdir(parents=True)
    (tmp / "docs" / "specs" / "2026-08-31-corpus-x.md").write_text("# X\n\n**Status:** done\n")
    (tmp / "docs" / "RAG.md").write_text(
        "# RAG\n\n- [x](specs/2026-08-31-corpus-x.md)\n" if rag_links else "# RAG\n\nnothing\n"
    )
    man = {
        "collection": "docs_rag_v1",
        "embedder": "nvidia/nemotron-3-embed-1b",
        "dim": 2048,
        "docs": {"2026-08-31-corpus-x": {"n_chunks": 1, "chunk_ids": ["2026-08-31-corpus-x#c000"]}},
    }
    if not manifest_ok:
        man["docs"] = {}
    (tmp / "tools" / "docs-rag" / "manifest.json").write_text(json.dumps(man))


def test_check_fails_on_unlinked_doc():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _fixture(tmp, rag_links=False)
        v = check(tmp)
        assert any("not linked" in x for x in v)


def test_check_fails_on_missing_manifest_entry():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _fixture(tmp, manifest_ok=False)
        v = check(tmp)
        assert any("missing from" in x for x in v)


def test_check_green_on_fixture():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _fixture(tmp)
        assert check(tmp) == []


# ---------- corrective grader (Phase A) ----------


class _FakeGrader:
    """Stand-in for the NLI cross-encoder: fixed entailment scores."""

    def __init__(self, scores):
        self._scores = list(scores)

    def predict(self, inputs):
        import math

        import numpy as np

        # NLI head order: [contradiction, entailment, neutral].
        # With lc=ln=0, P(entailment)=e^le/(2+e^le) -> le=log(2p/(1-p)).
        rows = [[0.0, math.log(2 * s / (1 - s)), 0.0] for s in self._scores]
        return np.array(rows)


def test_grade_applies_threshold():
    from correct import IRRELEVANT, RELEVANT, grade  # noqa: E402

    g = _FakeGrader([0.95, 0.50, 0.91, 0.89])
    out = grade(g, "q", ["c1", "c2", "c3", "c4"], threshold=0.90)
    assert out == [RELEVANT, IRRELEVANT, RELEVANT, IRRELEVANT]


def test_grade_empty_chunks():
    from correct import grade  # noqa: E402

    assert grade(_FakeGrader([]), "q", [], threshold=0.9) == []


def test_entailment_scores_use_middle_logit():
    from correct import entailment_scores  # noqa: E402

    scores = entailment_scores(_FakeGrader([0.9, 0.1]), [("q", "c")] * 2)
    assert scores[0] > 0.7 > scores[1]


def test_precision_recall_math():
    from correct import precision_recall  # noqa: E402

    scores = [0.95, 0.80, 0.60, 0.40]
    labels = ["relevant", "irrelevant", "relevant", "irrelevant"]
    prec, rec = precision_recall(scores, labels, 0.70)
    assert prec == 0.5  # 1 tp / 2 flagged
    assert rec == 0.5  # 1 tp / 2 relevant
    prec, rec = precision_recall(scores, labels, 0.99)
    assert (prec, rec) == (0.0, 0.0)


def test_calibrate_reports_all_thresholds():
    from correct import calibrate  # noqa: E402

    pairs = [{"question": "q", "text": "c", "label": lab} for lab in ["relevant", "irrelevant"]]
    rows, scores = calibrate(_FakeGrader([0.9, 0.1]), pairs, [0.5, 0.8])
    assert [r["threshold"] for r in rows] == [0.5, 0.8]
    assert rows[0]["precision"] == 1.0 and rows[0]["recall"] == 1.0
    assert len(scores) == 2


def test_section_blocklist():
    from build_dev_set import _is_header_only, _section_of, _substance_chunk  # noqa: E402

    assert _substance_chunk("d#c000", "[T / Solution] prose here")
    assert _substance_chunk("d#c001", "[T / Problem] prose here")
    assert not _substance_chunk("d#c002", "[T / Scope] new tool")
    assert not _substance_chunk("d#c003", "[T / Non-goals] no")
    assert not _substance_chunk("d#c004", "[T / See also] link")
    assert not _substance_chunk("d#c005", "[T] # T\n**Date:** x")
    assert _section_of("[T / Non-goals] x") == "Non-goals"
    assert _section_of("[T / See also] x") == "See also"
    assert _section_of("[T] # T body") == ""
    assert _is_header_only("[T] # T\n**Date:** x")
    assert not _is_header_only("[T / Solution] prose here")


def test_dev_set_contract():
    dev = json.loads((REPO / "tools" / "docs-rag" / "dev_set.json").read_text())
    assert dev["n_pairs"] == len(dev["pairs"]) >= 100
    assert dev["n_relevant"] + dev["n_irrelevant"] == dev["n_pairs"]
    assert "label_quality_note" in dev and dev["label_quality_note"]
    for p in dev["pairs"]:
        assert p["label"] in ("relevant", "irrelevant")
        assert p["text"] and p["question"] and p["chunk_id"]
