#!/usr/bin/env python3
"""Build the Phase-A grader dev set: 120 labeled (query, chunk) pairs.

Olga (Zion Data), for the corrective-RAG plan (claude-setup#422).

Labeling rule (a construct, not a truth — see LABEL_QUALITY_NOTE):
  - RELEVANT: chunk is from the question's ground-truth doc AND was
    retrieved in the top-10 by the live pipeline. Prefer lowest rank.
  - IRRELEVANT: chunk is from a non-truth doc AND was retrieved in the
    top-10 for the same question (a hard negative: the embedder thought
    it was similar).

Full chunk text is rebuilt locally with chunk_doc (deterministic, same
chunk_ids as the index); Qdrant only supplies the ranked chunk_ids.

Reproducible: fixed question order, fixed rank order, no sampling
randomness. Re-run with the same index state to regenerate.

Usage:
  python tools/docs-rag/build_dev_set.py --repo . [--limit 10] [--per-class 3]
Writes tools/docs-rag/dev_set.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import (  # noqa: E402
    COLLECTION_DEFAULT,
    EMBEDDER_MODEL,
    NVIDIA_HOST,
    chunk_doc,
    qd,
)

LABEL_QUALITY_NOTE = (
    "Labels are a construct: 'relevant' means 'from the ground-truth doc, "
    "retrieved top-10, AND from a substance section (Problem/Solution/..."
    " — Scope, Non-goals, See-also, and title-only header chunks are "
    "excluded by rule because they carry no answer content by construction'. "
    "Residual noise is possible (a Problem chunk may motivate without "
    "answering); a manual audit sample is recorded in docs/RAG.md. "
    "'Irrelevant' means 'non-truth chunk retrieved top-10 (hard negative)'."
)

# Sections that carry no answer content by construction.
SECTION_BLOCKLIST = {"scope", "non-goals", "see also"}


def _section_of(chunk_text):
    """'Title / Section' from the '[Title / Section]' context prefix."""
    if not chunk_text.startswith("["):
        return ""
    head = chunk_text[1:].split("]", 1)[0]
    parts = [p.strip() for p in head.split("/")]
    return parts[-1] if len(parts) > 1 else ""


def _is_header_only(chunk_text):
    """True when the chunk body is just the doc title header."""
    body = chunk_text.split("]", 1)[-1].strip() if "]" in chunk_text else chunk_text
    return body.startswith("# ")


def _substance_chunk(chunk_id, chunk_text):
    sec = _section_of(chunk_text).lower()
    return sec not in SECTION_BLOCKLIST and not _is_header_only(chunk_text)


def _embed_query_nvidia(text):
    import urllib.error
    import urllib.request

    from ingest import _surrogate

    add_surrogate_to_request = _surrogate()
    body = json.dumps(
        {"model": EMBEDDER_MODEL, "input": [text[:8000]], "input_type": "query"}
    ).encode()
    req = urllib.request.Request(
        f"https://{NVIDIA_HOST}/v1/embeddings",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    add_surrogate_to_request(req, "custom.nvidia", allowed_hosts=[NVIDIA_HOST])
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            d = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"NVIDIA embed HTTP {e.code}: {e.read().decode()[:200]}")
    import numpy as np

    v = np.array(d["data"][0]["embedding"], dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def search_chunks(collection, qvec, limit):
    r = qd(
        "POST",
        f"/collections/{collection}/points/search",
        {"vector": qvec, "limit": limit, "with_payload": ["doc_id", "chunk_id", "title"]},
    )
    return [(h["payload"]["doc_id"], h["payload"]["chunk_id"]) for h in r["result"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--collection", default=COLLECTION_DEFAULT)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--per-class", type=int, default=3)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    tool_dir = repo / "tools" / "docs-rag"
    eval_set = json.loads((tool_dir / "eval_set.json").read_text())

    # local chunk index: chunk_id -> full text (deterministic rebuild)
    chunk_text = {}
    for doc_path in sorted(repo.glob("docs/specs/2026-*-corpus-*.md")):
        doc_id = doc_path.stem
        text = doc_path.read_text(encoding="utf-8")
        for c in chunk_doc(doc_id, text):
            chunk_text[c["chunk_id"]] = c["text"]

    pairs = []
    for q in eval_set:
        truth = set(q["ground_truth"])
        qvec = _embed_query_nvidia(q["question"])
        hits = search_chunks(args.collection, qvec, args.limit)
        time.sleep(0.2)
        rel_all = [(d, c) for d, c in hits if d in truth]
        irr = [(d, c) for d, c in hits if d not in truth][: args.per_class]
        rel = [(d, c) for d, c in rel_all if _substance_chunk(c, chunk_text[c])][: args.per_class]
        for rank, (doc_id, chunk_id) in enumerate(rel):
            pairs.append(
                {
                    "id": f"{q['id']}-R{rank}",
                    "question": q["question"],
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text[chunk_id],
                    "label": "relevant",
                    "label_rule": "truth-doc chunk, top-%d, substance section "
                    "(Scope/Non-goals/See-also/header-only excluded)" % args.limit,
                }
            )
        for rank, (doc_id, chunk_id) in enumerate(irr):
            pairs.append(
                {
                    "id": f"{q['id']}-I{rank}",
                    "question": q["question"],
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text[chunk_id],
                    "label": "irrelevant",
                    "label_rule": "non-truth chunk retrieved in top-%d (hard negative)"
                    % args.limit,
                }
            )

    out = {
        "n_pairs": len(pairs),
        "n_relevant": sum(1 for p in pairs if p["label"] == "relevant"),
        "n_irrelevant": sum(1 for p in pairs if p["label"] == "irrelevant"),
        "label_quality_note": LABEL_QUALITY_NOTE,
        "pairs": pairs,
    }
    (tool_dir / "dev_set.json").write_text(json.dumps(out, indent=1))
    print(
        f"wrote {len(pairs)} pairs "
        f"({out['n_relevant']} relevant / {out['n_irrelevant']} irrelevant) "
        f"-> tools/docs-rag/dev_set.json"
    )


if __name__ == "__main__":
    main()
