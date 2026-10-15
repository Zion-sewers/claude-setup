"""Generated map of every document in the estate, with its status and its lane-letter scheme.

WHY A GENERATOR AND NOT AN INDEX. `docs/INDEX.md` is hand-curated and reaches 29 documents. The
estate holds 309. That is 9% coverage, and the 280 unreachable documents are findable only by
knowing they exist. A hand-written index is what decayed to 9%, so replacing it with a better
hand-written index would decay the same way. This mirrors the contract `tools/map/codemap.py`
already enforces for directories: the map is GENERATED, a hand edit reads as drift, and the gate
fails on a document whose status cannot be derived or declared.

INDEX.md is not deleted by this. It stays as the curated reading ORDER, which is a different job
from an inventory, and this tool measures how much of the estate it reaches.

WHAT STATUS MEANS, AND WHY IT IS DERIVABLE. The objection to recording status is that it needs a
human. Mostly it does not, because the document classes already encode it:

  adr/          declares `Status: accepted|proposed|superseded` in its own header. Authoritative.
  prd/ specs/   declare `Status:` when the author bothered. UNDECLARED when not, which is a finding.
  HANDOFF-*     historical-record ALWAYS. A handoff is a point-in-time artifact; it was true when
                written and is not a current-state claim. Marking one "stale" would be a category
                error, which is the mistake that makes people distrust status fields.
  analysis/     dated-snapshot ALWAYS, same reasoning. CLAUDE.md says several are knowingly stale.
  reflections/  dated-snapshot ALWAYS.
  prior-art/    carries `recheck_after`, so current-or-expired is arithmetic.

Only the residue needs a human, and that goes in `docs/doc-status.txt`, the same registry shape
`codemap.py` uses for directory purposes.

THE LANE-SCHEME COLUMN IS THE POINT. Lane letters were renumbered on 2026-07-30 (ADR-0016): the
harness moved from B to A, resume C to B, learning D to C, content E to D. Every document written
before that date uses the old letters and IS CORRECT FOR ITS DATE. A reader who does not know
which scheme a document was written under will silently misread `lane B` as the resume engine in a
handoff that meant the harness. `tools/lib/lanes.py` already owns the cutover, so this reuses it
rather than re-deriving the boundary.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path

MAP_PATH = "docs/DOCMAP.md"
REGISTRY_PATH = "docs/doc-status.txt"
INDEX_PATH = "docs/INDEX.md"

# Class is decided by path, in this order. First match wins, so the more specific prefixes lead.
CLASSES: tuple[tuple[str, str], ...] = (
    ("docs/adr/", "adr"),
    ("docs/prd/", "prd"),
    ("docs/specs/", "spec"),
    ("docs/analysis/", "analysis"),
    ("docs/reflections/", "reflection"),
    ("docs/prior-art/", "prior-art"),
    ("docs/standards/", "standard"),
    ("work-docs/", "work-doc"),
)

# Classes whose status is fixed by what the class IS, not by anything in the file.
#
# The DEFINITION classes are here because of a scoping error worth recording. The first version
# demanded a lifecycle status from 480 files that are definitions rather than documents: 134
# SKILL.md, 38 rules, 23 agent definitions, 7 slash commands, and the dot-codex equivalents. A
# skill does not have a status in this sense. What it has is a DEPLOYMENT state, live versus
# canonical, and `tools/audit/skills_sync.py check` already measures exactly that. Two tools
# measuring one property is how they drift apart and one of them goes stale, which is the argument
# `docs/QUALITY-CONTRACT.md` already makes against duplicated purpose strings.
#
# So definitions are LISTED, for coverage, and never required to declare anything. This tool
# reports on artifacts with a lifecycle: ADRs, PRDs, specs, standards, analyses, handoffs.
ALWAYS: dict[str, str] = {
    "vendored": "third-party-not-ours",
    "skill": "definition-see-skills_sync",
    "command": "definition-see-skills_sync",
    "agent": "definition-see-skills_sync",
    "rule": "definition-see-skills_sync",
    "corpus": "corpus-material",
    "automation-output": "run-log",
    "automation-prompt": "definition-see-skills_sync",
    "generated-output": "generated",
    "analysis": "dated-snapshot",
    "reflection": "dated-snapshot",
    "work-doc": "dated-snapshot",
    "archived": "historical-record",
}

DATE_IN_NAME = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
STATUS_LINE = re.compile(r"^[-*\s]*\**status\**\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
# ADRs 0001 through 0015 use an INLINE form: `Date: 2026-07-24. Status: accepted.` on one
# line. The anchored pattern above cannot see it, which made 15 of 17 ADRs read as
# UNDECLARED when every one of them declares a status. Tried second, so a proper header
# line still wins when both are present.
STATUS_INLINE = re.compile(r"\bstatus\s*[:=]\s*([A-Za-z][A-Za-z0-9 _-]{0,40})",
                          re.IGNORECASE)
SUPERSEDED_BY = re.compile(r"superseded\s+by\s+([A-Za-z0-9/._-]+)", re.IGNORECASE)
SUPERSEDES = re.compile(r"supersedes\s+([A-Za-z0-9/._ ,-]+)", re.IGNORECASE)
# A document referring to a lane by letter is the thing that can be misread across the cutover.
LANE_MENTION = re.compile(r"\blane\s+([A-E])\b", re.IGNORECASE)


def _load_lanes(project: Path):
    """Reuse tools/lib/lanes.py rather than re-deriving the cutover date."""
    p = project / "tools" / "lib" / "lanes.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("_docmap_lanes", p)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Doc:
    path: str
    cls: str
    date: str
    status: str
    source: str          # where the status came from: header, class, registry, arithmetic
    supersedes: str
    superseded_by: str
    lane_scheme: str     # 1, 2, or "n/a" when the document never names a lane by letter
    in_index: bool


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def tracked_docs(project: Path) -> list[str]:
    """Only tracked files. An untracked document is invisible to review and is a separate finding."""
    out = subprocess.run(
        ["git", "-C", str(project), "ls-files", "docs", "work-docs", "*.md"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout
    keep = []
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.endswith(".md") or (line.startswith("docs/prior-art/") and line.endswith(".json")):
            # The generated map itself is excluded, or it would describe its own staleness.
            if line != MAP_PATH:
                keep.append(line)
    return sorted(set(keep))


# Vendored third-party material. Listed for coverage, never required to declare a status, because
# it is not ours to declare one for. The same reasoning `codemap.py prior-art` already applies when
# it marks these directories out of scope.
VENDORED = (
    "skills/deep-research/", "skills/ui-ux-pro-max/", "/.system/", "skills/heygen-skills/",
)


def classify(path: str) -> str:
    if any(v in path for v in VENDORED):
        return "vendored"
    if path.startswith(("docs/HANDOFF", "docs/archive/HANDOFF")):
        return "handoff"
    # docs/archive/ holds what used to sit at the docs/ root: handoffs, pasted model
    # transcripts, one-off notes. Moved there 2026-08-23 when the root held 43 files and
    # 25 of them were point-in-time. Everything under it is a historical record by
    # construction, same reasoning as a handoff; a file that is still live does not
    # belong there and the move is the bug.
    if path.startswith("docs/archive/"):
        return "archived"
    # Definition classes, checked before the docs/ prefixes so a rule or skill living anywhere is
    # recognised. Order matters: a SKILL.md inside a skills/ dir is a skill, and the other files
    # beside it are that skill's own references, which are equally not lifecycle documents.
    if "/skills/" in path or path.startswith("skills/"):
        return "skill"
    if "/commands/" in path or path.startswith("commands/"):
        return "command"
    if "/agents/" in path or path.startswith("agents/"):
        return "agent"
    if "/rules/" in path or path.startswith("rules/"):
        return "rule"
    # Corpus material. It has no lifecycle status and, per the shrinkage analysis, it should not
    # live in this repo at all. Classing it honestly is a prerequisite for moving it.
    if path.startswith(("research-papers/", "master-plans/")):
        return "corpus"
    # Automation RUN OUTPUT. 154 of these are tracked, dated May to July 2026, one per cron
    # firing, with a single sweep accounting for about 45. They are logs. A log does not have a
    # lifecycle status, and tracking one per run forever is the bloat this repo is trying to shed,
    # so the class exists partly to make that count visible in the map.
    if "/automations/last-messages/" in path:
        return "automation-output"
    if "/automations/prompts/" in path:
        return "automation-prompt"
    # Generated tool output. Its status is whether the tool ran, which the tool's own exit code
    # already answers, and a status field here would be a second answer that can disagree.
    if "/out/" in path:
        return "generated-output"
    # A prior-art RECORD is a .json file. A README living in that directory is documentation
    # about the records, and classing it as a record made it read as malformed JSON.
    if path.startswith("docs/prior-art/") and not path.endswith(".json"):
        return "doc"
    for prefix, name in CLASSES:
        if path.startswith(prefix):
            return name
    if "/" not in path:
        return "root"
    return "doc"


def date_for(path: str, text: str) -> str:
    m = DATE_IN_NAME.search(Path(path).name)
    if m:
        return "-".join(m.groups())
    m = re.search(r"^[-*\s]*\**date\**\s*[:=]\s*(20\d{2}-\d{2}-\d{2})", text,
                  re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1)
    m = DATE_IN_NAME.search(text[:2000])
    return "-".join(m.groups()) if m else ""


def prior_art_status(text: str, today: str) -> tuple[str, str]:
    try:
        rec = json.loads(text)
    except ValueError:
        return "MALFORMED", "arithmetic"
    until = str(rec.get("recheck_after", ""))
    if not until:
        return "UNDECLARED", "arithmetic"
    return ("current" if until >= today else "EXPIRED " + until), "arithmetic"


def status_for(path: str, cls: str, text: str, registry: dict[str, str],
               today: str) -> tuple[str, str]:
    if path in registry:
        return registry[path], "registry"
    if cls == "handoff":
        return "historical-record", "class"
    if cls in ALWAYS:
        return ALWAYS[cls], "class"
    if cls == "prior-art":
        return prior_art_status(text, today)
    m = STATUS_LINE.search(text)
    if m:
        raw = m.group(1).strip().strip("*_`.").lower()
        # Keep the author's word, normalised only in case and punctuation.
        return (raw[:60] or "UNDECLARED"), "header"
    m = STATUS_INLINE.search(text[:1200])
    if m:
        return m.group(1).strip().strip("*_`.").lower()[:60], "header-inline"
    return "UNDECLARED", "none"


def lane_scheme_for(date: str, text: str, lanes) -> str:
    if not LANE_MENTION.search(text):
        return "n/a"
    if not date:
        return "1?"
    if lanes is not None:
        return str(lanes.scheme_for(date + "T00:00:00Z"))
    return "2" if date >= "2026-07-30" else "1"


def index_targets(project: Path) -> set[str]:
    """Paths docs/INDEX.md actually links to, normalised to repo-relative."""
    text = read(project / INDEX_PATH)
    out: set[str] = set()
    for m in re.finditer(r"\]\(([^)]+)\)", text):
        t = m.group(1).split("#")[0].strip()
        if not t or t.startswith("http"):
            continue
        if t.startswith("../"):
            out.add(t[3:])
        else:
            out.add("docs/" + t if not t.startswith("docs/") else t)
    return out


def read_registry(project: Path) -> dict[str, str]:
    """`docs/doc-status.txt`, one `path | status` per line. Same shape as dir-purpose.txt."""
    reg: dict[str, str] = {}
    p = project / REGISTRY_PATH
    if not p.exists():
        return reg
    for line in read(p).split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        k, _, v = line.partition("|")
        if k.strip():
            reg[k.strip()] = v.strip()
    return reg


def inventory(project: Path) -> list[Doc]:
    today = datetime.date.today().isoformat()
    registry = read_registry(project)
    idx = index_targets(project)
    lanes = _load_lanes(project)
    docs: list[Doc] = []
    for path in tracked_docs(project):
        text = read(project / path)
        cls = classify(path)
        date = date_for(path, text)
        status, source = status_for(path, cls, text, registry, today)
        sb = SUPERSEDED_BY.search(text)
        sp = SUPERSEDES.search(text)
        docs.append(Doc(
            path=path, cls=cls, date=date, status=status, source=source,
            supersedes=(sp.group(1).strip()[:40] if sp else ""),
            superseded_by=(sb.group(1).strip()[:40] if sb else ""),
            lane_scheme=lane_scheme_for(date, text, lanes),
            in_index=path in idx,
        ))
    return docs


def cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def render(docs: list[Doc]) -> str:
    by_class: dict[str, list[Doc]] = {}
    for d in docs:
        by_class.setdefault(d.cls, []).append(d)
    reached = sum(1 for d in docs if d.in_index)
    undeclared = [d for d in docs if d.status == "UNDECLARED"]
    expired = [d for d in docs if d.status.startswith("EXPIRED")]
    scheme1 = [d for d in docs if d.lane_scheme == "1"]

    out = [
        "# DOCMAP",
        "",
        "GENERATED by `python tools/docmap/docmap.py write`. A hand edit reads as drift and fails",
        "`docmap.py check`. To change a document's status, declare it in that document's own header,",
        "or add a row to `docs/doc-status.txt` when the class cannot derive it.",
        "",
        "`docs/INDEX.md` is not replaced by this. It is the curated reading ORDER, a different job",
        "from an inventory. This file measures how much of the estate that order reaches.",
        "",
        "## Coverage",
        "",
        f"- documents: **{len(docs)}**",
        f"- reachable from `docs/INDEX.md`: **{reached}** ({100 * reached // max(len(docs), 1)}%)",
        f"- status UNDECLARED: **{len(undeclared)}**",
        f"- prior-art records EXPIRED: **{len(expired)}**",
        f"- written under the pre-2026-07-30 lane scheme: **{len(scheme1)}**",
        "",
        "### Lane scheme, and why the column exists",
        "",
        "ADR-0016 renumbered the lanes on 2026-07-30: harness B to A, resume C to B, learning D to",
        "C, content E to D. A document written before that date uses the old letters and **is",
        "correct for its date**. Scheme `1` means read its lane letters through",
        "`tools/lib/lanes.py`; `n/a` means it never names a lane by letter.",
        "",
    ]
    for cls in sorted(by_class):
        rows = sorted(by_class[cls], key=lambda d: (d.date or "0000", d.path))
        out += [f"## {cls} ({len(rows)})", "",
                "| document | date | status | from | lane scheme | in INDEX | supersedes | superseded by |",
                "|---|---|---|---|---|---|---|---|"]
        for d in rows:
            out.append("| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                cell(d.path), d.date or "-", cell(d.status), d.source, d.lane_scheme,
                "yes" if d.in_index else "-", cell(d.supersedes) or "-",
                cell(d.superseded_by) or "-"))
        out.append("")
    return "\n".join(out) + "\n"


def evaluate(project: Path) -> dict:
    docs = inventory(project)
    return {
        "docs": docs,
        "rendered": render(docs),
        "on_disk": read(project / MAP_PATH),
    }


def problems(state: dict) -> list[str]:
    out: list[str] = []
    docs: list[Doc] = state["docs"]
    if state["rendered"] != state["on_disk"]:
        out.append("{} is not what the documents imply. Run `python tools/docmap/docmap.py write`."
                   .format(MAP_PATH))
    undeclared = [d.path for d in docs if d.status == "UNDECLARED"]
    if undeclared:
        out.append("{} document(s) with no derivable status. Declare `Status:` in the document, or "
                   "add a row to {}:\n  {}".format(
                       len(undeclared), REGISTRY_PATH, "\n  ".join(undeclared[:12])))
    expired = [d.path + " (" + d.status + ")" for d in docs if d.status.startswith("EXPIRED")]
    if expired:
        out.append("{} prior-art record(s) past recheck_after:\n  {}".format(
            len(expired), "\n  ".join(expired[:12])))
    malformed = [d.path for d in docs if d.status == "MALFORMED"]
    if malformed:
        out.append("{} prior-art record(s) are not valid JSON:\n  {}".format(
            len(malformed), "\n  ".join(malformed)))
    return out


def selftest(project: Path) -> int:
    """Prove the derivations work. Exit code is the failure count."""
    fails: list[str] = []

    def check(label: str, got, want) -> None:
        if got != want:
            fails.append("{}: got {!r} want {!r}".format(label, got, want))

    # Class assignment. A handoff must never be classed as a current-state document.
    check("handoff by prefix", classify("docs/HANDOFF-2026-07-30-x.md"), "handoff")
    check("archived handoff keeps its class", classify("docs/archive/HANDOFF-2026-07-30-x.md"), "handoff")
    check("docs/archive is archived", classify("docs/archive/gemini-code-1.md"), "archived")
    check("archived status is class-derived",
          status_for("docs/archive/note.md", "archived", "", {}, "2026-08-23"),
          ("historical-record", "class"))
    check("adr", classify("docs/adr/0016-x.md"), "adr")
    check("prior-art json", classify("docs/prior-art/tools-bus.json"), "prior-art")
    check("work-doc", classify("work-docs/x.md"), "work-doc")
    check("root", classify("CLAUDE.md"), "root")

    # A handoff's status comes from its CLASS and cannot be overridden by a stray Status: line,
    # because a handoff is a record of a moment, not a claim about now.
    st, src = status_for("docs/HANDOFF-x.md", "handoff", "Status: accepted\n", {}, "2026-07-30")
    check("handoff status is class-derived", (st, src), ("historical-record", "class"))

    # An ADR's status comes from its header.
    st, src = status_for("docs/adr/0001-x.md", "adr", "- Status: accepted\n", {}, "2026-07-30")
    check("adr status from header", (st, src), ("accepted", "header"))

    # The registry wins over everything, so a human can always override.
    st, src = status_for("docs/x.md", "doc", "Status: accepted", {"docs/x.md": "retired"},
                         "2026-07-30")
    check("registry overrides", (st, src), ("retired", "registry"))

    # prior-art expiry is arithmetic, both directions.
    st, _ = prior_art_status('{"recheck_after": "2026-12-01"}', "2026-07-30")
    check("prior-art current", st, "current")
    st, _ = prior_art_status('{"recheck_after": "2026-01-01"}', "2026-07-30")
    check("prior-art expired", st.startswith("EXPIRED"), True)
    st, _ = prior_art_status("not json", "2026-07-30")
    check("prior-art malformed", st, "MALFORMED")

    # Lane scheme. This is the column the whole tool exists for.
    lanes = _load_lanes(project)
    check("pre-cutover doc naming a lane is scheme 1",
          lane_scheme_for("2026-07-29", "handing to lane B", lanes), "1")
    check("post-cutover doc naming a lane is scheme 2",
          lane_scheme_for("2026-07-31", "lane B owns this", lanes), "2")
    check("a doc that names no lane is n/a",
          lane_scheme_for("2026-07-29", "no lanes mentioned here", lanes), "n/a")

    # Date extraction prefers the filename, which is the convention in this tree.
    check("date from filename", date_for("docs/analysis/2026-07-25-x.md", "Date: 2020-01-01"),
          "2026-07-25")
    check("date from body when filename has none", date_for("docs/x.md", "- Date: 2026-07-30"),
          "2026-07-30")

    # A generated map must not describe itself, or `check` can never converge.
    check("the map excludes itself", MAP_PATH in tracked_docs(project), False)

    # ---- added 2026-07-31, because a mutation control proved these were unguarded ----
    # tools/audit/mutations/docmap.py applied 8 mutations and FOUR SURVIVED. Everything
    # above this line asserts on a pure function; nothing reached evaluate() or
    # problems(), so the two functions that turn derivations into a verdict could each be
    # replaced with a stub and this selftest stayed green. That is the same defect
    # tools/trycmd was written to expose in the other five oracles, found here in the
    # newest one within a day of it becoming a required gate domain.

    # The INLINE form. 15 of 17 ADRs write `Date: X. Status: Y.` on one line, and a
    # literal 0x08 byte in this regex once reported 6 of them as UNDECLARED while every
    # one declared a status. The multiline form above does not exercise this path.
    st, src = status_for("docs/adr/0002-x.md", "adr",
                         "Date: 2026-07-30. Status: superseded by 0003.\n", {}, "2026-07-30")
    check("adr status from the INLINE form", src, "header-inline")
    check("adr inline status value", st.startswith("superseded"), True)

    # evaluate() is what makes DOCMAP.md a projection rather than a wish. Its contract is
    # three keys, and `rendered` versus `on_disk` is the whole drift comparison.
    state = evaluate(project)
    check("evaluate returns a populated inventory", len(state["docs"]) > 100, True)
    check("evaluate renders the map", state["rendered"].startswith("#"), True)
    # NOT asserted here: that rendered == on_disk. That is a property of the WORKING TREE,
    # not of this code, and asserting it made the selftest fail whenever the map was stale,
    # which broke the unit domain and the trycmd CLI snapshot within an hour of being added.
    # A selftest tests logic; `check` tests the tree. Drift detection is covered below by
    # feeding a synthetic on_disk instead.

    # problems() turns state into what a caller acts on. It reads status off the Doc
    # objects, so a synthetic undeclared document has to be a Doc, not a path string.
    docs = state["docs"]
    broken = dict(state, docs=list(docs) + [dc_replace(
        docs[0], path="docs/INVENTED-NOT-ON-DISK.md", status="UNDECLARED")])
    check("an undeclared document becomes a problem",
          any("INVENTED-NOT-ON-DISK" in msg for msg in problems(broken)), True)
    check("a drifted map becomes a problem",
          any("not what the documents imply" in msg
              for msg in problems(dict(state, on_disk="hand edited"))), True)
    # A clean state produces no problems. Built synthetically from the live docs by
    # repairing every bad status in memory and marking the map fresh, so this asserts
    # on problems()'s logic rather than on the working tree. Asserting the live tree
    # here was the same defect the comment above records for rendered == on_disk: a
    # selftest tests logic, `check` tests the tree, and the tree assertion broke the
    # falsifiability baseline on 2026-10-10 (stale DOCMAP.md, 9 undeclared documents,
    # 1 expired prior-art record), failing every PR's mutate step for the six weeks
    # the tree had drifted.
    repaired = [
        dc_replace(d, status="active")
        if d.status in ("UNDECLARED", "MALFORMED") or d.status.startswith("EXPIRED")
        else d
        for d in docs
    ]
    clean = dict(state, docs=repaired, on_disk=state["rendered"])
    check("a clean state produces no problems", problems(clean), [])

    for f in fails:
        print("FAIL " + f)
    print("docmap selftest: {} checks failed".format(len(fails)))
    return len(fails)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("check", "write", "selftest"))
    ap.add_argument("--project", type=Path, default=Path.cwd())
    a = ap.parse_args()
    project = a.project.resolve()

    if a.mode == "selftest":
        return 1 if selftest(project) else 0

    state = evaluate(project)
    if a.mode == "write":
        (project / MAP_PATH).write_text(state["rendered"], encoding="utf-8")
        n = len(state["docs"])
        reached = sum(1 for d in state["docs"] if d.in_index)
        undeclared = sum(1 for d in state["docs"] if d.status == "UNDECLARED")
        print("wrote {} ({} documents, {} reachable from INDEX, {} undeclared)".format(
            MAP_PATH, n, reached, undeclared))
        return 0

    probs = problems(state)
    if not probs:
        print("docmap clean: {} documents, all with a derivable or declared status".format(
            len(state["docs"])))
        return 0
    for p in probs:
        print("FAIL " + p)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
