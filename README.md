# claude-setup

**A personal agent OS you can inspect.**

![claude-setup](docs/hero.webp)

[![CI](https://github.com/Zion-sewers/claude-setup/actions/workflows/ship-gate.yml/badge.svg)](https://github.com/Zion-sewers/claude-setup/actions/workflows/ship-gate.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is one person's Claude configuration turned into a self-verifying control plane. Two halves:

- **Payload:** `dot-claude/` — 73+ authored skills, hooks, standing agent personas, slash commands (`/diverge`, `/commit-push-pr`), a committed copy of `~/.claude`. Plus `dot-codex/`, `dot-agents/`.
- **Harness:** `tools/` — a 12-domain quality gate, a dual-reviewer panel with agreement gates, a hash-chained intent/task ledger, mutation + dead-path audits, append-only state.

The organizing principle: **prose is not enforcement.** Every claim this repo makes about itself is checked by something that runs. The gate runs against the repo itself on every change.

## 60 seconds

```bash
git clone https://github.com/Zion-sewers/claude-setup && cd claude-setup
python3 tools/gate/gate.py run --project . -v     # the full 12-domain contract, on itself
python3 tools/gate/gate.py selftest               # every oracle's own selftest
python3 -m pytest tests/ -q
```

The honest demo is the gate passing on its own repo: a dotfiles repo that audits itself.

## What it does

- 12-domain quality gate (`tools/gate/gate.py`) with per-oracle selftests.
- Dual-reviewer panel (`tools/review`) with agreement gates.
- Hash-chained intent ledger (`tools/bus`) — history you cannot silently rewrite.
- Skill sync checks, mutation audits, ADRs and PRDs under `docs/`.

## Honest limits

- The `dot-*` trees are payload, not live config: editing `dot-claude/` changes nothing in a running session until deployed to `~/.claude`.
- Built around one person's machine and workflows. Not a general-purpose agent framework, and there is no `pip install` story for your setup.
- Wide parts matrix for a dotfiles repo: Python gate, Rust hot paths, Tauri dashboard.

---

Part of [Zion](https://github.com/Zion-sewers) — public infrastructure for trust in agents. The eval layer for the long-horizon era.
