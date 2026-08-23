#!/usr/bin/env python3
"""Multi-reviewer-with-history arm: N reviewers per round, each independent
of the others (no shared context, ever), but each tracks ITS OWN review
history across rounds (driver_lib.reviewer_history_context()) — every prior
version of the paper and that same reviewer's own past reviews of it.
Isolates "does per-reviewer memory help" from run_multireviewer.py's
"does averaging N blind independent judges help."

Seeds from an existing multi-reviewer run's v1.md + round1_review_{i}.json
(default: local-rev-e1-4reviewers) so round 1 is identical across both
arms — a controlled comparison, same trick as local-rev-e1-seed-nohistory-v1
for the single-reviewer arms. Reviewer history only starts mattering from
round 2 onward (reviewer_history_context() returns "" for k=1 regardless).

Agent dir: local-rev-e1-{N}reviewers-history (or -a{author}-r{reviewer}
suffix if either model is overridden).

Usage: run_multireviewer_history.py [--episodes S] [--rounds K]
                                    [--n-reviewers N] [--seed-from DIR]
                                    [--author-model M] [--reviewer-model M]
"""
import argparse
import shutil

import driver_lib as lib
from driver_lib import ml


def seed_from(src_dir, dst_dir, n_reviewers, episodes):
    """Copy v1.md + round1_review_1..N.json for each paper-s from src_dir
    into dst_dir, so round 1 starts identical across both arms. Also
    back-fills the round-1 ledger rows (matching what run_episode_multi_
    reviewer_history would have recorded, had it generated round 1 itself)
    so rounds.jsonl stays complete for later analysis/plotting."""
    ledger = dst_dir / "data/autoresearch/rounds.jsonl"
    existing = set()
    if ledger.exists():
        import json
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add((row.get("episode"), row.get("round"), row.get("reviewer")))
    for s in range(episodes):
        src_pdir = src_dir / f"data/autoresearch/paper-{s}"
        dst_pdir = dst_dir / f"data/autoresearch/paper-{s}"
        v1 = src_pdir / "v1.md"
        if not v1.exists():
            continue
        dst_pdir.mkdir(parents=True, exist_ok=True)
        if not (dst_pdir / "v1.md").exists():
            shutil.copy(v1, dst_pdir / "v1.md")
        for i in range(1, n_reviewers + 1):
            src_rev = src_pdir / f"round1_review_{i}.json"
            dst_rev = dst_pdir / f"round1_review_{i}.json"
            if src_rev.exists() and not dst_rev.exists():
                shutil.copy(src_rev, dst_rev)
                if (s, 1, i) not in existing:
                    import json as _json
                    review = _json.loads(src_rev.read_text(encoding="utf-8"))
                    lib.record(ledger, episode=s, round=1, reviewer=i,
                              version="v1.md", rating=review.get("rating"),
                              confidence=review.get("confidence"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None,
                    help="override S_EPISODES (default: full design value)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override K_ROUNDS (default: full design value)")
    ap.add_argument("--n-reviewers", type=int, default=4)
    ap.add_argument("--seed-from", default="local-rev-e1-4reviewers",
                    help="agent dir to seed v1.md/round1_review_*.json from "
                         "(default: local-rev-e1-4reviewers)")
    ap.add_argument("--author-model", default="sonnet")
    ap.add_argument("--reviewer-model", default="sonnet")
    args = ap.parse_args()
    if args.episodes is not None:
        lib.S_EPISODES = args.episodes
    if args.rounds is not None:
        lib.K_ROUNDS = args.rounds

    name = "local-rev-e1"
    if args.author_model != "sonnet" or args.reviewer_model != "sonnet":
        name += f"-a{args.author_model}-r{args.reviewer_model}"
    name += f"-{args.n_reviewers}reviewers-history"
    d = lib.setup_local(name)
    src = lib.PAPERENA_REPO_LOCAL / "agents" / args.seed_from
    if not src.exists():
        raise SystemExit(f"--seed-from dir not found: {src}")
    seed_from(src, d, args.n_reviewers, lib.S_EPISODES)

    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(lib.S_EPISODES):
        try:
            lib.run_episode_multi_reviewer_history(d, s, None, ledger,
                                                    args.n_reviewers,
                                                    author_model=args.author_model,
                                                    reviewer_model=args.reviewer_model)
            ml.log(f"multireviewer-history: episode {s} done")
        except Exception as exc:
            ml.log(f"multireviewer-history: episode {s} FAILED: {exc!r} — "
                   f"continuing to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"multireviewer-history: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("multireviewer-history: COMPLETE")


if __name__ == "__main__":
    main()
