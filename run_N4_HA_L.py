#!/usr/bin/env python3
"""N4_HA_L: N independent reviewers per round, each with its own
persistent per-paper review memory (driver_lib.reviewer_history_context(),
same as run_N4_HA_NL.py — never seeing the other N-1 reviewers' opinions,
past or present), PLUS a persistent lessons.md — same skills-style
mechanism as run_skills.py, not e2's per-round distill: after each paper's
FULL review-revise cycle completes, the author agent itself (real
Read/Write tool access) reflects on every version and every reviewer's
review of that paper and rewrites lessons.md
(driver_lib.reflect_and_update_skills()); the NEXT episode's author reads
it via its own Read tool before drafting v1
(driver_lib.author_with_tools(skills=True)). Lessons accumulate only once
per paper cycle, not every round. Combines N independent judges,
per-reviewer memory, AND cross-episode author lessons.

Seeds from an existing multi-reviewer run's v1.md + round1_review_{i}.json
(default: local-rev-e1-4reviewers), same as run_N4_HA_NL.py, so round 1 is
identical across the multi-reviewer arms.

Agent dir: local-rev-e1-{N}reviewers-history-lessons (or -a{author}-r-
{reviewer}-{N}reviewers-history-lessons if either model is overridden).

Usage: run_N4_HA_L.py [--episodes S] [--rounds K] [--n-reviewers N]
                      [--seed-from DIR]
                      [--author-model M] [--reviewer-model M]
"""
import argparse

import driver_lib as lib
from driver_lib import ml
from run_N4_HA_NL import seed_from


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5,
                    help="number of papers (default: 5)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override K_ROUNDS (default: full design value)")
    ap.add_argument("--n-reviewers", type=int, default=4)
    ap.add_argument("--seed-from", default="local-rev-e1-4reviewers",
                    help="agent dir to seed v1.md/round1_review_*.json from "
                         "(default: local-rev-e1-4reviewers)")
    ap.add_argument("--author-model", default="sonnet")
    ap.add_argument("--reviewer-model", default="sonnet")
    args = ap.parse_args()
    lib.S_EPISODES = args.episodes
    if args.rounds is not None:
        lib.K_ROUNDS = args.rounds

    name = "local-rev-e1"
    if args.author_model != "sonnet" or args.reviewer_model != "sonnet":
        name += f"-a{args.author_model}-r{args.reviewer_model}"
    name += f"-{args.n_reviewers}reviewers-history-lessons"
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
                                                    reviewer_model=args.reviewer_model,
                                                    skills_on=True)
            ml.log(f"N4_HA_L: episode {s} done")
        except Exception as exc:
            ml.log(f"N4_HA_L: episode {s} FAILED: {exc!r} — continuing to "
                   f"next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"N4_HA_L: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("N4_HA_L: COMPLETE")


if __name__ == "__main__":
    main()
