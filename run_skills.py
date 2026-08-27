#!/usr/bin/env python3
"""Skills-loop arm: reviewer is no-history (single reviewer, sees only the
current version each round — same as run_N1_HB_NL.py), but the author
agent maintains a persistent lessons.md file across episodes. After each
paper's full review-revise cycle, the author agent itself (real Read/Write
tool access, not a separate distill call) reflects on every version and
review of that paper and rewrites lessons.md with general, reusable
lessons (driver_lib.reflect_and_update_skills()). The NEXT episode's
author reads lessons.md via its own Read tool before drafting v1
(driver_lib.author_with_tools(skills=True)).

NOTE: this reuses the exact same on-disk filename (lessons.md) as e2's
`lessons_on` channel elsewhere in this codebase — a different mechanism
(ml.chat distill call vs. the author's own Read/Write access). No
collision in practice since this script never sets lessons_on=True in the
same agent dir, but don't combine them.

Hypothesis: v1 (round-1) ratings trend upward across episodes as
lessons.md accumulates, even though the reviewer never sees history and
each paper is on its own topic.

Agent dir: local-rev-e1-skills (or -a{author}-r{reviewer}-skills if either
model is overridden).

Usage: run_skills.py [--episodes S] [--rounds K]
                     [--author-model M] [--reviewer-model M]
"""
import argparse

import driver_lib as lib
from driver_lib import ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None,
                    help="override S_EPISODES (default: full design value)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override K_ROUNDS (default: full design value)")
    ap.add_argument("--author-model", default="sonnet")
    ap.add_argument("--reviewer-model", default="sonnet")
    ap.add_argument("--start-episode", type=int, default=0,
                    help="episode index to start from (default 0) — use to "
                         "skip papers seeded in from elsewhere (e.g. paper-0 "
                         "copied in whole from another run, already fully "
                         "reviewed/reflected on)")
    args = ap.parse_args()
    if args.episodes is not None:
        lib.S_EPISODES = args.episodes
    if args.rounds is not None:
        lib.K_ROUNDS = args.rounds

    name = "local-rev-e1"
    if args.author_model != "sonnet" or args.reviewer_model != "sonnet":
        name += f"-a{args.author_model}-r{args.reviewer_model}"
    name += "-skills"
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(args.start_episode, lib.S_EPISODES):
        try:
            lib.run_episode(d, s, None, False, ledger,
                            author_model=args.author_model,
                            reviewer_model=args.reviewer_model,
                            reviewer_history=False, skills_on=True)
            ml.log(f"skills: episode {s} done")
        except Exception as exc:
            ml.log(f"skills: episode {s} FAILED: {exc!r} — continuing "
                   f"to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"skills: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("skills: COMPLETE")


if __name__ == "__main__":
    main()
