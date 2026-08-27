#!/usr/bin/env python3
"""No-history + lessons arm (e2): reviewer sees only the current version
each round, never prior versions or reviews of the same paper
(driver_lib.history_context() skipped, same as run_N1_HB_NL.py), PLUS the
lessons channel — each round's review is distilled into lessons.md (stock
DISTILL_PROMPT) and injected into subsequent authoring and revise prompts.
Isolates the lessons-channel effect from reviewer history-awareness: the
reviewer itself never remembers anything about a specific paper, but the
author still accumulates general lessons across episodes. Agent dir:
local-rev-e1-nohistory-lessons (or -a{author}-r{reviewer}-nohistory-lessons
if either model is overridden).

Usage: run_N1_HB_L.py [--episodes S] [--rounds K]
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
    args = ap.parse_args()
    if args.episodes is not None:
        lib.S_EPISODES = args.episodes
    if args.rounds is not None:
        lib.K_ROUNDS = args.rounds

    name = "local-rev-e1"
    if args.author_model != "sonnet" or args.reviewer_model != "sonnet":
        name += f"-a{args.author_model}-r{args.reviewer_model}"
    name += "-nohistory-lessons"
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(lib.S_EPISODES):
        try:
            lib.run_episode(d, s, None, True, ledger,
                            author_model=args.author_model,
                            reviewer_model=args.reviewer_model,
                            reviewer_history=False)
            ml.log(f"nohistory-lessons: episode {s} done")
        except Exception as exc:
            ml.log(f"nohistory-lessons: episode {s} FAILED: {exc!r} — "
                   f"continuing to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"nohistory-lessons: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("nohistory-lessons: COMPLETE")


if __name__ == "__main__":
    main()
