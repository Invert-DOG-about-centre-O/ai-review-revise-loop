#!/usr/bin/env python3
"""Standard + lessons arm (e2): history-aware reviewer, one reviewer per
round, PLUS the lessons channel — each round's review is distilled into
lessons.md (stock DISTILL_PROMPT) and injected into subsequent authoring
and revise prompts. Otherwise identical to run_N1_HA_NL.py: from round 2 on
the reviewer also sees every prior version + its review
(driver_lib.history_context()) on top of the lessons channel, so this arm
combines both history-awareness AND cross-episode lessons. Agent dir:
local-rev-e1-lessons (or -a{author}-r{reviewer} if either model is
overridden).

Usage: run_N1_HA_L.py [--episodes S] [--rounds K]
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
    name += "-lessons"
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(lib.S_EPISODES):
        try:
            lib.run_episode(d, s, None, True, ledger,
                            author_model=args.author_model,
                            reviewer_model=args.reviewer_model,
                            reviewer_history=True)
            ml.log(f"standard-lessons: episode {s} done")
        except Exception as exc:
            ml.log(f"standard-lessons: episode {s} FAILED: {exc!r} — "
                   f"continuing to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"standard-lessons: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("standard-lessons: COMPLETE")


if __name__ == "__main__":
    main()
