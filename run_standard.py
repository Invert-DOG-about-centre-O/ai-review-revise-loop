#!/usr/bin/env python3
"""Standard arm: history-aware reviewer, one reviewer per round. Author
writes v1, then K rounds of review -> revise on the same paper; from round 2
on the reviewer also sees every prior version + its review
(driver_lib.history_context()). Agent dir: local-rev-e1 (or
-a{author}-r{reviewer} if either model is overridden).

Usage: run_standard.py [--episodes S] [--rounds K]
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
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(lib.S_EPISODES):
        try:
            lib.run_episode(d, s, None, False, ledger,
                            author_model=args.author_model,
                            reviewer_model=args.reviewer_model,
                            reviewer_history=True)
            ml.log(f"standard: episode {s} done")
        except Exception as exc:
            ml.log(f"standard: episode {s} FAILED: {exc!r} — continuing "
                   f"to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"standard: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("standard: COMPLETE")


if __name__ == "__main__":
    main()
