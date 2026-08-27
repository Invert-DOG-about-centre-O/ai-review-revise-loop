#!/usr/bin/env python3
"""No-history arm: reviewer sees only the current version each round, never
prior versions or reviews of the same paper (driver_lib.history_context()
is skipped). Otherwise identical to run_N1_HA_NL.py. Agent dir:
local-rev-e1-nohistory (or -a{author}-r{reviewer}-nohistory if either model
is overridden).

Usage: run_N1_HB_NL.py [--episodes S] [--rounds K]
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
    name += "-nohistory"
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(lib.S_EPISODES):
        try:
            lib.run_episode(d, s, None, False, ledger,
                            author_model=args.author_model,
                            reviewer_model=args.reviewer_model,
                            reviewer_history=False)
            ml.log(f"nohistory: episode {s} done")
        except Exception as exc:
            ml.log(f"nohistory: episode {s} FAILED: {exc!r} — continuing "
                   f"to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"nohistory: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("nohistory: COMPLETE")


if __name__ == "__main__":
    main()
