#!/usr/bin/env python3
"""Multi-reviewer arm: N independent no-history reviewers per round (each
reads only the current version, with no knowledge of the others or of prior
rounds); the author revises against all N reviews at once
(driver_lib.run_episode_multi_reviewer()). Agent dir:
local-rev-e1-{N}reviewers (or -a{author}-r{reviewer}-{N}reviewers if either
model is overridden).

Usage: run_multireviewer.py [--episodes S] [--rounds K] [--n-reviewers N]
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
    ap.add_argument("--n-reviewers", type=int, default=4)
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
    name += f"-{args.n_reviewers}reviewers"
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(lib.S_EPISODES):
        try:
            lib.run_episode_multi_reviewer(d, s, None, ledger,
                                           args.n_reviewers,
                                           author_model=args.author_model,
                                           reviewer_model=args.reviewer_model)
            ml.log(f"multireviewer: episode {s} done")
        except Exception as exc:
            ml.log(f"multireviewer: episode {s} FAILED: {exc!r} — "
                   f"continuing to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"multireviewer: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("multireviewer: COMPLETE")


if __name__ == "__main__":
    main()
