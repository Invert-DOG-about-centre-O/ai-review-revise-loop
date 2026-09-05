#!/usr/bin/env python3
"""N4_HB_L: N independent no-history reviewers per round (each reads only
the current version, with no knowledge of the others or of prior rounds —
same as run_N4_HB_NL.py), PLUS a persistent lessons.md — same skills-style
mechanism as run_skills.py, not e2's per-round distill: after each paper's
FULL review-revise cycle completes, the author agent itself (real
Read/Write tool access) reflects on every version and every reviewer's
review of that paper and rewrites lessons.md
(driver_lib.reflect_and_update_skills()); the NEXT episode's author reads
it via its own Read tool before drafting v1
(driver_lib.author_with_tools(skills=True)). Lessons accumulate only once
per paper cycle, not every round. Isolates the lessons-channel effect from
independent-reviewer aggregation.

Agent dir: local-rev-e1-{N}reviewers-lessons (or -a{author}-r{reviewer}-
{N}reviewers-lessons if either model is overridden; add a slug suffix if
--topic is set, so a custom-topic run never shares a directory — and
therefore never shares an in-progress lessons.md — with the default-topic
run).

Usage: run_N4_HB_L.py [--episodes S] [--rounds K] [--n-reviewers N]
                      [--topic TEXT] [--start-episode S]
                      [--author-model M] [--reviewer-model M]
"""
import argparse
import re

import driver_lib as lib
from driver_lib import ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5,
                    help="number of papers (default: 5)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override K_ROUNDS (default: full design value)")
    ap.add_argument("--n-reviewers", type=int, default=4)
    ap.add_argument("--topic", default=None,
                    help="override driver_lib.TOPICS with this single "
                         "topic for every episode (default: the standard "
                         "2-topic rotation)")
    ap.add_argument("--start-episode", type=int, default=0,
                    help="episode index to start from (default 0) — use "
                         "to resume after manually deleting/renumbering "
                         "earlier papers (e.g. to restore strict lessons.md "
                         "ordering after mid-run failures)")
    ap.add_argument("--author-model", default="sonnet")
    ap.add_argument("--reviewer-model", default="sonnet")
    args = ap.parse_args()
    lib.S_EPISODES = args.episodes
    if args.rounds is not None:
        lib.K_ROUNDS = args.rounds
    if args.topic:
        lib.TOPICS = [args.topic]

    name = "local-rev-e1"
    if args.author_model != "sonnet" or args.reviewer_model != "sonnet":
        name += f"-a{args.author_model}-r{args.reviewer_model}"
    name += f"-{args.n_reviewers}reviewers-lessons"
    if args.topic:
        slug = re.sub(r"[^a-z0-9]+", "-", args.topic.lower()).strip("-")
        name += f"-topic-{slug}"
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(args.start_episode, lib.S_EPISODES):
        try:
            lib.run_episode_multi_reviewer(d, s, None, ledger,
                                           args.n_reviewers,
                                           author_model=args.author_model,
                                           reviewer_model=args.reviewer_model,
                                           skills_on=True)
            ml.log(f"N4_HB_L: episode {s} done")
        except Exception as exc:
            ml.log(f"N4_HB_L: episode {s} FAILED: {exc!r} — "
                   f"continuing to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"N4_HB_L: COMPLETE with failures in episodes {failed}")
    else:
        ml.log("N4_HB_L: COMPLETE")


if __name__ == "__main__":
    main()
