#!/usr/bin/env python3
"""05-revision-loop: flexible/advanced CLI for one-off variants that don't
warrant their own dedicated script — cross-model pairings,
--lenient-on-execution, --tag (e.g. seeding from another run's v1/round1
files), e2 (lessons channel). For the 8 main experimental arms (2 reviewer
counts x history-aware/blind x lessons/no-lessons), use the dedicated
run_N{1|4}_{HA|HB}_{L|NL}.py scripts instead — see driver_lib.py's
docstring for the full list.

All share the same engine (driver_lib.py) — see its docstring for the full
list of deviations from the stock research-loop and the resumability
model.

E1 (arm e1): author writes an empirical paper, then K rounds of
review -> revise on the SAME paper. Reviewer rating recorded every round
(plus a final review of the last version). No lessons. S episodes.

E2 (arm e2): E1 + the lessons channel — each round's review is distilled
into lessons.md (stock distill prompt), lessons injected into subsequent
authoring AND revise prompts (stock header). Episodes share lessons, so
later episodes should start higher / improve faster.

--lenient-on-execution adds EXECUTION_LENIENCY_NOTE to the reviewer prompt.
Originally written for when the author had no compute at all; now that
author/revise actually execute code, it's mostly stale (see the note's own
docstring in driver_lib.py) — kept opt-in for the case where execution
genuinely wasn't feasible and the author said so honestly, writes to a
separate -lenient agent dir so default runs stay untouched and directly
comparable.

--author-model / --reviewer-model override the claude -p model per role
(default: sonnet for both); non-default combos get a distinguishing agent
dir suffix (e.g. local-rev-e1-ahaiku-rsonnet).

Usage:  driver.py {e1|e2} [--gpu N] [--lenient-on-execution]
                  [--author-model M] [--reviewer-model M]
"""
import argparse

import driver_lib as lib
from driver_lib import ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["e1", "e2"])
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--episodes", type=int, default=None,
                    help="override S_EPISODES (default: full design value)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override K_ROUNDS (default: full design value)")
    ap.add_argument("--lenient-on-execution", action="store_true",
                    help="add EXECUTION_LENIENCY_NOTE to the reviewer prompt "
                         "(runs to a separate -lenient agent dir, so it "
                         "stays directly comparable to the default arm)")
    ap.add_argument("--author-model", default="sonnet",
                    help="claude -p model for author_with_tools/"
                         "revise_with_tools/enforce_char_limit/distill "
                         "(default: sonnet)")
    ap.add_argument("--reviewer-model", default="sonnet",
                    help="claude -p model for review_with_tools "
                         "(default: sonnet)")
    ap.add_argument("--no-reviewer-history", action="store_true",
                    help="reviewer does not see prior versions/reviews of "
                         "the paper it's reviewing (runs to a separate "
                         "-nohistory agent dir) — prefer run_N1_HB_NL.py "
                         "unless you also need another flag here")
    ap.add_argument("--tag", default=None,
                    help="extra suffix appended to the agent dir name, for "
                         "variants that don't fit the other flags (e.g. a "
                         "run seeded from another run's v1/round1 files)")
    ap.add_argument("--n-reviewers", type=int, default=1,
                    help="if >1, use N independent no-history reviewers per "
                         "round instead of one — prefer run_N4_HB_NL.py "
                         "unless you also need another flag here. "
                         "Incompatible with --lenient-on-execution, "
                         "--no-reviewer-history (always no-history in this "
                         "mode), and e2 lessons.")
    args = ap.parse_args()
    if args.n_reviewers > 1 and (args.lenient_on_execution or
                                 args.no_reviewer_history or args.arm == "e2"):
        ap.error("--n-reviewers >1 is incompatible with "
                 "--lenient-on-execution, --no-reviewer-history, and arm e2")
    if args.episodes is not None:
        lib.S_EPISODES = args.episodes
    if args.rounds is not None:
        lib.K_ROUNDS = args.rounds
    name = f"local-rev-{args.arm}"
    if args.lenient_on_execution:
        name += "-lenient"
    if args.author_model != "sonnet" or args.reviewer_model != "sonnet":
        name += f"-a{args.author_model}-r{args.reviewer_model}"
    if args.no_reviewer_history:
        name += "-nohistory"
    if args.n_reviewers > 1:
        name += f"-{args.n_reviewers}reviewers"
    if args.tag:
        name += f"-{args.tag}"
    d = lib.setup_local(name)
    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in range(lib.S_EPISODES):
        try:
            if args.n_reviewers > 1:
                lib.run_episode_multi_reviewer(d, s, args.gpu, ledger,
                                               args.n_reviewers,
                                               author_model=args.author_model,
                                               reviewer_model=args.reviewer_model)
            else:
                lib.run_episode(d, s, args.gpu, args.arm == "e2", ledger,
                                lenient=args.lenient_on_execution,
                                author_model=args.author_model,
                                reviewer_model=args.reviewer_model,
                                reviewer_history=not args.no_reviewer_history)
            ml.log(f"{args.arm}: episode {s} done")
        except Exception as exc:
            # One episode's failure (e.g. a subprocess timeout) shouldn't
            # cost the rest of a multi-hour run — every step is resumable,
            # so log and move on; rerun the same command to retry episode s.
            ml.log(f"{args.arm}: episode {s} FAILED: {exc!r} — continuing "
                   f"to next episode (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"{args.arm}: COMPLETE with failures in episodes {failed}")
    else:
        ml.log(f"{args.arm}: COMPLETE")


if __name__ == "__main__":
    main()
