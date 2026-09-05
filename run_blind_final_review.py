#!/usr/bin/env python3
"""One-off evaluation, not part of the main S/K design: N independent BLIND
reviews of the FINAL (v4) paper from each episode of the history-aware
4-reviewer arm (default source: local-rev-e1-4reviewers-history).

"Blind" here means: no revision history, no prior reviews, no knowledge that
this was ever revised — each reviewer sees v4.md as if it were a fresh v1
submission, exactly like round-1 review mechanics in the main loop. This is
different from that arm's own round4_review_*.json files, which ARE
history-aware (the reviewer_history_context() the arm is named for) and were
generated seeing all of rounds 1-3.

For each source paper-s, copies v4.md (renamed v1.md) plus every non-review,
non-version supporting file (experiment code, data, logs) into a fresh
isolated directory under a new agent dir, so a reviewer can still verify
results but can't see the original review/revision trail. Then runs
--n-reviewers independent parallel review_with_tools(k=1) calls against it.

Usage: run_blind_final_review.py [--source DIR] [--n-reviewers N]
                                  [--episodes S] [--reviewer-model M]
Agent dir: <source>-blind-final-review
"""
import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import driver_lib as lib
from driver_lib import ml, review_with_tools, record


def seed_final(src_dir, dst_dir, episodes):
    """Copy each paper-s's v4.md (as v1.md) plus all non-review, non-version
    files into dst_dir/paper-s, skipping papers whose v4.md doesn't exist."""
    seeded = []
    for s in range(episodes):
        src_pdir = src_dir / f"data/autoresearch/paper-{s}"
        v4 = src_pdir / "v4.md"
        if not v4.exists():
            ml.log(f"blind-final-review: paper-{s} has no v4.md, skipping")
            continue
        dst_pdir = dst_dir / f"data/autoresearch/paper-{s}"
        dst_pdir.mkdir(parents=True, exist_ok=True)
        if not (dst_pdir / "v1.md").exists():
            shutil.copy(v4, dst_pdir / "v1.md")
        for item in src_pdir.iterdir():
            if item.name == "__pycache__":
                continue
            if item.name.startswith("v") and item.name.endswith(".md"):
                continue  # v1-v4.md — only the renamed v4 copy belongs here
            if item.name.startswith("round") and "review" in item.name:
                continue  # exclude every prior review — must stay blind
            dst = dst_pdir / item.name
            if dst.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy(item, dst)
        seeded.append(s)
    return seeded


def review_one_paper(pdir, n_reviewers, reviewer_model, ledger, s):
    review_names = [f"blind_review_{i}.json" for i in range(1, n_reviewers + 1)]
    to_generate = [(i, name) for i, name in enumerate(review_names, start=1)
                   if not (pdir / name).exists()]
    if to_generate:
        ml.log(f"{pdir.name}: generating {len(to_generate)} blind final "
               f"review(s) in parallel")
        with ThreadPoolExecutor(max_workers=len(to_generate)) as ex:
            futures = {
                ex.submit(review_with_tools, pdir, 1, model=reviewer_model,
                          out_name=name, parallel=True): (i, name)
                for i, name in to_generate}
            wait(futures)
            errors = []
            for fut, (i, name) in futures.items():
                exc = fut.exception()
                if exc is not None:
                    errors.append((i, exc))
                    continue
                review = fut.result()
                (pdir / name).write_text(json.dumps(review, indent=2), encoding="utf-8")
                record(ledger, paper=s, reviewer=i, version="v4-blind",
                       rating=review.get("rating"), confidence=review.get("confidence"))
            if errors:
                i, exc = errors[0]
                raise RuntimeError(f"{pdir.name}: reviewer {i} failed: {exc!r}") from exc
    return [json.loads((pdir / name).read_text(encoding="utf-8"))
            for name in review_names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="local-rev-e1-4reviewers-history",
                    help="agent dir to pull v4.md from")
    ap.add_argument("--n-reviewers", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=10,
                    help="how many paper-s (0..episodes-1) to look for")
    ap.add_argument("--reviewer-model", default="sonnet")
    args = ap.parse_args()

    src = lib.PAPERENA_REPO_LOCAL / "agents" / args.source
    if not src.exists():
        raise SystemExit(f"--source dir not found: {src}")

    d = lib.setup_local(f"{args.source}-blind-final-review")
    seeded = seed_final(src, d, args.episodes)

    ledger = d / "data/autoresearch/rounds.jsonl"
    failed = []
    for s in seeded:
        pdir = d / f"data/autoresearch/paper-{s}"
        try:
            reviews = review_one_paper(pdir, args.n_reviewers,
                                       args.reviewer_model, ledger, s)
            ratings = [r.get("rating") for r in reviews]
            ml.log(f"blind-final-review: paper-{s} ratings {ratings} "
                   f"(mean {sum(ratings)/len(ratings):.1f})")
        except Exception as exc:
            ml.log(f"blind-final-review: paper-{s} FAILED: {exc!r} — "
                   f"continuing (rerun to retry this one)")
            failed.append(s)
    if failed:
        ml.log(f"blind-final-review: COMPLETE with failures in papers {failed}")
    else:
        ml.log("blind-final-review: COMPLETE")


if __name__ == "__main__":
    main()
