#!/usr/bin/env python3
"""Plot mean +/- std of reviewer ratings per round, pooling across every
rating recorded for that round in rounds.jsonl. For a multi-reviewer run
this pools across BOTH episodes and reviewers — e.g. round 1 with 6 papers
x 4 reviewers pools all 24 ratings into one mean/variance point. For a
single-reviewer run it just pools across episodes (papers).

Usage:
    plot_scores.py AGENT_DIR [AGENT_DIR ...] [--out FILE.png]

AGENT_DIR may be a bare name (resolved under paperena-agent/agents/, same
as driver_lib.PAPERENA_REPO_LOCAL) or a full path. Multiple dirs are
plotted as separate lines on the same axes, for comparing arms.
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import driver_lib as lib


def load_ratings_by_round(agent_dir: Path):
    ledger = agent_dir / "data/autoresearch/rounds.jsonl"
    by_round = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_round.setdefault(row["round"], []).append(row["rating"])
    return by_round


def resolve(name: str) -> Path:
    p = Path(name)
    if "/" in name or "\\" in name:
        return p
    return lib.PAPERENA_REPO_LOCAL / "agents" / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_dirs", nargs="+")
    ap.add_argument("--out", default="scores.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in args.agent_dirs:
        d = resolve(name)
        by_round = load_ratings_by_round(d)
        rounds = sorted(by_round)
        means = [statistics.mean(by_round[r]) for r in rounds]
        stds = [statistics.stdev(by_round[r]) if len(by_round[r]) > 1 else 0.0
                for r in rounds]
        n = [len(by_round[r]) for r in rounds]
        label = f"{Path(name).name} (n={n[0]}/round)" if len(set(n)) == 1 \
            else f"{Path(name).name} (n={n})"
        ax.errorbar(rounds, means, yerr=stds, marker="o", capsize=4, label=label)
        for r, m, cnt in zip(rounds, means, n):
            print(f"{Path(name).name}: round {r}: mean={m:.2f} "
                  f"std={statistics.stdev(by_round[r]) if len(by_round[r]) > 1 else 0.0:.2f} "
                  f"n={cnt}")

    ax.set_xlabel("round")
    ax.set_ylabel("reviewer rating")
    ax.set_title("Mean reviewer rating per round (error bars = std dev)")
    ax.set_xticks(sorted({r for name in args.agent_dirs
                          for r in load_ratings_by_round(resolve(name))}))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
