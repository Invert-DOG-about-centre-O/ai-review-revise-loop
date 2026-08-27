#!/usr/bin/env bash
# Standard pipeline: S=1, K=3, author=sonnet, reviewer=sonnet, reviewer
# history-aware (the default arm, no leniency/no-history/tag variants).
# No failure detection or retry logic — just launches the one standard
# call. Re-run the same command to resume a partial run.
#
# Requirements (checked below before launching):
#   - python3 (or python) on PATH
#   - claude CLI on PATH and already authenticated (`claude auth login`)
#   - a "paperena-agent" checkout next to this monkey-experiments checkout
#     (../../paperena-agent from repo root), or set PAPERENA_AGENT_DIR to
#     point at a different location.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "error: no python3/python found on PATH." >&2
    exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
    echo "error: 'claude' CLI not found on PATH. Install it and run" \
         "'claude auth login' first." >&2
    exit 1
fi

REPO_ROOT="$(cd ../../.. && pwd)"
PAPERENA_AGENT_DIR="${PAPERENA_AGENT_DIR:-$(cd "$REPO_ROOT/.." 2>/dev/null && pwd)/paperena-agent}"
if [ ! -d "$PAPERENA_AGENT_DIR/agents/local-rev-template" ]; then
    echo "error: paperena-agent checkout not found at" \
         "'$PAPERENA_AGENT_DIR' (expected agents/local-rev-template" \
         "inside it). Clone it next to this repo, or set" \
         "PAPERENA_AGENT_DIR to point at your checkout." >&2
    exit 1
fi
export PAPERENA_AGENT_DIR

"$PYTHON" run_N1_HA_NL.py --episodes 1 --rounds 3 \
    --author-model sonnet --reviewer-model sonnet
