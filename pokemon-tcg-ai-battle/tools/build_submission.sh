#!/usr/bin/env bash
# Pack an agent (main.py + deck.csv + cg engine) into submission.tar.gz.
# The cg/ engine is copied from cg-lib/cg (contains libcg.so for Linux x86_64,
# the platform the Kaggle ladder runs on).
#
# Usage: bash tools/build_submission.sh [agent_dir]   (default: agents/baseline)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT="${1:-$ROOT/agents/baseline}"
AGENT="$(cd "$AGENT" && pwd)"

if [ ! -f "$AGENT/main.py" ] || [ ! -f "$AGENT/deck.csv" ]; then
  echo "agent dir must contain main.py and deck.csv: $AGENT" >&2
  exit 1
fi
if [ ! -d "$ROOT/cg-lib/cg" ]; then
  echo "engine missing: $ROOT/cg-lib/cg" >&2
  exit 1
fi

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cp "$AGENT/main.py" "$TMP/main.py"
cp "$AGENT/deck.csv" "$TMP/deck.csv"
cp -r "$ROOT/cg-lib/cg" "$TMP/cg"

( cd "$TMP" && tar -czf "$AGENT/submission.tar.gz" . )
echo "Built: $AGENT/submission.tar.gz"
echo "---contents---"
tar -tzf "$AGENT/submission.tar.gz" | sort
