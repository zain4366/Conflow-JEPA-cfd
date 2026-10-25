#!/usr/bin/env bash
# Commit one case's out/ directory to the `results` branch (runs/<run_id>/<case> and latest/<case>).
# usage: publish.sh OUT_DIR CASE RUN_ID      (needs GH_TOKEN and GITHUB_REPOSITORY)
#        publish.sh OUT_DIR reference RUN_ID  -> reference/ (experimental data and manifest)
set -euo pipefail
OUT=$(realpath "$1"); CASE=$2; RUN=$3
URL="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
D=$(mktemp -d)
for attempt in $(seq 1 12); do
  rm -rf "$D/res"
  if git ls-remote --exit-code --heads "$URL" results >/dev/null 2>&1; then
    git clone -q --depth 1 -b results "$URL" "$D/res"
  else
    git init -q "$D/res" && git -C "$D/res" checkout -q -b results
    printf "# SU2 results\n\nWritten by the su2 workflow. SuperWing excerpts: CC BY-SA 4.0, Yang et al. (2025), huggingface.co/datasets/yunplus/SuperWing.\n" > "$D/res/README.md"
    git -C "$D/res" remote add origin "$URL"
  fi
  cd "$D/res"
  git config user.name "conflow-cfd-runner"; git config user.email "actions@users.noreply.github.com"
  if [ "$CASE" = "reference" ]; then DSTS="reference"; else DSTS="runs/$RUN/$CASE latest/$CASE"; fi
  for dst in $DSTS; do rm -rf "$dst"; mkdir -p "$(dirname "$dst")"; cp -r "$OUT" "$dst"; done
  git add -A
  git commit -q -m "SU2 $CASE, run $RUN" || { echo "nothing to commit"; exit 0; }
  if git push -q origin results; then echo "published $CASE"; exit 0; fi
  cd /; sleep $((RANDOM % 60 + 10 * attempt))
done
echo "publish failed after 12 attempts (the collect job still commits this case from its artifact)" >&2; exit 1
