#!/usr/bin/env bash
# Push code to deepdish (never touches remote results/ or checkpoints/).
cd "$(dirname "$0")/.."
rsync -az --exclude .venv --exclude checkpoints --exclude .git --exclude results --exclude logs --exclude __pycache__ --exclude report \
  ./ "${DEEPDISH_HOST:?set DEEPDISH_HOST=user@host}":~/mlds_a2/
