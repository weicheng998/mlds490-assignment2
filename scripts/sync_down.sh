#!/usr/bin/env bash
# Pull logs/results (and small weight files, not replay buffers) from deepdish.
cd "$(dirname "$0")/.."
H=${DEEPDISH_HOST:?set DEEPDISH_HOST=user@host}
rsync -az $H:~/mlds_a2/results/ results/
rsync -az $H:~/mlds_a2/logs/ logs/
rsync -az --include '*/' --include 'best.pt' --include 'final.pt' --include 'config.json' --exclude '*' $H:~/mlds_a2/checkpoints/ checkpoints/
