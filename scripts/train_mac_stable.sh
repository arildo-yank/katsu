#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-4}"
cargo run --release -p katsu-core --example train_native --locked -- \
 --tokenizer runs/native-smoke-20261024/tokenizer.bpe \
 --corpus corpus/machado-dom-casmurro.txt --profile tiny --batch-size 2 \
 --steps 2000 --output "${1:-runs/mac-cpu-stable}"
