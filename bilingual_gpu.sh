#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TOKENIZER=${TOKENIZER:-runs/bpe-arandu-v1/tokenizer.bpe}
export TRAIN_DATA=${TRAIN_DATA:-data/corpus-bilingual-arandu-v1/sft/arandu_sft.jsonl}
export EVAL_DATA=${EVAL_DATA:-data/corpus-bilingual-arandu-v1/sft/arandu_eval.jsonl}
export OUTPUT=${OUTPUT:-runs/hostinger-bilingual-$(date -u +%Y%m%dT%H%M%SZ)}
export STEPS=${STEPS:-500}
exec bash hostinger_setup.sh "${1:---preflight}"
