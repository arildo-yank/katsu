#!/usr/bin/env bash
# SFT de contexto e recusa sobre o 134M já pré-treinado (1B de tokens), num pod novo. Roda no pod.
#
# No Mac, monte o pacote e suba para /root/up (o envio Mac→pod é o gargalo: ~1 MB/s em 2026-10-27):
#   code.tar.gz   git archive do KATSU
#   data.tar.gz   data/sft/$DATA.{train,eval}.jsonl e runs/pretrain-1b-base/{config.json,tokenizer.bpe}
#   model.ckpt    o adapted.ckpt do run gpu-pretrain-134m-1b (só pré-treino)
#   SHA256SUMS    dos três
# Depois, no pod:
#   mkdir -p /root/KATSU && tar -xzf /root/up/code.tar.gz -C /root/KATSU && /root/KATSU/deploy/cuda/runpod_context_sft.sh
#
# 1 época, batch 64, motor GPU em TF32. No Mac, scripts/pod_autopilot.py baixa o resultado e desliga o
# pod (com teto de custo).
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA=${DATA:-mix_context_general_v2}
# pasta do modelo de partida (config.json e tokenizer.bpe vêm no data.tar.gz; o model.ckpt, em /root/up)
BASE=${BASE:-runs/pretrain-1b-base}
OUTPUT=${OUTPUT:-runs/gpu-context-sft-v2}
LR=${LR:-0.0003}
BATCH=${BATCH:-64}
(cd /root/up && sha256sum -c SHA256SUMS)
tar -xzf /root/up/data.tar.gz
cp /root/up/model.ckpt "$BASE/model.ckpt"
[[ -e "$OUTPUT" ]] && { echo "$OUTPUT já existe" >&2; exit 1; }
if ! command -v cargo >/dev/null && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
cargo build --release --locked -p katu-server --features cuda --bin train_sft
train=$(grep -c . "data/sft/$DATA.train.jsonl")
steps=$(( (train + BATCH - 1) / BATCH ))
echo "SFT: $train exemplos, 1 época, batch $BATCH = $steps passos, lr $LR -> $OUTPUT"
export KATU_ENGINE=gpu KATU_GEMM=tf32 KATU_BACKEND=cuda KATU_BATCH="$BATCH" KATU_EVAL_SAMPLES=1000
export KATU_CHECKPOINT_EVERY=100 KATU_MEMORY_BUDGET_MIB=16384
target/release/train_sft "$BASE" "data/sft/$DATA.train.jsonl" "data/sft/$DATA.eval.jsonl" \
  "data/sft/$DATA.train.jsonl" "$OUTPUT" "$steps" 0 "$LR"
(cd "$OUTPUT" && sha256sum model.ckpt config.json tokenizer.bpe sft-report.json metrics.jsonl > SHA256SUMS)
echo "pronto: $OUTPUT/sft-report.json"
