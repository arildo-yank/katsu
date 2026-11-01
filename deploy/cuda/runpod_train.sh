#!/usr/bin/env bash
# Treino SFT do 134M num pod com GPU (RunPod e afins). Por padrão usa o motor inteiro na GPU
# (ENGINE=gpu: forward, backward e Adam na VRAM, GEMM tiled64) com batch 64; ENGINE=hybrid volta
# ao caminho antigo (pesos residentes, demais operações na CPU).
#
#   EPOCHS=3 BATCH=64 LR=0.0003 MAX_HOURS=6 ./deploy/cuda/runpod_train.sh
#
# Batch 64 no 134M (256 tokens): ~17 GB de VRAM com pesos, gradientes, Adam e ativações; cabe
# nos 24 GB de uma RTX 4090. O trainer confere a VRAM livre antes de alocar e recusa se faltar.
# GEMM=tf32 usa os Tensor Cores (produtos em TF32, ~1e-3 relativo); GEMM=reference16 é o da Etapa 1.
#
# Prova de conceito (10k exemplos, gerados com scripts/sample_sft_subset.py), como rodou em 2026-10-26:
#   DATA=data/sft/distill_arandu_10k EPOCHS=1 BATCH=16 MAX_HOURS=3 OUTPUT=runs/poc-10k-b16 ./deploy/cuda/runpod_train.sh
# São 625 passos × 12,2 s ≈ 2 h 07 min, mais ~10 min medindo a loss em 3 × EVAL_SAMPLES antes do
# passo 1 e ~6 min depois do último. MAX_HOURS=2 cortaria perto do passo 540, sem sft-report.json.
#
# Antes, suba para o pod (scp), dentro desta pasta:
#   data/sft/distill_arandu_50k.train.jsonl e .eval.jsonl
#   runs/hostinger-sft-134m-20261025/{model.ckpt,config.json,tokenizer.bpe}
#
# Passos = ceil(EPOCHS × exemplos de treino ÷ BATCH). MAX_HOURS corta o treino pelo relógio para
# não estourar o orçamento; o latest.ckpt (a cada CHECKPOINT_EVERY passos) permite retomar com
# KATU_RESUME=<run>/latest.ckpt e um OUTPUT novo. Com batch 64 uma época tem 4× menos passos que
# com 16; a LR é um ponto de partida e não foi recalibrada para o batch maior.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Requer Linux/NVIDIA.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-64}
ENGINE=${ENGINE:-gpu}
GEMM=${GEMM:-tiled64}
LR=${LR:-0.0003}
MAX_HOURS=${MAX_HOURS:-6}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-200}
EVAL_SAMPLES=${EVAL_SAMPLES:-1000}
DATA=${DATA:-data/sft/distill_arandu_50k}
BASE=${BASE:-runs/hostinger-sft-134m-20261025}
OUTPUT=${OUTPUT:-runs/gpu-sft-134m-$(date -u +%Y%m%dT%H%M%SZ)}

for f in "$DATA.train.jsonl" "$DATA.eval.jsonl" "$BASE/model.ckpt" "$BASE/config.json" "$BASE/tokenizer.bpe"; do
  [[ -f "$f" ]] || { echo "falta $f (suba com scp antes)" >&2; exit 1; }
done
[[ -e "$OUTPUT" ]] && { echo "$OUTPUT já existe" >&2; exit 1; }
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

train=$(grep -c . "$DATA.train.jsonl")
eval=$(grep -c . "$DATA.eval.jsonl")
steps=$(( (EPOCHS * train + BATCH - 1) / BATCH ))
echo "motor $ENGINE ($GEMM), treino $train, validação $eval (medida em $EVAL_SAMPLES), $EPOCHS épocas, batch $BATCH, lr $LR: $steps passos, corte em ${MAX_HOURS} h -> $OUTPUT"

cargo build --release --locked -p katu-server --features cuda --bin train_sft
case "$ENGINE" in
  gpu) export KATU_ENGINE=gpu KATU_GEMM="$GEMM" ;;
  hybrid) export KATU_RESIDENT=1 ;;
  *) echo "ENGINE deve ser gpu ou hybrid" >&2; exit 1 ;;
esac
export KATU_BACKEND=cuda KATU_BATCH="$BATCH" KATU_EVAL_SAMPLES="$EVAL_SAMPLES"
export KATU_CHECKPOINT_EVERY="$CHECKPOINT_EVERY" KATU_MEMORY_BUDGET_MIB=${MEMORY_BUDGET_MIB:-16384}
status=0
timeout --signal=INT "${MAX_HOURS}h" target/release/train_sft \
  "$BASE" "$DATA.train.jsonl" "$DATA.eval.jsonl" "$DATA.train.jsonl" "$OUTPUT" "$steps" 0 "$LR" || status=$?
if [[ $status == 124 || $status == 130 ]]; then
  echo "corte por tempo (${MAX_HOURS} h): retome com KATU_RESUME=$OUTPUT/latest.ckpt e um OUTPUT novo" >&2
  exit 124
fi
[[ $status == 0 ]] || exit $status
(cd "$OUTPUT" && sha256sum model.ckpt tokenizer.bpe config.json metrics.jsonl sft-report.json > SHA256SUMS)
echo "pronto: $OUTPUT/sft-report.json"
