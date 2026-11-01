#!/usr/bin/env bash
# Pré-treino do 134M do zero no corpus em português (Wikipedia PT + literatura em domínio público)
# e, no mesmo run, o SFT nos 50k de destilação. Motor inteiro na GPU (Tensor Cores TF32), batch 64
# janelas de 256 tokens = 16.384 tokens por passo.
#
#   TOKENS=900000000 PRETRAIN_HOURS=8 ./deploy/cuda/runpod_pretrain.sh
#
# Antes, suba para o pod (scp), dentro desta pasta, e confira os hashes:
#   data/pretrain/{train.bin,val.bin,SHA256SUMS}      (scripts/build_pretrain_corpus.py + tokenize_corpus)
#   data/sft/distill_arandu_50k.train.jsonl e .eval.jsonl
#   runs/hostinger-sft-134m-20261025/{config.json,tokenizer.bpe}   (só a arquitetura e o tokenizador:
#   os pesos começam aleatórios, KATU_INIT_RANDOM=1)
#
# Teto de custo: PRETRAIN_HOURS para a fase de corpus pelo relógio do processo
# (KATU_DEADLINE_SECONDS). Ao estourar, grava latest.ckpt e sai com erro, sem SFT; retome com
# KATU_RESUME=<run>/latest.ckpt e um OUTPUT novo. MAX_HOURS é o corte duro do processo inteiro.
# A loss do corpus de validação (janelas fixas) vai para <run>/corpus-eval.jsonl a cada checkpoint.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Requer Linux/NVIDIA.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TOKENS=${TOKENS:-900000000}
BATCH=${BATCH:-64}
GEMM=${GEMM:-tf32}
CORPUS_LR=${CORPUS_LR:-0.0003}
SFT_LR=${SFT_LR:-0.0003}
SFT_EPOCHS=${SFT_EPOCHS:-3}
PRETRAIN_HOURS=${PRETRAIN_HOURS:-8}
MAX_HOURS=${MAX_HOURS:-10}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
EVAL_SAMPLES=${EVAL_SAMPLES:-1000}
EVAL_WINDOWS=${EVAL_WINDOWS:-256}
CORPUS=${CORPUS:-data/pretrain}
DATA=${DATA:-data/sft/distill_arandu_50k}
BASE=${BASE:-runs/hostinger-sft-134m-20261025}
OUTPUT=${OUTPUT:-runs/gpu-pretrain-134m-$(date -u +%Y%m%dT%H%M%SZ)}

for f in "$CORPUS/train.bin" "$CORPUS/val.bin" "$CORPUS/SHA256SUMS" "$DATA.train.jsonl" "$DATA.eval.jsonl" \
  "$BASE/config.json" "$BASE/tokenizer.bpe"; do
  [[ -f "$f" ]] || { echo "falta $f (suba com scp antes)" >&2; exit 1; }
done
[[ -e "$OUTPUT" ]] && { echo "$OUTPUT já existe" >&2; exit 1; }
(cd "$CORPUS" && sha256sum -c SHA256SUMS) || { echo "corpus com hash diferente do Mac" >&2; exit 1; }
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

available=$(( $(stat -c %s "$CORPUS/train.bin") / 2 ))
(( TOKENS <= available )) || { echo "TOKENS=$TOKENS, mas o corpus tem $available" >&2; exit 1; }
adapt=$(( TOKENS / (BATCH * 256) ))
train=$(grep -c . "$DATA.train.jsonl")
steps=$(( (SFT_EPOCHS * train + BATCH - 1) / BATCH ))
echo "pré-treino: $adapt passos × $BATCH × 256 = $(( adapt * BATCH * 256 )) tokens de $available (lr $CORPUS_LR, teto ${PRETRAIN_HOURS} h)"
echo "SFT: $steps passos ($SFT_EPOCHS épocas de $train, lr $SFT_LR) -> $OUTPUT"

cargo build --release --locked -p katu-server --features cuda --bin train_sft
export KATU_ENGINE=gpu KATU_GEMM="$GEMM" KATU_BACKEND=cuda KATU_BATCH="$BATCH" KATU_INIT_RANDOM=1
export KATU_EVAL_SAMPLES="$EVAL_SAMPLES" KATU_CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
export KATU_CORPUS_LR="$CORPUS_LR" KATU_CORPUS_EVAL="$CORPUS/val.bin" KATU_CORPUS_EVAL_WINDOWS="$EVAL_WINDOWS"
export KATU_DEADLINE_SECONDS=$(( PRETRAIN_HOURS * 3600 )) KATU_MEMORY_BUDGET_MIB=${MEMORY_BUDGET_MIB:-16384}
status=0
timeout --signal=INT "${MAX_HOURS}h" target/release/train_sft \
  "$BASE" "$DATA.train.jsonl" "$DATA.eval.jsonl" "$CORPUS/train.bin" "$OUTPUT" "$steps" "$adapt" "$SFT_LR" || status=$?
if [[ $status != 0 ]]; then
  echo "parou (status $status): se foi o teto de tempo, retome com KATU_RESUME=$OUTPUT/latest.ckpt e um OUTPUT novo" >&2
  exit $status
fi
(cd "$OUTPUT" && sha256sum model.ckpt pretrain-final.ckpt tokenizer.bpe config.json metrics.jsonl corpus-eval.jsonl sft-report.json > SHA256SUMS)
echo "pronto: $OUTPUT/sft-report.json"
