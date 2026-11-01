#!/usr/bin/env bash
# Piloto da V2: 1B de parâmetros (d = 2048, 18 camadas, 16 cabeças, contexto 256) do zero, com 1B de
# tokens em português e inglês e, no mesmo run, o SFT de destilação. Motor inteiro na GPU, BF16 nos
# Tensor Cores, Adam FP32, batch 8: a configuração mais rápida medida numa RTX 4090 (5.990 tokens/s,
# 21.932 MiB; ARANDU/docs/ARANDU_V2_PLAN.md, 31/10/2026). Plano e custo: ARANDU/docs/PROMPT_TREINO_1B.md.
#
# Em duas etapas, para não pagar dois dias antes de ver uma hora:
#   GATE_HOURS=1 ./deploy/cuda/train_v2_1b.sh
#       roda a paridade, treina 1 h, grava latest.ckpt e sai com erro de propósito (teto de tempo).
#   KATU_RESUME=$RUNS/<run>/latest.ckpt GATE_HOURS=60 OUTPUT=$RUNS/<run>-2 ./deploy/cuda/train_v2_1b.sh
#       continua do mesmo passo até o fim.
#
# Antes, suba para o pod (rsync --partial), dentro desta pasta:
#   data/v2-1b/{manifest.json,shards/,SHA256SUMS}   ARANDU V2/corpus/v2-1b, sem o dedup.sqlite
#   data/pretrain/val.bin                           validação (só português; não há validação em inglês)
#   data/sft/distill_arandu_50k.{train,eval}.jsonl
#   runs/mac-variada-v1/tokenizer.bpe               o BPE com que o corpus foi tokenizado
#
# Os pesos e o checkpoint ficam em $RUNS, que precisa ser um volume: o Stop apaga o disco do container.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Requer Linux/NVIDIA.' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TOKENS=${TOKENS:-1000000000}
BATCH=${BATCH:-8}
GEMM=${GEMM:-bf16}
# O mesmo pico do pré-treino do 134M, que era em batch 64. Para este modelo e este batch, não foi calibrado.
CORPUS_LR=${CORPUS_LR:-0.0003}
SFT_LR=${SFT_LR:-0.0003}
SFT_EPOCHS=${SFT_EPOCHS:-3}
GATE_HOURS=${GATE_HOURS:-1}
MAX_HOURS=${MAX_HOURS:-$(( GATE_HOURS + 6 ))}
# Cada checkpoint tem ~12,5 GB (pesos + Adam). A cada 10.000 passos, uma queda perde menos de 1 h (~US$ 0,85).
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10000}
EVAL_SAMPLES=${EVAL_SAMPLES:-1000}
EVAL_WINDOWS=${EVAL_WINDOWS:-256}
MIN_FREE_GB=${MIN_FREE_GB:-40}
RUNS=${RUNS:-/workspace/runs}
CORPUS=${CORPUS:-data/v2-1b}
VAL=${VAL:-data/pretrain/val.bin}
DATA=${DATA:-data/sft/distill_arandu_50k}
TOKENIZER=${TOKENIZER:-runs/mac-variada-v1/tokenizer.bpe}
OUTPUT=${OUTPUT:-$RUNS/v2-1b-$(date -u +%Y%m%dT%H%M%SZ)}

for f in "$CORPUS/manifest.json" "$CORPUS/SHA256SUMS" "$VAL" "$DATA.train.jsonl" "$DATA.eval.jsonl" "$TOKENIZER"; do
  [[ -f "$f" ]] || { echo "falta $f (suba com rsync antes)" >&2; exit 1; }
done
[[ -e "$OUTPUT" ]] && { echo "$OUTPUT já existe" >&2; exit 1; }
mkdir -p "$RUNS"
free_gb=$(df -BG --output=avail "$RUNS" | tail -1 | tr -dc 0-9)
(( free_gb >= MIN_FREE_GB )) || { echo "$RUNS tem $free_gb GB livres; o run precisa de $MIN_FREE_GB" >&2; exit 1; }
(cd "$CORPUS" && sha256sum --quiet -c SHA256SUMS) || { echo "corpus com hash diferente do Mac" >&2; exit 1; }

if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
command -v cc >/dev/null || { apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config; }
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

adapt=$(( TOKENS / (BATCH * 256) ))
train=$(grep -c . "$DATA.train.jsonl")
steps=$(( (SFT_EPOCHS * train + BATCH - 1) / BATCH ))
echo "pré-treino: $adapt passos × $BATCH × 256 = $(( adapt * BATCH * 256 )) tokens (lr $CORPUS_LR, teto ${GATE_HOURS} h)"
echo "SFT: $steps passos ($SFT_EPOCHS épocas de $train, lr $SFT_LR) -> $OUTPUT"

# A paridade roda só na primeira etapa: se falhar, nada de treino pago.
if [[ -z ${KATU_RESUME:-} ]]; then
  mkdir -p "$RUNS/parity"
  {
    cargo test --release --locked -p katu-core --features cuda --lib gpu_train -- --nocapture
    cargo test --release --locked -p katu-core --features cuda --test gpu_train -- --nocapture --test-threads=1
  } 2>&1 | tee "$RUNS/parity/parity.txt"
  if grep -q "sem GPU" "$RUNS/parity/parity.txt"; then
    echo 'Testes pulados: a GPU não foi vista.' >&2
    exit 1
  fi
fi
cargo build --release --locked -p katu-server --features cuda --bin train_sft

BASE="$RUNS/v2-1b-base"
mkdir -p "$BASE"
python3 - "$TOKENIZER" "$BASE" <<'PY'
import json, shutil, struct, sys
from pathlib import Path
src, base = Path(sys.argv[1]), Path(sys.argv[2])
data = src.read_bytes()
if data[:8] not in (b"KATSUBP1", b"KATSUBP2"):
    raise SystemExit("Tokenizer inválido")
vocab = 256 + struct.unpack_from("<I", data, 8)[0] + (4 if data[:8] == b"KATSUBP2" else 0)
shutil.copyfile(src, base / "tokenizer.bpe")
(base / "config.json").write_text(json.dumps(dict(vocab_size=vocab, block_size=256, embed_dim=2048, layers=18, heads=16)))
PY

export KATU_ENGINE=gpu KATU_GEMM="$GEMM" KATU_BACKEND=cuda KATU_BATCH="$BATCH" KATU_INIT_RANDOM=1
export KATU_EVAL_SAMPLES="$EVAL_SAMPLES" KATU_CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
export KATU_CORPUS_LR="$CORPUS_LR" KATU_CORPUS_EVAL="$VAL" KATU_CORPUS_EVAL_WINDOWS="$EVAL_WINDOWS"
# A conta de orçamento do train_sft vale para host + GPU (pede 49.560 MiB neste modelo); como no probe, não limita aqui.
export KATU_DEADLINE_SECONDS=${GATE_SECONDS:-$(( GATE_HOURS * 3600 ))} KATU_MEMORY_BUDGET_MIB=${MEMORY_BUDGET_MIB:-999999}
status=0
timeout --signal=INT "${MAX_HOURS}h" target/release/train_sft \
  "$BASE" "$DATA.train.jsonl" "$DATA.eval.jsonl" "$CORPUS" "$OUTPUT" "$steps" "$adapt" "$SFT_LR" || status=$?
if [[ $status != 0 ]]; then
  echo "parou (status $status): se foi o teto de tempo, retome com KATU_RESUME=$OUTPUT/latest.ckpt e um OUTPUT novo" >&2
  exit $status
fi
(cd "$OUTPUT" && sha256sum model.ckpt pretrain-final.ckpt tokenizer.bpe config.json metrics.jsonl corpus-eval.jsonl sft-report.json > SHA256SUMS)
echo "pronto: $OUTPUT/sft-report.json"
