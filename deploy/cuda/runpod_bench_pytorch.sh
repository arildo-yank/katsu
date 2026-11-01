#!/usr/bin/env bash
# Katu × PyTorch no mesmo pod: mesmo modelo de 134M, mesmos lotes e mesmos pesos de partida.
# Roda dentro do pod, com a imagem runpod/pytorch, a partir da raiz do repositório:
#   OUT=/root/out ./deploy/cuda/runpod_bench_pytorch.sh
# Cada modo roda REPS vezes, em processos separados, porque os passos variam entre execuções.
set -euxo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-/root/out}
REPS=${REPS:-2}
STEPS=${STEPS:-100}
BASE=${BASE:-runs/bench-base}
TRAIN=data/sft/distill_arandu_10k.train.jsonl
EVAL=data/sft/distill_arandu_10k.eval.jsonl
mkdir -p "$OUT"
# Quem acompanha de fora lê este arquivo: pgrep -f encontraria a própria sessão SSH.
trap 'echo "$?" > "$OUT/EXIT_CODE"' EXIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$OUT/gpu.txt"
python3 -c "import torch; print(torch.__version__, torch.version.cuda)" >> "$OUT/gpu.txt"
git_rev=${GIT_REV:-desconhecido}
echo "katu $git_rev" >> "$OUT/gpu.txt"

if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
command -v cc >/dev/null || { apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config; }
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
cargo build --release --locked -p katu-server --features cuda --bin train_sft
BIN=target/release/train_sft

# Pesos sorteados com semente fixa: o mesmo ponto de partida em todos os modos. Mede velocidade, não qualidade.
export KATU_INIT_RANDOM=1 KATU_ENGINE=gpu KATU_BACKEND=cuda KATU_BATCH=64 KATU_FUSED=1
export KATU_MEMORY_BUDGET_MIB=${MEMORY_BUDGET_MIB:-22528}

katu() { # nome, kernel de GEMM, otimizador, repetição
  local run="$OUT/katu-$1-$4"
  KATU_GEMM=$2 KATU_OPTIM=$3 "$BIN" "$BASE" "$TRAIN" "$EVAL" "$TRAIN" "$run" "$STEPS" 0 0.0003 > "$run.log" 2>&1
  # O checkpoint do fim não serve aqui, e poucos enchem o disco do pod.
  find "$run" -name '*.ckpt' ! -name migrated.ckpt -delete
  [ "$4" = 0 ] || rm -f "$run/migrated.ckpt"
}

# A primeira execução grava os lotes e os pesos que todas as outras usam.
KATU_DUMP_BATCHES="$OUT/batches.jsonl" katu tiled64-dump tiled64 adam 0
WEIGHTS="$OUT/katu-tiled64-dump-0/migrated.ckpt"
cp "$BASE/config.json" "$OUT/katu-tiled64-dump-0/config.json" 2>/dev/null || true

for rep in $(seq 1 "$REPS"); do
  katu tiled64 tiled64 adam "$rep"
  katu tf32 tf32 adam "$rep"
  katu bf16 bf16 adam "$rep"
  katu bf16-adam8 bf16 adam8 "$rep"
  for mode in fp32 tf32 bf16; do
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python3 scripts/bench_pytorch.py "$OUT/batches.jsonl" --mode "$mode" --weights "$WEIGHTS" \
      --out "$OUT/torch-$mode-$rep.json" > "$OUT/torch-$mode-$rep.log" 2>&1
  done
done
rm -f "$OUT"/katu-*/*.ckpt
touch "$OUT/DONE"
