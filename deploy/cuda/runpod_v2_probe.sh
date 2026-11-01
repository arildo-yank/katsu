#!/usr/bin/env bash
# V2 na GPU: primeiro prova a correção dos kernels novos (BF16 nos Tensor Cores, Adam de 8 bits,
# recomputação de ativações) contra a referência em CPU; só se todos passarem, mede VRAM e
# tokens/s do 1B (d = 2048, 18 camadas, 16 cabeças, ~1,04B parâmetros), com pesos aleatórios e
# KATU_BENCHMARK_ONLY, contexto 256 como a medida FP32 de 28/10/2026 (runpod_1b_probe.sh).
#
#   ./deploy/cuda/runpod_v2_probe.sh        (no pod, na raiz do KATSU; mesmos arquivos de dados
#                                            que o runpod_1b_probe.sh)
#
# Saída em $OUT (padrão runs/probe-v2-<data>):
#   parity.txt     testes de paridade (KATU_PARITY_VERBOSE: diferença de cada tensor)
#   summary.jsonl  uma linha por configuração: s/passo, tokens/s, pico de VRAM, erro
# Os checkpoints de cada execução são apagados logo após a medida.
set -uo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Requer Linux/NVIDIA.' >&2; exit 1; }
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
command -v cc >/dev/null || { apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config; }

OUT=${OUT:-runs/probe-v2-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
TOKENIZER=runs/bpe-arandu-v1/tokenizer.bpe
TRAIN=data/corpus-bilingual-arandu-v1/sft/arandu_sft.jsonl
EVAL=data/corpus-bilingual-arandu-v1/sft/arandu_eval.jsonl

# 1. Correção. Um teste que falha encerra o script antes de qualquer medida.
set -e
{
  echo "== kernel adam8 contra adam8::apply da CPU (bit a bit)"
  cargo test --release --locked -p katu-core --features cuda --lib gpu_train -- --nocapture
  echo "== GpuTrainer contra o autograd da CPU (antigos + V2)"
  KATU_PARITY_VERBOSE=1 cargo test --release --locked -p katu-core --features cuda --test gpu_train -- --nocapture --test-threads=1
} 2>&1 | tee "$OUT/parity.txt"
grep -q "sem GPU" "$OUT/parity.txt" && { echo 'Testes pulados: a GPU não foi vista.' >&2; exit 1; }
cargo build --release --locked -p katu-server --features cuda --bin train_sft
set +e

probe() {  # rótulo largura camadas cabeças batch passos, e as opções em variáveis de ambiente
  local label=$1 d=$2 layers=$3 heads=$4 batch=$5 steps=$6
  local base="$OUT/base-$d-$layers" run="$OUT/run-$label-b$batch"
  if [[ ! -f $base/config.json ]]; then
    mkdir -p "$base"
    python3 - "$TOKENIZER" "$base" "$d" "$layers" "$heads" <<'PY'
import json, shutil, struct, sys
from pathlib import Path
src, base, d, layers, heads = Path(sys.argv[1]), Path(sys.argv[2]), *map(int, sys.argv[3:])
data = src.read_bytes()
if data[:8] not in (b"KATSUBP1", b"KATSUBP2"):
    raise SystemExit("Tokenizer inválido")
vocab = 256 + struct.unpack_from("<I", data, 8)[0] + (4 if data[:8] == b"KATSUBP2" else 0)
shutil.copyfile(src, base / "tokenizer.bpe")
(base / "config.json").write_text(json.dumps(dict(vocab_size=vocab, block_size=256, embed_dim=d, layers=layers, heads=heads)))
PY
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$run.mem" &
  local sampler=$!
  KATU_INIT_RANDOM=1 KATU_BACKEND=cuda KATU_ENGINE=gpu KATU_BENCHMARK_ONLY=1 KATU_BATCH=$batch \
    KATU_CHECKPOINT_EVERY=100000 KATU_MEMORY_BUDGET_MIB=999999 \
    timeout 1200 target/release/train_sft "$base" "$TRAIN" "$EVAL" "$TRAIN" "$run" 0 "$steps" 0.00005 > "$run.log" 2>&1
  local rc=$?
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  python3 - "$label" "$batch" "$rc" "$run" "${KATU_GEMM:-tiled64}" "${KATU_OPTIM:-adam}" "${KATU_RECOMPUTE:-0}" >> "$OUT/summary.jsonl" <<'PY'
import json, statistics, sys
from pathlib import Path
label, batch, rc, run, gemm, optim, recompute = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), *sys.argv[5:]
mem = [int(x) for x in Path(f"{run}.mem").read_text().split() if x.isdigit()]
row = {"config": label, "gemm": gemm, "optim": optim, "recompute": recompute == "1", "batch": batch,
       "exit_code": rc, "peak_vram_mib": max(mem) if mem else None}
config = run / "config.json"
if config.is_file():
    row["vram_after_alloc_mib"] = json.loads(config.read_text()).get("vram_used_after_alloc_mib")
metrics = run / "metrics.jsonl"
rows = [json.loads(l) for l in metrics.read_text().splitlines() if l.strip()] if metrics.is_file() else []
steady = rows[3:] or rows  # sem os primeiros passos (aquecimento)
if steady:
    secs = statistics.median(r["step_seconds"] for r in steady)
    tokens = sum(r.get("input_tokens", 0) for r in steady) / sum(r["step_seconds"] for r in steady)
    row.update(steps=len(rows), median_step_seconds=round(secs, 4), tokens_per_second=round(tokens),
               first_loss=rows[0].get("loss"), last_loss=rows[-1].get("loss"))
if rc != 0:
    row["error"] = Path(f"{run}.log").read_text()[-600:]
print(json.dumps(row, ensure_ascii=False))
PY
  tail -1 "$OUT/summary.jsonl"
  rm -f "$run"/*.ckpt
  return $rc
}

# 2. Medidas do 1B, contexto 256.
# Referência no mesmo pod: o motor como estava (FP32, Adam FP32, sem recomputação), batch 8.
probe fp32 2048 18 16 8 15
# Cada peça sozinha, batch 8, para separar o efeito de cada uma na memória e no tempo.
KATU_GEMM=bf16 probe bf16 2048 18 16 8 15
KATU_OPTIM=adam8 probe adam8 2048 18 16 8 15
KATU_RECOMPUTE=1 probe recompute 2048 18 16 8 15
# As três juntas, subindo o batch até não caber.
for batch in 1 8 16 32 64; do
  KATU_GEMM=bf16 KATU_OPTIM=adam8 KATU_RECOMPUTE=1 probe v2 2048 18 16 "$batch" 15 || break
done
echo "Resultado: $OUT/summary.jsonl e $OUT/parity.txt"
