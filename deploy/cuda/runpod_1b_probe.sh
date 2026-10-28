#!/usr/bin/env bash
# Mede se o 1B da V2 (d = 2048, 18 camadas, 16 cabeças, ~1,04B parâmetros) cabe e roda numa GPU com o
# motor GPU atual (FP32, Adam na GPU), sem treinar de verdade: pesos aleatórios e KATSU_BENCHMARK_ONLY.
# Usa a fase de corpus (janelas cheias de 256 tokens × batch), onde está o pico de memória.
# O 134M entra como controle (~24.000 tokens/s medidos na RTX 4090 em 26/10/2026).
#
#   ./deploy/cuda/runpod_1b_probe.sh        (no pod, na raiz do KATSU)
#
# Saída em $OUT (padrão runs/probe-1b-<data>): summary.jsonl, uma linha por configuração, com
# s/passo, tokens/s, pico de VRAM e o erro, se houver. Os checkpoints de cada execução são apagados
# logo após a medida (o do 1B tem ~4 GB).
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

OUT=${OUT:-runs/probe-1b-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$OUT"
TOKENIZER=runs/bpe-arandu-v1/tokenizer.bpe
TRAIN=data/corpus-bilingual-arandu-v1/sft/arandu_sft.jsonl
EVAL=data/corpus-bilingual-arandu-v1/sft/arandu_eval.jsonl

set -e
cargo test --release --locked -p katsu-core --features cuda --test cuda -- --include-ignored 2>&1 | tee "$OUT/cuda-tests.txt" | tail -3
cargo build --release --locked -p katsu-server --features cuda --bin train_sft
set +e

probe() {  # nome largura camadas cabeças batch passos
  local name=$1 d=$2 layers=$3 heads=$4 batch=$5 steps=$6
  local base="$OUT/base-$name" run="$OUT/run-$name-b$batch"
  rm -rf "$base" && mkdir -p "$base"
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
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$run.mem" &
  local sampler=$!
  KATSU_INIT_RANDOM=1 KATSU_BACKEND=cuda KATSU_ENGINE=gpu KATSU_BENCHMARK_ONLY=1 KATSU_BATCH=$batch \
    KATSU_CHECKPOINT_EVERY=100000 KATSU_MEMORY_BUDGET_MIB=999999 \
    timeout 1200 target/release/train_sft "$base" "$TRAIN" "$EVAL" "$TRAIN" "$run" 0 "$steps" 0.00005 > "$run.log" 2>&1
  local rc=$?
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null
  python3 - "$name" "$batch" "$rc" "$run" >> "$OUT/summary.jsonl" <<'PY'
import json, statistics, sys
from pathlib import Path
name, batch, rc, run = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
mem = [int(x) for x in Path(f"{run}.mem").read_text().split() if x.isdigit()]
row = {"model": name, "batch": batch, "exit_code": rc, "peak_vram_mib": max(mem) if mem else None}
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

probe 134m 768 12 12 16 20
for batch in 1 2 4 8; do
  probe 1b 2048 18 16 "$batch" 15 || break  # o primeiro batch que não cabe encerra a varredura
done
echo "Resultado: $OUT/summary.jsonl"
