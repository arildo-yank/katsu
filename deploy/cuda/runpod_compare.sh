#!/usr/bin/env bash
# Benchmark híbrido × residente dentro de um pod (RunPod e afins), onde não há Docker: os mesmos
# passos do `hostinger_setup.sh --compare`, executados direto no container com CUDA.
#
#   ./deploy/cuda/runpod_compare.sh
#
# Precisa de: Linux, driver NVIDIA (nvidia-smi) e as bibliotecas CUDA 12 com NVRTC (imagens
# *-cuda-devel ou runpod/pytorch). Instala o Rust 1.88 em ~/.cargo se faltar.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ $(uname -s) == Linux ]] || { echo 'Requer Linux/NVIDIA.' >&2; exit 1; }
nvidia-smi
python3 scripts/verify_gpu_data.py
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
command -v cc >/dev/null || { apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config; }

export TOKENIZER=runs/bpe-arandu-v1/tokenizer.bpe
export TRAIN_DATA=data/corpus-bilingual-arandu-v1/sft/arandu_sft.jsonl
export EVAL_DATA=data/corpus-bilingual-arandu-v1/sft/arandu_eval.jsonl
export CORPUS=data/corpus-bilingual-arandu-v1/train.txt
python3 scripts/verify_gpu_data.py --selected
OUTPUT=${OUTPUT:-runs/compare-$(date -u +%Y%m%dT%H%M%SZ)}

cargo test --release --locked -p katsu-core --features cuda --test cuda -- --include-ignored
cargo test --release --locked -p katsu-server --features cuda --bin train_sft

# Pesos aleatórios com o tokenizer lacrado: o benchmark mede velocidade, não qualidade.
BASE="$OUTPUT-init"
python3 - "$TOKENIZER" "$BASE" <<'PY'
import json, shutil, struct, sys
from pathlib import Path
src, base = Path(sys.argv[1]), Path(sys.argv[2])
base.mkdir(parents=True, exist_ok=False)
data = src.read_bytes()
if data[:8] not in (b"KATSUBP1", b"KATSUBP2"):
    raise SystemExit("Tokenizer inválido")
vocab = 256 + struct.unpack_from("<I", data, 8)[0] + (4 if data[:8] == b"KATSUBP2" else 0)
shutil.copyfile(src, base / "tokenizer.bpe")
(base / "config.json").write_text(json.dumps(dict(vocab_size=vocab, block_size=256, embed_dim=768, layers=12, heads=12)))
PY
export KATSU_INIT_RANDOM=1 KATSU_BACKEND=cuda KATSU_BENCHMARK_ONLY=1 KATSU_CHECKPOINT_EVERY=1000
export KATSU_MEMORY_BUDGET_MIB=${MEMORY_BUDGET_MIB:-12288}
train() {
  cargo run --release --locked -p katsu-server --features cuda --bin train_sft -- \
    "$BASE" "$TRAIN_DATA" "$EVAL_DATA" "$CORPUS" "$1" 0 100 0.00005
}
KATSU_RESIDENT= train "$OUTPUT-hybrid"
KATSU_RESIDENT=1 train "$OUTPUT-resident"
python3 - "$OUTPUT-hybrid" "$OUTPUT-resident" <<'PY'
import json, statistics, sys
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(run + "/metrics.jsonl")][5:]  # sem os passos de aquecimento
    secs = statistics.median(r["step_seconds"] for r in rows)
    moved = [r["transfer_bytes"] for r in rows if r.get("transfer_bytes") is not None]
    gb = f"{statistics.median(moved) / 1e9:.2f} GB/passo (store)" if moved else "sem store"
    tokens = rows[0]["input_tokens"]
    print(f"{run}: mediana {secs:.3f} s/passo, {tokens / secs:.0f} tokens/s, {gb}")
PY
