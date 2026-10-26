#!/usr/bin/env bash
# Prepara um pod novo para o runpod_pretrain.sh sem subir o corpus do Mac (o envio Mac→pod mediu
# ~80 KB/s em 2026-10-27): baixa o dump da Wikipedia PT no próprio pod, confere o SHA-256, monta e
# tokeniza o corpus e confere os .bin contra os hashes gerados no Mac.
#
#   ./deploy/cuda/runpod_pretrain_setup.sh
#
# Antes, suba (scp) e extraia nesta pasta o código (git archive) e o pacote pequeno de dados:
#   data/corpus-bilingual-v1/{manifest.json,documents/<obras pt>}
#   data/sft/distill_arandu_50k.{train,eval}.jsonl
#   runs/hostinger-sft-134m-20261025/{config.json,tokenizer.bpe}
#   data/pretrain/SHA256SUMS   (train.bin e val.bin montados no Mac com os mesmos scripts)
set -euo pipefail
cd "$(dirname "$0")/../.."
DUMP=ptwiki-20251229-cirrussearch-content.json.gz
DUMP_SHA256=09900536de15513f4f8bd6f3f6f5ec637ccd032bc298c2c548138f593d8dd620
# Espelho da lista oficial da Wikimedia: ~66 MB/s no pod, contra ~1,5 MB/s em dumps.wikimedia.org.
MIRROR=${MIRROR:-https://mirror.accum.se/mirror/wikimedia.org/other/cirrussearch/20251229}
TOKENIZER=${TOKENIZER:-runs/hostinger-sft-134m-20261025/tokenizer.bpe}

[[ -f data/pretrain/SHA256SUMS ]] || { echo "falta data/pretrain/SHA256SUMS (do Mac)" >&2; exit 1; }
mkdir -p data/pretrain/raw
if ! echo "$DUMP_SHA256  data/pretrain/raw/$DUMP" | sha256sum -c --status 2>/dev/null; then
  curl -fsS -o "data/pretrain/raw/$DUMP" "$MIRROR/$DUMP"
  echo "$DUMP_SHA256  data/pretrain/raw/$DUMP" | sha256sum -c
fi

if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
cargo build --release --locked -p katsu-server --features cuda --bin train_sft --bin tokenize_corpus &
build=$!

# O limite real de CPU do pod vem do cgroup (ex.: 10,2 vCPUs), não do nproc.
workers=$(awk '{ if ($1 == "max") print 8; else print int($1 / $2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 8)
python3 scripts/build_pretrain_corpus.py --wiki "data/pretrain/raw/$DUMP" --out data/pretrain --workers "$workers"
wait "$build"
target/release/tokenize_corpus "$TOKENIZER" data/pretrain/train.jsonl data/pretrain/train.bin
target/release/tokenize_corpus "$TOKENIZER" data/pretrain/val.jsonl data/pretrain/val.bin
(cd data/pretrain && sha256sum -c SHA256SUMS)
echo "corpus pronto e igual ao do Mac"
