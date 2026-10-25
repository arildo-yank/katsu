#!/bin/bash
# Benchmark do motor GPU: 15 passos por configuração, sem medições de loss.
cd /root/katsu-dev
source $HOME/.cargo/env
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH CARGO_TARGET_DIR=/root/katsu-dev-target
nice -n 19 cargo build -q --release --locked -p katsu-server --features cuda --bin train_sft || exit 1
BIN=/root/katsu-dev-target/release/train_sft
run() { # nome batch gemm fused
  rm -rf runs/bench-$1
  ( while sleep 2; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits; done ) > runs/bench-$1.mem 2>/dev/null &
  mon=$!
  KATSU_ENGINE=gpu KATSU_BACKEND=cuda KATSU_BATCH=$2 KATSU_GEMM=$3 KATSU_FUSED=$4 KATSU_BENCHMARK_ONLY=1 KATSU_MEMORY_BUDGET_MIB=16384 \
    timeout 900 $BIN runs/base data/sft/10k.train.jsonl data/sft/10k.eval.jsonl data/sft/10k.train.jsonl runs/bench-$1 15 0 0.0003 > runs/bench-$1.log 2>&1
  kill $mon
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json,sys,collections
name,b,g,f=sys.argv[1:]
try: r=[json.loads(l) for l in open(f"runs/bench-{name}/metrics.jsonl")]
except Exception as e: print(name,"FALHOU",open(f"runs/bench-{name}.log").read()[-300:]); sys.exit()
s=r[3:]; T=sum(x["step_seconds"] for x in s)
peak=collections.defaultdict(int)
for l in open(f"runs/bench-{name}.mem"):
    p=l.split(",")
    if len(p)==2: peak[p[0].strip()]=max(peak[p[0].strip()],int(p[1]))
print(f"{name:22s} batch {b:>2} gemm {g:11s} fused {f}: {T/len(s):.3f} s/passo, {sum(x['input_tokens'] for x in s)/T:7.0f} tok/s reais, {sum(x['padded_tokens'] for x in s)/T:7.0f} tok/s c/ padding, VRAM por processo (MiB): {dict(peak)}")
PY
}
run s1-ref16-unfused-b64 64 reference16 0
run s2-ref16-fused-b64 64 reference16 1
run s3-tiled64-b64 64 tiled64 1
run s3b-tf32-b64 64 tf32 1
run s3-tiled64-b16 16 tiled64 1
