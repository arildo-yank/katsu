#!/usr/bin/env bash
# Execute no host Linux/NVIDIA, dentro de uma cópia do repositório KATSU.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:---help}
if [[ "$MODE" == --help ]]; then
  cat <<'HELP'
Uso: ./hostinger_setup.sh --preflight | --check | --benchmark | --compare | --run
Pré-requisitos do host: Linux, driver NVIDIA, Docker e NVIDIA Container Toolkit.
--preflight: verifica host e arquivos, sem instalar nem treinar.
--check: valida CPU×CUDA e executa 10 passos SFT do modelo 134M.
--benchmark: executa exatamente 100 passos no corpus; sem avaliação SFT.
--compare: testes CUDA e 100 passos no modo híbrido e no residente (pesos e Adam
  na VRAM, KATSU_RESIDENT=1), com tokens/s e GB por passo de cada um.
--run: valida CUDA, treina no corpus e depois faz SFT (piloto experimental).
Variáveis:
 TOKENIZER=runs/bpe-arandu-v1/tokenizer.bpe (caminho relativo ao repo)
 BASE= (checkpoint pré-treinado compatível; caminho relativo ao repo)
 TRAIN_DATA=data/corpus-bilingual-arandu-v1/sft/arandu_sft.jsonl EVAL_DATA=data/corpus-bilingual-arandu-v1/sft/arandu_eval.jsonl
 STEPS=500 PRETRAIN_STEPS=500 LR=0.00005 MEMORY_BUDGET_MIB=12288
 CORPUS=data/corpus-bilingual-arandu-v1/train.txt
 CHECKPOINT_EVERY=25 (pesos, Adam, RNG e fase/passo; gravação atômica)
 KATSU_WARMUP_RATIO=0.05 KATSU_MIN_LR_RATIO=0.1 KATSU_CORPUS_LR=0.0001
 KATSU_RESUME= (latest.ckpt da mesma configuração; OUTPUT deve ser novo)
 OUTPUT=runs/hostinger-sft-139m-<UTC>
 KATSU_RESIDENT=1 (pesos, gradientes e Adam na VRAM; KATSU_RESIDENT_REST=device
   mantém somas, atenção e entropia na GPU em vez do host)
Sem BASE, inicializa pesos aleatórios. --check é smoke; --run é um piloto de
pré-treinamento e SFT, não um modelo final. Quantidades devem caber no orçamento.
Perfil nominal 139M: 12 camadas, 768 dimensões, 12 heads, contexto 256.
A contagem real é registrada pelo treinador (embeddings não compartilhados).
HELP
  exit 0
fi
[[ "$MODE" == --preflight || "$MODE" == --check || "$MODE" == --benchmark || "$MODE" == --compare || "$MODE" == --run ]] || { echo 'Modo inválido' >&2; exit 2; }
[[ $(uname -s) == Linux ]] || { echo 'Requer host Linux/NVIDIA; não inicia treino no Mac.' >&2; exit 1; }
python3 scripts/verify_gpu_data.py
command -v docker >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi
docker info >/dev/null
TOKENIZER=${TOKENIZER:-runs/bpe-arandu-v1/tokenizer.bpe}
BASE=${BASE:-}
STEPS=${STEPS:-500}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-500}
CORPUS=${CORPUS:-data/corpus-bilingual-arandu-v1/train.txt}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-25}
if [[ "$MODE" == --check ]]; then STEPS=10; PRETRAIN_STEPS=0; fi
KATSU_BENCHMARK_ONLY=0
if [[ "$MODE" == --benchmark || "$MODE" == --compare ]]; then STEPS=0; PRETRAIN_STEPS=100; KATSU_BENCHMARK_ONLY=1; fi
KATSU_RESIDENT=${KATSU_RESIDENT:-}
KATSU_RESIDENT_REST=${KATSU_RESIDENT_REST:-}
COMPARE=0; [[ "$MODE" == --compare ]] && COMPARE=1
export KATSU_RESIDENT KATSU_RESIDENT_REST COMPARE
KATSU_WARMUP_RATIO=${KATSU_WARMUP_RATIO:-0.05}
KATSU_MIN_LR_RATIO=${KATSU_MIN_LR_RATIO:-0.1}
KATSU_CORPUS_LR=${KATSU_CORPUS_LR:-0.0001}
KATSU_EVAL_SAMPLES=${KATSU_EVAL_SAMPLES:-25}
KATSU_RESUME=${KATSU_RESUME:-}
export KATSU_WARMUP_RATIO KATSU_MIN_LR_RATIO KATSU_CORPUS_LR KATSU_EVAL_SAMPLES KATSU_RESUME
export KATSU_BENCHMARK_ONLY
LR=${LR:-0.00005}
MEMORY_BUDGET_MIB=${MEMORY_BUDGET_MIB:-12288}
OUTPUT=${OUTPUT:-runs/hostinger-sft-139m-$(date -u +%Y%m%dT%H%M%SZ)}
TRAIN_DATA=${TRAIN_DATA:-data/corpus-bilingual-arandu-v1/sft/arandu_sft.jsonl}
EVAL_DATA=${EVAL_DATA:-data/corpus-bilingual-arandu-v1/sft/arandu_eval.jsonl}
export TOKENIZER BASE STEPS LR MEMORY_BUDGET_MIB OUTPUT TRAIN_DATA EVAL_DATA PRETRAIN_STEPS CORPUS CHECKPOINT_EVERY
python3 scripts/verify_gpu_data.py --selected
python3 - <<'PY'
import os, json, math
from pathlib import Path
root=Path.cwd().resolve()
for key in ('TOKENIZER','BASE','OUTPUT','TRAIN_DATA','EVAL_DATA','CORPUS','KATSU_RESUME'):
    val=os.environ[key]
    if not val: continue
    p=Path(val)
    if p.is_absolute() or '..' in p.parts or not (root/p).resolve().is_relative_to(root):
        raise SystemExit(f'{key} deve permanecer dentro do repositório')
for name in (os.environ['TOKENIZER'],os.environ['TRAIN_DATA'],os.environ['EVAL_DATA']):
    if not Path(name).is_file(): raise SystemExit(f'Arquivo ausente: {name}')
if Path(os.environ['OUTPUT']).exists(): raise SystemExit('OUTPUT já existe')
if int(os.environ['STEPS'])<0 or int(os.environ['STEPS'])+int(os.environ['PRETRAIN_STEPS'])<=0: raise SystemExit('Passos inválidos')
if int(os.environ['PRETRAIN_STEPS'])<0 or int(os.environ['CHECKPOINT_EVERY'])<=0: raise SystemExit('Intervalos inválidos')
if int(os.environ['PRETRAIN_STEPS']) and not Path(os.environ['CORPUS']).is_file(): raise SystemExit('Corpus de treino ausente')
for key, maximum in [('KATSU_WARMUP_RATIO',1),('KATSU_MIN_LR_RATIO',1)]:
    value=float(os.environ[key])
    if not math.isfinite(value) or not 0<value<=maximum or (key=='KATSU_WARMUP_RATIO' and value==1): raise SystemExit('Scheduler inválido')
if os.environ['KATSU_RESUME'] and not Path(os.environ['KATSU_RESUME']).is_file(): raise SystemExit('Checkpoint de retomada ausente')
lr=float(os.environ['LR'])
if not math.isfinite(lr) or lr<=0: raise SystemExit('LR inválida')
available=next(int(l.split()[1])//1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
budget=int(os.environ['MEMORY_BUDGET_MIB'])
if budget<8192 or available<budget+2048: raise SystemExit('Memória host insuficiente: reserve >=8192 MiB e 2048 MiB de margem')
if os.environ['BASE']:
    b=Path(os.environ['BASE']);c=json.loads((b/'config.json').read_text())
    for k,v in dict(layers=12,embed_dim=768,heads=12,block_size=256).items():
        if c[k]!=v: raise SystemExit(f'BASE incompatível: {k}')
    if (b/'tokenizer.bpe').read_bytes()!=Path(os.environ['TOKENIZER']).read_bytes():
        raise SystemExit('BASE usa outro tokenizer: IDs incompatíveis; não migre só pelo tamanho')
    for n in ('model.ckpt','tokenizer.bpe'):
        if not (b/n).is_file(): raise SystemExit(f'BASE ausente: {n}')
PY
[[ "$MODE" != --preflight ]] || exit 0
docker build -t katsu-cuda -f deploy/cuda/Dockerfile .
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e CARGO_HOME=/tmp/cargo -e CARGO_TARGET_DIR=/workspace/target-hostinger \
  -e KATSU_WARMUP_RATIO -e KATSU_MIN_LR_RATIO -e KATSU_CORPUS_LR -e KATSU_EVAL_SAMPLES -e KATSU_RESUME \
  -e TOKENIZER -e BASE -e STEPS -e LR -e MEMORY_BUDGET_MIB -e OUTPUT -e TRAIN_DATA -e EVAL_DATA -e PRETRAIN_STEPS -e CORPUS -e CHECKPOINT_EVERY -e KATSU_BENCHMARK_ONLY \
  -e KATSU_RESIDENT -e KATSU_RESIDENT_REST -e COMPARE \
  -v "$PWD:/workspace" katsu-cuda bash -c '
set -euo pipefail
cargo test --release --locked -p katsu-core --features cuda --test cuda -- --include-ignored
cargo test --release --locked -p katsu-server --features cuda --bin train_sft
python3 - <<"PY"
import os,json,struct,shutil
from pathlib import Path
if not os.environ["BASE"]:
    p=Path(os.environ["OUTPUT"]+"-init");p.mkdir(parents=True,exist_ok=False)
    src=Path(os.environ["TOKENIZER"]);data=src.read_bytes()
    if data[:8] not in (b"KATSUBP1",b"KATSUBP2"): raise SystemExit("Tokenizer inválido")
    # Header has little-endian u32 merge count after magic.
    vocab=256+struct.unpack_from("<I",data,8)[0]+(4 if data[:8]==b"KATSUBP2" else 0)
    shutil.copyfile(src,p/"tokenizer.bpe")
    (p/"config.json").write_text(json.dumps(dict(vocab_size=vocab,block_size=256,embed_dim=768,layers=12,heads=12)))
PY
if [[ -z "$BASE" ]]; then BASE="${OUTPUT}-init"; export KATSU_INIT_RANDOM=1; fi
export KATSU_BACKEND=cuda KATSU_MEMORY_BUDGET_MIB="$MEMORY_BUDGET_MIB" KATSU_CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
train() {
  cargo run --release --locked -p katsu-server --features cuda --bin train_sft -- \
   "$BASE" "$TRAIN_DATA" "$EVAL_DATA" "$CORPUS" "$1" "$STEPS" "$PRETRAIN_STEPS" "$LR"
  (cd "$1" && sha256sum model.ckpt tokenizer.bpe config.json metrics.jsonl sft-report.json > SHA256SUMS)
}
if [[ "$COMPARE" == 1 ]]; then
  KATSU_RESIDENT= train "$OUTPUT-hybrid"
  KATSU_RESIDENT=1 train "$OUTPUT-resident"
  python3 - "$OUTPUT-hybrid" "$OUTPUT-resident" <<"PY"
import json,statistics,sys
for run in sys.argv[1:]:
    rows=[json.loads(l) for l in open(run+"/metrics.jsonl")][5:]  # sem os passos de aquecimento
    secs=statistics.median(r["step_seconds"] for r in rows)
    moved=[r["transfer_bytes"] for r in rows if r.get("transfer_bytes") is not None]
    gb=f"{statistics.median(moved)/1e9:.2f} GB/passo (store)" if moved else "sem store"
    tokens=rows[0]["input_tokens"]
    print(f"{run}: mediana {secs:.3f} s/passo, {tokens/secs:.0f} tokens/s, {gb}")
PY
else
  train "$OUTPUT"
fi
'
