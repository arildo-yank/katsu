#!/usr/bin/env bash
# Treino do 134M nos 50k de destilação numa RTX 4090: 3 épocas, batch 64, motor inteiro na GPU com
# Tensor Cores (GEMM=tf32), corte em 1 h. É o runpod_train.sh com essas escolhas fixadas.
#
#   ./deploy/cuda/runpod_train_50k.sh
#
# Antes, suba para o pod (scp), dentro desta pasta (o pod não alcança o Mac):
#   data/sft/distill_arandu_50k.train.jsonl e .eval.jsonl
#   runs/hostinger-sft-134m-20261025/{model.ckpt,config.json,tokenizer.bpe}
#
# Estimativa: ~45 mil exemplos de treino × 3 ÷ 64 ≈ 2.100 passos × 0,42 s ≈ 15 min, mais a
# compilação e a medida da loss de validação (1.000 exemplos) antes e depois. MAX_HOURS corta só o
# treino; o pod cobra também o envio, a compilação e o tempo até o Stop.
#
# KATU_RESIDENT=1 é o motor híbrido antigo (ENGINE=hybrid), não o motor GPU: não exporte aqui.
# Qualquer variável abaixo pode ser sobrescrita na chamada (ex.: MAX_HOURS=2 ./deploy/...).
set -euo pipefail
export DATA=${DATA:-data/sft/distill_arandu_50k}
export EPOCHS=${EPOCHS:-3}
export BATCH=${BATCH:-64}
export ENGINE=${ENGINE:-gpu}
export GEMM=${GEMM:-tf32}
export MAX_HOURS=${MAX_HOURS:-1}
export OUTPUT=${OUTPUT:-runs/gpu-sft-134m-50k-$(date -u +%Y%m%dT%H%M%SZ)}
exec "$(dirname "$0")/runpod_train.sh"
