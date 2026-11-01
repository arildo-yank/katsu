#!/usr/bin/env bash
# Espelha um run de treino de um pod (via SSH) para KATSU/runs/<nome>, a cada MIRROR_EVERY segundos
# (padrão 30), para o painel Dev mostrar a curva ao vivo e, se o collect_telemetry.py estiver
# rodando no pod, o hardware. Aponta o active-run.json para o espelho uma vez, no início; depois a
# escolha do run no painel é sua.
#
#   scripts/mirror_pod_run.sh root@IP PORTA /root/katu-gpu-v2/runs/gpu-sft-134m-... gpu-sft-4090
#
# Para quando o run tiver sft-report.json (fim) ou com Ctrl-C.
set -euo pipefail
[[ $# == 4 ]] || { echo "uso: $0 user@host porta pasta_remota nome_local" >&2; exit 2; }
host=$1 port=$2 remote=$3 name=$4
KATSU="$(cd "$(dirname "$0")/.." && pwd)"
local="$KATSU/runs/$name"
mkdir -p "$local"
ssh_() { ssh -i "${SSH_KEY:-$HOME/.ssh/arandu_gpu}" -p "$port" -o BatchMode=yes -o ConnectTimeout=10 "$host" "$@"; }
pointed=0
while true; do
  for f in metrics.jsonl config.json launch.json sft-report.json system_telemetry.json corpus-eval.jsonl; do
    # só substitui quando o conteúdo mudou: a data do arquivo local é o que o painel usa para
    # dizer que o treino parou; regravar a mesma cópia escondia um processo morto
    if ssh_ "cat '$remote/$f' 2>/dev/null" > "$local/$f.tmp" && [[ -s "$local/$f.tmp" ]] \
        && ! cmp -s "$local/$f.tmp" "$local/$f"; then
      mv "$local/$f.tmp" "$local/$f"
    else
      rm -f "$local/$f.tmp"
    fi
  done
  if [[ $pointed == 0 && -s "$local/metrics.jsonl" ]]; then
    printf '{"metrics": "%s", "name": "%s"}\n' "$local/metrics.jsonl" "$name" > "$KATSU/active-run.json.tmp"
    mv "$KATSU/active-run.json.tmp" "$KATSU/active-run.json"
    pointed=1
    echo "painel apontado para $name"
  fi
  [[ -s "$local/sft-report.json" ]] && { echo "run concluído: $local/sft-report.json"; exit 0; }
  sleep "${MIRROR_EVERY:-30}"
done
