#!/usr/bin/env bash
set -euo pipefail
# Sem instalar driver, provisionar máquina, enviar dados ou iniciar modelos grandes.
nvidia-smi
command -v nvcc
cargo test -p katsu-core --release --features cuda --locked
# O segundo comando é obrigatório: os testes reais de GPU são ignored por padrão.
# Sem GPU/NVRTC ele falha; não transforma ausência de hardware em aprovação.
cargo test -p katsu-core --release --features cuda --test cuda --locked -- --ignored --nocapture --test-threads=1
