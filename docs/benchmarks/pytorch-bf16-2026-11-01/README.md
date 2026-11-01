# Katu × PyTorch com o BF16 do Katu: 134M, RTX 4090, 01/11/2026

Mesmo modelo (134.367.744 parâmetros), mesmos 100 lotes de 64 exemplos (`distill_arandu_10k`, gravados com `KATU_DUMP_BATCHES`)
e mesmos pesos de partida. PyTorch 2.8.0+cu128, driver 595.91.07, Katu no commit `e4a2b6c`. Cada modo rodou duas vezes, em
processos separados, no mesmo pod. Script: `deploy/cuda/runpod_bench_pytorch.sh`.

| Motor | Modo | Tokens reais/s, 1ª e 2ª | Média | s/passo |
|---|---|---:|---:|---:|
| Katu | FP32 (tiled64) | 23.143 e 22.937 | 23.040 | 0,478 |
| Katu | TF32 | 25.135 e 24.954 | 25.045 | 0,440 |
| Katu | BF16 | 28.460 e 28.384 | 28.422 | 0,388 |
| Katu | BF16 + Adam de 8 bits | 28.098 e 28.270 | 28.184 | 0,391 |
| PyTorch | FP32, atenção explícita | 34.230 e 34.182 | 34.206 | 0,322 |
| PyTorch | TF32 + SDPA | 56.223 e 56.171 | 56.197 | 0,196 |
| PyTorch | BF16 + FlashAttention + Adam fused | 92.029 e 92.036 | 92.032 | 0,120 |

Katu: passos 6 a 100, média. PyTorch: 95 passos cronometrados, depois de 5 de aquecimento.

## O que os números dizem

- **O PyTorch segue mais rápido em todos os modos:** 1,5× em FP32, 2,2× em TF32 e **3,2× em BF16**, modo contra modo.
- **O BF16 fechou pouco:** a distância no melhor modo de cada um caiu de 3,4× (26/10, contra o Katu em TF32) para 3,2×. Os dois
  números vêm de pods diferentes.
- **O BF16 ajudou o Katu:** 13% acima do TF32 e 23% acima do FP32 neste 134M.
- **A mesma conta:** a perda do passo 1 em FP32 coincide (Katu 10,549962, PyTorch 10,549963). Em BF16 fica 10,549807 e 10,549859. Depois
  as curvas se separam, porque o Katu usa aquecimento e cosseno na taxa de aprendizado, e o script do PyTorch usa taxa constante.

## Limites

- **Memória não é comparável:** o Katu relata VRAM do processo no `nvidia-smi` (16.811 MiB em FP32, TF32 e BF16; 16.043 MiB com Adam de 8
  bits). O PyTorch relata `max_memory_allocated` (20.491, 18.337 e 13.648 MiB), sem o contexto CUDA.
- **Esta medida não explica a causa.** O PyTorch em BF16 também usa FlashAttention e Adam fused, e o gargalo só um perfil dos
  kernels, como o Nsight, pode apontar.
- **134M, não 1B:** não diz qual seria a distância no modelo de 1B.

## Arquivos

`torch-*.json`, `katu-*/sft-report.json`, `katu-*/metrics.jsonl` (ignorados pelo git, como os demais `.jsonl`), logs e `gpu.txt`.
