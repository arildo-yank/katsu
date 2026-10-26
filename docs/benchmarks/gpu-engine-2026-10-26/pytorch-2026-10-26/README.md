# Katsu × PyTorch: 134M, RTX 4090, 2026-10-26

Mesmo modelo (134.367.744 parâmetros, 12 camadas, 768 dims, vocabulário de 32k), mesmos 100 lotes
de 64 exemplos (gravados pelo Katsu com `KATSU_DUMP_BATCHES` sobre `distill_arandu_10k`) e mesmos
pesos de partida (`migrated.ckpt` do Katsu). PyTorch 2.8.0+cu128, driver 595.91.07, `scripts/bench_pytorch.py`.

| Motor | Precisão | s/passo | Tokens reais/s | Memória de pico |
|---|---|---:|---:|---:|
| Katsu | FP32 (tiled64) | 0,453 | 24.338 | 16,4 GiB* |
| Katsu | TF32 (Tensor Cores) | 0,408 | 26.981 | 16,4 GiB* |
| PyTorch | FP32, atenção explícita | 0,330 | 33.389 | 20,0 GiB** |
| PyTorch | TF32 + SDPA | 0,200 | 54.990 | 17,9 GiB |
| PyTorch | BF16 + FlashAttention + Adam fused | 0,122 | 90.539 | 13,3 GiB |

Passos 6 a 100 no Katsu e os 95 passos cronometrados do PyTorch (5 de aquecimento): os dois
excluem a compilação e o aquecimento.

\* VRAM do processo no `nvidia-smi` (16.811 MiB), contexto CUDA incluído.
\*\* `torch.cuda.max_memory_allocated`, sem o contexto CUDA nem a memória reservada e não usada.
Os dois números não são da mesma régua; o do PyTorch subestima o uso real.

## O que os números dizem

- **Velocidade: o PyTorch é mais rápido em todos os modos.** 1,4× no FP32, 2,0× no TF32 (contra o
  Katsu TF32) e 3,4× no BF16 (contra o Katsu TF32, o modo mais rápido do Katsu).
- **Memória: o Katsu usa menos que o PyTorch em FP32 e TF32, e mais que o PyTorch em BF16.** No
  FP32 com atenção explícita, a primeira tentativa do PyTorch no batch 64 deu *out of memory*
  (23,1 GiB em uso, 5,7 GiB reservados e não usados: fragmentação). Com
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` o mesmo batch coube, com 20,0 GiB de pico.
  O OOM foi do alocador padrão, não um limite do PyTorch: não dá para dizer que "o PyTorch não roda
  batch 64".
- **Mesma conta:** a perda do passo 1 coincide (Katsu 9,352478, PyTorch FP32 9,352478). Depois as
  curvas se separam porque o `bench_pytorch.py` usa LR constante de 3e-4 e o Katsu usa aquecimento
  linear e cosseno (6e-5 no passo 1); isso muda a perda, não o tempo por passo.

## Arquivos

`torch-*.json` (resumo e cada passo), `katsu-*-sft-report.json`, `katsu-*-metrics.jsonl`
(gitignored, como os demais `.jsonl`) e `gpu.txt`.
