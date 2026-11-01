# Pesos e Adam residentes no dispositivo (25/10/2026)

## Problema

O backend CUDA híbrido copia entradas e saídas a cada chamada: no 134M, batch 4 × 256, um passo
movia **9,37 GB** entre host e GPU (medido com `examples/transfer_budget.rs`). A L40S treinava
a 82 tokens/s, contra 305 no Mac com Accelerate.

## O que mudou

- `ParamStore` (`src/resident.rs`): pesos, gradientes e momentos do Adam numa arena por buffer.
  Os parâmetros residentes são folhas `Tensor::resident` sem cópia no host; matmul e linear com
  peso residente rodam no store, e o backward soma o gradiente lá mesmo.
- `KatuLmConfig::resident_layout`: os 110M parâmetros que só aparecem como lado direito de matmul
  (Q, K, V, Wo, MLP e cabeça de saída). Embeddings, LayerNorm e bias (~25M) ficam no host. Os Wo de
  uma camada são vizinhos na arena: o concat das cabeças vira uma faixa, sem cópia.
- `Tensor::matmul_many_with_backend` + `Tensor::columns`: Q, K e V de todas as cabeças numa
  chamada, com a entrada subindo uma vez (antes: 36 vezes por camada). Vale também no caminho CPU,
  que segue as mesmas contas do residente.
- `Adam::step_resident`, `offload_moments`, `save_state_resident`: o Adam roda na arena inteira;
  o checkpoint tem o mesmo formato de sempre. Um Adam novo zera os momentos do store.
- `HostParamStore`: referência em memória do host, com as contas do `Backend` dado e a contagem
  dos bytes que uma GPU trocaria. `CudaParamStore` (feature `cuda`): a mesma arena na VRAM, com
  kernels `gemm_acc`, `adam` e `sumsq`.
- `train_sft`: `KATU_RESIDENT=1` liga o modo residente (store CUDA com `KATU_BACKEND=cuda`;
  senão o de referência). Somas, atenção e entropia rodam no host
  (`KATU_RESIDENT_REST=device` as mantém no backend). `metrics.jsonl` ganha `transfer_bytes`.

## Medido

Tráfego host↔dispositivo por passo, 134M, batch 4 × 256 (`transfer_budget`):

| Caminho | Por passo |
|---|---|
| Híbrido (antes) | 9,37 GB |
| Pesos + Adam residentes | 8,06 GB |
| + Q/K/V numa chamada | **4,09 GB** (1,74 GB do store + 2,35 GB de atenção, somas e entropia no backend) |
| + resto no host (`train_sft` residente) | **~1,74 GB** |

Ponta a ponta no 134M real, 6 passos de SFT no Mac (Accelerate), caminho normal × residente:
**checkpoints idênticos byte a byte**, mesmas losses e validação; ~0,7–0,8 GB por passo com os
exemplos de SFT (mais curtos que 256 tokens).

Testes: `tests/resident.rs` (treino, checkpoint, retomada e troca de Adam bit a bit iguais ao
CPU; tráfego; mau uso falha alto). `tests/cuda.rs::cuda_resident_store_matches_host_reference`
compara o store CUDA com o de referência, mas **precisa de uma NVIDIA**: aqui só compila.

## O que ainda não se sabe

- **Velocidade na L40S.** O store CUDA não rodou numa GPU. O próximo passo é um benchmark curto:
  `cargo test --features cuda -- --ignored` e algumas centenas de passos do `train_sft` com
  `KATU_RESIDENT=1 KATU_BACKEND=cuda`, lendo `step_seconds` e `transfer_bytes`.
- **O teto agora é o host.** LayerNorm, GELU, softmax e atenção rodam na CPU (no Mac, ~1,5 s
  de um passo de ~3 s). Com 1,74 GB por passo no PCIe, o limite deixa de ser a cópia dos pesos e
  passa a ser essas operações e o GEMM de referência (tiles 16×16, sem tensor cores). Ativações
  residentes e kernels para essas operações são a fase seguinte.
