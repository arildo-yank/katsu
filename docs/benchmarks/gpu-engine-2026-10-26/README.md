# Motor inteiro na GPU: benchmark por etapa (RTX 4090, 26/10/2026)

Modelo de 134M (d 768, 12 camadas, 12 cabeças, vocabulário 32.000), SFT com os 10k exemplos de
`distill_arandu_10k`, `KATSU_ENGINE=gpu`, `KATSU_BENCHMARK_ONLY=1` (sem as medições de loss),
15 passos por configuração, com média dos passos 4 a 15. Script: `bench.sh` (abaixo).

**Ressalva:** a GPU estava compartilhada com a prova `poc-10k-b16` (motor híbrido antigo, ~10%
da GPU, 2,5 GiB). Os números com a GPU livre devem ser iguais ou um pouco melhores.

"tok/s reais" conta só os tokens de entrada dos exemplos; "com padding" inclui o preenchimento até
o maior exemplo do lote, que a GPU também processa.

| Configuração | Batch | s/passo | tok/s reais | tok/s com padding | VRAM do trainer |
|---|---:|---:|---:|---:|---:|
| Antes: híbrido residente (prova `poc-10k-b16`) | 16 | 11,3–12,2 | 223 | n/d | ~2,5 GiB |
| Etapa 1: tudo na GPU, GEMM 16×16, sem fusão | 64 | 2,579 | 4.229 | 5.609 | 16,4 GiB |
| Etapa 2: + fusões no epílogo do GEMM | 64 | 2,571 | 4.241 | 5.625 | 16,4 GiB |
| Etapa 3: + GEMM 64×64 com registradores (FP32) | 64 | 0,453 | 24.064 | 31.921 | 16,4 GiB |
| Etapa 3b: Tensor Cores TF32 | 64 | 0,421 | 25.876 | 34.324 | 16,4 GiB |
| Etapa 3, batch 16 | 16 | 0,114 | 24.142 | 29.899 | 5,9 GiB |

VRAM medida por processo com `nvidia-smi --query-compute-apps` a cada 2 s (pico). No batch 64:
16.802 MiB do trainer + 2.562 MiB da prova = 18,9 GiB de 24.

## O que os números dizem

- **Tirar tudo da CPU (Etapa 1) é o primeiro salto:** de 223 para ~4.200 tok/s (~19×), mesmo com o
  GEMM ingênuo. No batch 16, o mesmo motor fez 4.540 tok/s num teste anterior.
- **As fusões (Etapa 2) não mudaram o tempo de forma mensurável:** com o GEMM 16×16, quase todo o
  passo é GEMM, e as passadas elemento a elemento que a fusão remove pesam pouco. Elas economizam
  memória de banda e vão pesar mais quando o GEMM ficar mais rápido; não medimos ainda a Etapa 3
  sem fusão.
- **O GEMM com tiles maiores (Etapa 3) é o ganho grande:** 5,7× sobre o 16×16, chegando a
  **24.064 tok/s reais, ~108× a prova híbrida** e ~93× os 259 tok/s do modo residente com batch 1.
- **TF32 rendeu só 7% a mais:** o kernel WMMA ainda carrega os tiles elemento a elemento pela
  memória compartilhada, sem cargas vetorizadas nem pipeline; ele é o próximo alvo.
- **Batch 64 não acelerou sobre batch 16** (24.064 contra 24.142 tok/s): com o motor na GPU, o
  batch 16 já ocupa a placa. O batch maior serve para estabilidade do gradiente, não para
  velocidade, e muda a otimização (4× menos passos por época).

## Paridade

`crates/katsu-core/tests/gpu_train.rs` compara perda, gradiente de cada tensor e pesos depois de um
passo do Adam com o autograd da CPU, em todas as variantes. Pior diferença relativa de gradiente:
5,9e-7 (FP32 tiled64) e 1,0e-3 (TF32, esperado para 10 bits de mantissa). Um erro de 0,1% na escala
da atenção faz o teste falhar. Na prova real, o passo 1 deu a mesma perda nos dois motores (9,2653).

## Reproduzir

```bash
KATSU_ENGINE=gpu KATSU_BACKEND=cuda KATSU_BATCH=64 KATSU_GEMM=tiled64 KATSU_FUSED=1 \
KATSU_BENCHMARK_ONLY=1 target/release/train_sft runs/base data/sft/10k.train.jsonl \
  data/sft/10k.eval.jsonl data/sft/10k.train.jsonl runs/bench 15 0 0.0003
```

Arquivos: `bench-*/metrics.jsonl` (um passo por linha), `bench-*/config.json`,
`bench-*/sft-report.json`, `bench-*.log`, `bench-*.mem` (VRAM por processo).
