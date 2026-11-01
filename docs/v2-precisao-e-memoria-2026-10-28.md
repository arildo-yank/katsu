# V2: BF16, memória e smoke test no Mac (28/10/2026)

Medidas feitas num MacBook (Apple M3, 8 GB de RAM, 8 núcleos), sem GPU. Os números de tempo
no Mac são ruidosos: o sistema já usava a memória quase toda (servidor Katu, API e navegador
abertos), e os passos variaram até 2× entre execuções iguais.

## 1. Precisão mista BF16 (`katu-core`)

O que existe:

- `Bf16Buffer`: armazenamento BF16 de 2 bytes por número; conversão com arredondamento para o
  par mais próximo (`f32_to_bf16`, `bf16_to_f32`, `round_bf16`).
- `matmul_bf16`: matmul de referência com operandos BF16 e acumulação FP32.
- `Bf16Backend`: embrulha um backend FP32 (Accelerate no Mac, CPU no Linux). Os operandos de
  cada matmul vão para BF16, a acumulação é FP32, a saída (ativação) volta para BF16. O
  gradiente de peso fica em FP32. Pesos-mestre e momentos do Adam ficam em FP32; LayerNorm,
  softmax, GELU e entropia cruzada rodam em FP32.

O que **não** existe:

- **Kernels CUDA em BF16.** Não há GPU neste Mac para testá-los; o `GpuTrainer` continua em
  TF32/FP32. Este é o bloqueio principal para medir a velocidade da V2 na GPU.
- **Economia de memória na CPU.** O grafo do autograd guarda `Vec<f32>`: os valores BF16 ocupam
  4 bytes cada. No Mac o BF16 é só numérico (serve para a paridade) e custa conversões.

### Paridade no 134M

Teste: `cargo run --release -p katu-core --example v2_step -- {fp32|bf16} 768 12 12 32004 256 1 8`
(134.373.888 parâmetros, contexto 256, batch 1, 8 passos de Adam com lr 5e-4, sempre o mesmo
lote, mesma inicialização). Cada precisão rodou duas vezes, em processos separados.

| Passo | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Perda FP32 | 10,532 | 9,275 | 8,463 | 9,433 | 8,130 | 11,782 | 7,441 | 7,690 |
| Perda BF16 | 10,533 | 9,274 | 8,463 | 9,463 | 8,142 | 11,764 | 7,439 | 7,682 |
| Diferença relativa | 0,00% | 0,01% | 0,01% | 0,32% | 0,15% | 0,15% | 0,04% | 0,11% |

**Tolerância declarada: 1% por passo. Maior diferença medida: 0,32%.** As duas execuções de
cada precisão deram perdas idênticas bit a bit. A perda oscila porque o lr não tem aquecimento;
cai de 10,53 para 7,69 nos dois casos.

| Precisão | s/passo (média dos passos 2–8), execução 1 / 2 | Pico de memória do processo |
|---|---|---|
| FP32 (Accelerate) | 2,23 / 1,60 | 2.280 / 2.239 MB |
| BF16 misto (emulado) | 2,20 / 4,74 | 2.320 / 2.287 MB |

Leitura: no Mac o BF16 **não** economiza memória nem tempo, como esperado (emulação sobre o
sgemm FP32). A comparação de tempo serve só para mostrar que o custo da emulação não é
proibitivo para testes. O ganho de BF16 (velocidade nos Tensor Cores e metade da memória das
ativações) só pode ser medido com kernels CUDA, que não existem.

Teste automático (roda no hook): `tests/bf16.rs`, modelo pequeno, 8 passos, tolerância de 1%.

## 2. Memória para 1B: Adam de 8 bits e recomputação de ativações

O BF16 sozinho não faz o 1B caber numa RTX 4090 (24 GB): com pesos-mestre e Adam em FP32, o
estado continua em ~16 bytes/parâmetro. Entraram as duas opções mais baratas que rodam e se
testam no Mac:

- `Adam8bit`: momentos em 8 bits por blocos de 256, com uma escala FP32 por bloco e compansão
  por raiz quadrada (mais resolução perto de zero). Estado: **2,03 bytes/parâmetro** (medido
  com `state_bytes()`), contra 8 do Adam FP32. A conta do passo é FP32.
- `KatuLmConfig::loss_backward_recompute`: guarda só a entrada de cada camada; o backward
  refaz uma camada por vez. Custa um forward a mais.

O Adam na RAM do host não entrou: na CPU não muda nada (já é tudo host) e, sem GPU, não há
como testar o ganho no dispositivo. Fica como alternativa para o executor CUDA.

Testes no hook (`tests/v2_memory.rs`): a recomputação dá a mesma perda (1e-6) e os mesmos
gradientes (1e-4 da escala) que o backward normal; o Adam de 8 bits fica a 3% da perda do Adam
FP32 em 25 passos, e os dois caem mais de 30%.

### Medida no Mac (batch 4 × 256, FP32, Accelerate, 4 passos)

`KATU_OPTIM={adam|adam8} KATU_RECOMPUTE={0|1} v2_step fp32 D CAMADAS CABEÇAS 32004 256 4 4`

| Modelo | Otimizador | Recomputação | Pico do processo | Pico / parâmetro | s/passo |
|---|---|---|---|---|---|
| 58M (512 × 8) | Adam FP32 | não | 1.502 MB | 25,9 B | 0,92 |
| 58M | Adam FP32 | sim | 1.147 MB | 19,7 B | 0,99 |
| 58M | Adam 8 bits | não | 977 MB | 16,8 B | 0,85 |
| 58M | Adam 8 bits | sim | **561 MB** | **9,7 B** | 1,03 |
| 134M (768 × 12) | Adam FP32 | não | 3.006 MB | 22,4 B | 2,59 |
| 134M | Adam FP32 | sim | 2.289 MB | 17,0 B | 3,10 |
| 134M | Adam 8 bits | não | 1.927 MB | 14,3 B | 2,60 |
| 134M | Adam 8 bits | sim | **1.488 MB** | **11,1 B** | 3,33 |

As perdas dos quatro modos, em cada tamanho, coincidiram nas duas primeiras casas decimais nos
4 passos. "Pico / parâmetro" inclui ativações, logits e o processo; é um teto, não o estado.

Leitura: o estado medido é pesos FP32 (4) + gradientes FP32 (4) + Adam 8 bits (2,03) =
**10,03 bytes/parâmetro**; o resto do pico do 134M com recomputação (~140 MB) são ativações e
logits. Sem recomputação, as ativações medidas foram ~0,39 MB/token no 58M e ~0,57 MB/token no
134M. A recomputação custou 7–28% a mais de tempo por passo.

### Extrapolação para 1B (estimativa, não medida)

Configuração de referência: d = 2048, 18 camadas, 16 cabeças, vocabulário 32.000, contexto
1.024: **1.039.474.688 parâmetros**.

| Arranjo | Bytes/parâmetro | Estado do 1B |
|---|---|---|
| Adam FP32, pesos e gradientes FP32 (hoje) | 16 | 16,6 GB |
| Adam 8 bits, pesos-mestre e gradientes FP32 (medido acima) | 10,03 | 10,4 GB |
| GPU: o mesmo + cópia BF16 dos pesos | 12,03 | 12,5 GB |
| GPU: só pesos e gradientes BF16 no dispositivo, mestre e Adam no host | 4 | 4,2 GB |

Na 4090 sobram ~11 GB para ativações no arranjo de 12 bytes/parâmetro. Com recomputação, o
maior consumo deixa de ser o grafo das camadas e passa a ser os logits (batch × contexto ×
32.000 × 4 bytes: ~1 GB para 8 × 1.024, mais o gradiente) e a atenção sem fusão (matrizes
T × T por cabeça). **Estimativa:** cabe com batch pequeno. Só uma medida na GPU confirma, e o
`GpuTrainer` ainda não tem BF16, Adam 8 bits nem recomputação.

## 3. Smoke test do 1B no Mac

`sysctl hw.memsize` = 8.589.934.592 (8 GB). Antes do teste o swap já tinha 4,4 GB em uso, com
o servidor Katu (8792) e a API (8766) no ar. O 1B precisa de ~10,4 GB só de estado no melhor
arranjo medido (seção 2): **não cabe neste Mac**, e tentar empurraria o sistema para o swap
num disco do sistema quase cheio. O 1B não foi rodado.

Rodou, no lugar, a **largura real do 1B** (d = 2048, 16 cabeças, vocabulário 32.000) com 1 e 2
camadas, BF16 misto + Adam 8 bits + recomputação, contexto 128, batch 1, 6 passos, sempre o
mesmo lote:

`KATU_OPTIM=adam8 KATU_RECOMPUTE=1 v2_step bf16 2048 {1|2} 16 32000 128 1 6`

| Camadas | Parâmetros | Pico do processo | s/passo (passos 2–6) | Perda: passo 1 → 6 |
|---|---|---|---|---|
| 1 | 181.688.320 | 1.871 MB | 0,66 | 10,746 → 0,0003 |
| 2 | 232.038.400 | 2.391 MB | 0,99 | 10,716 → 0,0006 |

A perda cai nos dois (o modelo decora um lote de 128 tokens; o teste confere que o treino
funciona, não que o modelo aprende língua). Depois do segundo, o swap estava em 5,3 de 6 GB:
parei aí de propósito, sem tentar 3 ou mais camadas.

**Extrapolação para as 18 camadas do 1B (estimativa, não medida):** cada camada de largura 2048
somou 50,35M parâmetros, **520 MB** de pico (10,3 bytes/parâmetro, o que a seção 2 prevê) e
~0,33 s por passo no Mac. Para 18 camadas: ~1,87 + 17 × 0,52 ≈ **10,7 GB** de pico com
contexto 128 e batch 1, e ~0,66 + 17 × 0,33 ≈ **6 s por passo de 128 tokens** (~20 tokens/s)
na CPU do Mac. Treinar o 1B no Mac não é viável; o teste valida o caminho (BF16, Adam 8 bits,
recomputação) na largura do 1B, não a velocidade na GPU.

## 4. As três peças no `GpuTrainer` (CUDA), 28/10/2026

Código pronto e compilado no Mac (`cargo clippy --features cuda`), **sem execução em GPU**: este
Mac não tem NVIDIA nem nvcc, e os kernels são compilados em tempo de execução pelo NVRTC. Os
testes da GPU são pulados aqui. **A correção só fica provada depois da rodada no pod**
(`deploy/cuda/runpod_v2_probe.sh`), que roda a paridade antes de medir qualquer coisa.

As opções ficam em `GpuTrainOptions`, todas desligadas por padrão (o treino atual não muda), e no
`train_sft` por variável de ambiente:

| Peça | Opção / variável | O que faz na GPU |
|---|---|---|
| BF16 | `GemmKernel::Bf16` / `KATU_GEMM=bf16` | GEMM em Tensor Cores (WMMA m16n16k16, `cuda/train_bf16.cu`), entradas FP32 convertidas para BF16 na carga, acumulação FP32. A mistura é a do `Bf16Backend` da CPU: o produto de ativação e de gradiente de ativação volta a BF16, o gradiente de peso fica FP32. Precisa de sm_80+. |
| Adam de 8 bits | `adam8: true` / `KATU_OPTIM=adam8` | Kernel `adam8` em `cuda/train.cu`: um bloco de 256 threads por bloco de quantização, a mesma conta e o mesmo formato do `Adam8bit` (`adam8.rs`), com os blocos sobre a arena da GPU. Estado 2 bytes + 8 bytes por bloco de 256. O checkpoint sai dequantizado em FP32 (formato do `Adam`) e volta quantizado. |
| Recomputação | `recompute: true` / `KATU_RECOMPUTE=1` | Guarda só a entrada de cada camada; as demais ativações (QKV, atenção T×T, MLP, LayerNorms) existem para uma camada só. O backward refaz o forward de cada camada antes do backward dela. |

O que **não** mudou: os buffers continuam FP32. O BF16 aqui muda a conta (e talvez a velocidade
nos Tensor Cores), **não a memória**; a economia de memória vem do Adam de 8 bits e da
recomputação. Atenção, LayerNorm, GELU e entropia cruzada continuam em FP32 (na CPU, as matmuls
da atenção também passam por BF16: os dois lados não são idênticos). Contexto 1.024 com atenção
fundida e leitura de vários shards continuam pendentes.

### Testes de paridade (rodam no pod; pulados sem GPU)

| Teste | Contra | Critério |
|---|---|---|
| `gpu_train::tests::adam8_kernel_is_bit_identical_to_cpu` | `adam8::apply` da CPU, 6 passos, 3⅓ blocos (um com gradientes 1.000× menores) | pesos, momentos e escalas **iguais bit a bit** (operações `_rn` sem FMA) |
| `adam8_on_gpu_tracks_cpu_adam8bit` | `Adam8bit` da CPU, 8 passos | perda a 0,5%; pesos a ≤ meio passo máximo do Adam por passo |
| `adam8_state_survives_export_and_reload` | o próprio estado exportado | momentos a um degrau de quantização; mesma perda depois de recarregar |
| `bf16_tensor_cores_match_cpu_autograd_within_bf16_precision` | autograd FP32 da CPU | perda a 1%, gradientes a 25% do máximo de cada tensor |
| `recompute_matches_cpu_autograd` | autograd FP32 da CPU (Tiled64 estrito e BF16) | as tolerâncias de cada GEMM |
| `recompute_gives_the_same_gradients_as_stored_activations` | o `GpuTrainer` sem recomputação | perda igual; gradientes a 1e-6 |
| `v2_options_together_track_fp32_cpu_loss` | treino FP32 da CPU com Adam FP32, 8 passos | perda a **1% em cada passo** (o critério do 134M no Mac) |

As tolerâncias do BF16 são palpites a confirmar no pod; se um teste falhar, o número que sai em
`parity.txt` (com `KATU_PARITY_VERBOSE`) diz se é tolerância apertada ou kernel errado.

### Estimativa de memória do 1B no pod (não medida)

Contexto 256, batch 8, as três peças: estado de 10,03 bytes/parâmetro × 1,04B ≈ **9,7 GiB**, mais
~1 GiB de ativações (entradas das 18 camadas, uma camada inteira, logits de 8 × 256 × 32.000) e o
contexto CUDA. Total esperado ~11–12 GiB, contra 21,4 GiB medidos em FP32 no mesmo batch.
**Velocidade: só a medida diz.** A recomputação custa um forward a mais por passo; os Tensor Cores
em BF16 podem compensar ou não.
