# V2: BF16, memória e smoke test no Mac (28/10/2026)

Medidas feitas num MacBook (Apple M3, 8 GB de RAM, 8 núcleos), sem GPU. Os números de tempo
no Mac são ruidosos: o sistema já usava a memória quase toda (servidor Katsu, API e navegador
abertos), e os passos variaram até 2× entre execuções iguais.

## 1. Precisão mista BF16 (`katsu-core`)

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

Teste: `cargo run --release -p katsu-core --example v2_step -- {fp32|bf16} 768 12 12 32004 256 1 8`
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
