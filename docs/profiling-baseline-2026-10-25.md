# Perfil do passo de treino — linha de base (2026-10-25)

Apple M3, 8 núcleos, 8 GB (3 GB de swap em uso durante a medição), CPU apenas.
Perfil 134M, contexto 256, dados aleatórios. Comando:

```
cargo run --release -p katu-core --example profile_step -- <batch> 134m 2
```

Referência para medir cada otimização antes/depois.

## Batch 1

```
perfil 134m: 134373888 parâmetros, batch 1 × 256 tokens, init 0.1s, threads rayon 8
passo 1 (aquecimento): 12.37s

passo 2: loss 10.572, |g| 7.355, pesos finitos true
fase                              s       %
forward + cross-entropy       1.209   10.3%
backward                      6.170   52.3%
norma do gradiente            0.171    1.5%
Adam (host)                   4.109   34.8%
checagem de pesos             0.135    1.1%
total                        11.795   22 tokens/s na CPU deste computador

chamada do Backend        vezes     GFLOP   sobe MB  desce MB  s (CPU)
fwd add                     350       0.1     456.1     228.1    0.020
fwd cross_entropy             1       0.0      32.8       0.0    0.016
fwd matmul                  889      58.5     901.3     268.7    0.784
bwd add_backward            350       0.1    1140.3     389.5    0.090
bwd cross_entropy_backward      1       0.0      32.8      32.8    0.033
bwd matmul_backward         889     117.0    1170.0     901.3    5.546
total                      2480     175.6    3733.2    1820.3    6.489

fora do Backend no forward+backward (LayerNorm, GELU, softmax, reshape, grafo): 0.890s
por token: 0.69 GFLOP, 21.7 MB copiados entre host e GPU, 2480 chamadas de kernel por passo
```

## Batch 4

```
perfil 134m: 134373888 parâmetros, batch 4 × 256 tokens, init 0.1s, threads rayon 8
passo 1 (aquecimento): 39.69s

passo 2: loss 10.511, |g| 3.999, pesos finitos true
fase                              s       %
forward + cross-entropy       7.518   12.0%
backward                     46.895   74.6%
norma do gradiente            0.305    0.5%
Adam (host)                   7.949   12.7%
checagem de pesos             0.166    0.3%
total                        62.835   16 tokens/s na CPU deste computador

chamada do Backend        vezes     GFLOP   sobe MB  desce MB  s (CPU)
fwd add                     350       0.2    1824.5     912.3    0.168
fwd cross_entropy             1       0.1     131.1       0.0    0.122
fwd matmul                 1753     233.9    2290.9    1074.8    4.817
bwd add_backward            350       0.2    4561.3    1444.2    0.905
bwd cross_entropy_backward      1       0.1     131.1     131.1    0.212
bwd matmul_backward        1753     467.9    3365.7    2290.9   40.424
total                      4208     702.5   12304.6    5853.2   46.649

fora do Backend no forward+backward (LayerNorm, GELU, softmax, reshape, grafo): 7.764s
por token: 0.69 GFLOP, 17.7 MB copiados entre host e GPU, 4208 chamadas de kernel por passo
```

## Depois: matmul_backward com dot SIMD e Adam paralelo

Mudanças: `Kernel::dot` (NEON/AVX2) no gradiente da entrada do `matmul_backward`; Adam lê os
gradientes sem copiar e atualiza momentos e pesos em blocos paralelos (rayon). Os pesos continuam
sendo tensores novos a cada passo, porque o backward depende de dados imutáveis.

Versão antiga e nova rodadas em sequência, na mesma sessão. O Mac estava com 3 a 5 GB de swap,
então os tempos variam bastante entre execuções; a loss e a norma do gradiente saem idênticas.

| batch | versão | forward | backward | matmul_backward | Adam | total | tokens/s |
|---|---|---|---|---|---|---|---|
| 1 | antiga | 1,23 s | 6,64 s | 5,84 s | 1,26 s | 9,36 s | 27 |
| 1 | nova | 1,64 s | 3,03 s | 1,99 s | 0,95 s | 6,05 s | 42 |
| 4 | antiga | 9,80 s | 44,82 s | 38,78 s | 7,98 s | 63,01 s | 16 |
| 4 | nova | 8,99 s | 18,76 s | 12,76 s | 3,48 s | 31,72 s | 32 |

Numa execução com menos swap (só o dot aplicado), batch 1 chegou a 4,18 s e 61 tokens/s, com o
matmul_backward em 1,21 s (~97 GFLOP/s, acima dos ~85 do matmul do forward).

## Memória do passo (batch 4 sem swap)

O `profile_step` agora mede a memória física do processo ao fim de cada fase e o pico
(`proc_pid_rusage`, o mesmo "peak memory footprint" do `/usr/bin/time -l`; inclui o que o
macOS comprimiu ou mandou para o swap). Perfil 134M, contexto 256, mesmo Mac (M3, 8 GB).

### Onde estava a memória (antes, batch 4)

| momento | memória | o que entrou |
|---|---|---|
| fim do forward | 4.609 MB | ~3,0 GB de ativações sobre pesos + momentos do Adam (1,6 GB) |
| fim do backward | 5.403 MB | gradientes (537 MB) + adjuntos; o grafo inteiro continuava vivo |
| fim do Adam | 5.862 MB | um conjunto novo de pesos (537 MB) ao lado do antigo |

As ativações eram ~2,9 MB por token: cada operação guarda a própria saída, mesmo quando
nenhum backward a usa. As maiores: a projeção de saída por cabeça com uma soma residual por
cabeça (~35%), a atenção com scores, ×escala, +máscara e softmax (~25%) e as cópias do reshape
(~14%, incluindo duas cópias dos logits).

### O que mudou

- `reshape` compartilha o buffer do pai (`TensorNode.data` virou `Arc<Vec<f32>>`).
- `Tensor::backward_release` consome o grafo: cada nó é liberado logo depois de propagar o
  gradiente. `Optimizer::step_owned` / Adam atualiza no próprio buffer os pesos sem outro dono e
  solta o gradiente na hora. `train_sft` e `profile_step` usam os dois.
- Operações fundidas que guardam só a saída: `causal_attention_with_backend` (só as
  probabilidades), `linear_with_backend` (`x·W + b`) e `layer_norm_affine`. Fazem as mesmas
  contas, na mesma ordem, que as compostas (testes em `tests/memory.rs` comparam bit a bit).
- A projeção de saída da atenção junta as cabeças (`Tensor::concat`) e faz um matmul só. É a
  única mudança que altera a ordem das somas: num SFT de 100 passos a loss difere no máximo
  2e-7 (relativo) do run anterior; loss e |g| do profile_step não mudam nas casas impressas.

### Antes e depois

| batch | versão | fim do forward | fim do backward | fim do Adam | pico | tempo do passo |
|---|---|---|---|---|---|---|
| 1 | antes | 2.313 MB | 2.980 MB | 3.483 MB | 3.483 MB | 8,7 s (29 tok/s) |
| 1 | depois | 1.889 MB | 2.189 MB | 1.695 MB | 2.294 MB | 3,7 s (70 tok/s) |
| 4 | antes | 4.609 MB | 5.403 MB | 5.862 MB | 5.862 MB | 60,3 s (17 tok/s) |
| 4 | depois | 2.669 MB | 2.322 MB | 1.734 MB | 2.997 MB | 34,1 s (30 tok/s) |

Batch 4 com `/usr/bin/time -l`, antes e depois em sequência (59% de memória livre no sistema):

| versão | RSS máximo | pico de footprint | fora da RAM (comprimido ou swap) | tempo |
|---|---|---|---|---|
| antes | 1.558 MB | 5.809 MB | ~4.251 MB | 48,7 s |
| depois | 2.196 MB | 3.001 MB | ~804 MB | 23,5 s |

Os tempos variam muito: outra sessão compilava Rust durante as medições (carga 5 a 13). O
`vm.swapusage` do sistema também mudava com os outros processos (cresceu 800 MB durante um
batch 1 de 2,3 GB de pico), por isso a comparação usa a memória do próprio processo.

### O piso que sobra

Pesos (537 MB) + momentos do Adam (1.075 MB) ficam o passo todo: 1,6 GB. No pico, durante o
backward do batch 4, somam-se ~1,0 GB de ativações (~1 MB por token) e os primeiros gradientes.
Os ~800 MB fora da RAM podem ser páginas comprimidas pelo macOS ou em swap; estas medições não
separam os dois. O guard do `train_sft` (params × 48 bytes + 2 GB ≈ 8,4 GB para
o 134M, exige CUDA) continua igual; a medição real no batch 1 é ~2,3 GB (~17 bytes por parâmetro).

## Velocidade: Accelerate, lotes e operações do host

### Por que lotes não rendiam

Com o kernel próprio, o GFLOP/s cai quando a matriz cresce (ele percorre B inteira para cada
linha de saída): matmul 70 → 33 GFLOP/s e matmul_backward 69 → 23 GFLOP/s do batch 1 para o
batch 4. Batch maior era mais lento por token.

### Teto medido no M3 (`cblas_sgemm` do Accelerate × kernel próprio)

| formato | kernel próprio | Accelerate |
|---|---|---|
| 256×768×768 (proj, batch 1) | 113 GFLOP/s | 1.471 GFLOP/s |
| 1024×768×768 (proj, batch 4) | 99 GFLOP/s | 1.192 GFLOP/s |
| 1024×768×3072 (MLP, batch 4) | 105 GFLOP/s | 932 GFLOP/s |
| 1024×768×32004 (vocabulário, batch 4) | 51 GFLOP/s | 812 GFLOP/s |

Erro relativo < 1e-6 (só a ordem das somas muda).

### O que mudou

- `AccelerateBackend` (macOS): matmul e matmul_backward pelo Accelerate, com as transposições
  pelo próprio sgemm; o resto delega ao `CpuBackend`, que segue como referência exata. Uso:
  `KATU_BACKEND=accelerate` no `train_sft`; 4º argumento `accelerate` no `profile_step`.
- `train_sft` em lotes: `KATU_BATCH` (1 a 64). Exemplos completados no fim até o maior do lote;
  com atenção causal o padding não afeta os tokens reais. A perda é a média das perdas de cada
  exemplo, o mesmo objetivo do batch 1 (batch 1 é bit a bit igual ao caminho antigo; teste
  `batch_loss_is_the_mean_of_single_example_losses`).
- O guard do modelo grande aceita `accelerate` com estimativa medida: 16 bytes/parâmetro +
  300 MiB por janela + 512 MiB (134M, batch 4: 3.762 MiB). Ainda exige
  `KATU_MEMORY_BUDGET_MIB` acima disso. CUDA mantém a estimativa antiga.
- GELU, cross-entropy e softmax da atenção em paralelo por linha (rayon), e o primeiro gradiente
  de cada nó vira o adjunto em vez de zerar um buffer e somar. Mesmas contas por elemento.

### Medições (134M, profile_step, dados aleatórios)

| backend | batch | passo | tokens/s | pico |
|---|---|---|---|---|
| CPU (kernel próprio) | 1 | 3,7 s | 70 | 2.294 MB |
| Accelerate | 1 | 1,3 s | 196 | 2.282 MB |
| Accelerate | 4 | 3,1 s | 332 | 2.862 MB |
| Accelerate | 8 | 9,6 s | 212 | 4.233 MB |

Batch 8 piora: ~4,2 GB de pico neste Mac de 8 GB com outras apps abertas.

As operações do host em paralelo foram medidas alternando versões, com carga 8–9 (outra sessão
compilando): tempo fora dos kernels no batch 4 1,9–2,7 s → 1,2–1,5 s; passo 4,1–6,4 s → 3,6–4,6 s.

### SFT real do 134M no Mac (`train_sft`, Accelerate, dados da destilação, 20 passos)

| batch | passo típico | tokens de entrada/s | pico |
|---|---|---|---|
| 1 | ~1,1 s (~165 tokens) | ~150 | 2,29 GB |
| 4 | ~3,7 s (~660 tokens) | ~180 | 2,79 GB |

No batch 4 houve passos de 13–31 s no meio do run, junto com carga 10–12 na máquina e o disco
interno com 2–3 GB livres (o swap do macOS fica nele). O padding até o maior exemplo do lote
custa ~30% nos dados da destilação.

### O que falta para milhares de tokens/s

- Com o Accelerate, matmul fica em ~1,3 s de ~3,1 s no batch 4; o resto é host (LayerNorm,
  adjuntos, cópias do concat e do reshape no backward), Adam (0,3 s, limitado por memória) e
  norma do gradiente (0,1 s).
- No Mac CPU, Accelerate e GPU compartilham a mesma memória: não há cópia CPU↔GPU a evitar.
  "Residência no dispositivo" importa para CUDA (as colunas "sobe/desce MB" do profile_step: ~9 GB
  por passo no batch 4 com o backend atual, que recebe e devolve buffers do host).
- Pesos e Adam residentes exigem que o trait `Backend` passe a trabalhar com buffers do
  dispositivo (handles) em vez de `&[f32]`/`Vec<f32>`, e que o `Tensor` guarde um buffer de host
  ou de dispositivo. É uma reforma do núcleo, a fazer junto com um backend Metal (ou para o CUDA
  em VRAM), não um ajuste.
