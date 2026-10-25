# Perfil do passo de treino — linha de base (2026-10-25)

Apple M3, 8 núcleos, 8 GB (3 GB de swap em uso durante a medição), CPU apenas.
Perfil 134M, contexto 256, dados aleatórios. Comando:

```
cargo run --release -p katsu-core --example profile_step -- <batch> 134m 2
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
