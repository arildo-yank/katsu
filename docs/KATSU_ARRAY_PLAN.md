# Plano: o `katsu-core` como base de computação científica

*01/11/2026. Aprovado pelo dono do projeto. As fases que não estão marcadas como feitas são plano, sem prazo
medido. As medidas de partida são do Mac M3, no mesmo dia.*

**A meta é ter, em código nosso, o que o NumPy e o SciPy dão a quem constrói IA:** arrays, álgebra linear
no nível do BLAS e do LAPACK, números aleatórios e estatística, FFT, otimização e matrizes esparsas.

## A divisão

| Peça | Papel | Equivale a |
|---|---|---|
| `katsu-core` | Arrays, autograd, GEMM, fatorações, CUDA | NumPy, e o BLAS e o LAPACK por baixo |
| `katsu-math` | Expressões, derivada simbólica, primitivas conferidas, EDO | SymPy |
| crates novas, ao lado do núcleo, quando chegar a vez | FFT, estatística, otimização, esparso | SciPy |

## A regra da independência

- **O que vai no binário é nosso.** A única dependência externa do núcleo continua sendo o `rayon`.
- **O NumPy, o SciPy e o LAPACK entram só como gabarito.** Um script gera os casos de referência uma vez e
  os grava em `.npy`, no repositório. Os testes do Katsu comparam com esses arquivos e não precisam de Python
  para rodar.
- **Toda tolerância é declarada.** Operação elemento a elemento: igualdade exata. Soma e redução: erro
  relativo, porque a ordem da soma muda o último bit. Fatoração: resíduo, como ‖A − LU‖ / ‖A‖.

## Fases

Cada fase tem um portão medido contra o NumPy na mesma máquina. A fase só fecha quando passa no portão.

| Fase | O que entra | Para que serve em IA | Portão | Situação |
|---|---|---|---|---|
| **0. Array completo** | Tipos f64, f32, i64 e bool; *views* com *strides* sem cópia; fatias com passo, inclusive negativo; transposição; broadcasting; promoção de tipos; reduções; leitura e escrita de `.npy` | Ciência precisa de f64; o pipeline de dados precisa de fatias sem cópia | Gabaritos do NumPy passando | **Quase fechada** (01/11/2026): 106 casos passando; falta a troca no `Tensor` |
| **1. GEMM no nível BLAS** | Empacotamento e blocagem por cache no estilo BLIS, microkernels NEON e AVX2/AVX-512, f32 e f64 | Atenção, MLP, e a base das fatorações | 1024×1024 no tempo do OpenBLAS (partida: Katsu 16,1 ms contra 8,1 ms) | **f32 no Mac no portão** (01/11/2026); faltam f64, a medida em x86 e ligar ao treino |
| **2. Fatorações** | LU com pivô, Cholesky, QR de Householder, autovalores simétricos, SVD, em blocos sobre o GEMM | SVD para LoRA e compressão; QR para inicialização ortogonal; PCA | Resíduo no nível do LAPACK, inclusive em matrizes mal condicionadas; tempo até 2× o do NumPy | Falta |
| **3. Aleatório e estatística** | Gerador reproduzível igual na CPU e na GPU (Philox), distribuições, funções especiais (gamma, erf) | Inicialização, amostragem, intervalos de confiança | Testes estatísticos; funções especiais contra o SciPy | Falta |
| **4. FFT** | Radix misto e FFT real, na CPU e depois na GPU | Áudio e fala, convoluções longas, modelos de sequência | Erro e tempo contra o `numpy.fft` | Falta |
| **5. Otimização e EDO** | L-BFGS, RK45 adaptativo, método implícito para EDO rígida | Ajuste de hiperparâmetros, modelos físicos | Problemas de referência do SciPy | Falta |
| **6. Esparso** | CSR, produto esparso por denso | Grafos, RAG, embeddings grandes | Contra o `scipy.sparse` | Falta |
| **7. Python** | `katsu-python` expondo o array e as fatorações | Quem vem do NumPy usa o Katsu sem trocar de linguagem | Os mesmos gabaritos pelo Python | Falta |

**Ordem:** as fases 0, 1 e 2 vêm primeiro, nessa ordem, porque todo o resto se apoia nelas. As fases 3 a 6
entram conforme o uso: a 3 primeiro, porque o treino já precisa dela.

## Medidas de partida (01/11/2026, Mac M3, NumPy 1.26.4 com OpenBLAS)

| Operação | Katsu | NumPy |
|---|---|---|
| Produto 512×512, f32 (`katsu-core`, CPU) | 2,17 ms | 1,39 ms |
| Produto 1024×1024, f32 (`katsu-core`, CPU) | 16,1 ms | 8,1 ms |
| Produto 64×64, f64 (`katsu-math`) | 219 µs | 21 µs |
| Determinante 64×64, f64 (`katsu-math`) | 65 µs | 18 µs |

Contexto e o restante das medidas: `ARANDU/docs/KATSU_MATH_FEASIBILITY.md`, que fica no repositório
privado.

## Fase 0: decisões

- **Um tipo novo, `NdArray`, ao lado do `Tensor`.** O `Tensor` é o nó do autograd: só f32, contíguo e usado
  no treino que está rodando. Mexer nele agora arriscaria o treino. O `NdArray` traz os tipos, os *strides*
  e as *views*. Quando estiver maduro, o `Tensor` passa a guardar um `NdArray`.
- **Os dados ficam num buffer compartilhado (`Arc`).** Uma view é forma, *strides* e deslocamento sobre o
  mesmo buffer, como no NumPy. Escrever numa view não é permitido ainda: as views são só de leitura, e a
  cópia é explícita.
- **A promoção de tipos segue o NumPy 1.26 para estes quatro tipos** (conferido no NumPy em 01/11/2026):
  - bool + bool dá bool (OU), bool × bool dá bool (E) e bool − bool é erro;
  - i64 com i64 dá i64, com estouro dando a volta, como no NumPy;
  - i64 com f32 dá f64; f32 com f32 dá f32; qualquer operação com f64 dá f64;
  - a divisão dá f64, exceto f32 / f32, que dá f32;
  - a soma de bool ou de i64 dá i64; a média de bool ou de i64 dá f64.
- **`.npy` na versão 1.0, little-endian:** `<f4`, `<f8`, `<i8` e `|b1`. A leitura aceita a ordem C e a
  ordem Fortran; a escrita é sempre na ordem C.

## Fase 0: situação em 01/11/2026

**Feito:** `NdArray` em `crates/katsu-core/src/array.rs`, sem dependência nova. Os testes estão em
`crates/katsu-core/tests/array.rs`, e os gabaritos foram gerados pelo NumPy 1.26.4 em
`tests/fixtures/array/gen.py`. São 106 casos, mais a tabela de promoção.

| Peça | Conferido como |
|---|---|
| Tipos bool, u8, i32, i64, f16, bf16, f32 e f64, com `astype` | Contra o NumPy. Float fora da faixa do inteiro (NaN inclusive) é recusado |
| Promoção de tipos | As 49 combinações do NumPy (`promotion.txt`). bf16 com f16 ou u8 dá f32, como no `ml_dtypes` |
| f16 | Os 65.536 valores voltam iguais; conversão de f64 com um arredondamento só |
| bf16 | Igual à conversão do motor de treino (`f32_to_bf16`) em 200 mil f32 aleatórios e nos casos de borda |
| Fatias, transposição, permutação e broadcasting, sem cópia | Contra o NumPy, e conferido que o buffer é o mesmo |
| Soma, subtração, produto e divisão; comparações | Igualdade exata, inclusive a volta de u8 e i32 e a conta de f16 feita em f32 |
| Soma, média, variância e desvio padrão (com `ddof`), máximo, mínimo, `argmax` e `argmin` | Por eixo ou no total. NaN contamina o máximo; o `argmax` aponta o primeiro NaN e o primeiro empate |
| Seleção por máscara, `take`, `concatenate` e `stack` | Contra o NumPy, inclusive com tipos misturados |
| `.npy` versões 1 a 3, ordem C e Fortran | A escrita gera os mesmos bytes que o NumPy |

**Diferenças conhecidas do NumPy:**
- a soma de u8 dá i64, e o NumPy dá u64, com os mesmos valores;
- bf16 não existe no NumPy nem no `.npy`: a referência é o `ml_dtypes` e a conversão do motor.

**Falta para fechar a fase 0:**
- **o `Tensor` guardar um `NdArray`.** Fica para depois da avaliação do 1B (prevista para 03/11). O `Tensor` é
  o nó do autograd usado pelo treino e pelo servidor que vão medir o 1B, e trocar o armazenamento dele agora
  arriscaria essa medida. A troca vai numa branch própria, e os testes de paridade CPU contra GPU precisam
  passar numa RTX 4090 antes do merge;
- escrever numa view (hoje as views são só de leitura);
- percorrer views não contíguas sem montar a lista de posições, que hoje custa memória.

## Fase 1: situação em 01/11/2026

**Feito:** `sgemm` em `crates/katsu-core/src/gemm.rs`, no esquema do BLIS. B é empacotado em fatias de NR
colunas e A em fatias de MR linhas, com blocos de KC = 256, e as tarefas paralelas são blocos de 64 linhas
de C. Há três microkernels:
- NEON 8×12 (aarch64), com 24 acumuladores e FMA por lane;
- AVX2+FMA 6×16 (x86_64);
- um escalar 4×8 de reserva.

Os testes comparam cada microkernel disponível com uma referência em f64, em formatos que não são múltiplos
dos blocos, de 1×1×1 a 130×513×37. A tolerância é 2·k·ε do f32 vezes a soma de |a|·|b|.

**Medida no Mac M3, f32, matrizes quadradas.** São 3 rodadas alternadas com o NumPy 1.26.4 (OpenBLAS, 8
threads), com mediana de 21 repetições (7 no 2048). O Mac estava ocupado e o tempo variou entre as rodadas;
a tabela mostra a faixa das medianas.

| n | GEMM antigo do backend | `sgemm` | NumPy (OpenBLAS) | NumPy, melhor tempo |
|---|---|---|---|---|
| 512 | 2,5–4,0 ms | 1,22–1,42 ms | 1,47–4,65 ms | 1,18 ms |
| 1024 | 24,7–30,2 ms | 7,2–8,5 ms | 12,4–17,6 ms | 7,6 ms |
| 2048 | 212–302 ms | 66–93 ms | 64–78 ms | 60 ms |

`cargo run -p katsu-core --release --example bench_gemm` mede o lado do Katsu.

**O que isso quer dizer:** no 1024, o `sgemm` fica no tempo do OpenBLAS ou abaixo, e passa no portão. No 2048
ele empata. É de 2× a 4× mais rápido que o GEMM antigo do backend.

**Falta para fechar a fase 1:**
- f64 (`dgemm`);
- medir o microkernel AVX2 numa máquina x86. Ele só passou nos testes de exatidão; o CI roda em x86;
- matrizes retangulares do treino, por exemplo (lote × contexto) × 768 por 768 × 3072;
- ligar o `sgemm` ao `CpuBackend`. Muda a ordem das somas, então espera a avaliação do 1B, como a troca do
  `Tensor`.
