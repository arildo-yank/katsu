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
| **0. Array completo** | Tipos f64, f32, i64 e bool; *views* com *strides* sem cópia; fatias com passo, inclusive negativo; transposição; broadcasting; promoção de tipos; reduções; leitura e escrita de `.npy` | Ciência precisa de f64; o pipeline de dados precisa de fatias sem cópia | Gabaritos do NumPy passando | **Primeira parte feita** (01/11/2026): 50 casos passando |
| **1. GEMM no nível BLAS** | Empacotamento e blocagem por cache no estilo BLIS, microkernels NEON e AVX2/AVX-512, f32 e f64 | Atenção, MLP, e a base das fatorações | 1024×1024 no tempo do OpenBLAS (partida: Katsu 16,1 ms contra 8,1 ms) | Falta |
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
`tests/fixtures/array/gen.py`.

| Peça | Conferido como |
|---|---|
| Tipos bool, i64, f32 e f64, com `astype` | Contra o NumPy; NaN ou infinito para i64 é recusado |
| Fatias com passo, inclusive negativo, índices negativos e fora do intervalo | 7 fatias de um cubo 3×4×5 |
| Transposição e permutação de eixos, sem cópia | Contra o NumPy, e conferido que o buffer é o mesmo |
| Broadcasting, soma, subtração, produto e divisão verdadeira, com a promoção de tipos | 11 casos, igualdade exata |
| Comparações | `<` e `==`, com promoção |
| Soma (em pares, como o NumPy), média, máximo e mínimo, por eixo ou no total | 20 casos, com NaN contaminando o máximo |
| `.npy` versões 1 a 3, ordem C e Fortran | A escrita gera os mesmos bytes que o NumPy |

**Falta para fechar a fase 0:**
- escrever numa view (hoje as views são só de leitura);
- índice por máscara e por lista de inteiros;
- concatenar e empilhar; `argmax`, soma acumulada, produto, variância e desvio padrão;
- os tipos i32, u8, f16 e bf16;
- percorrer views não contíguas sem montar a lista de posições, que hoje custa memória;
- o `Tensor` guardar um `NdArray`.
