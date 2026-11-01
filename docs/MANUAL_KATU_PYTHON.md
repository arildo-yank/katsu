# Katu para Python

`katu` é a ponte Python do motor Katu. A conta acontece em Rust; o Python só chama.

- **`katu.Array`:** array N-dimensional com tipo, no modelo do NumPy. Fatias e transposição não copiam dados.
- **`katu.math`:** derivada simbólica, primitivas conferidas, integral, raízes e EDO. Responde exato ou recusa.
- **`katu.plot`:** gráficos 2D de funções, com polos, saltos e lacunas marcados pelo motor. Exporta SVG, CSV e JSON.

*Situação em 01/11/2026: funciona compilando a partir do código. Ainda não está no PyPI. O nome `katu` está
livre lá, mas ainda não foi registrado. Os exemplos abaixo foram rodados no Python 3.13, num Mac M3, e estão
em `python/tests/test_katu.py`.*

## Instalação

Precisa do Rust (`rustup`) e de Python 3.10 ou mais novo.

**Para testar sem instalar nada** (compila e roda os testes):

```bash
PYTHON=python3.13 sh scripts/test_python.sh
```

Depois disso, `import katu` funciona de dentro da pasta `python/`.

**Para gerar o pacote instalável (wheel):** precisa do [maturin](https://www.maturin.rs).

```bash
pip install maturin
maturin build --release
pip install target/wheels/katu-*.whl
```

O wheel é `abi3`: o mesmo arquivo serve para qualquer Python a partir do 3.10, na mesma plataforma.

## Gráficos

```python
import katu

katu.plot.plot("1/x", "x^3", x=(-1, 1)).save("curvas.svg")

fig = katu.plot.figure(x=(0, 6.28), theme="dark", width=1000, height=600)
fig.function("sin(x)", label="seno").function("a*cos(x)", label="cosseno")
```

Variável além de `x` vai em `params`, como em `katu.plot.figure(x=(0, 1), params={"a": "1/3"})`.

| O que `build()` devolve | Conteúdo |
|---|---|
| `.series`, `.axes`, `.warnings`, `.alt_text` | Partes do artefato `katu.plot/1` |
| `.data`, `.to_json()` | O artefato inteiro |
| `.svg`, `.csv` | Exportações geradas pelo motor |
| `.save(caminho)` | Grava `.svg`, `.csv` ou `.json`, conforme a extensão |

No Jupyter, o gráfico aparece sozinho, em SVG. Recusa levanta `katu.plot.PlotRefused`, um `ValueError` com `status` e `code`
do contrato, sem gráfico parcial. Só 2D, em escala linear; não há PNG nem 3D. Contrato em
[KATU_PLOT_CONTRATO.md](KATU_PLOT_CONTRATO.md).

## Primeiros passos

```python
import katu

a = katu.array([1, 2, 3])
b = katu.array([4, 5, 6])
print(a + b)                       # Array([5, 7, 9], dtype='i64')

m = katu.array([[1.0, 2.0], [3.0, 4.0]], dtype="f32")
print(m @ m)                       # Array([[7.0, 10.0], [15.0, 22.0]], dtype='f32')
print(m.T)                         # transposta, sem cópia
print(m[:, ::-1])                  # colunas invertidas, sem cópia
print(m.sum(), m.mean(axis=0))     # 10.0 Array([2.0, 3.0], dtype='f32')
print(m[m > 2.0])                  # Array([3.0, 4.0], dtype='f32')
```

## Arrays

### Criar

| Função | Faz |
|---|---|
| `katu.array(dados, dtype=None)` | De listas aninhadas. Sem `dtype`: bool, i64 ou f64, conforme os valores |
| `katu.zeros(forma, dtype="f64")` | Zeros |
| `katu.arange(n)` | 0, 1, …, n − 1, em i64 |
| `katu.load(caminho)` | Lê um `.npy` do NumPy ou do Katu |

**Tipos:** `bool`, `u8`, `i32`, `i64`, `f16`, `bf16`, `f32` e `f64`. Também aceitam os nomes do NumPy
(`"float32"`, `"int64"` e os outros).

### Ler

`a.shape`, `a.dtype`, `a.ndim`, `a.size`, `len(a)` e `a.tolist()`.

### Fatiar

Como no NumPy: `a[1]`, `a[1:5:2]`, `a[::-1]`, `a[0, :, -1]`.

- Uma fatia é uma *view*: lê o mesmo buffer, sem copiar.
- Um Array de bool seleciona por máscara: `a[a > 2]`.
- Uma lista de inteiros seleciona linhas: `a[[3, 0]]`.
- Selecionar um único elemento devolve um número do Python.

### Contas

`+`, `-`, `*`, `/`, comparações (`<`, `==` e as outras) e `-a`, com broadcasting.

**A promoção de tipos é a do NumPy 1.26**, conferida nas 49 combinações:
- `i64 + f32` dá f64;
- a divisão de inteiros dá f64;
- `bool + bool` dá bool (OU), e `bool - bool` é erro.

**Um número do Python fica com o tipo do array**, como no NumPy 2:
- `katu.array([1.5], dtype="f32") + 1` continua f32;
- um float com um array de inteiros dá f64.

### Produto de matrizes

`a @ b` ou `katu.matmul(a, b)`, em arrays 2-D de **f32**. Usa o GEMM em blocos do Katu: num Mac M3, o
1024×1024 levou de 7,2 a 8,5 ms, contra 12,4 a 17,6 ms do NumPy com OpenBLAS. Outros tipos dão `TypeError`;
converta antes com `a.astype("f32")`. O GEMM em f64 ainda não existe.

### Reduções

`sum`, `mean`, `var`, `std`, `max`, `min`, `argmax` e `argmin`.
- Todas aceitam `axis=`; `var` e `std` aceitam também `ddof=`.
- Sem `axis`, o resultado é um número do Python.
- Um NaN contamina o máximo, e `argmax` aponta o primeiro NaN, como no NumPy.

### Forma e tipo

`reshape([...])`, `transpose()`, `transpose(2, 0, 1)`, `broadcast_to([...])`, `astype("f16")`,
`katu.concatenate([a, b], axis=0)` e `katu.stack([a, b], axis=1)`.

### Arquivos

`a.save("x.npy")` grava num formato que o `numpy.load` lê, com os mesmos bytes que o NumPy escreveria.
bf16 não existe no formato `.npy` e é recusado.

## Cálculo: `katu.math`

```python
import katu

katu.math.derivative("x^2 - 5*x")               # '2 * x - 5'
katu.math.antiderivative("x^2 - 5*x")           # 'x^3 / 3 - 5 * x^2 / 2'  (sem o + C)
katu.math.integrate("x^2 - 5*x", 0, 3)          # -13.5
katu.math.root("x^2 - 2", 1.0)                  # 1.4142135623730951
katu.math.ode("y", 0, 1, 1)                     # 2.718281828459…  (dy/dt = y, y(0) = 1, em t = 1)
katu.math.evaluate("a*x + 1", x=2, a=3)         # 7.0
katu.math.latex("x^2/3")                        # '\\frac{x^{2}}{3}'
```

- **Escreva a multiplicação:** `5*x`, não `5x`.
- **Ângulos são em radianos.**
- **Integral, raiz e EDO são aproximações numéricas,** com tolerância ou número de passos que você pode
  passar (`tol=`, `steps=`).
- **O que o motor não sabe fazer é recusado com `ValueError`, nunca chutado.** Exemplos: divisão por zero,
  variável sem valor, primitiva que ele não sabe escrever em forma fechada, ou método que não converge.
  As primitivas são conferidas derivando o resultado antes de sair.

## Por que usar o Katu

- **Não usa OpenBLAS nem outra biblioteca de álgebra externa.** O GEMM, os arrays e o cálculo são código do
  projeto. O núcleo Rust depende de 6 crates (o `rayon` e as que ele traz); o `katu-math` não depende de
  nenhuma.
- **Roda sem internet**, e nada sai da máquina.
- **É auditável:** cada operação está no repositório, conferida contra gabaritos gerados pelo NumPy.

**Onde o NumPy ainda é melhor:** ele tem muito mais funções; tem GEMM em f64; no 2048×2048 o NumPy empata
com o Katu; e ele é instalável com um comando. O Katu ainda não substitui o NumPy num projeto inteiro: serve
onde as peças acima bastam.

## Limites conhecidos

- Ainda não há escrita em views (`a[0] = 1`); as views são só de leitura.
- O produto de matrizes é só em f32 e 2-D.
- O autograd e o treino do Katu ainda não estão expostos no Python.
- A soma de u8 dá i64; no NumPy dá u64, com os mesmos valores.
