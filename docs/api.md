# Referência da API Rust

[Índice](README.md) · [Tensor](../crates/katsu-core/src/tensor.rs) · [Conv2D](../crates/katsu-core/src/conv2d.rs)

Importação pública: `use katsu_core::{Tensor, TensorError, Op, Conv2D};`.
Nas tabelas, `Result<T>` abrevia `std::result::Result<T, TensorError>`;
não há alias Result exportado pelo crate. `&self` foi omitido onde indicado.

## Tensor

Handle clonável para um nó compartilhado. `clone()` preserva identidade; não faz
cópia independente. Shape e dados são privados, acessíveis por slices imutáveis.

| Método | Retorno | Contrato |
| --- | --- | --- |
| `Tensor::new(data: Vec<f32>, shape: &[usize])` | `Result<Tensor>` | Assume ownership dos dados; valida tamanho e copia shape |
| `Tensor::zeros(shape: &[usize])` | `Result<Tensor>` | Aloca buffer preenchido com zero |
| `shape()` | `&[usize]` | Dimensões |
| `data()` | `&[f32]` | Dados contíguos row-major |
| `rank()` | `usize` | Número de dimensões |
| `numel()` | `usize` | Quantidade de elementos |
| `op()` | `Op` | Operação que produziu o nó |
| `parents()` | `&[Tensor]` | Pais na ordem dos operandos |
| `add(rhs: &Tensor)` | `Result<Tensor>` | Soma; shapes idênticos |
| `mul(rhs: &Tensor)` | `Result<Tensor>` | Produto elemento a elemento; shapes idênticos |
| `matmul(rhs: &Tensor)` | `Result<Tensor>` | Somente matrizes: `[m,k] @ [k,n]` |
| `sum()` | `Result<Tensor>` | Soma em escalar com shape `[]`; vazio soma zero |
| `backward()` | `Result<()>` | Saída com um elemento; seed 1 |
| `backward_with(seed: &[f32])` | `Result<()>` | VJP com seed achatado do tamanho da saída |
| `grad()` | `Result<Ref<'_, Option<Vec<f32>>>>` | Guard de leitura; inicialmente None |
| `zero_grad()` | `Result<()>` | Define grad deste nó como None |
| `retain_grad()` | `()` | Faz backward publicar o grad deste intermediário |
| `retains_grad()` | `bool` | Verdadeiro para folhas e nós com `retain_grad` |

Shape `[]` exige um elemento. Shape com alguma dimensão zero produz tensor vazio.
Operações não fazem broadcasting. NaN/Inf seguem aritmética f32 e não são rejeitados.
Todos os operandos são registrados; não existe flag `requires_grad`.
Backward publica `grad` só em folhas e em nós marcados com `retain_grad()`;
a saída de `Conv2D::forward` já vem marcada.
Libere o guard de grad antes de chamar backward ou zero_grad.

## Op

Enum público com `Debug`, `Clone`, `Copy`, `PartialEq` e `Eq`:
`Leaf`, `Add`, `Mul`, `Matmul`, `Sum`, `Conv2D { stride: usize, padding: usize }`.

O método que aplica as derivadas locais é interno. O usuário inspeciona `op()` e
`parents()`, mas não injeta operações ou modifica o grafo. Conv2D registra os pais
na ordem `[input, weights]`; as operações binárias usam `[left, right]`.

## Conv2D

Os parâmetros numéricos abaixo são `usize`, exceto seed (`u64`).
Kernel é `(usize, usize)`; stride/padding são iguais nos dois eixos.

| Método | Retorno | Contrato |
| --- | --- | --- |
| `Conv2D::new(in_channels, out_channels, kernel_size, stride, padding)` | `Result<Conv2D>` | Pesos uniformes; seed 42 |
| `Conv2D::new_seeded(in_channels, out_channels, kernel_size, stride, padding, seed)` | `Result<Conv2D>` | Inicialização reproduzível |
| `Conv2D::from_weights(weights: Tensor, stride, padding)` | `Result<Conv2D>` | Pesos OIHW; preserva o nó |
| `weights()` | `&Tensor` | Nó dos pesos |
| `in_channels()` / `out_channels()` | `usize` | Canais de entrada/saída |
| `kernel_size()` | `(usize, usize)` | Altura/largura do kernel |
| `stride()` / `padding()` | `usize` | Configuração espacial |
| `forward(input: &Tensor)` | `Result<Tensor>` | Entrada NCHW, saída NCHW com grafo |
| `backward(grad_output: &Tensor)` | `Result<(Tensor, Tensor)>` | dX e dW do último forward válido; não acumula grad |
| `get_activation_map()` | `Option<Tensor>` | Handle da última saída pré-ativação |
| `clear_cache()` | `()` | Descarta input/output retidos pela camada |

Construtores exigem canais/kernel positivos e stride > 0. Forward exige rank 4,
canais compatíveis e H/W positivos. Batch zero é aceito. Grad_output local exige
shape completo igual à saída. Backward local retorna tensores folha, sem grafo
de derivadas superiores. Nenhum desses métodos realiza atualização de pesos.

## TensorError

Implementa `Debug`, `Display` e `std::error::Error`.

| Variante | Significado |
| --- | --- |
| `ShapeOverflow` | Produto/tamanho/aritmética espacial não representável |
| `InvalidConv(&'static str)` | Configuração ou shapes inválidos da convolução |
| `MissingForward` | Conv2D sem forward válido no cache |
| `NonScalarBackward` | `backward()` chamado em saída com numel diferente de 1 |
| `GradientBorrowed` | Gradiente ainda emprestado quando seria alterado |
| `DataLength { expected, actual }` | Dados ou seed com quantidade incorreta |
| `ShapeMismatch` | Operação elemento a elemento com shapes distintos |
| `RankMismatch { left, right }` | Matmul com rank diferente de 2 |
| `MatmulMismatch { left_columns, right_rows }` | Dimensões internas incompatíveis |
| `Allocation(TryReserveError)` | Reserva falível de buffer não pôde ser atendida |

Use `?` para propagar erros. `Allocation` expõe a causa via `Error::source()`.
A variante não cobre todas as alocações internas ou as feitas pelo chamador.
As verificações de shape não proíbem NaN/Inf; verifique a estabilidade numérica da aplicação.

## Rustdoc

```sh
cargo doc -p katsu-core --no-deps --open
```

O [guia de testes](testing.md) também mostra como gerar rustdoc sem resolver o workspace.
