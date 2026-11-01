# Testes, evidências e limites

[Índice](README.md)

## Verificação padrão

Na raiz, com Rust/Cargo disponíveis e acesso às dependências do workspace:

```sh
cargo fmt --all -- --check
cargo test -p katu-core
cargo clippy -p katu-core --all-targets -- -D warnings
cargo run -p katu-core --release --example conv2d
cargo doc -p katu-core --no-deps
```

O workflow [Rust core](../.github/workflows/ci.yml) executa formatação, testes,
Clippy e o exemplo Conv2D em Ubuntu com Rust stable. Ele não verifica wheels,
Python, outras plataformas, versões mínimas de Rust ou performance.

## Core offline com Cargo

O core agora depende de Rayon. Depois de preencher o cache com `cargo fetch --locked`,
use `cargo test -p katu-core --offline --locked`. Compilar o arquivo isolado com
`rustc` sem resolver dependências não é mais uma alternativa suportada.

## Cobertura existente

| Suíte | Testes | Casos |
| --- | --- | --- |
| [Tensor](../crates/katu-core/tests/tensor.rs) | 6 | Shapes, overflow, soma, matmul, escalares e vazios |
| [Autograd](../crates/katu-core/tests/autograd.rs) | 8 | Nós compartilhados, seeds, acumulação, borrows, retain_grad, grafo profundo e diferenças finitas |
| [Operações](../crates/katu-core/tests/ops.rs) | 11 | Broadcasting, transpose, reshape, GELU, softmax, LayerNorm, máscara causal, cross-entropy, Adam e um bloco de atenção, todos com diferenças finitas |
| [Katu LM](../crates/katu-core/tests/lm.rs) | 11 | Tokenizer, shapes, perda inicial, causalidade, diferenças finitas (1 e 4 cabeças), aprendizado, geração e checkpoint |
| [Treinamento](../crates/katu-core/tests/training.rs) | 6 | SGD à mão, erros, convergência de regressão linear, divergência, JSON e ReLU |
| [Conv2D](../crates/katu-core/tests/conv2d.rs) | 7 | Forward conhecido, padding/stride, dX/dW, cache, batch e inicialização |
| [Escala](../crates/katu-core/tests/scaling.rs) | 10 | SIMD/Rayon, grafo compartilhado, batches, gather, CE fundida, BPE, perfis e rejeição de CUDA |

As diferenças finitas da Conv2D perturbam entradas e pesos, com batch > 1,
múltiplos canais e kernel retangular. Testes de valor conhecido complementam
essa checagem: uma derivada consistente, sozinha, não prova a convenção do forward.
Os testes usam tolerâncias para f32; não prometem igualdade bit a bit entre plataformas.

## Evidência disponível

59 testes locais passaram via Cargo. O core passou pelo clippy-driver sem avisos;
há benchmark de matmul e treino Rust pequeno em [Escala](scaling.md).
A execução remota do GitHub Actions e o build Python completo ainda não foram
verificados. Não há medição comparativa com PyTorch.

## Antes de afirmar performance

Registre hardware, toolchain, flags de compilação, shapes, dtype, número de threads,
warmup, quantidade de repetições e medida de dispersão. Compare forward e backward
separadamente, em cargas equivalentes. Inclua a memória temporária e verifique os
resultados numéricos. Não derive promessas de desempenho da linguagem utilizada.
