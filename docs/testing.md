# Testes, evidências e limites

[Índice](README.md)

## Verificação padrão

Na raiz, com Rust/Cargo disponíveis e acesso às dependências do workspace:

```sh
cargo fmt --all -- --check
cargo test -p katsu-core
cargo clippy -p katsu-core --all-targets -- -D warnings
cargo run -p katsu-core --release --example conv2d
cargo doc -p katsu-core --no-deps
```

O workflow [Rust core](../.github/workflows/ci.yml) executa formatação, testes,
Clippy e o exemplo Conv2D em Ubuntu com Rust stable. Ele não verifica wheels,
Python, outras plataformas, versões mínimas de Rust ou performance.

## Core offline com rustc

O core não importa crates externas. O workspace também contém PyO3; por isso
`cargo test -p katsu-core --offline` pode falhar se PyO3 não estiver no cache.
A alternativa abaixo compila os fontes reais e roda as três suítes de integração,
sem substituir os testes por implementações equivalentes.

```sh
mkdir -p target/manual
rustc --edition=2021 --crate-name katsu_core --crate-type rlib \
  crates/katsu-core/src/lib.rs --out-dir target/manual
for suite in tensor autograd conv2d training ops lm; do
  rustc --edition=2021 --test "crates/katsu-core/tests/$suite.rs" \
    --extern katsu_core=target/manual/libkatsu_core.rlib \
    -o "target/manual/$suite-tests" && "target/manual/$suite-tests" || break
done
rustdoc --edition=2021 --crate-name katsu_core \
  crates/katsu-core/src/lib.rs -o target/manual/docs
```

Execute os comandos em um shell Unix; se uma compilação falhar, corrija-a antes
de prosseguir. Essa alternativa valida o core, não o empacotamento Cargo/PyO3.

## Cobertura existente

| Suíte | Testes | Casos |
| --- | --- | --- |
| [Tensor](../crates/katsu-core/tests/tensor.rs) | 6 | Shapes, overflow, soma, matmul, escalares e vazios |
| [Autograd](../crates/katsu-core/tests/autograd.rs) | 8 | Nós compartilhados, seeds, acumulação, borrows, retain_grad, grafo profundo e diferenças finitas |
| [Operações](../crates/katsu-core/tests/ops.rs) | 11 | Broadcasting, transpose, reshape, GELU, softmax, LayerNorm, máscara causal, cross-entropy, Adam e um bloco de atenção, todos com diferenças finitas |
| [Katsu LM](../crates/katsu-core/tests/lm.rs) | 11 | Tokenizer, shapes, perda inicial, causalidade, diferenças finitas (1 e 4 cabeças), aprendizado, geração e checkpoint |
| [Treinamento](../crates/katsu-core/tests/training.rs) | 6 | SGD à mão, erros, convergência de regressão linear, divergência, JSON e ReLU |
| [Conv2D](../crates/katsu-core/tests/conv2d.rs) | 7 | Forward conhecido, padding/stride, dX/dW, cache, batch e inicialização |

As diferenças finitas da Conv2D perturbam entradas e pesos, com batch > 1,
múltiplos canais e kernel retangular. Testes de valor conhecido complementam
essa checagem: uma derivada consistente, sozinha, não prova a convenção do forward.
Os testes usam tolerâncias para f32; não prometem igualdade bit a bit entre plataformas.

## Evidência disponível

49 testes locais passaram via Cargo. O core passou pelo clippy-driver sem avisos;
o exemplo RGB 224×224 executou em build otimizado. Isso não equivale a benchmark.
A execução remota do GitHub Actions e o build Python completo ainda não foram
verificados. Não há medição comparativa com PyTorch.

## Antes de afirmar performance

Registre hardware, toolchain, flags de compilação, shapes, dtype, número de threads,
warmup, quantidade de repetições e medida de dispersão. Compare forward e backward
separadamente, em cargas equivalentes. Inclua a memória temporária e verifique os
resultados numéricos. Não derive promessas de desempenho da linguagem utilizada.
