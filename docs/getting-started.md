# Começar

[Índice](README.md) · [API](api.md)

## Pré-requisitos

Use um toolchain Rust stable com Cargo, rustfmt e Clippy. A versão mínima de Rust
não foi estabelecida por uma matriz de testes; o CI está configurado para stable.
Python só é necessário para experimentar o módulo PyO3.

```sh
rustc --version
cargo --version
```

No macOS, se Rust já estiver instalado pelo rustup mas não estiver no PATH:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
```

Se não houver Rust instalado, siga o instalador oficial em <https://rustup.rs/>.
Os comandos abaixo devem ser executados na raiz do workspace Katsu.

## Obter o código

```sh
git clone https://github.com/arildo-yank/katsu.git
cd katsu
```

## Executar os exemplos

```sh
cargo test -p katsu-core
cargo run -p katsu-core --example basic
cargo run -p katsu-core --example autograd
cargo run -p katsu-core --release --example conv2d
```

| Exemplo | O que demonstra | Resultado de referência |
| --- | --- | --- |
| `basic` | Soma e matmul | Matmul: `[58, 64, 139, 154]` |
| `autograd` | `sum((a+b)²)` | Loss 52; dA e dB `[8, 12]` |
| `conv2d` | Imagem RGB e oito filtros | `[1,3,224,224]` → `[1,8,224,224]` |

Os fontes estão em [examples](../crates/katsu-core/examples).
A perda da demonstração Conv2D não é uma métrica de acurácia; a imagem é sintética.

## Usar o core em outro projeto local

No `Cargo.toml` da aplicação, ajuste o caminho real:

```toml
[dependencies]
katsu-core = { path = "../KATSU/crates/katsu-core" }
```

Não se pressupõe que o pacote esteja publicado no crates.io.
Para compor uma rede, as operações já propagam gradientes; não há ainda API de
atualização de parâmetros nem otimizador. Evite apresentar isso como treino completo.

## Python experimental

O scaffold exige Python >= 3.10 e Maturin. O único atributo público atual é
`__version__`. O build completo ainda não foi validado neste ambiente.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'maturin>=1.9.4,<2'
maturin develop --release
python -c 'import katsu; print(katsu.__version__)'
```

Não existe `katsu.Tensor` nesta versão. Não use `pip install katsu` como instrução
de instalação deste repositório: a titularidade do nome no PyPI não foi verificada.
No Windows, a ativação do ambiente virtual usa um comando diferente do exemplo Unix.

## Problemas comuns

| Sintoma | Ação |
| --- | --- |
| `cargo: command not found` | Confira instalação e PATH; no macOS tente o comando acima |
| Falha de DNS em crates.io | Resolva o acesso à rede ou use a [validação offline do core](testing.md) |
| `NonScalarBackward` | Use `sum()?.backward()` ou `backward_with(&seed)` |
| `GradientBorrowed` | Libere o guard de `grad()` antes de `backward`/`zero_grad` |
| `MissingForward` | Execute um forward válido antes do backward local da camada |
| `InvalidConv` | Confira NCHW, OIHW, canais, stride e tamanho do kernel |
| `katsu.Tensor` não encontrado | A classe Python ainda não está implementada |
