# Katsu LM: modelo de linguagem em português

[Índice](README.md)

**Katsu LM v0**: modelo próprio do projeto. Código, treino e pesos são feitos aqui,
sem pesos ou bibliotecas de terceiros; os dados são textos brasileiros em domínio público.
A arquitetura segue a família transformer, técnica publicada abertamente.

O primeiro modelo de linguagem do Katsu: um transformer decoder-only que aprende a
prever o próximo caractere. Tudo em Rust, sem dependências, sobre o mesmo autograd.

## Arquitetura

```text
tokens → one_hot @ wte  +  one_hot(posições) @ wpe
       → [LayerNorm → atenção causal (h cabeças de d/h) → + resíduo
          LayerNorm → MLP (d → 4d → GELU → d)   → + resíduo] × camadas
       → LayerNorm → @ cabeça → logits [t, vocabulário]
```

Cada cabeça tem suas projeções `wq, wk, wv` (d → d/h) e `wo` (d/h → d); somar as
saídas das cabeças equivale a concatená-las e multiplicar por uma Wo [d, d].

| Versão | Contexto | Largura | Camadas × cabeças | Parâmetros | Corpus |
| --- | --- | --- | --- | --- | --- |
| **v0** | 64 | 64 | 2 × 1 | ~116 mil | Dom Casmurro |
| **v1** | 96 | 96 | 4 × 4 | ~481 mil | [22 obras brasileiras](corpus.md) |

Nas duas: lote de 8 janelas por passo, Adam com lr 3e-3. Com uma cabeça, o layout e
a inicialização são os do v0, então um treino v0 continua reproduzível.

Inicialização no estilo GPT-2 (desvio 0.02; projeções de saída escaladas por
1/√(2·camadas)), reproduzível pela seed.

## Treinar

Baixe o [corpus](corpus.md) e treine com a pasta inteira (v1 é o padrão):

```sh
scripts/fetch-corpus.sh
cargo run --release -p katsu-core --example katsu_lm -- corpus metrics.jsonl 6000 v1
```

O primeiro modelo: `-- corpus/machado-dom-casmurro.txt metrics.jsonl 3000 v0`.

- `metrics.jsonl`: uma linha JSON por passo; o monitor arandu (`/dev`) lê esse arquivo ao vivo.
- stderr: a cada 100 passos, perda de validação (últimos 10% de cada obra) e uma
  amostra que continua "Capitu".
- `metrics.jsonl.ckpt`: pesos finais (`save_params` / `load_params`).
- `metrics.jsonl.ckpt.vocab.txt`: vocabulário; `CharTokenizer::from_text` o reconstrói.
- Cabeçalho e rodapé de licença do Gutenberg são removidos; parágrafos são normalizados.

Use `--release`: em debug o treino é ~10× mais lento.

## API

| Item | Contrato |
| --- | --- |
| `CharTokenizer::from_text(&str)` | Vocabulário ordenado dos caracteres do texto |
| `encode(&str)` / `decode(&[usize])` | `IndexOutOfRange` para caractere fora do vocabulário |
| `KatsuLmConfig { vocab_size, block_size, embed_dim, layers, heads }` | Hiperparâmetros; `heads` divide `embed_dim` |
| `init(seed)` | `Vec<Tensor>` de parâmetros, no layout esperado |
| `forward(params, tokens)` | Logits `[t, vocab]`, 1 ≤ t ≤ block_size |
| `loss(params, windows)` | Entropia cruzada média do próximo caractere |
| `generate(params, prompt, count, temperature, seed)` | Amostragem; temperatura ≤ 0 = mais provável |
| `param_count()` / `param_tensors()` | Números treináveis / quantidade de tensores |
| `save_params(params, writer)` / `load_params(reader)` | Formato binário little-endian `KATSUCK1` |

## Verificação

[tests/lm.rs](../crates/katsu-core/tests/lm.rs): tokenizer com acentos, shapes e contagem
de parâmetros, perda inicial ≈ ln(vocabulário), causalidade (tokens futuros não
alteram logits passados), gradientes conferidos por diferenças finitas, treino que
aprende a completar uma frase, geração reproduzível, checkpoint ida e volta e,
para várias cabeças, validação de `heads`, layout, causalidade e diferenças finitas.

## Limites

Tokens por caractere; lote processado janela a janela (sem
matmul em lote); CPU em uma thread. O matmul usa laços vetorizáveis (SIMD automático),
sem blocagem de cache. Modelos de milhões de parâmetros exigem as próximas etapas
de desempenho do [roadmap](roadmap.md).
