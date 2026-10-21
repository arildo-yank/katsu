# GPT de caracteres em português

[Índice](README.md)

O primeiro modelo de linguagem do Katsu: um transformer decoder-only que aprende a
prever o próximo caractere. Tudo em Rust, sem dependências, sobre o mesmo autograd.

## Arquitetura

```text
tokens → one_hot @ wte  +  one_hot(posições) @ wpe
       → [LayerNorm → atenção causal (1 cabeça) → + resíduo
          LayerNorm → MLP (d → 4d → GELU → d)   → + resíduo] × camadas
       → LayerNorm → @ cabeça → logits [t, vocabulário]
```

| Item | Valor no exemplo `gpt_pt` |
| --- | --- |
| Contexto (`block_size`) | 64 caracteres |
| Largura (`embed_dim`) | 64 |
| Camadas | 2 |
| Parâmetros | ~110 mil (depende do vocabulário) |
| Lote | 8 janelas por passo |
| Otimizador | Adam, lr 3e-3 |

Inicialização no estilo GPT-2 (desvio 0.02; projeções de saída escaladas por
1/√(2·camadas)), reproduzível pela seed.

## Treinar

Com um texto em domínio público, por exemplo *Dom Casmurro* (Machado de Assis,
Project Gutenberg #55752) salvo em `corpus/dom-casmurro.txt`:

```sh
cargo run --release -p katsu-core --example gpt_pt -- corpus/dom-casmurro.txt metrics.jsonl 2000
```

- `metrics.jsonl`: uma linha JSON por passo; o monitor arandu (`/dev`) lê esse arquivo ao vivo.
- stderr: a cada 100 passos, perda de validação (últimos 10% do texto) e uma amostra
  que continua "Capitu".
- `metrics.jsonl.ckpt`: pesos finais (`save_params` / `load_params`).
- Cabeçalho e rodapé de licença do Gutenberg são removidos; parágrafos são normalizados.

Use `--release`: em debug o treino é ~10× mais lento.

## API

| Item | Contrato |
| --- | --- |
| `CharTokenizer::from_text(&str)` | Vocabulário ordenado dos caracteres do texto |
| `encode(&str)` / `decode(&[usize])` | `IndexOutOfRange` para caractere fora do vocabulário |
| `GptConfig { vocab_size, block_size, embed_dim, layers }` | Hiperparâmetros |
| `init(seed)` | `Vec<Tensor>` de parâmetros, no layout esperado |
| `forward(params, tokens)` | Logits `[t, vocab]`, 1 ≤ t ≤ block_size |
| `loss(params, windows)` | Entropia cruzada média do próximo caractere |
| `generate(params, prompt, count, temperature, seed)` | Amostragem; temperatura ≤ 0 = mais provável |
| `param_count()` / `param_tensors()` | Números treináveis / quantidade de tensores |
| `save_params(params, writer)` / `load_params(reader)` | Formato binário little-endian `KATSUCK1` |

## Verificação

[tests/gpt.rs](../crates/katsu-core/tests/gpt.rs): tokenizer com acentos, shapes e contagem
de parâmetros, perda inicial ≈ ln(vocabulário), causalidade (tokens futuros não
alteram logits passados), gradientes conferidos por diferenças finitas, treino que
aprende a completar uma frase, geração reproduzível e checkpoint ida e volta.

## Limites

Uma cabeça de atenção; tokens por caractere; lote processado janela a janela (sem
matmul em lote); CPU em uma thread. O matmul usa laços vetorizáveis (SIMD automático),
sem blocagem de cache. Modelos de milhões de parâmetros exigem as próximas etapas
de desempenho do [roadmap](roadmap.md).
