# Escala do Katu: CPU, BPE e batches

Esta etapa usa o motor Rust e seu autograd/Adam, sem PyTorch ou pesos externos.
O núcleo numérico continua em f32; Rayon é a primeira dependência do core.

## O que roda

- `CpuBackend`: matmul forward e backward, linhas independentes via Rayon.
- NEON em aarch64 e AVX2 em x86_64, detectados em runtime, com caudas escalares.
  Intrinsics ficam isolados em `simd.rs`; o restante do crate nega unsafe.
- `Tensor: Send + Sync`: `Arc` para ownership, `RwLock` para gradientes,
  `AtomicBool` para retenção. Forward/VJPs podem trabalhar em paralelo; a publicação
  dos gradientes é serializada e acumulativa. Um guard externo de `grad()` pode
  causar `GradientBorrowed`, sem publicação parcial. Libere-o antes de backward.
- `bmm` em `[B,M,K] × [B,K,N]`, transpose nas últimas dimensões e gradientes.
- `gather_rows` com scatter-add no backward; embeddings dispensam one-hot.
- Cross-entropy fundida, sem materializar one-hot ou grafo de log-probabilidades.
- `forward_batch` retorna `[B,T,V]`; projeções operam em `[B*T,D]`, atenção mantém
  batches separados. `loss` usa esse caminho quando as janelas têm tamanho igual;
  janelas de tamanhos diferentes mantêm a média por janela do contrato anterior.
- BPE byte-level puro Rust, determinístico, serializável e com fallback por byte.
  Não baixa corpus, tokenizer ou pesos. Caracteres não vistos continuam codificáveis.

O layout dos parâmetros v0/v1 e o formato `KATSUCK1` permanecem compatíveis.
Os novos perfis precisam do tokenizer e `config.json` correspondentes ao checkpoint.

## Treinar BPE nas obras locais

```sh
cargo run --release -p katu-core --example train_bpe -- corpus corpus-32k.bpe 32000
```

O corpus é lido em ordem alfabética. Cada obra é dividida em 90% treino e 10%
validação antes do BPE; só o treino aprende merges. A separação ocorre em espaço
UTF-8 válido. O programa remove o envelope Gutenberg, mas não verifica direitos
ou qualidade editorial. Não há tokens artificiais para completar o vocabulário:
se faltarem pares repetidos, ele termina abaixo da meta.

Os exemplos leem o corpus em RAM. A implementação de BPE atualiza contagens só
nas palavras afetadas e usa heap de frequências; não rescaneia todo o corpus por
merge. A unidade de pré-tokenização é palavra incluindo whitespace seguinte.
Não é compatível com o tokenizer HuggingFace do protótipo Python do Arandu.

## Treino real pequeno no Mac

```sh
RAYON_NUM_THREADS=4 cargo run --release -p katu-core --example train_native -- \
  --corpus corpus --tokenizer corpus-32k.bpe --profile tiny \
  --batch-size 2 --steps 10 --backend cpu --output runs/native-smoke-001
```

Salva `model.ckpt`, `tokenizer.bpe`, `config.json` e `metrics.jsonl` no diretório novo.
A loss de validação final fica no config. O exemplo usa Adam e atualiza pesos;
não é uma simulação de métricas. É teste de engenharia, não um assistente treinado.
Somente os pesos são salvos: estado do Adam e retomada ainda não implementados.
`epoch` nas métricas representa passo de otimização, como no exemplo anterior.

## Perfis grandes: descrever antes de alocar

```sh
cargo run --release -p katu-core --example train_native -- \
  --tokenizer corpus-32k.bpe --profile 139m --describe
cargo run --release -p katu-core --example train_native -- \
  --tokenizer corpus-32k.bpe --profile 349m --describe
```

Os nomes são alvos aproximados: o número real depende do vocabulário aprendido.
`native_139m`: 13 camadas, largura 768, 12 cabeças, contexto 512; **138.577.152**
parâmetros com 30.000 tokens. `native_349m`: 22 camadas, largura 1024, 16 cabeças,
contexto 512; **348.856.320** parâmetros com 34.816 tokens. Embedding e cabeça não
compartilham pesos neste layout. Doze camadas/768 não dão 139M para todo vocabulário.

`--describe` não inicializa pesos. Treino grande não foi executado nem considerado
viável em tempo útil no Mac. Só pesos FP32 custam ~554 MB/~1,4 GB nesses perfis;
gradientes, Adam, cópias do otimizador e ativações exigem múltiplos desse tamanho.
Reshape/transpose ainda copiam buffers; atenção não usa FlashAttention, e não há
mixed precision, checkpointing de ativações ou inferência sem grafo.

## Backend CUDA experimental

`Backend: Send + Sync` seleciona matmul, add e cross-entropy e seus VJPs.
O backend escolhido fica no nó e é reutilizado no backward. A feature `cuda`
usa cudarc 0.19.9 para carregar kernels próprios via NVRTC em Linux/NVIDIA.
O caminho é híbrido: demais operações e Adam continuam CPU, com transferências
por operação. Sem GPU/feature, retorna erro explícito, sem fallback silencioso.

Veja [ambiente e comparação CPU×CUDA](../deploy/cuda/README.md). A compilação Rust
foi verificada no Mac, mas os kernels NVRTC e os dois testes NVIDIA não foram
executados: não há máquina NVIDIA disponível. O Dockerfile também não foi executado.

Os perfis 139M/349M não foram iniciados neste Mac de 8 GiB. O preflight estimou
4.24 GiB e 10.26 GiB para batch 1, sequência 16; ambos excederam o orçamento
com margem da memória disponível. `--sequence-length` permite smoke curto mantendo
a tabela de posições original; `--memory-budget-mib` bloqueia alocação acima do
orçamento informado. Estimativas não garantem o pico real.

O [exportador GGUF](gguf.md) preserva pesos e BPE nativos, usando arquitetura
customizada Katu que requer loader próprio no consumidor.

## Verificação e benchmark

```sh
cargo test -p katu-core --release --locked
cargo clippy -p katu-core --all-targets --locked -- -D warnings
cargo fmt --all -- --check
cargo run --release -p katu-core --example bench_matmul -- 256
```

O benchmark compara a referência escalar, SIMD sem threads e SIMD com Rayon.
A referência escalar pode ser autovetorizada pelo compilador; SIMD explícito não
é presumido mais rápido. Publique shape, dtype, arquitetura, threads e flags.
O caminho AVX2 precisa de execução em máquina x86_64; a validação local é arm64/NEON.

Testes cobrem gradientes numéricos já existentes, escalar×paralelo com caudas,
backward concorrente em grafo compartilhado, isolamento de batches, igualdade de
loss/gradientes com janelas separadas, gather com índices repetidos, CE fundida,
BPE determinístico com UTF-8 não visto e rejeição explícita de GPU indisponível.

## Medição local desta entrega (2026-10-24)

Rust 1.88.0, macOS arm64, release com LTO thin do workspace, 8 threads Rayon,
sem `RUSTFLAGS` adicional. Média de 5 execuções após warm-up, matmul 512×512 f32:

| Caminho | Tempo por matmul |
| --- | ---: |
| Referência escalar (autovetorização permitida) | 10,712 ms |
| NEON, uma thread | 10,355 ms |
| NEON + Rayon | 2,157 ms |

São medições locais (~5× no kernel combinado), sem promessa de ganho no treino
inteiro ou em outro hardware. O branch AVX2 não foi executado neste Mac.

BPE: 23 arquivos locais, 6.266.567 bytes de treino, 696.221 bytes de validação,
32.000 tokens, 1,56 s. O corpus disponível tem uma obra a mais que o treino v1
histórico descrito como 22 obras; nenhum livro foi baixado nesta entrega.

Smoke Rust: 2.061.152 parâmetros, batch 2, contexto 16, dez passos Adam,
loss inicial 10,35808, final 10,346785 e validação final 10,341413.
Essa execução prova atualização/checkpoint, não qualidade de linguagem.
Com o tokenizer de 32k, os perfis nominalmente 139m/349m têm exatamente
141.649.152 e 343.089.152 parâmetros, respectivamente. Só foram descritos,
sem alocar/treinar esses modelos grandes.
