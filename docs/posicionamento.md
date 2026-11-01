# Posicionamento do Katu

[Índice](README.md)

*31/10/2026. Os números marcados como medidos têm comando ou arquivo de origem. O resto é
estimativa ou plano, e está dito assim.*

## O nome

**Katu** vem do tupi e quer dizer "bom". O motor se chamava Katsu até 01/11/2026. A troca veio
porque `katsu` já era um pacote de outra pessoa no PyPI, e porque o projeto passou a usar nomes de
línguas indígenas brasileiras, como o modelo Arandu ("sabedoria", em tupi-guarani). O pedido de
registro da marca no INPI ainda não foi feito.

## Katu e Arandu: as duas camadas

**O Katu é a tecnologia; o Arandu é o produto.**

| | Katu (o motor) | Arandu (o assistente) |
|---|---|---|
| O que é | Infraestrutura de IA em Rust | App e API para quem trabalha com documentos |
| Peças | Núcleo de tensores e autograd, GEMM próprio, CUDA e BF16, `katu-math`, busca na biblioteca (`katu-rag`), `katu-server` e o pacote Python `katu` | App desktop, usuários com papéis, filtro de dados pessoais, roteamento entre motores, trilha de auditoria encadeada e medidor de substituição |
| Quem usa | Equipe técnica que integra IA no próprio sistema | Funcionário que faz perguntas |
| Como se usa | HTTP (com rota compatível com a OpenAI) ou `import katu` | Pelo app |

**Duas formas de comprar:**
- **só o motor:** a empresa integra o Katu no próprio sistema e cuida da interface e das regras;
- **o produto completo:** os funcionários usam o Arandu.

**O que o Arandu faz hoje, com o Katu como motor (01/11/2026):**
- responde perguntas sobre a biblioteca citando o trecho e a fonte;
- faz contas pelo `katu-math`;
- recusa dados pessoais;
- tudo isso sem sair da máquina.

**Copiloto, não piloto automático.** O Arandu acelera a leitura, a busca de provas e as contas; a redação e a decisão continuam com o profissional. Em laudos, isso quer dizer preencher campos conferíveis num modelo que o profissional controla, e não escrever a conclusão.

**O que ele ainda não faz:** juntar vários documentos numa resposta escrita, ou entender um problema descrito com palavras. Isso exige um modelo que entenda o pedido. Hoje esse modelo é o DeepSeek, na nuvem, com o filtro de dados pessoais na frente. O modelo próprio de 1B ainda está em treino e não foi medido.

## Por que o Katu existe

**Soberania.** Um motor de IA feito no Brasil, que um órgão ou empresa brasileira possa auditar
inteiro, compilar sozinho e rodar sem internet, sem depender de engenharia estrangeira que possa
chegar a dados sigilosos.

O Katu não tenta ser maior que o PyTorch. Tenta ser **pequeno o bastante para ser auditado** e
**completo o bastante para treinar e rodar um modelo de linguagem de verdade**.

## O que já é verificável

| Afirmação | Situação | Como conferir |
|---|---|---|
| O núcleo (`katu-core`) depende de 6 crates externas: `rayon`, `rayon-core`, `crossbeam-deque`, `crossbeam-epoch`, `crossbeam-utils`, `either` | medido | `cargo tree -p katu-core -e normal` |
| O servidor com CUDA (`katu-server --features cuda`) soma 75 crates, a maior parte por `tokio`, `hyper` e `serde` | medido | `cargo tree -p katu-server --features cuda -e normal` |
| Nenhuma crate de cliente HTTP ou TLS (`reqwest`, `ureq`, `rustls`, `openssl`); o `hyper` só atende, não chama ninguém | medido | `cargo tree --workspace -e normal` |
| ~16 mil linhas de Rust no workspace, ~13 mil no núcleo | medido | `find crates -name '*.rs' \| xargs cat \| wc -l` |
| Treina um GPT de 134M em CPU (Mac M3) e um de 1B em GPU NVIDIA | medido | `docs/katu-lm.md`, `docs/v2-precisao-e-memoria-2026-10-28.md` |
| O motor CUDA bate com o CPU: 16 testes de paridade passaram numa RTX 4090, perda igual a ~0,03% por passo | medido, 31/10/2026 | `deploy/cuda/runpod_v2_probe.sh` |
| BF16 + Adam de 8 bits + recomputação levam o 1B de 21,9 GiB para 11,4 GiB de VRAM reservada | medido numa 4090 | `docs/v2-precisao-e-memoria-2026-10-28.md` |
| O 1B cabe numa GPU de 12 GB (RTX 4070) | **estimativa**, no limite; nunca rodou numa 4070 | — |
| Velocidade contra o PyTorch: no 134M numa 4090, mesmos lotes e mesmos pesos, o PyTorch foi 1,4× mais rápido em FP32 e 3,4× no seu melhor modo (BF16 + FlashAttention) contra o melhor do Katu na época (TF32); a perda do passo 1 coincide nos dois | medido, 26/10/2026, antes do BF16 do Katu | `docs/benchmarks/gpu-engine-2026-10-26/pytorch-2026-10-26/` |
| Distância para o PyTorch com o BF16 atual do Katu | **não medido** | falta repetir o comparativo |
| O servidor aceita o formato da API da OpenAI (`/v1/chat/completions`, `/v1/models`): o cliente oficial funciona trocando só a `base_url`, e o que não há (stream, JSON mode, tools) é recusado com erro | medido, 31/10/2026, fora do Docker | `crates/katu-server/src/openai.rs` |
| Isso cobre o formato, não a qualidade: o 134M é leitor de documentos (`/rag/*`), não chat geral. Numa primeira conferência, acertou 1 de 5 respostas com gabarito | medido, 31/10/2026, amostra pequena | `ARANDU/docs/INTEGRATION_GUIDE.md` |

## Quem mais está nesse espaço

| | Linguagem | Força | Diferença para o Katu |
|---|---|---|---|
| **PyTorch** | Python/C++ | padrão do mercado, várias GPUs, todo modelo | centenas de dependências, milhões de linhas, governança estrangeira |
| **Burn** (Tracel) | Rust | muitos backends (CUDA, ROCm, Metal, WebGPU), DDP em construção na 0.21 | genérico e grande; empresa estrangeira |
| **Candle** (Hugging Face) | Rust | roda Llama, Whisper e Stable Diffusion; foco em inferência | treino é secundário; empresa estrangeira |
| **tinygrad** | Python | pequeno, muitos backends | depende de Python |
| **Katu** | Rust | pequeno, auditável, treino e inferência do modelo inteiro, documentação em português | um backend de GPU (CUDA), uma família de modelo, uma GPU por vez |

Em amplitude, o Burn e o Candle estão à frente hoje. O Katu compete em outra coisa: **ser
auditável por inteiro e entregue com código-fonte a quem precisa de soberania**.

## O que soberania exige, além de um motor próprio

Um motor próprio não basta. O que entrega dado sigiloso para fora quase sempre é a rede, não o
framework.

| Requisito | Situação |
|---|---|
| Operar sem internet (air-gapped): sem telemetria, sem atualização automática | **verificado**: compila, testa e treina com a rede cortada pelo sistema (`scripts/soberania/offline.sh`) |
| Dependências travadas e guardadas localmente (`Cargo.lock`, `cargo vendor`) | **feito**: `scripts/soberania/vendor.sh`, com hash de cada arquivo |
| Lista pública do que entra (SBOM) | **feito**: `docs/soberania/sbom.json` (CycloneDX), conferido no CI |
| Build reproduzível: quem compila o fonte obtém o mesmo binário | **verificado na mesma máquina** em caminhos diferentes; entre máquinas, ainda não |
| Licença clara | MIT |
| Pesos e dados de treino de origem conhecida | sem pesos de terceiros; corpus documentado em `docs/corpus.md` |
| Não depender de um fabricante de GPU | **depende**: CUDA é da NVIDIA, código fechado. O CPU é a alternativa que já roda |

**Limites que nenhum framework resolve:** o driver CUDA, o macOS e o Accelerate são fechados e
estrangeiros; o compilador Rust e as crates vêm de fora, embora abertos e auditáveis. Para
ambiente sigiloso, o caminho é CPU ou GPU em máquina sem rede, com o binário compilado pelo
próprio órgão.

## O que falta para chamar de alternativa

Em ordem:

1. **Repetir o comparativo com o PyTorch** com o BF16 do Katu: o script já existe
   (`scripts/bench_pytorch.py`, mesmos lotes e mesmos pesos). A medida de 26/10 deixou o Katu
   3,4× atrás no melhor modo de cada um; falta saber quanto o BF16 fechou. ~15 minutos numa 4090,
   menos de US$ 1. Até lá, a única afirmação de desempenho honesta é "mais lento que o PyTorch".
2. **Pacote de soberania:** feito em 31/10/2026 ([`soberania/`](soberania/README.md)). Falta
   comparar o build entre duas máquinas (macOS e Linux, ou dois Linux).
3. **Uso por terceiros:** `pip install katu` a partir do `katu-python`, exemplos, documentação da
   API.
4. **Mais modelos:** hoje o Katu treina uma família (o GPT do Arandu). Um framework precisa servir
   a modelos que não foram previstos.
5. **Várias GPUs:** ver `ARANDU/docs/DDP_ROADMAP_V3.md`, ~3–4 semanas (estimativa).
6. **Outra GPU além da NVIDIA:** para reduzir a dependência de um fabricante. Projeto grande, sem
   estimativa.
7. **Katu Embedding Model (busca por sentido):** um codificador treinado no próprio Katu, sem
   PyTorch nem pesos de terceiros. A busca por palavras parou em 12/26 no 1.º trecho na régua da
   NR-12, e o dicionário de sinônimos não mudou isso (31/10/2026). Ver [roadmap](roadmap.md),
   seção 5. Semanas.

## A frase para investidor ou órgão público

> Núcleo com 6 dependências, todas auditáveis. Roda sem internet. Treina e roda um modelo de
> linguagem de 1B. Código-fonte entregue a quem compra.

Desde 31/10/2026 a frase se confere com os scripts de [`soberania/`](soberania/README.md). O que ainda
falta para uma frase de desempenho é o item 1.
