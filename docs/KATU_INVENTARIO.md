# Inventário de contratos, consumidores e processos

Data: 01/11/2026. Tarefas A2 e A3 do [marco 1](KATU_MARCO1_TAREFAS.md). Levantado por leitura de código e de processos, sem
reiniciar nada. Retrato deste dia: conferir de novo antes de mudar o que depende dele.

## Rotas HTTP do `katu-server`

| Rota | Depende do modelo | Consumidor conhecido | Regra |
|---|---|---|---|
| `POST /math` | Não | Arandu: `math_tool.py`, tela Matemática, ferramenta do DeepSeek | Preservar; recusa é HTTP 400 com `{"error"}` |
| `POST /math/ask` | Não | Arandu: `api.py`, conversa | Preservar; `understood: false` manda seguir como conversa |
| `POST /v1/compute` | Não | Nenhum ainda | Novo, `katu.result/1`; recusa é HTTP 200 |
| `POST /generate` | Sim | Arandu, testes de medição | Fora deste trabalho |
| `POST /rag/ask`, `/rag/search`, `/rag/documents` | Em parte | Arandu | Fora deste trabalho |
| `GET /health` | | Arandu, app | Fora deste trabalho |
| `POST /v1/chat/completions`, `GET /v1/models` | Sim | Formato OpenAI | Fora deste trabalho |

O servidor só sobe com `config.json`, `tokenizer.bpe` e `model.ckpt` válidos, mesmo para responder contas (tarefa A6).

## Campos de `/math` lidos pelos consumidores

| Campo | Quem lê | Observação |
|---|---|---|
| `error` | `math_tool.py` | Vira HTTP 422 em `/api/math` e informação para o modelo |
| `result`, `result_latex`, `input_latex`, `input` | `Matematica.jsx`, modelo | |
| `approximate` | `Matematica.jsx`: selo "Aproximado" ou "Calculado" | Desde a Fase 1, `eval` com decimais aparece como "Aproximado" |
| `steps` | `Matematica.jsx` | LaTeX |
| `value_at`, `value_at_approximate`, `method`, `note`, `elapsed_us` | Modelo, pelo JSON da ferramenta | `value_at_approximate` é novo e opcional |
| `understood`, `ok`, `markdown` | `api.py`, em `/math/ask` | |

Texto de erro é contrato de fato: o Arandu mostra e o modelo lê. A comparação das cadeias de `katu-math` e de `math.rs`
entre `71e2863` e a mudança para erro tipado não encontrou cadeia removida; só foram acrescentadas.

## Python (`import katu`)

Matemática: `derivative`, `antiderivative`, `evaluate`, `integrate`, `root`, `ode`, `latex`; erros viram `ValueError` com a
mesma mensagem. Arrays: `array`, `zeros`, `arange`, `concatenate`, `stack`, `load`, `matmul`. O erro tipado só passou por
`cargo check` no Python: **a extensão não foi construída nem exercitada neste ciclo**, e o `newton` mais estrito agora
recusa assíntotas também em `katu.root`.

## Formatos e nomes que não mudam

| Item | Valor | Motivo |
|---|---|---|
| Tokenizador | Magias `KATSUBP1`, `KATSUBP2` | Modelos antigos precisam carregar |
| Checkpoint | Magia `KATSUCK1` | Idem |
| Pasta e repositório | `KATSU`, `arildo-yank/katsu` | Treino, backups e pods usam esses caminhos |
| Régua do piloto | `ARANDU/scripts/eval_knowledge.py` e JSONL | SHA256 congelado pelos critérios do piloto; usa `KATSU_URL` |

## Configuração de integração

- `KATU_URL`: servidor do modelo. `KATU_MATH_URL`: base do servidor de contas, sem `/math`, e pode ser outro.
- `arandu-web/start-api-v2.sh`: modelo na 8793, contas na 8792, por padrão.
- O app sobe o `katu-server` de um dos seguintes: o binário do pacote, ou `KATU_SRC/target/release/katu-server` com
  `KATU_MODEL_DIR`, lidos do `.env`. Por isso **uma compilação em `KATSU/target/release` muda o que o app executa na próxima
  abertura**. A worktree isolada compila em `KATSU-motor-fase1/target`, para isso não acontecer por acidente.

**Dois apps, uma porta.** `Arandu.app` (API 8766) e `Arandu V2.app` (API 8767) sobem o Katu na mesma 8792. O app vigia a porta
enquanto o app está aberto e deixa em paz um servidor que já responde. Quem abriu primeiro fica dono do servidor e o derruba ao
fechar; o outro assume em cerca de 5 segundos, medidos nos apps instalados em 01/11/2026 (ARANDU `b934966`). Log em
`ARANDU/data/logs/arandu-app-<porta da API>.log`.

## Processos em 01/11/2026, 20:05

| Processo | Depende do checkout KATSU? | Regra |
|---|---|---|
| `guarda_retoma.py` e `caffeinate`, em `ARANDU V2/medidas/15-treino-1b-2026-11-01` | Não pelo código-fonte; vigia o pod por SSH | Não tocar; após reinício do Mac, relançar este e nunca `guarda.py` |
| Treino 1B no pod `jfo7c9h0li3ua5` | Não usa o checkout do Mac durante a execução | Não tocar; estado conferido só pelo log do guarda |
| `Arandu V2.app`, PID 36101 | Só o binário, ao abrir | Sem reiniciar sem autorização |
| API em Python na `127.0.0.1:8767` | Fala por HTTP; depende da 8792 para contas | Sem reiniciar |
| Portas 8792 e 8793 | Nada escutando; V2 aberto sem filho `katu-server`, nem encerrado | Testes usam porta própria (18792), e o servidor temporário é desligado no fim |

Fica fora deste levantamento o que roda só no pod, além do que o guarda registra.

## Matriz de testes

| Conjunto | Comando | Mac M3, rustc 1.99.1 | Linux x86, GitHub Actions |
|---|---|---|---|
| katu-core | `cargo test -p katu-core --locked` | 116 aprovados | Não executado neste trabalho |
| katu-math | `cargo test -p katu-math --locked` | 37 aprovados | Não executado |
| Rota `/math` | `cargo test -p katu-server --locked math::` | 7 aprovados | Não executado |
| Rota `/v1/compute` | `cargo test -p katu-server --locked compute::` | 7 aprovados | Não executado |
| Consumidor | `arandu-web/.venv/bin/python -m unittest test_math` | 17 aprovados | Fora: o ARANDU não sai do Mac |
| Referência | `scripts/math_reference/check.py` | Sem divergência | Fora do CI: NumPy e SciPy não são dependência |
| CUDA | `cargo test -p katu-core --test cuda -- --ignored` | Não validado | Precisa de GPU NVIDIA em Linux |
