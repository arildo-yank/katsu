# Treino CPU e serviço local

```sh
bash scripts/train_mac_stable.sh
# O diretório precisa ser novo; alternativa: bash scripts/train_mac_stable.sh runs/novo-treino
python3 scripts/analyze_stability.py runs/mac-cpu-stable
RAYON_NUM_THREADS=4 cargo run --release -p katu-server --locked -- runs/mac-cpu-stable 8787
```

O servidor usa [Axum](https://docs.rs/axum/0.8.6/axum/), escuta somente em
127.0.0.1, aceita uma geração por vez (429 se ocupado) e limita tamanho do corpo,
prompt e saída. Não faz chamadas externas. Encerrar com Ctrl+C.

```sh
curl http://127.0.0.1:8787/health
curl -H 'Content-Type: application/json' http://127.0.0.1:8787/generate \
 -d '{"prompt":"Era uma vez","max_new_tokens":16,"temperature":0.7,"seed":42}'
python3 scripts/test_http_local.py -v
```

`text` contém só a continuação. A resposta inclui token_ids, prompt_tokens,
context_tokens, prompt_truncated e valid_utf8. Bytes UTF-8 incompletos são
substituídos na visualização, mas os IDs originais permanecem disponíveis.
Não há garantia factual na geração livre. O servidor carrega o checkpoint na
inicialização; reinicie para trocar o modelo. Usa apenas configurações pequenas.

## Execução observada em 2026-10-24

Modelo: 2.061.152 parâmetros, BPE 32k existente, Adam 0.001, batch 2, contexto 16,
seeds 1/2026, quatro threads Rayon. Pesos inicializados do zero; corpus local:
`corpus/machado-dom-casmurro.txt`, separado 90/10 antes da tokenização.

2.000 passos em aproximadamente 56,69 segundos (não foi um ensaio de várias horas).
Média dos primeiros 100 passos: 9,4574; dos últimos 100: 7,3346.
Loss inicial 10,3675, final 8,3762, EMA final 7,3718. O caminho não é monotônico.
Gradiente L2 máximo 11,1682; elemento absoluto máximo 3,6789. Nenhum NaN/Inf
observado em loss/gradientes/pesos. Validação em uma única janela: 8,1088;
não é avaliação representativa de perplexidade no corpus inteiro.

`runs/mac-cpu-stable/` contém métricas por passo, tokenizer, config e checkpoint.
`latest.ckpt` é backup de pesos a cada 500 passos; não salva o estado Adam
para retomada exata. `model.ckpt` é o resultado final. Não houve gradient clipping.

## GGUF

```sh
python3 scripts/export_to_gguf.py --run runs/mac-cpu-stable --output runs/mac-cpu-stable/Katu-stable-F32.gguf
```

O exportador existente lê o BPE nativo. Continua sendo GGUF de arquitetura
customizada `katu`; **não é um fallback utilizável pelo llama.cpp** sem implementar
suporte ao modelo/tokenizador. Veja [contrato GGUF](gguf.md). CUDA permanece sem
execução NVIDIA nesta etapa.

## Continuação

Veja [contexto 256 e SFT](sft-arandu.md) para o checkpoint seguinte, novos padrões
de amostragem HTTP e a avaliação de fidelidade ao JSON.
