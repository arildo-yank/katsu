# Dimensão efetiva dos pesos

`scripts/effective_dimension.py` mede quantas direções de cada matriz de pesos o modelo usa de fato,
pela SVD. Serve para comparar runs de larguras diferentes (2M com `d=32`, 134M com `d=768`).

```sh
python3 scripts/effective_dimension.py runs/mac-sft-special-v1 \
  --base runs/mac-sft-special-v1/migrated.ckpt \
  --json runs/mac-sft-special-v1/effective-dimension.json
python3 scripts/effective_dimension.py runs/<run-134m> --compare runs/mac-sft-special-v1/effective-dimension.json
```

- `erank`: posto efetivo, `exp(H(p))` com `p = s / Σs`.
- `PR`: razão de participação das variâncias, `(Σs²)² / Σs⁴`.
- `d90`: direções que guardam 90% da energia.
- `frac`: `erank / máximo`. É o número comparável entre tamanhos; cada linha mostra também o
  `frac` de uma matriz gaussiana do mesmo formato.
- `--base` (checkpoint de antes do SFT) separa os tokens cuja linha de entrada o SFT mudou:
  só tokens presentes nos dados recebem gradiente.
- Mede pesos, não ativações. Requer numpy. No 134M leva ~100 s e ~800 MB de RAM.

## Referência: `mac-sft-special-v1` (2M, d=32, 1 camada × 4 cabeças)

| Medida | Valor | Pesos aleatórios |
|---|---|---|
| Atenção por cabeça (`Q·Kᵀ`), frac média / mínima | 0,37 / 0,22 | 0,94 |
| MLP, frac média | 0,87 | 0,96 |
| Embeddings dos 178 tokens do SFT, PR | 14,0 de 32 | 27,1 |
| Saída: variância na direção dominante | 84,6 % | ~3 % |

- As cabeças usam 1,7 a 3,8 das 8 dimensões; a cabeça 0 põe 99,6 % numa única direção. Com tão
  pouco espaço para comparar conteúdo, o modelo não copia números do contexto (AranduBench 2/10).
- A direção dominante da saída tem correlação 0,997 com o logit médio: é a frequência do token,
  já que o modelo não tem bias de saída.
- Um checkpoint 134M com pesos aleatórios dá frac 0,98 em todos os grupos, igual à referência.
  Num 134M treinado, frac de atenção bem abaixo de 0,98 indica estrutura aprendida; perto de 0,37
  indica o mesmo colapso do 2M.
