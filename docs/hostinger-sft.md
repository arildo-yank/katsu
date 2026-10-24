# Execução GPU: preflight, smoke e treino piloto

Sem PyTorch. Backend CUDA híbrido: Adam e várias operações permanecem no host.
Ainda não houve validação em NVIDIA. O launcher não provisiona nem destrói instâncias.

```bash
./bilingual_gpu.sh --preflight
./bilingual_gpu.sh --check
PRETRAIN_STEPS=500 STEPS=500 ./bilingual_gpu.sh --run
```

- `--preflight`: verifica arquivos, host Linux, NVIDIA, Docker e orçamento RAM.
- `--check`: constrói ambiente, testa kernels e 10 passos CPU×CUDA do modelo pequeno;
  depois executa 10 passos SFT do modelo de 134.373.888 parâmetros, 12 camadas,
  768 dimensões, 12 heads, contexto 256 e vocab 32004.
- `--run`: repete testes CUDA e executa a fase de corpus seguida por SFT.
  Defaults de 500+500 passos são piloto, não promessa de qualidade final.

Dataset independente: `data/sft-bilingual-v2/`; tokenizer:
`runs/bpe-bilingual-v2/tokenizer.bpe`; corpus: `data/corpus-bilingual-v1/train.txt`.
Sem BASE, pesos são inicializados aleatoriamente. Com BASE, exige arquitetura
compatível e tokenizer idêntico. Adam é reiniciado entre fases.

LR do corpus: 1e-4; SFT: LR (padrão 5e-5). Métricas distinguem pretrain/adapt/sft.
`CHECKPOINT_EVERY=25` grava latest.ckpt via rename após salvar arquivo temporário.
São pesos intermediários, não estado completo do otimizador. Saída final inclui
model.ckpt, tokenizer, configuração, métricas, relatório e SHA256SUMS.

MEMORY_BUDGET_MIB padrão 12288: o preflight exige mais 2048 MiB disponíveis além
desse orçamento. VRAM não substitui RAM host. Estimativa não garante ausência de OOM.

Custos e encerramento pertencem ao controle do provedor. Limite de passos não é
limite financeiro. Antes de excluir a instância, copiar os artefatos ao iMac,
verificar SHA-256 e leitura do checkpoint. Excluir somente a instância identificada
para esta execução e verificar a confirmação no painel/API do provedor.

## Benchmark e limite da sessão

`./bilingual_gpu.sh --benchmark` executa exatamente 100 atualizações sobre janelas
256 do corpus, com zero atualizações SFT. Desativa as avaliações completas para
medir o processamento; o relatório deixa suas losses de avaliação como null.
Os testes CUDA do launcher continuam obrigatórios. `step_seconds` inclui forward,
backward, Adam e eventual checkpoint periódico; o tempo total também é registrado.

```bash
python3 scripts/report_cuda_benchmark.py runs/SAIDA-BENCHMARK --output /tmp/benchmark.json
```

O relatório distingue velocidade medida, overhead e projeções. Uma quantidade de
tokens equivalente ao corpus não garante cobertura integral, pois as janelas são
amostradas. O piloto padrão 500+500 não é um treinamento completo até convergência.
Velocidade de corpus não é medida direta de SFT; avaliação e coleta também custam tempo.

Limites autorizados para esta primeira sessão: US$10 e duas horas totais de instância.
Não iniciar o treino seguinte se sua estimativa, margem e coleta excederem o tempo
restante. O launcher não aluga nem exclui instâncias; o controle do provedor precisa
estar disponível antes de provisionar. Em OOM/falha CUDA, interromper e excluir a
instância desta execução imediatamente, conforme instrução do usuário.
