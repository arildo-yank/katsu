# Corpus PT/EN e SFT com resposta em português

Versão `bilingual-v1`, 24/10/2026. Implementação sem PyTorch e sem APIs de modelos.
O objetivo de responder em PT-BR a contextos em inglês é uma hipótese a avaliar;
o corpus e o tokenizer, por si, não demonstram transferência entre línguas.

## Dados e licenças

`data/corpus-bilingual-v1/manifest.json` registra documentos, autoria, DOI/URL,
licença, hashes de origem e texto limpo, rejeições, splits e proporções. Os XMLs
licenciados e as respostas de busca são preservados em `raw/`; os textos completos
limpos ficam em `documents/`. As 23 obras locais permanecem intactas em `corpus/`.

Busca: índice público NCBI PMC; texto integral: Europe PMC. A busca Europe PMC
retornou 503 e não foi usada como índice final. O coletor exige URL explícita
CC BY ou CC0 no bloco de permissões do artigo. Acesso aberto sem licença explícita,
CC BY-NC e CC BY-ND são recusados. Referências, figuras e tabelas não entram no texto
limpo. As transformações são extração de parágrafos, normalização de espaços e
remoção de duplicatas exatas. O manifesto e o XML mantêm atribuição e licença.

A seleção automática por tema/licença ainda não equivale a revisão científica de
qualidade. Estudos podem discordar, incluir limitações ou populações distintas.
O corpus inglês é texto publicado, não uma base de prescrições clínicas aprovadas.

## Mistura e avaliação

60% PT / 40% EN é uma proporção inicial de **bytes UTF-8 limpos**, não de documentos
ou tokens. O coletor divide os documentos em treino e validação antes da mistura,
com ordenação por hash. Seleciona prefixos proporcionais de cada documento, sem
repetir artigos para inflar volume. O conteúdo completo continua arquivado.
Duplicatas exatas são removidas; similaridade aproximada entre artigos não foi
eliminada. A proporção não é um ótimo demonstrado e deve ser comparada futuramente
com outras misturas em avaliação independente.

`train.txt` é o único arquivo utilizado para aprender os merges BPE. A compressão
em amostras de validação mede eficiência de tokenização; não mede compreensão.
Nunca passe o diretório inteiro ao exemplo `train_native`, pois ele leria também
`validation.txt`. Para experimentar esse exemplo, passe o arquivo `train.txt`
explicitamente; ele ainda reserva internamente seus últimos 10% para validação.

## Tokenizer e pesos

O treinador solicita 32000 tokens básicos e acrescenta 4 especiais: 32004 no total
se houver pares suficientes. `report.json` registra o tamanho realmente alcançado,
IDs especiais, round-trip PT/EN e tamanho dos exemplos SFT. Nenhum vocábulo técnico
precisa obrigatoriamente virar um token único: subpalavras são parte do BPE.

O novo vocabulário muda o significado dos IDs comuns. **Não carregar pesos antigos
com esse tokenizer**, mesmo que o número de tokens seja idêntico. O caminho desta
versão usa inicialização nova; o launcher GPU compara o tokenizer de BASE byte a
byte com o solicitado. O tokenizer e checkpoints anteriores ficam preservados.

## Dataset de instrução

`data/sft-bilingual-v2/` inclui os 302 exemplos PT existentes e 116 novos exemplos
sintéticos com contexto inglês/resposta PT. A avaliação inclui os 10 casos antigos
e 15 novos. Há fatos numéricos, unidades, síntese, distinção entre associação e
causalidade e ausência de informação. Nenhuma resposta foi gerada por serviço externo.

Os casos de validação usam valores ou enunciados separados dos exemplos do mesmo
tipo de treino. A separação não prova ausência de sobreposição semântica: parte dos
templates é compartilhada. A suíte ainda é pequena e não substitui um benchmark
externo de transferência cross-lingual. A inferência é acessada diretamente pelo serviço Rust; não há cliente de outra aplicação neste pipeline.

## Comandos

```bash
python3 scripts/build_bilingual_corpus.py --portuguese corpus --output data/corpus-bilingual-v1
python3 scripts/build_bilingual_sft.py --source data/sft --output data/sft-bilingual-v2
cargo run --release --locked -p katsu-server --bin train_bilingual_bpe -- \
  data/corpus-bilingual-v1 runs/bpe-bilingual-v2 data/sft-bilingual-v2
```

Saídas completas são protegidas contra sobrescrita. Para repetir a coleta, use
outra versão; o índice pode mudar. Para repetir exatamente, use o cache XML, os
hashes e os textos preservados. O SFT aceita caminhos explícitos com `--train` e
`--eval`; isso não altera os datasets padrão.

No futuro host Linux/NVIDIA:

```bash
./bilingual_gpu.sh --check
./bilingual_gpu.sh --run
```

`--preflight` verifica pré-requisitos sem treino. `--check` executa testes CUDA
mais dez passos do modelo maior. `--run` inclui a fase de corpus seguida de SFT.
Consulte [execução GPU](hostinger-sft.md) para os defaults, checkpoints e limites.
Esses modos foram preparados localmente; não houve execução NVIDIA.

O pré-treinamento autoregressivo em inglês também ensina a prever texto inglês;
“inglês somente leitura” não é uma separação garantida da arquitetura. A preferência
por saída em português precisa ser treinada e medida na geração livre. O percentual
global de conhecimento em inglês e a suposta ordem em que outros modelos aprendem
línguas não foram usados como fatos ou premissas mensuradas deste pipeline.

## Resultado local registrado

- 23 arquivos PT preservados e 146 artigos EN aceitos; 55 candidatos rejeitados.
- Treino: 6.303.175 bytes PT e 4.202.062 bytes EN; 60,0003% PT.
- Validação: 817.052 bytes PT e 544.702 bytes EN; aproximadamente 60% PT.
- 169 documentos limpos verificados por SHA-256, sem documento idêntico nos dois splits.
- BPE: 32004 tokens, incluindo IDs especiais 32000–32003; round-trip aprovado.
- SFT: 418 exemplos de treino e 25 de avaliação; máximo de 87 e 85 tokens.
- Dez passos CPU com pesos aleatórios: loss média de treino 10,36268 → 10,34883;
  loss de avaliação 10,36573 → 10,35381; verificações de finitude aprovadas.
- Os pesos/tokenizer antigos foram preservados. O manifesto histórico de integração foi arquivado durante a separação de projetos.
- Quatro testes Python do coletor e Clippy do treinador BPE passaram.
- GPU: não executada. O --check local recusa corretamente macOS.

Relatórios: `runs/bpe-bilingual-v2/report.json`,
`runs/bilingual-cpu-smoke-v1/sft-report.json` e
`runs/bpe-bilingual-v2/verification.json`.

A versão v2 remove vínculos de nomes e prompts com aplicações externas. Os números do smoke de dez passos acima pertencem à execução histórica v1, não a um retreino dos pesos v2.
