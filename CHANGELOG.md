# Changelog

## Unreleased

A versão nos manifests é 0.1.0. Ainda não foi estabelecida uma release publicada.

### Adicionado

- Workspace Cargo com core Rust e scaffold PyO3/Maturin.
- Tensores CPU f32, validação de shapes, soma, mul, matmul e sum.
- Grafo dinâmico com autograd reverso, VJP e acumulação de gradientes.
- Conv2D NCHW/OIHW com stride, padding e derivadas de entrada/pesos.
- Acesso ao mapa de ativação e limpeza do cache da camada.
- GPT de caracteres (`GptConfig`, `CharTokenizer`), geração com temperatura,
  checkpoint binário (`save_params`/`load_params`) e exemplo `gpt_pt`.
- Matmul e seu backward com laços vetorizáveis: ~11× mais rápido no treino do GPT.
- 46 testes de integração, incluindo diferenças finitas, grafo profundo e convergência.
- Broadcasting em `add`/`mul`; `transpose`, `reshape`, `gelu`, `softmax`, `log_softmax`,
  `layer_norm`, `cross_entropy`, `one_hot`, `causal_mask` e `scalar`.
- Otimizador `Adam` e trait `Optimizer`; `fit` recebe `&mut` de qualquer otimizador.
- Otimizador `Sgd`, loop `fit` e `EpochMetrics` em JSON Lines para o monitor arandu.
- Operação `relu` e exemplo `train` (regressão linear).
- `retain_grad()`: backward publica gradientes só em folhas e nós retidos.
- Destruição iterativa do grafo e adjuntos alocados sob demanda no backward.
- `Cargo.lock` versionado e hook local de pre-commit.
- Exemplos de tensores, autograd e imagem RGB 224×224.
- Workflow de CI do core, documentação técnica, guias de contribuição e SVGs.

### Limitações conhecidas

- Python expõe somente a versão; não há katsu.Tensor ou wheel validado.
- Sem mini-batches, GPU, Grad-CAM ou serviço HTTP; transpose/reshape copiam dados.
- CPU single-thread; sem comparação de performance com outros frameworks.
