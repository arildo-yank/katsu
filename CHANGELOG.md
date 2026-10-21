# Changelog

## Unreleased

A versão nos manifests é 0.1.0. Ainda não foi estabelecida uma release publicada.

### Adicionado

- Workspace Cargo com core Rust e scaffold PyO3/Maturin.
- Tensores CPU f32, validação de shapes, soma, mul, matmul e sum.
- Grafo dinâmico com autograd reverso, VJP e acumulação de gradientes.
- Conv2D NCHW/OIHW com stride, padding e derivadas de entrada/pesos.
- Acesso ao mapa de ativação e limpeza do cache da camada.
- 27 testes de integração, incluindo diferenças finitas, grafo profundo e convergência.
- Otimizador `Sgd`, loop `fit` e `EpochMetrics` em JSON Lines para o monitor arandu.
- Operação `relu` e exemplo `train` (regressão linear).
- `retain_grad()`: backward publica gradientes só em folhas e nós retidos.
- Destruição iterativa do grafo e adjuntos alocados sob demanda no backward.
- `Cargo.lock` versionado e hook local de pre-commit.
- Exemplos de tensores, autograd e imagem RGB 224×224.
- Workflow de CI do core, documentação técnica, guias de contribuição e SVGs.

### Limitações conhecidas

- Python expõe somente a versão; não há katsu.Tensor ou wheel validado.
- SGD com taxa fixa apenas; não há Adam, mini-batches, Grad-CAM ou serviço HTTP.
- CPU single-thread; sem comparação de performance com outros frameworks.
