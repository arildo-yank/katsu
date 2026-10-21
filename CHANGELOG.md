# Changelog

## Unreleased

A versão nos manifests é 0.1.0. Ainda não foi estabelecida uma release publicada.

### Adicionado

- Workspace Cargo com core Rust e scaffold PyO3/Maturin.
- Tensores CPU f32, validação de shapes, soma, mul, matmul e sum.
- Grafo dinâmico com autograd reverso, VJP e acumulação de gradientes.
- Conv2D NCHW/OIHW com stride, padding e derivadas de entrada/pesos.
- Acesso ao mapa de ativação e limpeza do cache da camada.
- 19 testes de integração, incluindo diferenças finitas.
- Exemplos de tensores, autograd e imagem RGB 224×224.
- Workflow de CI do core, documentação técnica, guias de contribuição e SVGs.

### Limitações conhecidas

- Python expõe somente a versão; não há katsu.Tensor ou wheel validado.
- Não há otimizador, treinamento completo, Grad-CAM ou serviço HTTP.
- CPU single-thread; sem comparação de performance com outros frameworks.
- Destruição de grafos Rc extremamente profundos pode consumir a pilha.
