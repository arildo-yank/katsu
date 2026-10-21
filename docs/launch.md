# Katsu — lançamento de prévia técnica

Status: textos preparados para revisão, ainda não publicados.

## Hacker News

Título:

Show HN: Katsu – A Rust tensor engine with autograd and Conv2D

Texto:

I'm building Katsu, an open-source Rust deep learning engine with a longer-term
focus on explainability. This is an early technical preview.

The current core implements contiguous CPU tensors, reverse-mode autograd, and
NCHW Conv2D with gradients for both inputs and weights. The repository includes
19 passing core tests, including finite-difference gradient checks, and a runnable
224×224 RGB convolution example.

Activations retain their graph identity and gradients, providing a foundation for
future attribution methods. Grad-CAM itself is not implemented yet. Python
packaging is scaffolded, but the Python Tensor API, optimizers, and an end-to-end
training example are still pending.

I'm sharing this stage to get feedback on the autograd API, gradient correctness,
and what would make a small Rust engine useful for learning and experimentation.
There are no PyTorch performance comparisons yet.

Repository: https://github.com/arildo-yank/katsu

## Texto para uma landing page

### Katsu — construa e inspecione os primeiros passos de uma rede neural

Um núcleo open-source em Rust com tensores, diferenciação automática e convolução
2D. Em desenvolvimento, com explicabilidade como direção do projeto.

**Explore o código. Execute uma convolução. Inspecione seus gradientes.**

O que funciona hoje:

- Tensores CPU f32 com operações básicas.
- Autograd reverso com acumulação em grafos compartilhados.
- Conv2D em NCHW com forward e backward.
- Acesso às ativações e aos seus gradientes.

O que estamos construindo a seguir:

- Otimizador e um exemplo completo de treinamento.
- API Python para Tensor.
- Grad-CAM e visualização de atribuições.
- Benchmarks reproduzíveis antes de anunciar ganhos de performance.

Chamada principal: **Ver o código no GitHub**
Chamada secundária: **Executar o exemplo Conv2D**

```sh
cargo run -p katsu-core --release --example conv2d
```

Licença MIT. Prévia técnica CPU single-thread; ainda não é uma alternativa completa
a frameworks de treinamento estabelecidos.

## Condições para publicar estes textos

- Substituir o campo de URL pelo repositório público real.
- Aguardar uma execução do CI antes de mostrar um badge de testes aprovados.
- Não anunciar `pip install katsu` até disponibilizar e verificar o pacote correto.
- Não usar “powered by Katsu” em projetos que ainda não executam o Katsu.
- Não apresentar mapas de atribuição como explicações exatas ou causais de decisões.
- Manter comparação com PyTorch para depois de medir cargas equivalentes.

A configuração do CI está no repositório, mas a execução remota ainda não foi
verificada. Não há promessa de alcance, posição na página inicial ou número de
upvotes suficiente para promoção no Hacker News.
