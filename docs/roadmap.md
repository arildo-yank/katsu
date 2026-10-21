# Roadmap

[Índice](README.md)

Não há datas prometidas. Cada marco precisa de implementação e validação;
documentação ou estrutura de pastas, isoladamente, não tornam uma funcionalidade pronta.

## Base disponível

- [x] Tensor CPU com validação de shape e kernels básicos.
- [x] Grafo dinâmico e backward com acumulação.
- [x] Conv2D forward/backward, stride e padding.
- [x] Acesso às ativações e seus gradientes.
- [x] Testes numéricos e exemplos executáveis.
- [x] Estrutura de documentação e CI do core.
- [ ] Confirmar uma execução remota do CI.

## 1. Demonstrar aprendizado

- [x] Definir API para parâmetros atualizáveis sem invalidar grafos anteriores.
- [x] Implementar SGD e uma política clara de zero_grad por passo.
- [x] Adicionar ReLU e perda para um problema pequeno.
- [x] Criar um treinamento reproduzível, com seed e dados de demonstração.
- [x] Emitir métricas por época (JSON Lines) para o monitor arandu.

Critério: o teste de integração executa múltiplos passos e demonstra redução da
perda dentro de tolerâncias documentadas. Ter backward não basta para concluir este marco.

## 1b. Peças do transformer

- [x] Broadcasting em add/mul com gradientes reduzidos nas dimensões expandidas.
- [x] transpose, reshape, softmax, log_softmax, LayerNorm, GELU e máscara causal.
- [x] cross_entropy, one_hot e Adam.
- [x] Bloco de atenção causal com gradientes verificados por diferenças finitas.

## Rumo à base da IA Arandu

Crescimento gradual, cada etapa com algo demonstrável:

1. **Katsu fala português**: GPT de caracteres com alguns milhões de parâmetros,
   treinado no Mac com textos brasileiros em domínio público, loss ao vivo no monitor arandu.
2. **Desempenho**: matmul blocado/SIMD, threads (`Arc` + `rayon`), f16 e quantização int8.
3. **Inferência de pesos abertos**: carregar safetensors e gerar texto no chat do Arandu.
4. **Escala em GPU alugada**: backend GPU e modelos de centenas de milhões a bilhões de parâmetros.
5. **Atribuição por token** no modelo próprio: explicabilidade real das respostas.

Ordem de grandeza: 1 bilhão de parâmetros em f16 ocupa ~2 GB; 1 trilhão, ~2 TB e
milhares de GPUs para treinar. Escalas acima de bilhões dependem de investimento.

## 2. Interface Python

- [ ] Expor Tensor e operações com PyO3.
- [ ] Mapear TensorError para exceções Python.
- [ ] Resolver afinidade de thread do core Rc/RefCell antes de liberar o GIL.
- [ ] Construir wheel e testar instalação em um ambiente limpo.

Critério: um exemplo Python usa a API real e reproduz o resultado Rust. Verificar
os nomes de distribuição antes de anunciar instalação pública.

## 3. Explicabilidade

- [ ] Selecionar um objetivo/score e uma camada de interesse.
- [ ] Implementar Grad-CAM a partir de features e gradientes.
- [ ] Validar a ponderação, normalização e o tratamento de mapas degenerados.
- [ ] Gerar overlay em uma imagem com exemplo reproduzível.

Critério: heatmap derivado do objetivo real, com testes e limitações explicadas.
Não anunciar causalidade ou explicação exata das decisões.

## 4. Performance e deploy

- [ ] Medir kernels e memória em cargas publicadas.
- [ ] Selecionar otimizações com base nos resultados (blocagem, SIMD, threads ou GPU).
- [ ] Definir inferência e modelo de concorrência antes de criar backend HTTP.
- [ ] Avaliar ownership/arena para grafos muito profundos.

Critério: benchmarks reproduzíveis e uma demonstração de inferência com limites
de recursos definidos. Nenhum ganho competitivo é presumido.
