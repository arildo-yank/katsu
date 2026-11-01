# Plano de evolução do Katu como motor computacional

Data: 01/11/2026
Estado: proposta de planejamento. Não autoriza nem representa implementação, alteração de serviços, treinamento, publicação ou gastos.

## 1. Objetivo e alcance

Evoluir o Katu para exercer, como ferramenta de um modelo de linguagem, o papel de um motor computacional semelhante ao Wolfram Alpha: interpretar pedidos estruturados, executar cálculos, consultar dados com procedência e devolver resultados verificáveis, gráficos e explicações reproduzíveis.

O foco inicial é katu-math e katu-core. O conjunto completo de requisitos recebidos inclui também conhecimento factual, dados atualizados, imagens e domínios especializados. Essas capacidades exigem componentes adjacentes; não devem ser colocadas indiscriminadamente dentro das duas bibliotecas matemáticas.

Há dois marcos distintos:
1. Motor matemático confiável para uso como ferramenta pelo Katu/Arandu.
2. Plataforma de conhecimento computável que cubra progressivamente todas as categorias solicitadas.

Equivalência ampla ao Wolfram Alpha é uma direção de produto, não uma capacidade atual nem uma promessa de prazo. O catálogo recebido não contém todos os exemplos concretos; categorias não detalhadas precisam de especificações antes da implementação.

## 2. Base existente e decisões a preservar

- Katu: `/Volumes/500G DEV/KATSU/`. KATSU é o nome histórico da pasta.
- Arandu: `/Volumes/500G DEV/ARANDU/`, consumidor do motor.
- Referência da auditoria: KATSU HEAD `2709b2bd1a1db6bd136eec9cdafa864e64b758ef`; ARANDU HEAD `1d3bac0df97be41cc4e4b000ddb6404d45c98b0f`. Verificar novamente ao iniciar qualquer mudança.
- Preservar formatos e identificadores de checkpoints e tokenizadores antigos, caminhos de modelos e arquivos históricos.
- Preservar alterações locais e arquivos não rastreados dos dois projetos; não resetar, limpar ou sobrescrever trabalho existente.
- Manter NdArray e o GEMM novo separados do caminho de treino até cumprir as condições registradas pelo Claude: avaliação do piloto 1B e testes de paridade pertinentes. A documentação descreve um treinamento; seu estado atual precisa ser confirmado, não presumido.
- Não trocar o armazenamento do Tensor nem o GEMM do CpuBackend como parte de uma correção matemática isolada.
- Preservar o contrato de integração: KATU_MATH_URL pode diferir de KATU_URL. O script V2 lido aponta matemática para 8792 e modelo para 8793; conferir configuração efetiva antes de testar uma implantação.
- Respeitar a direção de implementação própria: NumPy/SciPy e outros motores são referências independentes de validação, não dependências novas de produção por padrão. As dependências já presentes permanecem; novas exigem justificativa técnica e de licença.
- Planos anteriores continuam como referência: KATSU/docs/KATU_ARRAY_PLAN.md e KATSU/docs/KATU_MATH_PLAN.md. Esta proposta amplia o escopo de produto, inclusive além da exclusão anterior de conhecimento de mundo, sem modificar esses documentos.

## 3. Arquitetura proposta

| Componente | Responsabilidade | Limite |
|---|---|---|
| katu-core | Arrays, tensores, kernels, álgebra linear numérica e infraestrutura de execução | Não armazenar biografias, legislação ou dados de mercado |
| katu-math | Expressões, números exatos, álgebra simbólica, cálculo, condições de validade, unidades e resultados matemáticos | Não usar o texto de um LLM como prova de correção |
| Componentes científicos adjacentes | Estatística, otimização, FFT, esparsas e algoritmos especializados | Criar módulos/crates conforme necessidade comprovada; nomes finais a definir |
| Catálogo e provedores de dados | Entidades, fontes, versões, datas, cobertura geográfica e atualização | Separar dados curados de dados ao vivo; explicitar indisponibilidade |
| Camada de ferramentas/servidor | Contratos, limites de recursos, coordenação e respostas estruturadas | Preservar rotas existentes; novos contratos versionados |
| Modelo de linguagem | Traduzir intenção, pedir esclarecimentos, compor explicação | Não inventar números, fontes ou etapas ausentes do resultado |
| Consumidores, incluindo Arandu | Entrada, gráficos, apresentação e interação | Renderizar dados do motor, sem duplicar a lógica matemática |

Fluxo: pergunta → interpretação estruturada → validação de dados, domínio e unidades → execução → verificação apropriada → resultado estruturado → apresentação.

Para entrada matemática direta, o fluxo independe de um modelo. Para linguagem natural, avaliar separadamente interpretação e execução. O fato de um cálculo estar correto não valida a tradução da pergunta.

## 4. Contrato de resultado e confiança

Definir uma especificação antes de ampliar as ferramentas. Separar:
- Estado: sucesso, entrada inválida, esclarecimento necessário, não suportado, inconclusivo, limite de recursos ou dado indisponível.
- Natureza: exato, aproximação numérica, estimativa estatística ou dado consultado.
- Conteúdo: expressão interpretada, resultado, unidades, domínio, hipóteses e parâmetros utilizados.
- Evidência: método, tolerância solicitada, erro/resíduo observado ou limite demonstrado quando disponível. Não apresentar estimativa de erro como garantia universal.
- Procedência: versão do motor, fonte dos dados, data de referência, data de consulta, região e licença aplicável.
- Passos: transformações e regras executadas, com condições; o modelo pode explicá-las, mas não fabricá-las.
- Visualização: séries, malhas, intervalos, escalas, segmentos desconectados e avisos de amostragem.

Evitar um único selo genérico de “verificado”. Distinguir prova simbólica, substituição, resíduo numérico, comparação independente e simples consulta de fonte.

## 5. Sequência de execução

### Fase 0 — Preservação e linha de base

Entregas:
- Inventário de branches, mudanças locais, contratos HTTP/Python, formatos de modelo e consumidores.
- Identificação dos processos e treinamentos que dependem do checkout, por leitura, sem reiniciar nada.
- Checkout isolado para implementação futura; builds e testes separados dos sistemas em uso.
- Matriz de testes e resultados de referência, com versões e hardware.
- Casos de integração Arandu↔Katu com dados temporários e serviços simulados.

Portão: escopo de mudança e compatibilidade documentados, trabalho preexistente preservado e linha de base reproduzível.

### Fase 1 — Corrigir a confiabilidade atual

Resolver os seis defeitos reproduzidos na auditoria:
1. Bisseção aceita um polo como raiz: `1/(x-0.1)` em `[0,1]`.
2. Primitiva elimina coeficiente pequeno: `0.0000000001*x` vira zero.
3. Determinante de `diag(1e13,1)` retorna zero.
4. Derivada de `ln(x)` avaliada em `x=-1` ignora o domínio original.
5. NdArray aceita broadcast com produto de dimensões que transborda.
6. Concatenação de tensores vazios transborda a soma de dimensões.

Entregas: correções localizadas, testes de regressão, casos adversariais e inclusão explícita de katu-math e testes de integração matemática no CI. Revisar rótulos de precisão: determinístico não significa exato.

Portão: seis reproduções corrigidas, suíte existente aprovada e integração preservada. A referência executada foi 112 testes core + 22 math + 5 testes matemáticos do servidor = 139 aprovados; não é certificação da plataforma. CUDA não foi validado nessa auditoria.

### Fase 2 — Contratos e fundamentos matemáticos

- AST e representação de resultados extensíveis, com compatibilidade dos consumidores.
- Inteiros grandes, racionais exatos, decimais apropriados a dinheiro, complexos e precisão arbitrária, introduzidos em etapas independentes.
- Política explícita de arredondamento, precisão e conversão de tipos.
- Hipóteses sobre variáveis e domínio; expressões por partes; desigualdades e intervalos.
- Unidades, dimensões e conversões afins como temperatura; constantes com versão, unidade e incerteza quando pertinente.
- Limites de tamanho, memória, tempo e profundidade; cancelamento e controle de concorrência para requisições custosas.

Portão: operações suportadas com semântica definida e testes de propriedades; nenhuma promoção silenciosa que apresente aproximação como exatidão. Precisão arbitrária não implica precisão infinita nem custo ilimitado.

### Fase 3 — Computação científica no katu-core

Seguir a dependência estrutural do plano existente:
1. Fechar validações e operações de NdArray, melhorar percurso de views e medir alocações.
2. Expandir GEMM f64 e validar desempenho em ARM/x86 e formatos relevantes.
3. LU, Cholesky, QR, sistemas lineares, mínimos quadrados, autovalores e SVD, com resíduos e condicionamento.
4. Geradores reproduzíveis, distribuições, funções especiais e estatística.
5. FFT, otimização, EDOs adaptativas/rígidas e matrizes esparsas em módulos adjacentes conforme uso.
6. Expor APIs Python coerentes e testadas.

Mudanças de armazenamento do Tensor, backend de treino ou CUDA são etapas próprias, condicionadas ao piloto e à paridade. Não são pré-requisito para todo avanço de katu-math.

Portão por algoritmo: casos conhecidos, referências independentes, casos mal condicionados, política de erro e benchmarks reproduzíveis. Ganhos de desempenho não justificam regressão de correção.

### Fase 4 — Cobertura matemática e passos verificáveis

- Matemática elementar, frações, porcentagens, sinais e paridade.
- Simplificação, polinômios, fatoração, equações, sistemas e parâmetros.
- Domínio, imagem, limites, continuidade e singularidades.
- Derivadas de ordem superior, parciais, gradiente, divergente e rotacional.
- Integrais simbólicas por classes explicitamente suportadas; definidas, impróprias e múltiplas com diagnóstico de convergência.
- Séries Taylor/Laurent, somas, produtos, sequências e recorrências.
- Fourier/Laplace, EDOs e, em etapa avançada, cálculo fracionário e famílias delimitadas de EDPs.
- Geometria, trigonometria, teoria dos números, quatérnios, grupos, permutações e corpos finitos.
- Demonstrações apenas para classes suportadas, com regras rastreáveis; nenhuma promessa de provar qualquer enunciado.

Portão por família: lista de operações cobertas, condições, recusas e suíte independente. Passos devem reproduzir o resultado e declarar condições de equivalência.

### Fase 5 — Gráficos e apresentação matemática

Ordem:
1. Funções 2D, múltiplas curvas, intervalos e escalas logarítmicas/semilogarítmicas.
2. Polares, curvas paramétricas 2D/3D e conjuntos na reta.
3. Superfícies, malhas paramétricas e curvas de nível.
4. Equações implícitas, quádricas e regiões de inequações.

O motor gera geometria e metadados; a interface renderiza. Implementar amostragem adaptativa, separação de descontinuidades, domínio das escalas e limites de resolução. Nenhum método finito de amostragem deve prometer detectar toda singularidade de uma função arbitrária.

Portão: casos visuais e numéricos conhecidos, ausência de ligações falsas nos casos de singularidade testados, unidades/eixos corretos e rastreabilidade dos parâmetros. Testar o contrato do gráfico além de sua aparência.

### Fase 6 — Uso como ferramenta pelo modelo

Aproveitar a integração construída pelo Claude, em vez de substituí-la:
- Chamadas estruturadas e encadeamento de cálculos com unidades e resultados intermediários.
- Exposição da interpretação e esclarecimento quando houver ambiguidade material.
- Modelo recebe erros e recusas como resultados legítimos.
- Passos explicativos ligados a operações reais.
- Régua cega de português: problemas e gabaritos escritos independentemente do gerador/modelo.
- Separar taxa de tradução correta, execução correta e resposta final fiel.

Portão: cada resultado quantitativo declarado como calculado deve corresponder a uma chamada registrada; nenhuma operação recusada pode ser reapresentada como verificada. A substituição do intérprete atual pelo modelo Katu depende de avaliação própria de qualidade.

**Primeiro marco de produto:** fases 0–2 completas e um recorte explícito das fases 3–6: aritmética exata, álgebra básica, cálculo já suportado corrigido, unidades básicas, gráficos 2D e integração validada. Não esperar todas as áreas avançadas para entregar esse marco.

### Fase 7 — Física, engenharia, química e computação

Construir bibliotecas de fórmulas e modelos com hipóteses, unidades, validade e evidências:
- Física: mecânica, ondas, fluidos, termodinâmica, óptica, eletromagnetismo, relatividade e domínios quântico/nuclear/astrofísico delimitados.
- Engenharia: elétrica, mecânica, estruturas, controle, vapor, refrigeração, psicrometria, acústica e aeroespacial.
- Química: elementos, compostos, reações, estequiometria, soluções, termodinâmica, estrutura eletrônica e quimioinformática.
- Computação: autômatos, máquinas de Turing limitadas por orçamento, fractais, substituições, códigos corretores, hashes e processamento de imagens.

Portão: validação especializada por domínio. Fórmulas físicas devem mostrar hipóteses; previsões químicas não se resumem a balancear equações. Cargas admissíveis precisam explicitar modelo, material, condições e norma usada, quando aplicável. Não prometer decisão automática de problemas indecidíveis.

### Fase 8 — Base de conhecimento e dados atualizados

Antes de adquirir dados, definir para cada domínio: fontes, direitos de uso/redistribuição, atualização, custos, idioma, região, campos obrigatórios e verificação.

- Entidades e identificadores estáveis para pessoas, lugares, obras, empresas, substâncias e instrumentos.
- Separar fatos históricos, estimativas, dados temporais e convenções culturais.
- Provedores versionados com data de referência, cache, limites e falhas explícitas.
- Atualizações atômicas e possibilidade de reconstituir resultados históricos.
- Operação offline para capacidades locais; dados atuais somente quando disponíveis e com data informada.
- Conteúdo externo tratado como dados, nunca como instrução de execução.

Portão: fonte e data presentes nos fatos retornados; licença e cobertura verificadas; dado ausente não preenchido por invenção do modelo.

### Fase 9 — Domínios cotidianos, financeiros e culturais

Implementar sobre o catálogo e os cálculos validados: datas/fusos, finanças, nutrição/saúde, pessoas, artes, história, geografia, transportes, linguagem, entretenimento e usos domésticos. Escolher entregas fechadas por domínio, sem declarar uma categoria inteira pronta por possuir alguns exemplos.

Para saúde: estimativas com população, referência, faixa de validade e incerteza; gráficos de crescimento com referência identificada. Para finanças: dinheiro exato, datas/calendários, hipóteses de projeção e jurisdição/versionamento tributário. Conteúdo protegido requer fonte/licença adequada; consulta de obras não implica redistribuição integral.

Portão: avaliação por especialistas quando necessária, corpus de casos representativos e dados rastreáveis. Dados ao vivo, recomendações personalizadas e cálculo local têm critérios distintos.

## 6. Catálogo consolidado dos requisitos

| Área | Capacidades solicitadas |
|---|---|
| Passo a passo | Aritmética, estatística, álgebra, trigonometria, números, geometria, cálculo, EDOs, álgebra linear, provas por indução, convergência, química e física; referência de amplitude: mais de 200 tópicos, ainda a decompor |
| Matemática elementar | Operações, raízes, porcentagens/descontos, frações exatas, problemas escritos, valor posicional, paridade, sinal e tipos de números |
| Álgebra | Equações simbólicas/numéricas, sistemas, parâmetros, polinômios, funções racionais, frações parciais, simplificação, domínio/imagem, quatérnios, matrizes, grupos/permutação e corpos finitos |
| Cálculo e análise | Integrais simples/múltiplas/impróprias, cálculo fracionário/Caputo, derivadas, limites, somas/produtos, séries, sequências, recorrências, campos vetoriais, continuidade, Fourier/Laplace e aplicações geométricas |
| Gráficos | Funções reais/especiais, múltiplas curvas, intervalos, escalas, implícitas, quádricas, curvas de nível, paramétricos 2D/3D, superfícies, regiões, reta real e polares |
| Unidades e medidas | Conversões, comparação, cálculo dimensional, adimensionais, medidas industriais, fios/parafusos, tinta, instrumentos, tamanhos de anéis/calçados, estimativas de preenchimento/cobertura/contorno |
| Física | Mecânica, estatística, óptica, quântica, constantes, fluidos, oscilações, termodinâmica, relatividade, partículas, leis/efeitos, eletromagnetismo, nuclear, órbitas/cosmologia e buracos negros |
| Química | Elementos/íons/compostos, filtros e gráficos, reações e produtos, grafos moleculares/QSAR, nuclídeos, quantidades, termodinâmica/Arrhenius/Joback, orbitais, soluções, titulação/diluição, grupos protetores e hibridização |
| Engenharia | Aeronaves/foguetes/satélites/sondas, AC/motores/guias de onda, estruturas, vapor, energia, controle/Bode/Nyquist, fluidos, instrumentação, materiais, refrigeração/psicrometria e acústica/geração de som |
| Ciências computacionais | Autômatos, Turing, fractais, substituições, classes de complexidade, imagens/morfologia/cor, códigos algébricos e hashes/checksums |
| Pessoas | Perfis, comparações, datas/idades, lugares, obras, esportes, nomes, prêmios, parentescos, líderes/mandatos/sucessão e patrimônio estimado |
| Artes e mídia | Cinema, jogos, periódicos, curvas artísticas, TV, obras/artistas, quadrinhos/mangás, mitologia, literatura, personagens/Pokémon, música e frequências de notas |
| Datas e horários | Durações, calendário mensal/anual e conversões, eventos recorrentes, Unix, fusos, horários de oração/Qibla, história, eras geológicas, bodas, feriados, nascimentos/óbitos, flores/pedras tradicionais |
| Finanças pessoais | Crédito, juros, quitação, investimentos, câmbio, ações, custo de vida e combustíveis |
| Dinheiro e finanças | Valores presente/futuro, renda fixa, impostos/deduções, inflação, financiamento, opções/estratégias, blockchain, gorjetas/contas, salários e receita publicitária |
| Saúde pessoal | Gasto energético, características físicas, estimativas de perda de peso, frequência cardíaca, crescimento infantil, dentição, nutrientes/produtos e porções |
| Surpresas | Conversação, humor/enigmas, warp fictício, imagens, trava-línguas, matemática recreativa, curvas de personagens, alimentos, aforismos/cantigas/frases, manuscrito e animais/raças/visão/idades |
| Categorias adicionais da imagem | Geometria, equações diferenciais, estatística, ciências da Terra, materiais, transportes, palavras/linguística, alimentação/nutrição, geografia política, história, entretenimento, ciência/matemática doméstica, hobbies e atualidades |

Itens repetidos foram consolidados. As categorias adicionais são requisitos de categoria, não listas completas de funções. “Mais tópicos” e links para produtos Wolfram não foram tratados como autorização para copiar produtos ou como especificações detalhadas.

## 7. Estratégia de qualidade

Para cada recurso, entregar em conjunto:
1. Especificação de entrada, saída, hipóteses e limites.
2. Implementação e orçamento de recursos.
3. Casos exatos conhecidos e referências independentes quando cabíveis.
4. Casos de borda, entradas inválidas e testes de propriedades/metamórficos.
5. Verificação do contrato HTTP/Python e do consumidor aplicável.
6. Documentação de cobertura e limitações.

Métricas separadas: respostas corretas, respostas incorretas com aparência de certeza, recusas corretas, recusas desnecessárias, interpretação, fidelidade dos passos, erro/resíduo, latência e memória. Metas numéricas de qualidade de linguagem e desempenho serão fixadas com uma linha de base e conjunto de avaliação definido; não inventar percentuais globais agora.

Testes adversariais essenciais: descontinuidades, escalas muito pequenas/grandes, cancelamento numérico, matriz mal condicionada, unidades incompatíveis, calendário/fuso ambíguo, fonte vencida, overflow e explosão simbólica. Testes de referência passando não demonstram correção de todo o domínio.

## 8. Equipe, custos e cronograma

Responsabilidades necessárias, que podem ser acumuladas inicialmente: álgebra computacional, análise numérica/Rust, integração e visualização, qualidade independente, engenharia de dados e revisão especializada de domínio.

Não há estimativa responsável para todo o catálogo sem inventariar funções, equipe e fontes. Estimar em ciclos fechados após a fase 0: tarefa, dependências, esforço, evidência de conclusão e risco. Primeiro medir o custo real da fase 1 e do contrato da fase 2.

Gastos que precisam de orçamento antes de contratação: dados licenciados, provedores ao vivo, revisão especializada, infraestrutura de atualização e hardware/tempo de GPU quando necessário. Treinar um modelo maior não é pré-requisito para corrigir o motor matemático.

## 9. Próximo pacote de trabalho proposto

1. Reconfirmar estado dos dois repositórios e restrições do piloto.
2. Preparar ambiente isolado de correções, sem substituir serviços instalados.
3. Corrigir seis falhas com regressões e validar integração matemática.
4. Especificar contrato de resultados e o primeiro recorte de aritmética exata/unidades.
5. Decompor o primeiro marco de produto em tarefas estimáveis.

Este documento é o plano. Nenhuma dessas implementações foi iniciada nesta solicitação.
