# Primeiro marco de produto: decomposição em tarefas

Data: 01/11/2026. Marco definido na Fase 6 do [plano](KATU_MOTOR_COMPUTACIONAL_PLAN.md): fases 0 a 2 completas e um
recorte das fases 3 a 6, com aritmética exata, álgebra básica, cálculo já suportado corrigido, unidades básicas, gráficos
2D e integração validada. Contrato em [KATU_CONTRATO_RESULTADOS.md](KATU_CONTRATO_RESULTADOS.md).

## Estado do primeiro ciclo

| Tarefa | Estado |
|---|---|
| A1 | Feita: integrado em `feat/katsu-lm-v1`, sem push, compilado e no ar no `Arandu V2.app`, conferido em 01/11/2026. O binário anterior está em `katu-server.antes-2026-11-01` |
| A2, A3 | Feitas: [KATU_INVENTARIO.md](KATU_INVENTARIO.md) |
| B1, B4 | Feitas |
| C1 | Feita em 01/11/2026: det zero e polo como `inconclusive`; registrada no contrato |
| A7 | Feita e instalada |
| C2, C3 | Feitas e integradas, sem consumidor ainda; o Arandu segue em `/math` |
| C4, C5, C7, C8 | Feitas e integradas (`ca84f08`), compiladas às 21:11 e no ar só na próxima abertura do app: `katu_math::exact`, `katu_math::budget`, modos em `/v1/compute`. C7 só em `eval`; C8 sem teto de memória |

Medido neste ciclo: cerca de 10 minutos de relógio, com um agente assistido, e cerca de 1.140 linhas alteradas entre código,
testes e script. Estimado para o ciclo, nas faixas abaixo: B1 e B4 P, C2 e C3 M, cerca de 5 a 10 dias de uma pessoa. A diferença não
vale para o resto: foi troca mecânica e envelope sobre contas prontas, sem matemática nova. C4 e C6 são a próxima medida útil.

## Como ler as estimativas

Esforço em dias de trabalho concentrado de uma pessoa experiente na área: **P** até 1, **M** de 2 a 4, **G** de 5 a 10.
São estimativas de ordem de grandeza, sem medição própria, exceto a Fase 1. A referência medida é uma só:

| Fase 1 | Medido |
|---|---|
| Trabalho | 6 defeitos, 4 commits, cerca de 820 linhas alteradas, testes e script de referência incluídos |
| Tempo | cerca de 15 minutos de relógio entre a primeira regressão e o último commit, com um agente assistido |

Esse tempo não se transfere para o restante: correção localizada com reprodução pronta é o caso mais fácil. Reestimar ao
fim de cada ciclo, comparando com o previsto. Nenhuma tarefa daqui pede GPU, treino ou dado licenciado.

## Ciclo A: preservação e linha de base (Fase 0)

| ID | Tarefa | Depende | Esforço | Evidência de conclusão | Risco |
|---|---|---|---|---|---|
| A1 | Decidir a integração da branch de correções e reinstalar quando o usuário autorizar | | P | App responde às 4 reproduções na 8792 | Reinstalar pede permissão de volume no macOS |
| A2 | Inventário de contratos: HTTP, Python, formatos de modelo e consumidores | | P | Documento com a origem de cada campo | Consumidor esquecido |
| A3 | Inventário, por leitura, dos processos que dependem do checkout | | P | Lista conferida com `lsof` e `ps` | Mudar enquanto o piloto roda |
| A4 | Matriz de testes com versões e hardware; primeira execução do CI no Linux x86 | A1 | P | Execução verde registrada | Push exige autorização, repositório público |
| A5 | Casos Arandu↔Katu reais com servidor temporário, em script | | M | Script no ARANDU, sem rede externa | Servidor pede checkpoint para subir mesmo só para contas |
| A7 | Feita (ARANDU `b934966`). O app sobe o Katu de novo quando a 8792 cai e registra em `data/logs`. Medido nos apps instalados: 25 s após derrubar o servidor; 6 s e 5 s na troca de dono entre os dois apps | | P | Teste do laço e os três testes no app real | Dois apps ao mesmo tempo com a porta fechada carregam o modelo os dois; um sai |
| A6 | Separar o serviço matemático do carregamento do modelo | A2 | M | Rota `/math` sobe sem checkpoint e sem 540 MB | Muda inicialização; preservar a atual |

## Ciclo B: lacunas achadas na Fase 1

| ID | Tarefa | Depende | Esforço | Evidência | Risco |
|---|---|---|---|---|---|
| B1 | Feita. Newton aceitava assíntota como raiz (`1/x`, `exp(-x)`) | | P | Regressão; comparação com SciPy | Raiz muito íngreme ainda é recusada |
| B2 | Convenção de `0^0` declarada ou recusada em `/math` | C1 | P | Teste e nota | Quebrar consumidor |
| B3 | LaTeX não pode arredondar a entrada: `1.000000000000001` aparece como `1` | | P | Teste de ida e volta | Layout |
| B4 | Feita. Código e mensagem próprios para matriz quase singular | C2 | P | Teste | Baixo |
| B5 | Autograd de concat vazio com dimensão enorme não pode percorrer as linhas | Fim do piloto | P | Teste com `backward` | Caminho de treino; só depois do piloto |
| B6 | Integral: diagnóstico de polo em vez de "não convergiu" | C2 | M | Casos com polo, salto e imprópria | Falsos positivos |

## Ciclo C: contrato e fundamentos (Fase 2)

| ID | Tarefa | Depende | Esforço | Evidência | Risco |
|---|---|---|---|---|---|
| C1 | Revisar e aprovar o contrato | | P | Decisão registrada | Refazer rota mais tarde |
| C2 | Erro tipado no lugar de `Result<_, String>` | C1 | M | Todo `refusal.code` testado | Toca toda a API do crate; manter textos de `/math` |
| C3 | Rota `/v1/compute` envolvendo as operações atuais | C2 | M | Teste de contrato por operação; `/math` intacta | Divergência entre rotas |
| C4 | Feita. `Int` próprio: soma, produto, divisão, potência e texto | | G | Propriedades e comparação com Python | Desempenho e erros sutis de divisão |
| C5 | Feita, sem unidades. `Rational` e literal decimal exato | C4 | M | Gabaritos da seção 8 | Crescimento de denominadores |
| C6 | AST numérica: `Expr::Num(f64)` mantendo o literal original | C5 | G | Consumidores sem mudança | Maior risco de compatibilidade do marco |
| C7 | Parte feita. Modos `exact`, `numeric`, `auto` e política de conversão, em `eval`; demais operações dependem de C6 | C3, C6 | M | Nenhuma promoção silenciosa em testes de propriedade | Decidir `auto` caso a caso |
| C8 | Parte feita. Tempo, profundidade, concorrência e cancelamento em `/v1/compute`; falta teto de memória | C3 | M | Testes adversariais, `limit_*` | Cancelar laço de CPU no `spawn_blocking` |
| C9 | Hipóteses sobre variáveis; intervalos e desigualdades | C2 | G | Testes por família | Escopo cresce; fatiar |
| C10 | Unidades: dimensão, fatores, conversão afim | C5 | M | Gabaritos; fatores conferidos no NIST SP 811 | Temperatura e nomes em conflito com variáveis |
| C11 | Unidades em expressões e leitura em texto | C10 | M | Casos ambíguos pedem esclarecimento | `m` e `s` são variáveis válidas hoje |

## Ciclo D: recorte numérico (Fase 3)

Só o que o marco usa. NdArray e o GEMM novo seguem fora do caminho de treino até a avaliação do piloto 1B e os testes de
paridade.

| ID | Tarefa | Depende | Esforço | Evidência | Risco |
|---|---|---|---|---|---|
| D1 | Resíduo e estimativa de condicionamento em `det` e `inv` | C3 | M | Comparação com NumPy, casos mal condicionados | Custo de estimativa |
| D2 | Sistemas lineares por LU, com resíduo | D1 | M | Comparação com SciPy | Reaproveitar sem tocar no treino |
| D3 | Validações pendentes de NdArray: percurso de views e alocações | | M | Medição | Permanecer separado do treino |

## Ciclo E: álgebra e cálculo do marco (Fase 4)

| ID | Tarefa | Depende | Esforço | Evidência | Risco |
|---|---|---|---|---|---|
| E1 | Frações, porcentagens, sinal e paridade sobre o tipo exato | C7 | M | Suíte independente | Baixo |
| E2 | Polinômios em uma variável com coeficientes racionais | C5 | G | Comparação com SymPy como referência | Explosão de termos |
| E3 | Simplificação com condições declaradas | E2, C9 | G | `x^2/x` não vira `x` sem condição | Maior risco de correção do ciclo |
| E4 | Equações lineares e quadráticas, sistemas lineares exatos | E2 | M | Substituição de volta em toda resposta | Casos degenerados |
| E5 | Derivada e primitiva com coeficientes exatos; classes listadas | C6 | M | Passos reproduzem o resultado | Compatibilidade da forma impressa |
| E6 | Registro de passos com regra e condição | C3, E5 | M | Reconstituição automática testada | Passos virarem texto solto |

## Ciclo F: katu-plot, gráficos 2D (Fase 5, primeiro recorte)

Contrato em [KATU_PLOT_CONTRATO.md](KATU_PLOT_CONTRATO.md). A crate ainda não existe; criá-la depende de autorização.

| ID | Tarefa | Depende | Esforço | Evidência | Risco |
|---|---|---|---|---|---|
| F1 | Aprovar o contrato e decidir API e organização | C1 | P | Decisão registrada | Refazer a API |
| F2 | Avaliar renderizadores e PNG: Plotters, Vello, egui_plot e próprio, com licença e SBOM | F1 | P | Comparativo medido nos mesmos casos | Dependência grande entrar no motor |
| F3 | Crate do núcleo: `Figure`, `Axes`, `Series` e artefato `katu.plot/1` | F1, C3 | M | Teste de contrato; artefato idêntico byte a byte | Estado global voltar pela conveniência |
| F4 | Amostragem adaptativa, várias curvas, intervalos explícitos, faixa y robusta | F3 | M | Erro e contagem nos gabaritos | Custo por curva |
| F5 | Classificação de quebras, função pública no katu-math | F4 | M | Sem ligação falsa em `1/x`, `tan(x)`, `abs(x)/x`; `sin(1/x)` sai `undersampled` | Nunca prometer detectar tudo |
| F6 | Exportação SVG, JSON e CSV; temas e alternativa textual | F4 | M | XML válido, contraste, CSV igual ao JSON | Aparência depender de conferência humana |
| F7 | Arandu: apresentar, zoom visual e zoom calculado | F5, F6 | M | Teste de contrato; indicação de desatualizado; conferência visual | Duplicar lógica na interface |
| F8 | PNG, conforme F2 | F2, F6 | P | Compara com o SVG | Dependência de rasterização |
| F9 | Escalas log e semilog, com domínio | F4 | P | Recusa em valor não positivo | Fora do primeiro recorte; fecha o marco |

Interação além do zoom, 3D e GPU ficam para depois do marco.

## Ciclo G: uso como ferramenta (Fase 6)

| ID | Tarefa | Depende | Esforço | Evidência | Risco |
|---|---|---|---|---|---|
| G1 | `math_tool.py` aceita o contrato novo e mantém o antigo | C3 | M | `test_math.py` e novos testes; `/math` antiga segue funcionando | Regressão na integração instalada |
| G2 | Encadear cálculos, com unidades e resultados intermediários | G1, C10 | M | Rastro de chamadas | Limite de chamadas |
| G3 | Registro: todo número declarado como calculado liga-se a uma chamada | G1 | M | Portão da Fase 6 testado | Comparar números em texto livre |
| G4 | Régua cega em português, escrita por quem não conhece o gerador | | G | Gabaritos congelados com SHA256 | Exige autor independente |
| G5 | Avaliação separada de tradução, execução e fidelidade | G3, G4 | M | Três taxas, sem meta inventada | Custo de API do DeepSeek: pedir orçamento antes |

## Ordem e dependências

```text
A (paralelo) ─┐
C1 → C2 → C3 ─┼→ D1 → D2
              ├→ C8
C4 → C5 → C6 → C7 → E1
     └→ C10 → C11      E2 → E3, E4; C6 → E5 → E6
C3 → F1 → F2; F1 → F3 → F4 → F5, F6, F9; F5, F6 → F7; F6 → F8
C3 → G1 → G2, G3; G4 → G5
```

## Resumo do esforço, sem paralelismo

Somando as faixas: 15 tarefas P, 26 M e 6 G, ou cerca de 90 a 180 dias de trabalho de uma pessoa. As tarefas G concentram a
incerteza: C4, C6, C9, E2, E3 e G4. Essa soma é uma ordem de grandeza para decidir o próximo ciclo, não um prazo prometido, e
será refeita ao fim do ciclo C.

## Primeiro ciclo proposto, se autorizado

A1, A2, A3, B1, C1, C2 e C3: fecha a correção, entrega o contrato de ponta a ponta sem mudar a matemática e mede o custo de
uma mudança de API, que é o que falta para reestimar o resto.
