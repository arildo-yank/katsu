# Plano do `katu-math`

*31/10/2026. O que está marcado como feito tem teste em `crates/katu-math`. O resto é plano, sem estimativa medida.*

**A meta é um motor de cálculo auditável e offline, que responde exato ou recusa.** Ele começa pela entrada matemática direta,
porque o gargalo medido hoje é o modelo entender a pergunta, não a conta.

## O que motivou a ordem

Teste de 31/10/2026 com o 134M e a calculadora atual, em 6 perguntas escolhidas à mão: o modelo traduziu certo 2.

| Pergunta | O modelo escreveu | Problema |
|---|---|---|
| 10 dividido por 0 | `10^0` | A conta exata de uma expressão errada dá "1" |
| (12 + 8) × 3 | Texto, sem chamar a ferramenta | Inventou "56" |
| 12 toneladas em 7 viagens | Texto, sem chamar a ferramenta | Sem sentido |
| Seno de 30 graus | Texto, sem chamar a ferramenta | Inventou "60" |

**Um motor exato não elimina a alucinação.** Se a tradução sai errada, o motor devolve com precisão a resposta da pergunta errada, com aparência de verificada.
Isso pode ser pior do que uma resposta visivelmente ruim. Por isso a tradução precisa de régua própria, antes de qualquer promessa.

## Etapas

| # | Etapa | Depende do modelo? | Situação |
|---|---|---|---|
| 1 | Expressões, derivada simbólica, integral numérica, raízes, RK4 | Não | **Feito**, 10 testes |
| 2 | Rota `POST /math` e tela "Matemática" no Arandu | Não | **Feito** e testado no navegador; falta entrar no app instalado |
| 3 | Detector de contas diretas no chat, antes do modelo | Não | Falta |
| 4 | Vetores e matrizes | Não | **Parte feita:** determinante, inversa, produto de matrizes, escalar e vetorial. Faltam autovalores, gradiente, divergente e rotacional |
| 5 | Unidades de medida e constantes físicas com fonte | Não | Falta |
| 6 | Passo a passo das derivadas | Não | Falta |
| 7 | Álgebra simbólica: simplificação, limites, Taylor | Não | Falta |
| 8 | Integral simbólica, e só para classes definidas de funções | Não | **Parte feita:** somas, potências, 1/x, sqrt(x), ln(x), a^x, e sin, cos e exp de argumento linear, cada resultado conferido derivando. O caso geral é um projeto de anos |
| 9 | **Régua de tradução:** frases → expressão, escrita por quem não viu o gerador | Sim | Falta. Portão da etapa 10 |
| 10 | O modelo chama o motor como ferramenta | Sim | **Feito com o DeepSeek** (`ARANDU/arandu-web/math_tool.py`). Para o modelo do Katu fazer isso, ainda depende da 9 |

## Decisões

- **Sem dependências externas no núcleo.** Conta a favor da soberania, e o `calc.rs` já segue essa linha. A `nalgebra` só entra se a etapa 4 precisar de autovalores em escala, e com decisão registrada.
- **Crates sugeridas e descartadas:** `symbolic` é da Sentry e processa stack traces, não faz álgebra (conferido no crates.io em 31/10/2026). `math-rs` é um projeto pequeno, parado desde 2023.
- **Numérico é dito como numérico.** Integral e RK4 são aproximações, e a resposta diz isso, com a tolerância ou o número de passos.
- **Recusa em vez de palpite:** divisão por zero, domínio inválido, não convergência e variável sem valor são erro.
- **Radianos por padrão.** "Graus" precisa ser explícito, como `sin(30 * pi / 180)`.
- **Fora do escopo:** "Conhecimento de mundo" no estilo do Wolfram Alpha (pessoas, química, geografia, finanças). Isso é uma base de dados curada, não um motor de cálculo.

## O que ainda não se pode dizer

- "Wolfram Alpha brasileiro": hoje é uma fração pequena do que ele faz em matemática, e não tem a base de conhecimento.
- "Sem alucinação": só vale para a entrada matemática direta. Em linguagem natural, depende da etapa 9.
- "Resolve Cálculo 1 a 3 e Física 1 a 3": ainda não. A etapa 1 cobre derivadas e métodos numéricos de uma variável.
