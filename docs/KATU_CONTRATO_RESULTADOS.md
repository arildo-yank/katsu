# Contrato de resultados do Katu e primeiro recorte exato

Data: 01/11/2026. Estado: especificação, ainda sem implementação. Detalha a seção 4 do
[plano do motor computacional](KATU_MOTOR_COMPUTACIONAL_PLAN.md). Exemplos JSON são normativos quanto aos campos;
valores são ilustrativos, exceto na tabela de gabaritos.

## 1. Regras gerais

1. A rota `POST /math` e o `POST /math/ask` continuam como estão. O contrato novo vive em `POST /v1/compute`.
2. `schema` é `"katu.result/1"`. Campo novo opcional não muda a versão; remover ou mudar o sentido de um campo muda.
3. Toda resposta a um pedido bem formado é HTTP 200 com `status`. Recusa é resultado legítimo, não falha de transporte.
   HTTP 400 só para envelope malformado; 413 para corpo acima do limite; 429 e 503 ficam para a operação.
4. Não existe selo genérico de "verificado". `evidence.checks` lista o que foi feito, um item por verificação.
5. O modelo de linguagem recebe o resultado inteiro e não pode acrescentar número, fonte ou passo ausente dele.
6. Campo que não se aplica é omitido, nunca preenchido com zero, texto vazio ou palpite.

## 2. Envelope

```json
{
  "schema": "katu.result/1",
  "status": "success",
  "nature": "exact",
  "request": {"op": "eval", "expr": "1/3 + 1/6", "mode": "exact"},
  "content": {
    "interpreted": "1/3 + 1/6",
    "interpreted_latex": "\\frac{1}{3} + \\frac{1}{6}",
    "value": {"kind": "rational", "num": "1", "den": "2"},
    "value_latex": "\\frac{1}{2}",
    "decimal_preview": {"text": "0.5", "exact": true},
    "unit": null,
    "domain": null,
    "assumptions": [],
    "parameters": {}
  },
  "evidence": {
    "method": "aritmética racional exata",
    "tolerance_requested": null,
    "error_observed": null,
    "checks": [{"kind": "exact_arithmetic", "detail": "sem arredondamento"}]
  },
  "provenance": {"engine": "katu-math", "engine_version": "0.1.0", "git": "<commit>"},
  "steps": [],
  "warnings": [],
  "usage": {"elapsed_us": 12, "limit_hit": null}
}
```

Quando `status` não é `success`, `content.value` é omitido e o campo `refusal` é obrigatório:

```json
{"code": "pole_or_jump", "message": "a função troca de sinal perto de x = 0.1 sem passar por zero", "where": {"x": 0.1}, "retry": null}
```

## 3. Vocabulários fechados

### `status`

| Valor | Quando | Exemplo |
|---|---|---|
| `success` | Há resultado dentro da semântica declarada | `2 + 2` |
| `invalid_input` | O pedido não tem sentido ou está fora do domínio | `1/0`, `ln(-1)`, `m + s` |
| `clarification_needed` | Mais de uma leitura muda o resultado; `refusal.options` lista as leituras | `20 °C + 5 °C` |
| `unsupported` | Pedido válido, fora do que o motor sabe fazer | `∫exp(-x²)dx` em forma fechada |
| `inconclusive` | O método rodou e não permite afirmar | matriz sem pivô distinguível do arredondamento |
| `resource_limit` | Passou de tempo, memória, tamanho ou profundidade; `usage.limit_hit` diz qual | `2^(10^9)` |
| `data_unavailable` | Depende de dado que falta. Inclui variável sem valor; depois, fonte fora do ar | `x + 1` sem `x` |

### `nature`, obrigatório em `success`

| Valor | Significado | Não significa |
|---|---|---|
| `exact` | Valor matemático sem arredondamento | Que a pergunta em linguagem natural foi bem traduzida |
| `symbolic` | Expressão obtida por regras, com coeficientes como escritos | Equivalência em todo domínio, salvo condições listadas |
| `numeric` | Aproximação em ponto flutuante | Que o erro seja menor que a tolerância em qualquer caso |
| `statistical` | Estimativa com incerteza | Valor verdadeiro |
| `lookup` | Dado consultado | Verdade além do que a fonte afirma |

### `evidence.checks[].kind`

| Valor | Afirma | Não afirma |
|---|---|---|
| `exact_arithmetic` | Nenhuma operação arredondou | Correção da interpretação |
| `symbolic_proof` | Regras rastreáveis levam da entrada ao resultado, com condições | Validade fora das condições |
| `substitution` | Conferido em N pontos, com folga declarada | Identidade em todos os pontos |
| `numeric_residual` | Resíduo observado de tamanho R | Distância até a solução verdadeira |
| `sign_change_and_decay` | Troca de sinal com \|f\| caindo com o intervalo | Ausência de polo ou salto que a amostragem não veja |
| `independent_comparison` | Outro motor concordou, identificado com versão | Correção se ambos errarem igual |
| `source_lookup` | A fonte traz o dado | Que a fonte esteja certa |

Resíduo pequeno não é erro pequeno: a Fase 1 mediu um polinômio de grau 9 cujo sinal vira ruído numa faixa em volta da
raiz. Por isso `error_observed` diz o que foi medido (`{"kind": "residual", "value": ...}`) e nunca se chama "erro" sem
qualificador.

### `refusal.code`, primeiro conjunto

`parse_error`, `domain_error`, `division_by_zero`, `pole_or_jump`, `no_sign_change`, `not_converged`, `singular_matrix`,
`near_singular_matrix`, `not_representable`, `dimension_mismatch`, `ambiguous_temperature`, `unknown_unit`,
`no_closed_form`, `missing_variable`, `exact_mode_unavailable`, `limit_time`, `limit_memory`, `limit_size`, `limit_depth`.

Hoje o motor devolve `Result<_, String>`; o código precisa virar erro tipado (tarefa C2 da decomposição).

## 4. Como as operações atuais se encaixam

| Operação | `nature` | Verificação que existe de fato |
|---|---|---|
| `eval` só com inteiros abaixo de 2^53 | `exact` | `exact_arithmetic` |
| `eval` nos demais casos | `numeric` | nenhuma |
| `derive` sem ponto | `symbolic` | nenhuma automática |
| Valor de `derive` em ponto | `exact` ou `numeric`, em `value_nature` | domínio da função original |
| `antiderivative` | `symbolic` | `substitution` em até 7 pontos, mínimo 3, folga relativa 1e-7 |
| `integrate` | `numeric` | nenhuma; Simpson adaptativo, a tolerância pedida não é garantia |
| `root` por intervalo | `numeric` | `sign_change_and_decay` |
| `root` por Newton | `numeric` | `numeric_residual` (ver lacuna L1 no relatório da Fase 1) |
| `ode` | `numeric` | nenhuma; passo fixo, sem estimativa de erro |
| `det`, `inv`, `matmul`, `dot`, `cross` | `numeric` | `inv` deve incluir `numeric_residual` |

Mapeamento de compatibilidade de `/math`: `approximate: true` corresponde a `numeric`; `approximate: false` a `exact` ou
`symbolic`. Em `/v1/compute`, `det` sem pivô distinguível devolve `inconclusive`; em `/math` continua `0`, por
compatibilidade.

## 5. Passos e visualização

**Passos.** Cada passo vem de uma regra executada, nunca do modelo:
`{"rule": "power_rule", "from": "x^3", "to": "x^4 / 4", "conditions": ["n != -1"]}`. Concatenar os passos precisa
reproduzir o resultado; a suíte confere. Sem passos rastreáveis, `steps` fica vazio.

**Visualização**, especificada aqui e implementada na Fase 5:

```json
{
  "kind": "plot2d",
  "axes": {"x": {"label": "x", "scale": "linear", "range": [0, 1], "unit": null}, "y": {"label": "f(x)", "scale": "linear", "range": [-100, 100], "unit": null}},
  "series": [{"label": "1/(x-0.1)", "segments": [[[0.0, -10.0]], [[0.11, 100.0]]]}],
  "breaks": [{"x": 0.1, "reason": "pole_suspected"}],
  "sampling": {"method": "adaptive", "points": 400, "warning": "amostragem finita não detecta toda singularidade"}
}
```

Segmentos separados são a forma de não ligar um polo com uma reta falsa. O motor gera a geometria; a interface desenha.

## 6. Primeiro recorte: aritmética exata

**Modos.** `mode` aceita `exact`, `numeric` ou `auto`; o padrão de `/v1/compute` é `auto`.

- `exact`: ou sai exato, ou recusa com `exact_mode_unavailable`. Nunca cai para ponto flutuante.
- `numeric`: f64, como hoje.
- `auto`: tenta exato; se precisar aproximar, responde `nature: "numeric"` e um aviso dizendo qual operação forçou isso.

**Tipos**, sem dependência nova, implementados no katu-math: `Int` (inteiro de tamanho arbitrário) e `Rational`
(sempre em forma reduzida, com denominador positivo). Decimais de dinheiro, complexos e precisão arbitrária ficam em
etapas próprias da Fase 2.

**Semântica:**

| Entrada | Regra |
|---|---|
| Literal `0.1` | Em `exact` é 1/10, o decimal escrito, não o binário mais próximo |
| `+ - *` | Fechadas em racionais |
| `/` | Exata; divisor zero dá `division_by_zero` |
| `a^n`, `n` inteiro | Exata; `n` negativo inverte; `0^0` e `0^-n` são `invalid_input`, pois hoje `0^0` retorna 1 sem declarar a convenção |
| `a^(p/q)` não inteiro | `exact_mode_unavailable`, mesmo quando o resultado é racional. `sqrt(4)` também, neste recorte |
| `pi`, `e`, `sin`, `ln`, `exp` | `exact_mode_unavailable`, salvo `auto`, que aproxima e avisa |
| `abs`, negação, comparação | Exatas |
| Variáveis | Valores passados como texto (`"1/3"`, `"0.1"`) continuam exatos; número JSON passado em `exact` é recusado, porque já chegou binário |

**Conversões.** Exato para f64 só de forma explícita, em `decimal_preview` ou `auto`, com arredondamento para o mais
próximo e empate para par. Se a conversão perder valor, `decimal_preview.exact` é `false`. Nenhuma conversão de f64 para
exato acontece sozinha.

**Saída.** Inteiros grandes e racionais saem em JSON como texto, pois número JSON perderia dígitos.

**Limites**, ajustáveis após medição: entrada de até 500 bytes (atual), profundidade 48 (atual), 262.144 bits por
inteiro e estimativa de bits antes de calcular potência, para `2^(10^9)` sair como `limit_size` sem alocar nada.

## 7. Primeiro recorte: unidades

**Representação.** Quantidade é `(valor exato, vetor de 7 expoentes inteiros SI)`. Unidade é `(fator exato, deslocamento
exato, dimensão)`. Só entram fatores exatos por definição; constantes medidas, com incerteza, ficam fora do recorte.

| Dimensão | Unidades |
|---|---|
| Comprimento | m, km, cm, mm, in (0,0254 m), ft, yd, mi (1609,344 m) |
| Massa | kg, g, t, lb (0,45359237 kg) |
| Tempo | s, min, h, d |
| Volume | m³, L |
| Derivadas | N, J, W, Pa |
| Temperatura absoluta | K, degC, degF |
| Diferença de temperatura | delta_K, delta_degC, delta_degF |
| Adimensional | percent; rad e deg entram só em conversão |

**Regras:**

- Somar ou subtrair dimensões diferentes: `dimension_mismatch`, citando as duas dimensões.
- Conversão é afim onde preciso: `degF = degC * 9/5 + 32`.
- `20 degC + 5 degC`: `clarification_needed`, com opções de tratar o segundo termo como `delta_degC`.
- `degC + delta_degC` dá `degC`; `degC - degC` dá `delta_degC`; multiplicar temperatura com deslocamento dá `invalid_input`.
- `sin` e demais funções transcendentes exigem argumento adimensional. Ângulo em `deg` é convertido; essa conversão
  envolve pi e, portanto, deixa de ser `exact`.
- Unidade desconhecida nunca é tratada como variável quando vier no campo `unit`: dá `unknown_unit`.
- A entrada estruturada vem primeiro: `{"value": "100", "unit": "km/h", "to": "m/s"}`. Ler unidades dentro de texto livre
  é tarefa separada, porque `m` e `s` hoje são nomes válidos de variável.

**Fora do recorte:** fios, parafusos, tamanhos de anel e calçado, moedas, constantes físicas, logarítmicas (dB) e
unidades dependentes de data.

## 8. Gabaritos do recorte

Escritos à mão, a partir das definições. Entram como testes antes da implementação.

| Entrada | Modo | Resultado esperado |
|---|---|---|
| `0.1 + 0.2` | exact | `3/10` |
| `1/3 + 1/6` | exact | `1/2` |
| `2^100` | exact | `1267650600228229401496703205376` |
| `2^-3` | exact | `1/8` |
| `(1/3) * 3` | exact | `1` |
| `1/0` | exact | `invalid_input` / `division_by_zero` |
| `0^0` | exact | `invalid_input` / `domain_error` |
| `sqrt(2)` | exact | `unsupported` / `exact_mode_unavailable` |
| `sqrt(2)` | auto | `numeric`, com aviso |
| `2^(10^9)` | exact | `resource_limit` / `limit_size`, sem alocar |
| `1 in` para `cm` | exact | `127/50`, isto é, 2,54 |
| `100 km/h` para `m/s` | exact | `250/9` |
| `37 degC` para `degF` | exact | `493/5`, isto é, 98,6 |
| `37 degC` para `K` | exact | `6203/20`, isto é, 310,15 |
| `0 K` para `degC` | exact | `-5463/20`, isto é, -273,15 |
| `1 lb` para `g` | exact | `45359237/100000` |
| `1 m + 1 s` | exact | `invalid_input` / `dimension_mismatch` |
| `20 degC + 5 degC` | exact | `clarification_needed` / `ambiguous_temperature` |
| `5 foo` para `m` | exact | `invalid_input` / `unknown_unit` |

Referência independente, sem virar dependência de produção: Python `fractions` nas contas racionais e NIST SP 811 nos
fatores de unidades, a conferir antes da implementação.

## 9. Fora deste contrato

Interpretação de linguagem natural, catálogo de dados e procedência de fontes externas. `provenance` já reserva os
nomes `source`, `reference_date`, `retrieved_at`, `region` e `license`, a especificar na Fase 8.
