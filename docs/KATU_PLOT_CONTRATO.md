# katu-plot: contrato e primeiro recorte

Data: 01/11/2026. Estado: **primeiro recorte implementado** em `crates/katu-plot`, com a rota `POST /v1/plot` no katu-server
e criação autorizada pelo usuário em 01/11/2026. A seção 11 registra as decisões e o que ficou de fora. Detalha a Fase 5 e a seção 10 do [plano](KATU_MOTOR_COMPUTACIONAL_PLAN.md). Segue o
[contrato de resultados](KATU_CONTRATO_RESULTADOS.md) nos campos `status`, `refusal` e `warnings`.

Nada aqui foi medido. Nenhuma afirmação de velocidade ou qualidade em relação a Matplotlib ou a outra biblioteca vale sem
medição própria, com o mesmo gráfico nos dois lados.

## 1. Divisão de responsabilidades

| Componente | Faz | Não faz |
|---|---|---|
| katu-math | Interpreta e avalia a expressão, declara domínio e recusas | Não conhece eixos nem desenho |
| katu-core | Apoia cálculo quando necessário | Não é obrigatório em todo gráfico |
| katu-plot, núcleo | Amostragem, segmentos, escalas, especificação, geometria, metadados | Não duplica o motor simbólico; não abre janela nem usa GPU |
| katu-plot, renderizadores | Transformam a especificação em SVG, PNG ou tela | Não recalculam matemática nem decidem descontinuidades |
| Arandu e outros consumidores | Apresentam e tratam a interação | Não amostram funções por conta própria |

Dependência em uma direção só: `katu-plot → katu-math`, e nunca o contrário. As contas continuam funcionando sem gráficos.
Qualquer dependência gráfica fica só no renderizador, em crate ou recurso opcional, de modo que o núcleo não a carregue.

## 2. Decisões de desenho

| Tema | Proposta | Situação |
|---|---|---|
| API | Própria, inspirada no pyplot, com objetos explícitos `Figure`, `Axes` e `Series` e sem estado global | Proposta, decidir em F1 |
| Núcleo | Rust, junto do motor | Proposta, decidir em F1 |
| Renderizador | O plano não presume Rust em todo o renderizador. O SVG pode sair do núcleo; a tela interativa pode ser JS no consumidor | Aberto |
| Ordem | SVG primeiro, interação depois, GPU e 3D em etapa separada | Proposta |
| Bibliotecas | Plotters, Vello, egui_plot e renderização própria são opções a avaliar; escolha explícita, com licença, tamanho e soberania (SBOM) | Aberto, tarefa F2 |

Exemplo da API proposta, ilustrativo e sem compromisso de assinatura:

```rust
let mut figure = Figure::new();
let axes = figure.axes();
axes.x_range(-1.0, 1.0).y_range(-20.0, 20.0).x_label("x");
axes.function("1/x", "x")?.label("1/x");
axes.function("x^3", "x")?.label("x³");
let spec = figure.build(&Budget::default())?;
let svg = SvgRenderer::default().render(&spec, &Theme::light())?;
```

## 3. Pedido: `katu.plot.request/1`

```json
{
  "schema": "katu.plot.request/1",
  "axes": {
    "x": {"label": "x", "range": [-1, 1], "scale": "linear"},
    "y": {"label": "f(x)", "range": "auto", "scale": "linear"}
  },
  "series": [
    {"id": "a", "kind": "function", "expr": "1/x", "var": "x", "label": "1/x"},
    {"id": "b", "kind": "function", "expr": "x^3", "var": "x", "label": "x³"}
  ],
  "viewport": {"width_px": 800, "height_px": 500},
  "theme": "light"
}
```

Regras:

- `axes.x.range` é obrigatório, com dois finitos e início menor que o fim. O motor não inventa intervalo: sem ele,
  `clarification_needed`.
- `y.range` é `[min, max]` ou `"auto"`. Em `auto`, um polo não pode achatar o resto: usa uma faixa robusta e declara
  `y_auto: "robust"` e `clipped: true`, para ninguém achar que a curva acaba ali.
- Variável além de `var` precisa de valor em `params`; se faltar, `data_unavailable`, citando qual série.
- `deny_unknown_fields`, como a rota `/math`.

## 4. Artefato: `katu.plot/1`

Sobre o envelope de resultado, `nature` é `numeric` e `content.value` traz:

```json
{
  "kind": "plot2d",
  "schema": "katu.plot/1",
  "axes": {
    "x": {"label": "x", "scale": "linear", "range": [-1, 1], "ticks": [-1, -0.5, 0, 0.5, 1]},
    "y": {"label": "f(x)", "scale": "linear", "range": [-20, 20], "y_auto": "robust", "clipped": true, "ticks": [-20, 0, 20]}
  },
  "series": [
    {
      "id": "a",
      "label": "1/x",
      "label_latex": "\\frac{1}{x}",
      "style": {"color_token": "series-1", "dash": "solid"},
      "segments": [{"points": [[-1.0, -1.0], [-0.05, -20.0]]}, {"points": [[0.05, 20.0], [1.0, 1.0]]}],
      "breaks": [{"x_from": -0.0001, "x_to": 0.0001, "kind": "pole_suspected"}],
      "sampling": {"method": "adaptive", "evaluations": 812, "points": 240, "max_depth_hit": false}
    }
  ],
  "warnings": [{"code": "finite_sampling", "message": "amostragem finita não detecta toda singularidade"}],
  "alt_text": "Gráfico de 2 curvas, x de -1 a 1. 1/x tem polo suspeito perto de 0.",
  "request_hash": "sha256:...",
  "engine_version": "..."
}
```

- `segments` é a unidade de desenho: **o renderizador nunca liga dois segmentos**.
- `breaks[].kind`: `pole_suspected`, `jump_suspected`, `domain_gap` ou `undersampled`. "Suspeito" faz parte do nome,
  pois vem de amostragem finita.
- Pontos em JSON são f64. O gráfico é aproximação visual, mesmo com expressão exata.
- Mesmo pedido e mesma versão do motor produzem bytes idênticos. Sem relógio nem aleatoriedade sem semente.
- Cor é token, não valor. O tema resolve o token, para o artefato funcionar em claro e escuro.

## 5. Amostragem e descontinuidades

1. **Grade inicial** proporcional à largura em pixels, com deslocamento fixo por semente, para não coincidir com o período
   de uma função.
2. **Subdivisão** enquanto o erro em pixels da corda passar de um limiar, até o limite de profundidade. Se o limite for
   atingido em algum trecho, esse trecho sai como `undersampled`.
3. **Falha de avaliação** (`ln` de negativo, divisão por zero) vira `domain_gap`. Nunca vira zero nem é interpolada.
4. **Candidato a quebra:** salto grande entre pontos vizinhos, com subdivisão que não reduz esse salto.
5. **Classificação**, com o mesmo critério da bisseção corrigida: se |f| não cai junto com o intervalo, é quebra. Crescendo
   é `pole_suspected`; estável, `jump_suspected`. Só queda de escala continua como curva contínua.

A classificação precisa vir de uma função pública do katu-math, para gráfico e raiz não discordarem sobre o mesmo polo.

**Limite explícito.** Uma função com detalhe mais fino que a amostragem pode enganar o método, e o aviso `finite_sampling`
sai em todo gráfico de função. `sin(1/x)` perto de 0 precisa terminar como `undersampled`, não como curva lisa falsa.

## 6. Primeiro recorte

**Dentro:** funções reais de uma variável, várias curvas em um mesmo eixo, intervalos explícitos, escala linear, zoom,
exportação SVG, especificação JSON e dados CSV, temas claro e escuro, alternativa textual.

**Fora:** escalas log, polares, paramétricas, implícitas, superfícies e 3D, GPU, estatística, dados tabulares de entrada,
controles de parâmetros, rotação e animação.

### Zoom

Existem dois zooms, e a diferença é parte do contrato:

| Tipo | Como | Garantia |
|---|---|---|
| Ampliação visual | O consumidor reescala a geometria já recebida | Nenhum detalhe novo; serve só de resposta imediata |
| Zoom calculado | Novo pedido com outro `axes.x.range` | Reamostra, reclassifica quebras e substitui a geometria |

O consumidor mostra a ampliação visual e pede o zoom calculado ao soltar o gesto. Enquanto a resposta não chega, precisa
indicar que está desatualizado. Deslocar segue a mesma regra; inspeção de coordenadas e seleção de série são locais e não
inventam valores entre pontos sem avisar que são interpolados.

### Exportação

| Formato | Recorte | Observação |
|---|---|---|
| SVG | Sim | Gerado do artefato; com `<title>` e `<desc>` |
| JSON | Sim | O próprio artefato, com o pedido, reproduzível |
| CSV | Sim | `series,segment,x,y`; o segmento preserva as quebras |
| PNG | Decisão em F2 | O plano prevê PNG; depende do renderizador escolhido |

### Aparência e acessibilidade

- Tipografia legível, grade discreta, legenda, temas claro e escuro com contraste verificado.
- Séries diferem por tracejado ou marcador, além da cor, a partir da segunda.
- Rótulos em LaTeX quando disponíveis, com texto simples como alternativa.
- `alt_text` gerado dos dados e tabela de pontos disponível; teclado alcança zoom e deslocamento.
- Visual não é critério subjetivo solto: contraste e legibilidade têm teste, e a aparência final tem conferência humana.

### Limites iniciais, a ajustar com medição

Até 16 séries, 20.000 pontos e 200.000 avaliações por série, 2 s por pedido, viewport até 4096 px. Ao passar:
`resource_limit`, e sem artefato parcial que pareça completo.

## 7. Uso como ferramenta

O modelo pede o gráfico; o motor gera; o consumidor apresenta. Para não encher o contexto, o modelo recebe referência e
resumo:

```json
{"artifact_id": "plot_...", "summary": {"series": 2, "x_range": [-1, 1], "breaks": [{"series": "a", "kind": "pole_suspected", "near": 0.0}], "warnings": ["finite_sampling"]}}
```

O modelo só descreve o que está no resumo. Ele não afirma assíntota, máximo ou raiz por "olhar" o gráfico: para isso, chama
a ferramenta de cálculo. Recusa volta como resultado legítimo, como nas contas. Armazenamento e tempo de vida do artefato
ficam para o desenho da integração.

## 8. Gabaritos de aceitação

Testam o contrato, além da aparência.

| Caso | Esperado |
|---|---|
| `1/x` em `[-1,1]` | 2 segmentos, nenhum atravessa 0; `pole_suspected` perto de 0 |
| `tan(x)` em `[-5,5]` | 5 segmentos; `pole_suspected` perto de ±π/2 e ±3π/2 |
| `abs(x)/x` em `[-1,1]` | 2 segmentos em -1 e 1; `jump_suspected` |
| `sqrt(x)` em `[-1,1]` | `domain_gap` em x < 0; nenhum ponto ali |
| `ln(x)` em `[-1,0]` | Nenhum ponto; `invalid_input`, sem gráfico vazio com aparência de sucesso |
| `x^2` em `[-1,1]` | 1 segmento, sem quebra; erro de corda abaixo do limiar contra o valor analítico |
| `sin(x)` e `cos(x)` em `[0,6.28]` | 2 séries com estilos diferentes, sem quebra |
| `sin(1/x)` em `[0.0001,1]` | `undersampled` perto de 0; não sai como curva lisa |
| `0.000000001*x` em `[0,1]` | Eixo y ajustado à escala, sem achatar em zero |
| `1/x` com zoom em `[0.001,0.01]` | Reamostrado, sem quebra no novo intervalo |
| Intervalo ausente, invertido ou não finito | Recusa ou esclarecimento |
| `foo(x)` | `invalid_input`, citando a série |
| 17 séries | `resource_limit` |
| Mesmo pedido, duas vezes | Artefato idêntico byte a byte |
| CSV do artefato | Mesmos pontos e segmentos do JSON |
| SVG, tema claro e escuro | XML válido; contraste verificado; nenhum `<path>` liga segmentos |

Comparação independente: os valores dos pontos contra NumPy, por amostragem. Matplotlib pode servir de comparação visual e
de medição, nunca como dependência de produção.

## 9. Medições previstas, ainda não feitas

Tempo e memória por gráfico, número de avaliações, tamanho do SVG e tempo de renderização, nos mesmos casos e no mesmo
hardware. Só depois disso comparar com outra biblioteca.

Primeira medida, no Mac M, com a rota `/v1/plot` local, mediana de 7 pedidos, sem comparação com outra biblioteca:

| Pedido | Tempo do pedido | Avaliações | Pontos | SVG |
|---|---|---|---|---|
| `x^2` | 0,5 ms | 401 | 401 | 7 KB |
| `1/x` e `x^3` | 0,7 ms | 1.867 | 801 | 13 KB |
| `tan(x)` | 0,7 ms | 3.251 | 584 | 10 KB |
| `sin(1/x)` | 1,0 ms | 3.242 | 1.041 | 16 KB |
| `sin(1/x)`, viewport de 4096 px | 3,2 ms | 7.767 | 4.280 | 67 KB |
| 16 curvas `sin(1/x)` | 14,7 ms | 51.872 | 16.656 | 244 KB |

## 11. Decisões da implementação

| Tema | Decisão | Motivo |
|---|---|---|
| Organização | `crates/katu-plot`, com `katu-math`, `serde` e `serde_json`. Sem dependência gráfica | O núcleo não carrega renderizador |
| API | `PlotRequest` é a fonte; `Figure` é só uma forma de escrevê-lo em código. Sem estado global | Um só caminho até o artefato |
| Classificação | `katu_math::classify_bracket`, com o critério de queda de escala da bisseção | Gráfico e raiz não discordam sobre o mesmo polo |
| Polo de mesmo sinal | Uma segunda busca persegue o maior \|f\| | `1/x^2` não muda de sinal, e a busca por diferença o perdia de vista |
| SVG | Sai do núcleo, sem biblioteca | O plano já permitia. Tarefa F2 segue aberta para PNG e tela interativa |
| Exportação | Campo opcional `formats` no pedido (`svg`, `csv`), com resposta em `content.svg` e `content.csv` | Acréscimo a este contrato: não muda o artefato nem o `request_hash` |
| Rota | `POST /v1/plot`, no envelope `katu.result/1`; divide vagas e prazo de 2 s com `/v1/compute` | O contrato não definia o nome |
| `request_hash` | SHA-256 próprio, conferido com vetores do NIST | Identifica o pedido, sem papel de segurança; evita dependência |

**Acréscimos ao artefato:** a recusa por domínio de `ln(x)` em `[-1,0]` usa `domain_error`; `y_auto` é `"robust"` se algo ficou
fora da faixa, e `"full"` se não; sai o aviso `empty_series` quando só uma das curvas não tem ponto.

**Implementado e testado:** os gabaritos da seção 8, em `crates/katu-plot/tests/gabaritos.rs` e, contra o servidor real, em
`ARANDU/katu-qt/tests/test_gui.py`, onde os pontos também são comparados com NumPy.

**Limites conhecidos:**

- Curva contínua muito íngreme, com detalhe abaixo da profundidade de subdivisão dentro da vista, sai como `undersampled`.
- Um buraco isolado, como `x/x` em 0, só aparece se a amostragem cair nele; a linha passa por cima, e isso é parte do aviso
  `finite_sampling`.
- A legenda do SVG estima a largura do texto, porque SVG não mede texto.

**Python:** `katu.plot`, pelo pacote `katu`, chama o mesmo `build` da rota, no processo, com o mesmo prazo de 2 s.

**Fora, por decisão:** PNG no motor (F2, F8), escalas log (F9), integração com o Arandu e uso como ferramenta do modelo
(F7, seção 7).
