//! Do pedido ao artefato. Grade inicial, subdivisão enquanto a corda errar mais que meio pixel, e o que sobra vai para
//! `katu_math::classify_bracket`. Falha de avaliação vira lacuna, nunca zero nem interpolação.
use crate::artifact::{
    Artifact, AxesOut, AxisOut, BreakOut, Sampling, Segment, SeriesOut, Style, WarningOut,
};
use crate::request::{PlotRequest, SeriesIn, YRange, REQUEST_SCHEMA};
use crate::sha256::hex_digest;
use katu_math::{
    budget, classify_bracket, eval, latex, parse, Bracket, ErrorCode, Expr, MathError, Rational,
    Vars, Verdict,
};
use serde_json::Value;

const MAX_DEPTH: u32 = 14;
const CELL_BUDGET: u32 = 64;
const GAP_DEPTH: u32 = 2;
const PIXEL_TOLERANCE: f64 = 0.5;
/// Fixo, para a grade não coincidir com o período de uma função nem cair em cima de um polo. Faz o papel da semente.
const GRID_OFFSET: f64 = 0.1373;
/// Conferir o prazo a cada avaliação pesaria mais que a própria conta.
const BUDGET_EVERY: u64 = 256;
const SHORT_RUN: usize = 8;
const MARGINS_PX: u32 = 64;
const DASHES: [&str; 4] = ["solid", "dash", "dot", "dashdot"];
const MAX_LABEL: usize = 200;

/// Limites iniciais do contrato, a ajustar com medição. O tempo de 2 s fica com o orçamento de quem chama.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub series: usize,
    pub points_per_series: u64,
    pub evaluations_per_series: u64,
    pub viewport: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            series: 16,
            points_per_series: 20_000,
            evaluations_per_series: 200_000,
            viewport: 4096,
        }
    }
}

/// Recusa nos termos do contrato de resultados. Nunca há artefato parcial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlotError {
    pub status: String,
    pub code: String,
    pub message: String,
}

impl PlotError {
    fn new(status: &str, code: &str, message: impl Into<String>) -> Self {
        PlotError {
            status: status.into(),
            code: code.into(),
            message: message.into(),
        }
    }

    fn invalid(code: &str, message: impl Into<String>) -> Self {
        Self::new("invalid_input", code, message)
    }

    fn from_math(why: MathError, series: Option<&str>) -> Self {
        let message = match series {
            Some(id) => format!("Série {id}: {why}"),
            None => why.to_string(),
        };
        Self::new(why.code.status(), why.code.as_str(), message)
    }

    fn too_much(message: String) -> Self {
        Self::new(
            ErrorCode::LimitEvaluations.status(),
            ErrorCode::LimitEvaluations.as_str(),
            message,
        )
    }
}

enum Event {
    Point(f64, f64),
    Break(&'static str, f64, f64),
}

struct Sampler<'a> {
    id: &'a str,
    expr: &'a Expr,
    var: &'a str,
    vars: Vars,
    max_evaluations: u64,
    evaluations: u64,
    depth_hit: bool,
    events: Vec<Event>,
    grid: Vec<(f64, Option<f64>)>,
    window: (f64, f64),
    pixels_per_y: f64,
    budget: u32,
}

impl Sampler<'_> {
    fn f(&mut self, x: f64) -> Result<Option<f64>, PlotError> {
        self.evaluations += 1;
        if self.evaluations > self.max_evaluations {
            return Err(PlotError::too_much(format!(
                "Série {}: passou de {} avaliações.",
                self.id, self.max_evaluations
            )));
        }
        if self.evaluations % BUDGET_EVERY == 1 {
            budget::check().map_err(|why| PlotError::from_math(why, None))?;
        }
        match self.vars.get_mut(self.var) {
            Some(slot) => *slot = x,
            None => {
                self.vars.insert(self.var.to_owned(), x);
            }
        }
        match eval(self.expr, &self.vars) {
            Ok(y) if y.is_finite() => Ok(Some(y)),
            Ok(_) => Ok(None),
            Err(why) if why.code.is_point_failure() => Ok(None),
            Err(why) => Err(PlotError::from_math(why, Some(self.id))),
        }
    }

    fn sample_grid(&mut self, low: f64, high: f64, cells: u32) -> Result<(), PlotError> {
        let width = high - low;
        let mut xs = vec![low];
        for i in 1..cells {
            xs.push(low + width * (f64::from(i) + GRID_OFFSET) / f64::from(cells));
        }
        xs.push(high);
        for x in xs {
            let y = self.f(x)?;
            self.grid.push((x, y));
        }
        Ok(())
    }

    fn refine(&mut self, window: (f64, f64), pixels_per_y: f64) -> Result<(), PlotError> {
        self.window = window;
        self.pixels_per_y = pixels_per_y;
        if let (x, Some(y)) = self.grid[0] {
            self.events.push(Event::Point(x, y));
        }
        for index in 1..self.grid.len() {
            let ((xl, yl), (xr, yr)) = (self.grid[index - 1], self.grid[index]);
            self.budget = CELL_BUDGET;
            self.cell(xl, yl, xr, yr, 0)?;
            if let Some(y) = yr {
                self.events.push(Event::Point(xr, y));
            }
        }
        Ok(())
    }

    fn same_side_outside(&self, yl: f64, yr: f64) -> bool {
        let (low, high) = self.window;
        (yl > high && yr > high) || (yl < low && yr < low)
    }

    fn cell(
        &mut self,
        xl: f64,
        yl: Option<f64>,
        xr: f64,
        yr: Option<f64>,
        depth: u32,
    ) -> Result<(), PlotError> {
        if let (Some(a), Some(b)) = (yl, yr) {
            // Fora da vista nada precisa de precisão em pixels: só importa se há um polo escondido entre os dois.
            if self.same_side_outside(a, b) {
                return self.judge(xl, xr, a, b);
            }
        }
        if yl.is_none() && yr.is_none() && depth >= GAP_DEPTH {
            self.events.push(Event::Break("domain_gap", xl, xr));
            return Ok(());
        }
        let xm = (xl + xr) / 2.0;
        if depth >= MAX_DEPTH || self.budget == 0 || xm <= xl || xm >= xr {
            self.depth_hit = true;
            return match (yl, yr) {
                (Some(_), Some(_)) if self.budget == 0 => {
                    self.events.push(Event::Break("undersampled", xl, xr));
                    Ok(())
                }
                (Some(a), Some(b)) => self.judge(xl, xr, a, b),
                _ => {
                    self.events.push(Event::Break("domain_gap", xl, xr));
                    Ok(())
                }
            };
        }
        self.budget -= 1;
        let ym = self.f(xm)?;
        // O meio caiu em cima de um ponto indefinido, com a função definida dos dois lados: polo e salto têm nome
        // próprio. O resto, como o buraco de x/x em 0, segue a subdivisão e vira lacuna.
        if let (Some(_), None, Some(_)) = (yl, ym, yr) {
            let Bracket { verdict, from, to } = self.classify(xl, xr)?;
            let kind = match verdict {
                Verdict::Pole => Some("pole_suspected"),
                Verdict::Jump => Some("jump_suspected"),
                _ => None,
            };
            if let Some(kind) = kind {
                self.events.push(Event::Break(kind, from, to));
                return Ok(());
            }
        }
        if let (Some(a), Some(m), Some(b)) = (yl, ym, yr) {
            if (m - (a + b) / 2.0).abs() * self.pixels_per_y <= PIXEL_TOLERANCE {
                self.events.push(Event::Point(xm, m));
                return Ok(());
            }
        }
        self.cell(xl, yl, xm, ym, depth + 1)?;
        if let Some(m) = ym {
            self.events.push(Event::Point(xm, m));
        }
        self.cell(xm, ym, xr, yr, depth + 1)
    }

    fn classify(&mut self, xl: f64, xr: f64) -> Result<Bracket, PlotError> {
        let mut sampler_error = None;
        let found = classify_bracket(
            |x| match self.f(x) {
                Ok(y) => Ok(y),
                Err(why) => {
                    sampler_error = Some(why);
                    Err(MathError::new(ErrorCode::Cancelled, "interrompido"))
                }
            },
            xl,
            xr,
        );
        found.map_err(|why| {
            sampler_error.unwrap_or_else(|| PlotError::from_math(why, Some(self.id)))
        })
    }

    fn judge(&mut self, xl: f64, xr: f64, yl: f64, yr: f64) -> Result<(), PlotError> {
        let Bracket { verdict, from, to } = self.classify(xl, xr)?;
        let visible = !self.same_side_outside(yl, yr)
            && (yl - yr).abs() * self.pixels_per_y > 2.0 * PIXEL_TOLERANCE;
        let kind = match verdict {
            Verdict::Pole => "pole_suspected",
            Verdict::Jump if self.same_side_outside(yl, yr) => return Ok(()),
            Verdict::Jump => "jump_suspected",
            Verdict::Undefined => "domain_gap",
            Verdict::Oscillating => "undersampled",
            // Contínua, mas com detalhe que a subdivisão não resolveu dentro da vista.
            Verdict::Continuous if visible => "undersampled",
            Verdict::Continuous => return Ok(()),
        };
        self.depth_hit = self.depth_hit || kind == "undersampled";
        self.events.push(Event::Break(kind, from, to));
        Ok(())
    }

    /// Eventos em segmentos e quebras. Pontos antes e depois de uma quebra nunca ficam no mesmo segmento.
    fn geometry(&self) -> (Vec<Segment>, Vec<BreakOut>, u64) {
        let mut segments: Vec<Vec<[f64; 2]>> = Vec::new();
        let mut breaks: Vec<BreakOut> = Vec::new();
        let mut current: Vec<[f64; 2]> = Vec::new();
        for event in &self.events {
            match *event {
                Event::Point(x, y) => {
                    if current.last().is_none_or(|last| last[0] < x) {
                        current.push([x, y]);
                    }
                }
                Event::Break(kind, from, to) => {
                    let previous = breaks.last().map(|item| item.kind.as_str());
                    // Sobra de poucos pontos logo depois de um trecho subamostrado é parte do mesmo trecho.
                    if previous == Some("undersampled") && current.len() <= SHORT_RUN {
                        breaks.last_mut().unwrap().x_to = to;
                        current.clear();
                        continue;
                    }
                    if previous == Some(kind) && current.is_empty() {
                        breaks.last_mut().unwrap().x_to = to;
                        continue;
                    }
                    if !current.is_empty() {
                        segments.push(std::mem::take(&mut current));
                    }
                    breaks.push(BreakOut {
                        x_from: from,
                        x_to: to,
                        kind: kind.into(),
                    });
                }
            }
        }
        if !current.is_empty() {
            segments.push(current);
        }
        let points = segments.iter().map(|points| points.len() as u64).sum();
        (
            segments
                .into_iter()
                .map(|points| Segment { points })
                .collect(),
            breaks,
            points,
        )
    }
}

fn finite_pair(range: &[f64]) -> Option<(f64, f64)> {
    match range {
        [low, high] if low.is_finite() && high.is_finite() && low < high => Some((*low, *high)),
        _ => None,
    }
}

fn percentile(sorted: &[f64], share: f64) -> f64 {
    sorted[((share * sorted.len() as f64) as usize).min(sorted.len() - 1)]
}

/// Um polo não pode achatar o resto: de 5% a 95% dos valores, a não ser que o intervalo todo caiba sem achatar.
fn robust_window(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (low, high) = (percentile(&sorted, 0.05), percentile(&sorted, 0.95));
    let (first, last) = (sorted[0], sorted[sorted.len() - 1]);
    if last - first <= 4.0 * (high - low) {
        (first, last)
    } else {
        (low, high)
    }
}

fn widen(low: f64, high: f64) -> (f64, f64) {
    if high > low {
        let margin = (high - low) * 0.05;
        (low - margin, high + margin)
    } else {
        let pad = if low == 0.0 { 1.0 } else { low.abs() * 0.1 };
        (low - pad, high + pad)
    }
}

/// Marcadores em múltiplos de 1, 2 ou 5 vezes uma potência de 10, para o consumidor também usar em outra vista.
pub fn nice_ticks(low: f64, high: f64) -> Vec<f64> {
    const WANTED: f64 = 5.0;
    let raw = (high - low) / WANTED;
    let power = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|factor| factor * power)
        .find(|step| *step >= raw)
        .unwrap_or(10.0 * power);
    let (first, last) = (
        (low / step - 1e-9).ceil() as i64,
        (high / step + 1e-9).floor() as i64,
    );
    // Dividir por 10^n, em vez de multiplicar por 0,1, evita 0.30000000000000004 nos rótulos.
    let scale = if power < 1.0 {
        (1.0 / power).round()
    } else {
        1.0
    };
    let whole = (step * scale).round();
    (first..=last)
        .map(|i| (i as f64) * whole / scale + 0.0)
        .collect()
}

fn validate(
    request: &PlotRequest,
    limits: &Limits,
) -> Result<((f64, f64), Option<(f64, f64)>), PlotError> {
    if request.schema != REQUEST_SCHEMA {
        return Err(PlotError::invalid(
            "invalid_argument",
            format!("schema precisa ser {REQUEST_SCHEMA}"),
        ));
    }
    if request.axes.x.scale != "linear" || request.axes.y.scale != "linear" {
        return Err(PlotError::new(
            "unsupported",
            "unsupported_scale",
            "Só há escala linear neste recorte.",
        ));
    }
    let Some(x_range) = &request.axes.x.range else {
        return Err(PlotError::new(
            "clarification_needed",
            "missing_range",
            "Falta o intervalo de x: o motor não inventa um.",
        ));
    };
    let invalid_range = |axis: &str| {
        PlotError::invalid("invalid_range", format!("O intervalo de {axis} precisa de dois números finitos, com início menor que o fim."))
    };
    let x = finite_pair(x_range).ok_or_else(|| invalid_range("x"))?;
    if !(x.1 - x.0).is_finite() || (x.0 + (x.1 - x.0) / 4096.0) <= x.0 {
        return Err(PlotError::invalid(
            "invalid_range",
            "O intervalo de x é largo ou estreito demais para os números de ponto flutuante.",
        ));
    }
    let y = match &request.axes.y.range {
        None => None,
        Some(YRange::Named(name)) if name == "auto" => None,
        Some(YRange::Named(name)) => {
            return Err(PlotError::invalid(
                "invalid_range",
                format!("y.range é \"auto\" ou [min, max]; veio {name:?}."),
            ));
        }
        Some(YRange::Fixed(range)) => {
            let pair = finite_pair(range).ok_or_else(|| invalid_range("y"))?;
            if !(pair.1 - pair.0).is_finite() {
                return Err(invalid_range("y"));
            }
            Some(pair)
        }
    };
    if request.series.is_empty() {
        return Err(PlotError::invalid(
            "invalid_argument",
            "O gráfico precisa de ao menos uma série.",
        ));
    }
    if request.series.len() > limits.series {
        return Err(PlotError::new(
            "resource_limit",
            ErrorCode::LimitSize.as_str(),
            format!(
                "No máximo {} séries; vieram {}.",
                limits.series,
                request.series.len()
            ),
        ));
    }
    let viewport = request.viewport;
    if !(16..=limits.viewport).contains(&viewport.width_px)
        || !(16..=limits.viewport).contains(&viewport.height_px)
    {
        return Err(PlotError::new(
            "resource_limit",
            ErrorCode::LimitSize.as_str(),
            format!(
                "A área de desenho precisa ficar entre 16 e {} px.",
                limits.viewport
            ),
        ));
    }
    if !matches!(request.theme.as_str(), "light" | "dark") {
        return Err(PlotError::invalid(
            "invalid_argument",
            "theme precisa ser light ou dark.",
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for series in &request.series {
        let valid = !series.id.is_empty()
            && series.id.len() <= 32
            && series
                .id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !valid {
            return Err(PlotError::invalid(
                "invalid_argument",
                format!(
                    "id de série inválido: {:?}. Use letras, números, _ e -.",
                    series.id
                ),
            ));
        }
        if !seen.insert(series.id.as_str()) {
            return Err(PlotError::invalid(
                "invalid_argument",
                format!("id de série repetido: {}.", series.id),
            ));
        }
        if series.kind != "function" {
            return Err(PlotError::new(
                "unsupported",
                "unsupported_series",
                format!(
                    "Série {}: só há séries do tipo function neste recorte.",
                    series.id
                ),
            ));
        }
        if series
            .label
            .as_ref()
            .is_some_and(|label| label.chars().count() > MAX_LABEL)
        {
            return Err(PlotError::invalid(
                "invalid_argument",
                format!(
                    "Série {}: rótulo com mais de {MAX_LABEL} caracteres.",
                    series.id
                ),
            ));
        }
    }
    Ok((x, y))
}

fn parameters(request: &PlotRequest) -> Result<Vars, PlotError> {
    let mut vars = Vars::new();
    for (name, value) in &request.params {
        let number = match value {
            Value::Number(number) => number.as_f64(),
            Value::String(text) => Some(Rational::parse(text).and_then(|r| r.to_f64()).map_err(
                |why| PlotError::invalid(why.code.as_str(), format!("params.{name}: {why}")),
            )?),
            _ => None,
        };
        let Some(number) = number else {
            return Err(PlotError::invalid(
                "invalid_argument",
                format!("params.{name} precisa ser número ou texto com número."),
            ));
        };
        vars.insert(name.clone(), number);
    }
    Ok(vars)
}

/// Sem "-0.0000" no texto: abaixo de quatro casas, o ponto é 0.
fn shown_near(x: f64) -> f64 {
    if x.abs() < 5e-5 {
        0.0
    } else {
        x
    }
}

fn alt_text(x: (f64, f64), series: &[SeriesOut]) -> String {
    let name = |kind: &str| match kind {
        "pole_suspected" => "polo suspeito",
        "jump_suspected" => "salto suspeito",
        "domain_gap" => "fora do domínio",
        _ => "trecho subamostrado",
    };
    let parts: Vec<String> = series
        .iter()
        .map(|one| {
            if one.breaks.is_empty() {
                return format!("{}: sem quebra detectada", one.label);
            }
            let mut found: Vec<String> = one
                .breaks
                .iter()
                .take(4)
                .map(|item| {
                    format!(
                        "{} perto de {:.4}",
                        name(&item.kind),
                        shown_near((item.x_from + item.x_to) / 2.0)
                    )
                })
                .collect();
            if one.breaks.len() > 4 {
                found.push(format!("e mais {}", one.breaks.len() - 4));
            }
            format!("{}: {}", one.label, found.join(", "))
        })
        .collect();
    let plural = if series.len() == 1 { "curva" } else { "curvas" };
    format!(
        "Gráfico de {} {plural}, x de {} a {}. {}.",
        series.len(),
        x.0,
        x.1,
        parts.join(". ")
    )
}

/// O artefato, ou a recusa. Com orçamento ativo (`katu_math::budget::with_budget`), respeita prazo e cancelamento.
pub fn build(request: &PlotRequest, limits: &Limits) -> Result<Artifact, PlotError> {
    let (x, fixed_y) = validate(request, limits)?;
    let params = parameters(request)?;
    let cells = (request.viewport.width_px / 4).clamp(64, 1024);
    let height = f64::from(
        request
            .viewport
            .height_px
            .saturating_sub(MARGINS_PX)
            .max(16),
    );

    let mut parsed = Vec::with_capacity(request.series.len());
    for series in &request.series {
        let expr = parse(&series.expr).map_err(|why| {
            PlotError::from_math(why, Some(&format!("{} ({})", series.id, series.expr)))
        })?;
        parsed.push(expr);
    }
    let mut samplers: Vec<Sampler> = Vec::with_capacity(parsed.len());
    for (series, expr) in request.series.iter().zip(&parsed) {
        let mut sampler = Sampler {
            id: &series.id,
            expr,
            var: &series.var,
            vars: params.clone(),
            max_evaluations: limits.evaluations_per_series,
            evaluations: 0,
            depth_hit: false,
            events: Vec::new(),
            grid: Vec::new(),
            window: (0.0, 1.0),
            pixels_per_y: 1.0,
            budget: 0,
        };
        sampler.sample_grid(x.0, x.1, cells)?;
        samplers.push(sampler);
    }
    let values: Vec<Vec<f64>> = samplers
        .iter()
        .map(|one| one.grid.iter().filter_map(|point| point.1).collect())
        .collect();
    if values.iter().all(Vec::is_empty) {
        let names: Vec<&str> = request.series.iter().map(|one| one.expr.as_str()).collect();
        return Err(PlotError::invalid(
            ErrorCode::DomainError.as_str(),
            format!(
                "Nenhum ponto de {} está definido em x de {} a {}.",
                names.join(", "),
                x.0,
                x.1
            ),
        ));
    }
    let window = fixed_y.unwrap_or_else(|| {
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for one in values.iter().filter(|one| !one.is_empty()) {
            let (a, b) = robust_window(one);
            low = low.min(a);
            high = high.max(b);
        }
        widen(low, high)
    });
    let pixels_per_y = height / (window.1 - window.0);

    let mut out = Vec::with_capacity(samplers.len());
    let mut clipped = false;
    let mut warnings = vec![WarningOut {
        code: "finite_sampling".into(),
        message: "amostragem finita não detecta toda singularidade".into(),
    }];
    for (index, (mut sampler, (series, expr))) in samplers
        .drain(..)
        .zip(request.series.iter().zip(&parsed))
        .enumerate()
    {
        sampler.refine(window, pixels_per_y)?;
        let (segments, breaks, points) = sampler.geometry();
        if points > limits.points_per_series {
            return Err(PlotError::new(
                "resource_limit",
                ErrorCode::LimitSize.as_str(),
                format!(
                    "Série {}: {points} pontos, acima do limite de {}.",
                    series.id, limits.points_per_series
                ),
            ));
        }
        clipped = clipped
            || segments
                .iter()
                .flat_map(|one| &one.points)
                .any(|point| point[1] < window.0 || point[1] > window.1);
        let label = series.label.clone().unwrap_or_else(|| series.expr.clone());
        if points == 0 {
            warnings.push(WarningOut {
                code: "empty_series".into(),
                message: format!("{label}: nenhum ponto definido neste intervalo"),
            });
        }
        out.push(series_out(
            index, series, expr, label, segments, breaks, points, &sampler,
        ));
    }

    let mut hashed = request.clone();
    hashed.formats.clear();
    let canonical = serde_json::to_string(&hashed)
        .map_err(|why| PlotError::invalid("invalid_argument", why.to_string()))?;
    Ok(Artifact {
        kind: "plot2d".into(),
        schema: "katu.plot/1".into(),
        axes: AxesOut {
            x: AxisOut {
                label: request.axes.x.label.clone().unwrap_or_else(|| "x".into()),
                scale: "linear".into(),
                range: [x.0, x.1],
                y_auto: None,
                clipped: None,
                ticks: nice_ticks(x.0, x.1),
            },
            y: AxisOut {
                label: request
                    .axes
                    .y
                    .label
                    .clone()
                    .unwrap_or_else(|| "f(x)".into()),
                scale: "linear".into(),
                range: [window.0, window.1],
                y_auto: fixed_y.is_none().then(|| {
                    if clipped {
                        "robust".into()
                    } else {
                        "full".into()
                    }
                }),
                clipped: Some(clipped),
                ticks: nice_ticks(window.0, window.1),
            },
        },
        alt_text: alt_text(x, &out),
        series: out,
        warnings,
        request_hash: format!("sha256:{}", hex_digest(canonical.as_bytes())),
        engine_version: crate::VERSION.into(),
    })
}

#[allow(clippy::too_many_arguments)]
fn series_out(
    index: usize,
    series: &SeriesIn,
    expr: &Expr,
    label: String,
    segments: Vec<Segment>,
    breaks: Vec<BreakOut>,
    points: u64,
    sampler: &Sampler,
) -> SeriesOut {
    SeriesOut {
        id: series.id.clone(),
        label,
        label_latex: latex(expr),
        style: Style {
            color_token: format!("series-{}", index + 1),
            dash: DASHES[index % DASHES.len()].into(),
        },
        segments,
        breaks,
        sampling: Sampling {
            method: "adaptive".into(),
            evaluations: sampler.evaluations,
            points,
            max_depth_hit: sampler.depth_hit,
        },
    }
}
