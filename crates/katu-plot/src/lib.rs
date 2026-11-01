//! Gráficos 2D de funções, conforme `docs/KATU_PLOT_CONTRATO.md`: do pedido `katu.plot.request/1` ao artefato
//! `katu.plot/1`, com SVG e CSV gerados do artefato. Quem interpreta e avalia a expressão, e classifica as quebras, é
//! o katu-math. Sem estado global, sem relógio no resultado: mesmo pedido e mesma versão dão os mesmos bytes.
mod artifact;
mod request;
mod sampling;
mod sha256;
mod svg;

pub use artifact::{Artifact, AxisOut, BreakOut, Sampling, Segment, SeriesOut, Style, WarningOut};
pub use request::{Axes, AxisIn, PlotRequest, SeriesIn, Viewport, YRange, REQUEST_SCHEMA};
pub use sampling::{build, nice_ticks, Limits, PlotError};
pub use svg::{render_svg, Theme, TOKENS_DARK, TOKENS_LIGHT};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Um gráfico montado em código. É só um jeito de escrever o mesmo pedido que chega em JSON.
#[derive(Clone, Debug)]
pub struct Figure {
    request: PlotRequest,
}

impl Default for Figure {
    fn default() -> Self {
        Self::new()
    }
}

impl Figure {
    pub fn new() -> Self {
        Figure {
            request: PlotRequest::empty(),
        }
    }

    pub fn x_range(&mut self, low: f64, high: f64) -> &mut Self {
        self.request.axes.x.range = Some(vec![low, high]);
        self
    }

    pub fn y_range(&mut self, low: f64, high: f64) -> &mut Self {
        self.request.axes.y.range = Some(YRange::Fixed(vec![low, high]));
        self
    }

    pub fn x_label(&mut self, label: &str) -> &mut Self {
        self.request.axes.x.label = Some(label.to_owned());
        self
    }

    pub fn y_label(&mut self, label: &str) -> &mut Self {
        self.request.axes.y.label = Some(label.to_owned());
        self
    }

    pub fn function(&mut self, expr: &str, var: &str) -> &mut Self {
        let id = format!("s{}", self.request.series.len() + 1);
        self.request.series.push(SeriesIn::function(&id, expr, var));
        self
    }

    pub fn build(&self, limits: &Limits) -> Result<Artifact, PlotError> {
        build(&self.request, limits)
    }

    pub fn request(&self) -> &PlotRequest {
        &self.request
    }
}
