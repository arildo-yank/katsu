//! Pedido `katu.plot.request/1`. Campo desconhecido é erro de transporte, como na rota `/math`.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const REQUEST_SCHEMA: &str = "katu.plot.request/1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlotRequest {
    pub schema: String,
    pub axes: Axes,
    pub series: Vec<SeriesIn>,
    /// Valor das variáveis além de `var`. Número ou texto ("1/3"), como em `vars` de `/v1/compute`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, Value>,
    #[serde(default)]
    pub viewport: Viewport,
    #[serde(default = "default_theme")]
    pub theme: String,
    /// Exportações a gerar do artefato: "svg", "csv". Não muda o artefato nem o seu hash.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub formats: Vec<String>,
}

fn default_theme() -> String {
    "light".into()
}

impl PlotRequest {
    pub(crate) fn empty() -> Self {
        PlotRequest {
            schema: REQUEST_SCHEMA.into(),
            axes: Axes::default(),
            series: Vec::new(),
            params: BTreeMap::new(),
            viewport: Viewport::default(),
            theme: default_theme(),
            formats: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Axes {
    pub x: AxisIn<Vec<f64>>,
    #[serde(default)]
    pub y: AxisIn<YRange>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisIn<R> {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default = "none", skip_serializing_if = "Option::is_none")]
    pub range: Option<R>,
    #[serde(default = "linear")]
    pub scale: String,
}

fn none<R>() -> Option<R> {
    None
}

fn linear() -> String {
    "linear".into()
}

impl<R> Default for AxisIn<R> {
    fn default() -> Self {
        AxisIn {
            label: None,
            range: None,
            scale: linear(),
        }
    }
}

/// `"auto"` ou `[min, max]`. Outro texto vira recusa, e não erro de transporte.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum YRange {
    Fixed(Vec<f64>),
    Named(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesIn {
    pub id: String,
    pub kind: String,
    pub expr: String,
    #[serde(default = "default_var")]
    pub var: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

fn default_var() -> String {
    "x".into()
}

impl SeriesIn {
    pub fn function(id: &str, expr: &str, var: &str) -> Self {
        SeriesIn {
            id: id.to_owned(),
            kind: "function".into(),
            expr: expr.to_owned(),
            var: var.to_owned(),
            label: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Viewport {
    pub width_px: u32,
    pub height_px: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Viewport {
            width_px: 800,
            height_px: 500,
        }
    }
}
