//! Artefato `katu.plot/1`. A ordem dos campos é fixa, e o f64 sai na forma mais curta que volta ao mesmo número:
//! a serialização é parte da garantia de bytes idênticos.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub kind: String,
    pub schema: String,
    pub axes: AxesOut,
    pub series: Vec<SeriesOut>,
    pub warnings: Vec<WarningOut>,
    pub alt_text: String,
    pub request_hash: String,
    pub engine_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AxesOut {
    pub x: AxisOut,
    pub y: AxisOut,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AxisOut {
    pub label: String,
    pub scale: String,
    pub range: [f64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_auto: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clipped: Option<bool>,
    pub ticks: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeriesOut {
    pub id: String,
    pub label: String,
    pub label_latex: String,
    pub style: Style,
    /// A unidade de desenho: o renderizador nunca liga dois segmentos.
    pub segments: Vec<Segment>,
    pub breaks: Vec<BreakOut>,
    pub sampling: Sampling,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Style {
    pub color_token: String,
    pub dash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub points: Vec<[f64; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BreakOut {
    pub x_from: f64,
    pub x_to: f64,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sampling {
    pub method: String,
    pub evaluations: u64,
    pub points: u64,
    pub max_depth_hit: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WarningOut {
    pub code: String,
    pub message: String,
}

impl Artifact {
    /// `series,segment,x,y`. O segmento preserva as quebras.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("series,segment,x,y\n");
        for series in &self.series {
            for (number, segment) in series.segments.iter().enumerate() {
                for [x, y] in &segment.points {
                    out.push_str(&format!("{},{number},{x},{y}\n", series.id));
                }
            }
        }
        out
    }
}
