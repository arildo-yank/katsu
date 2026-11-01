//! SVG gerado do artefato: não recalcula nada, e cada segmento é um `<path>` próprio. A paleta resolve os tokens de
//! cor do artefato, em ordem fixa; da segunda série em diante, o tracejado também distingue as curvas.
use crate::artifact::Artifact;
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Theme {
    pub surface: &'static str,
    pub text: &'static str,
    pub muted: &'static str,
    pub grid: &'static str,
    pub axis: &'static str,
    pub series: [&'static str; 8],
}

pub const TOKENS_LIGHT: Theme = Theme {
    surface: "#fcfcfb",
    text: "#0b0b0b",
    muted: "#52514e",
    grid: "#e4e3df",
    axis: "#a8a7a0",
    series: [
        "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
    ],
};

pub const TOKENS_DARK: Theme = Theme {
    surface: "#1a1a19",
    text: "#f0efec",
    muted: "#c3c2b7",
    grid: "#33332f",
    axis: "#6b6a64",
    series: [
        "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767",
    ],
};

const LEFT: f64 = 64.0;
const RIGHT: f64 = 16.0;
const TOP: f64 = 12.0;
const BOTTOM: f64 = 40.0;
const LEGEND_ROW: f64 = 20.0;
const LEGEND_SAMPLE: f64 = 34.0;
const CHAR_WIDTH: f64 = 7.0;
const PIXEL_CLAMP: f64 = 100_000.0;
const FONT: &str = "system-ui, -apple-system, 'Segoe UI', sans-serif";

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            // Controle não é XML válido.
            c if (c as u32) < 0x20 && !matches!(c, '\t' | '\n' | '\r') => out.push(' '),
            c => out.push(c),
        }
    }
    out
}

fn dash_array(dash: &str) -> &'static str {
    match dash {
        "dash" => " stroke-dasharray=\"8 5\"",
        "dot" => " stroke-dasharray=\"2 5\"",
        "dashdot" => " stroke-dasharray=\"8 4 2 4\"",
        _ => "",
    }
}

fn color<'a>(theme: &'a Theme, token: &str) -> &'a str {
    // Da nona série em diante a cor repete: quem distingue é o tracejado e a legenda.
    let number: usize = token
        .rsplit('-')
        .next()
        .and_then(|n| n.parse().ok())
        .unwrap_or(1);
    theme.series[(number.max(1) - 1) % theme.series.len()]
}

/// O número do rótulo, sem ruído de ponto flutuante.
fn tick_text(value: f64) -> String {
    let text = format!("{value:.6}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if value != 0.0 && (value.abs() < 1e-4 || value.abs() >= 1e7) {
        format!("{value:.3e}")
    } else if trimmed == "-0" || trimmed.is_empty() {
        "0".into()
    } else {
        trimmed.into()
    }
}

/// `width` e `height` em pixels. A largura da legenda é estimada: SVG não mede texto.
pub fn render_svg(artifact: &Artifact, theme: &Theme, width: u32, height: u32) -> String {
    let (total_w, total_h) = (f64::from(width), f64::from(height));
    let inner = (total_w - LEFT - RIGHT).max(8.0);
    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut used = 0.0;
    for (index, series) in artifact.series.iter().enumerate() {
        let need = LEGEND_SAMPLE + series.label.chars().count() as f64 * CHAR_WIDTH + 16.0;
        if rows.is_empty() || used + need > inner {
            rows.push(Vec::new());
            used = 0.0;
        }
        rows.last_mut().unwrap().push(index);
        used += need;
    }
    let legend = rows.len() as f64 * LEGEND_ROW;
    let (left, top) = (LEFT, TOP + legend);
    let plot_h = (total_h - TOP - BOTTOM - legend).max(8.0);
    let (bottom, right) = (top + plot_h, left + inner);
    let [x_low, x_high] = artifact.axes.x.range;
    let [y_low, y_high] = artifact.axes.y.range;
    let to_x =
        |x: f64| (left + (x - x_low) / (x_high - x_low) * inner).clamp(-PIXEL_CLAMP, PIXEL_CLAMP);
    let to_y = |y: f64| {
        (bottom - (y - y_low) / (y_high - y_low) * plot_h).clamp(-PIXEL_CLAMP, PIXEL_CLAMP)
    };

    let mut out = String::new();
    let _ = write!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-labelledby=\"title desc\" font-family=\"{FONT}\" font-size=\"12\">"
    );
    let _ = write!(
        out,
        "<title id=\"title\">Gráfico do Katu</title><desc id=\"desc\">{}</desc>",
        escape(&artifact.alt_text)
    );
    let _ = write!(out, "<defs><clipPath id=\"plot-area\"><rect x=\"{left:.2}\" y=\"{top:.2}\" width=\"{inner:.2}\" height=\"{plot_h:.2}\"/></clipPath></defs>");
    let _ = write!(
        out,
        "<rect width=\"{width}\" height=\"{height}\" fill=\"{}\"/>",
        theme.surface
    );

    for tick in &artifact.axes.x.ticks {
        let px = to_x(*tick);
        let _ = write!(out, "<line x1=\"{px:.2}\" y1=\"{top:.2}\" x2=\"{px:.2}\" y2=\"{bottom:.2}\" stroke=\"{}\"/>", theme.grid);
        let _ = write!(
            out,
            "<text x=\"{px:.2}\" y=\"{:.2}\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
            bottom + 16.0,
            theme.muted,
            tick_text(*tick)
        );
    }
    for tick in &artifact.axes.y.ticks {
        let py = to_y(*tick);
        let _ = write!(out, "<line x1=\"{left:.2}\" y1=\"{py:.2}\" x2=\"{right:.2}\" y2=\"{py:.2}\" stroke=\"{}\"/>", theme.grid);
        let _ = write!(
            out,
            "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"end\" fill=\"{}\">{}</text>",
            left - 8.0,
            py + 4.0,
            theme.muted,
            tick_text(*tick)
        );
    }
    let _ = write!(out, "<line x1=\"{left:.2}\" y1=\"{bottom:.2}\" x2=\"{right:.2}\" y2=\"{bottom:.2}\" stroke=\"{}\"/>", theme.axis);
    let _ = write!(out, "<line x1=\"{left:.2}\" y1=\"{top:.2}\" x2=\"{left:.2}\" y2=\"{bottom:.2}\" stroke=\"{}\"/>", theme.axis);
    let _ = write!(
        out,
        "<text x=\"{:.2}\" y=\"{:.2}\" text-anchor=\"middle\" fill=\"{}\">{}</text>",
        (left + right) / 2.0,
        bottom + 34.0,
        theme.muted,
        escape(&artifact.axes.x.label)
    );

    out.push_str("<g clip-path=\"url(#plot-area)\">");
    for series in &artifact.series {
        for item in &series.breaks {
            let (from, to) = (to_x(item.x_from), to_x(item.x_to));
            if matches!(item.kind.as_str(), "domain_gap" | "undersampled") {
                let _ = write!(out, "<rect x=\"{from:.2}\" y=\"{top:.2}\" width=\"{:.2}\" height=\"{plot_h:.2}\" fill=\"{}\" fill-opacity=\"0.11\"/>", (to - from).max(1.0), theme.muted);
            } else {
                let center = (from + to) / 2.0;
                let _ = write!(out, "<line x1=\"{center:.2}\" y1=\"{top:.2}\" x2=\"{center:.2}\" y2=\"{bottom:.2}\" stroke=\"{}\" stroke-dasharray=\"2 3\"/>", theme.axis);
            }
        }
    }
    for series in &artifact.series {
        let stroke = color(theme, &series.style.color_token);
        let dash = dash_array(&series.style.dash);
        for segment in &series.segments {
            if let [[x, y]] = segment.points[..] {
                let _ = write!(
                    out,
                    "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"2.5\" fill=\"{stroke}\"/>",
                    to_x(x),
                    to_y(y)
                );
                continue;
            }
            out.push_str("<path d=\"");
            for (index, [x, y]) in segment.points.iter().enumerate() {
                let _ = write!(
                    out,
                    "{}{:.2} {:.2}",
                    if index == 0 { "M" } else { "L" },
                    to_x(*x),
                    to_y(*y)
                );
            }
            let _ = write!(out, "\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"{dash}/>");
        }
    }
    out.push_str("</g>");

    let mut row_y = 4.0;
    for row in rows {
        let mut x = left;
        for index in row {
            let series = &artifact.series[index];
            let stroke = color(theme, &series.style.color_token);
            let _ = write!(out, "<line x1=\"{x:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{stroke}\" stroke-width=\"2\"{}/>", row_y + 8.0, x + 26.0, row_y + 8.0, dash_array(&series.style.dash));
            let _ = write!(
                out,
                "<text x=\"{:.2}\" y=\"{:.2}\" fill=\"{}\">{}</text>",
                x + 32.0,
                row_y + 12.0,
                theme.text,
                escape(&series.label)
            );
            x += LEGEND_SAMPLE + series.label.chars().count() as f64 * CHAR_WIDTH + 16.0;
        }
        row_y += LEGEND_ROW;
    }
    out.push_str("</svg>");
    out
}
