//! Os gabaritos da seção 8 de `docs/KATU_PLOT_CONTRATO.md`.
use katu_plot::{
    build, render_svg, Artifact, Figure, Limits, PlotError, PlotRequest, TOKENS_DARK, TOKENS_LIGHT,
};
use serde_json::json;
use std::f64::consts::PI;

fn request(exprs: &[&str], range: serde_json::Value) -> PlotRequest {
    let series: Vec<_> = exprs
        .iter()
        .enumerate()
        .map(|(i, expr)| json!({"id": format!("s{i}"), "kind": "function", "expr": expr, "var": "x"}))
        .collect();
    serde_json::from_value(json!({
        "schema": "katu.plot.request/1",
        "axes": {"x": {"range": range}},
        "series": series,
    }))
    .unwrap()
}

fn plot(exprs: &[&str], low: f64, high: f64) -> Artifact {
    build(&request(exprs, json!([low, high])), &Limits::default())
        .unwrap_or_else(|why| panic!("{exprs:?}: {why:?}"))
}

fn refused(exprs: &[&str], range: serde_json::Value) -> PlotError {
    build(&request(exprs, range), &Limits::default()).unwrap_err()
}

fn kinds(artifact: &Artifact, series: usize) -> Vec<&str> {
    artifact.series[series]
        .breaks
        .iter()
        .map(|item| item.kind.as_str())
        .collect()
}

fn segments(artifact: &Artifact, series: usize) -> usize {
    artifact.series[series].segments.len()
}

#[test]
fn inverse_does_not_cross_its_pole() {
    let art = plot(&["1/x"], -1.0, 1.0);
    assert_eq!(segments(&art, 0), 2);
    assert_eq!(kinds(&art, 0), ["pole_suspected"]);
    let item = &art.series[0].breaks[0];
    assert!(
        item.x_from <= 0.0 && 0.0 <= item.x_to && item.x_to - item.x_from < 1e-3,
        "{item:?}"
    );
    assert!(art.series[0].segments[0].points.iter().all(|p| p[0] < 0.0));
    assert!(art.series[0].segments[1].points.iter().all(|p| p[0] > 0.0));
    assert_eq!(art.axes.y.clipped, Some(true));
    assert_eq!(art.axes.y.y_auto.as_deref(), Some("robust"));
    assert!(art.alt_text.contains("perto de 0.0000"), "{}", art.alt_text);
    assert!(!art.alt_text.contains("-0.0000"), "{}", art.alt_text);
}

#[test]
fn tangent_has_five_segments() {
    let art = plot(&["tan(x)"], -5.0, 5.0);
    assert_eq!(segments(&art, 0), 5);
    assert_eq!(kinds(&art, 0), ["pole_suspected"; 4]);
    for (item, pole) in art.series[0]
        .breaks
        .iter()
        .zip([-1.5 * PI, -0.5 * PI, 0.5 * PI, 1.5 * PI])
    {
        assert!(item.x_from <= pole && pole <= item.x_to, "{item:?} {pole}");
    }
}

#[test]
fn step_is_a_jump() {
    let art = plot(&["abs(x)/x"], -1.0, 1.0);
    assert_eq!(segments(&art, 0), 2);
    assert_eq!(kinds(&art, 0), ["jump_suspected"]);
    assert_eq!(art.series[0].segments[0].points[0][1], -1.0);
    assert_eq!(art.series[0].segments[1].points.last().unwrap()[1], 1.0);
}

#[test]
fn same_sign_poles_are_found_outside_the_view() {
    let art = plot(&["1/x^2"], -1.0, 1.0);
    assert_eq!(segments(&art, 0), 2);
    assert_eq!(kinds(&art, 0), ["pole_suspected"]);
    let art = plot(&["ln(abs(x))"], -1.0, 1.0);
    assert_eq!(segments(&art, 0), 2);
    assert_eq!(kinds(&art, 0), ["pole_suspected"]);
    let art = plot(&["1/(x-0.3)+1/(x+0.7)^2"], -2.0, 2.0);
    assert_eq!(kinds(&art, 0), ["pole_suspected", "pole_suspected"]);
}

#[test]
fn square_root_has_no_point_outside_its_domain() {
    let art = plot(&["sqrt(x)"], -1.0, 1.0);
    assert_eq!(kinds(&art, 0), ["domain_gap"]);
    assert_eq!(segments(&art, 0), 1);
    assert!(art.series[0].segments[0].points.iter().all(|p| p[0] >= 0.0));
    assert!(art.series[0].breaks[0].x_to < 1e-3);
}

#[test]
fn nothing_defined_is_a_refusal_and_not_an_empty_plot() {
    let why = refused(&["ln(x)"], json!([-1, 0]));
    assert_eq!(
        (why.status.as_str(), why.code.as_str()),
        ("invalid_input", "domain_error")
    );
}

#[test]
fn parabola_is_one_segment_within_tolerance() {
    let art = plot(&["x^2"], -1.0, 1.0);
    assert_eq!(segments(&art, 0), 1);
    assert!(kinds(&art, 0).is_empty());
    assert_eq!(art.axes.y.clipped, Some(false));
    let [low, high] = art.axes.y.range;
    let pixel = (high - low) / (500.0 - 64.0);
    let points = &art.series[0].segments[0].points;
    for pair in points.windows(2) {
        let middle = (pair[0][0] + pair[1][0]) / 2.0;
        assert!(((pair[0][1] + pair[1][1]) / 2.0 - middle * middle).abs() < 1.05 * pixel);
    }
    for [x, y] in points {
        assert_eq!(*y, x.powi(2));
    }
}

#[test]
fn two_series_have_different_styles() {
    let art = plot(&["sin(x)", "cos(x)"], 0.0, 6.28);
    assert!(kinds(&art, 0).is_empty() && kinds(&art, 1).is_empty());
    assert_ne!(
        art.series[0].style.color_token,
        art.series[1].style.color_token
    );
    assert_ne!(art.series[0].style.dash, art.series[1].style.dash);
}

#[test]
fn detail_finer_than_the_sampling_is_declared() {
    let art = plot(&["sin(1/x)"], 0.0001, 1.0);
    assert!(
        kinds(&art, 0).contains(&"undersampled"),
        "{:?}",
        kinds(&art, 0)
    );
    assert!(
        !kinds(&art, 0).contains(&"pole_suspected"),
        "{:?}",
        kinds(&art, 0)
    );
    assert!(art.series[0].breaks[0].x_from < 0.001);
    assert!(art.series[0].sampling.max_depth_hit);
}

#[test]
fn tiny_scale_is_not_flattened() {
    let art = plot(&["0.000000001*x"], 0.0, 1.0);
    let [low, high] = art.axes.y.range;
    assert!(high - low < 2e-9 && high - low > 5e-10);
    assert_eq!(segments(&art, 0), 1);
    assert!(art.axes.y.ticks.len() >= 2);
}

#[test]
fn computed_zoom_away_from_the_pole_has_no_break() {
    let art = plot(&["1/x"], 0.001, 0.01);
    assert!(kinds(&art, 0).is_empty());
    assert_eq!(segments(&art, 0), 1);
}

#[test]
fn bad_ranges_are_refused() {
    let series = json!({"id": "a", "kind": "function", "expr": "x"});
    let missing: PlotRequest = serde_json::from_value(
        json!({"schema": "katu.plot.request/1", "axes": {"x": {}}, "series": [series]}),
    )
    .unwrap();
    assert_eq!(
        build(&missing, &Limits::default()).unwrap_err().status,
        "clarification_needed"
    );
    for bad in [
        json!([1, 0]),
        json!([0, 0]),
        json!([0]),
        json!([0, 1, 2]),
        json!([1e308, 1e308]),
        json!([-1.7e308, 1.7e308]),
    ] {
        assert_eq!(
            refused(&["x"], bad.clone()).status,
            "invalid_input",
            "{bad}"
        );
    }
    assert_eq!(
        refused(&["x"], json!([1.0, 1.0000000000000002])).status,
        "invalid_input"
    );
}

#[test]
fn unknown_function_names_the_series() {
    let why = refused(&["x", "foo(x)"], json!([0, 1]));
    assert_eq!(
        (why.status.as_str(), why.code.as_str()),
        ("invalid_input", "parse_error")
    );
    assert!(
        why.message.contains("s1") && why.message.contains("foo(x)"),
        "{}",
        why.message
    );
}

#[test]
fn missing_variable_names_the_series() {
    let why = refused(&["x + a"], json!([0, 1]));
    assert_eq!(why.status, "data_unavailable");
    assert!(
        why.message.contains("s0") && why.message.contains('a'),
        "{}",
        why.message
    );
    let mut with_params = request(&["x + a"], json!([0, 1]));
    with_params.params.insert("a".into(), json!("1/3"));
    let art = build(&with_params, &Limits::default()).unwrap();
    assert_eq!(art.series[0].segments[0].points[0], [0.0, 1.0 / 3.0]);
}

#[test]
fn seventeen_series_hit_the_limit() {
    assert_eq!(refused(&["x"; 17], json!([0, 1])).status, "resource_limit");
}

#[test]
fn outside_the_cut_is_refused() {
    let mut log = request(&["x"], json!([1, 2]));
    log.axes.x.scale = "log".into();
    assert_eq!(
        build(&log, &Limits::default()).unwrap_err().status,
        "unsupported"
    );
    let mut data = request(&["x"], json!([1, 2]));
    data.series[0].kind = "data".into();
    assert_eq!(
        build(&data, &Limits::default()).unwrap_err().status,
        "unsupported"
    );
    let mut id = request(&["x"], json!([1, 2]));
    id.series[0].id = "a,b".into();
    assert_eq!(
        build(&id, &Limits::default()).unwrap_err().status,
        "invalid_input"
    );
    let unknown = serde_json::from_value::<PlotRequest>(
        json!({"schema": "x", "axes": {"x": {}}, "series": [], "nope": 1}),
    );
    assert!(unknown.is_err());
}

#[test]
fn evaluation_limit_gives_no_partial_artifact() {
    let limits = Limits {
        evaluations_per_series: 10,
        ..Limits::default()
    };
    let why = build(&request(&["x"], json!([0, 1])), &limits).unwrap_err();
    assert_eq!(why.status, "resource_limit");
}

#[test]
fn deadline_stops_the_plot() {
    let result = katu_math::budget::with_budget(std::time::Duration::ZERO, None, || {
        build(&request(&["sin(x)"], json!([0, 1])), &Limits::default())
    });
    let why = result.unwrap_err();
    assert_eq!(
        (why.status.as_str(), why.code.as_str()),
        ("resource_limit", "limit_time")
    );
}

#[test]
fn same_request_same_bytes() {
    let build_bytes =
        || serde_json::to_string(&plot(&["tan(x)", "x^2", "sin(1/x)"], -5.0, 5.0)).unwrap();
    assert_eq!(build_bytes(), build_bytes());
    let art = plot(&["x"], 0.0, 1.0);
    assert!(art.request_hash.starts_with("sha256:") && art.request_hash.len() == 71);
    let mut formats = request(&["x"], json!([0, 1]));
    assert_eq!(
        build(&formats, &Limits::default()).unwrap().request_hash,
        art.request_hash
    );
    formats.formats.push("svg".into());
    assert_eq!(
        build(&formats, &Limits::default()).unwrap().request_hash,
        art.request_hash
    );
    assert_ne!(plot(&["x"], 0.0, 2.0).request_hash, art.request_hash);
}

#[test]
fn csv_matches_the_artifact() {
    let art = plot(&["1/x", "x"], -1.0, 1.0);
    let csv = art.to_csv();
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some("series,segment,x,y"));
    for series in &art.series {
        for (number, segment) in series.segments.iter().enumerate() {
            for [x, y] in &segment.points {
                let line = lines.next().unwrap();
                let fields: Vec<&str> = line.split(',').collect();
                assert_eq!(fields[0], series.id);
                assert_eq!(fields[1].parse::<usize>().unwrap(), number);
                assert_eq!(fields[2].parse::<f64>().unwrap(), *x);
                assert_eq!(fields[3].parse::<f64>().unwrap(), *y);
            }
        }
    }
    assert_eq!(lines.next(), None);
}

/// Contraste WCAG entre duas cores #rrggbb.
fn contrast(a: &str, b: &str) -> f64 {
    let lum = |hex: &str| {
        let channel = |i: usize| {
            let value = f64::from(u8::from_str_radix(&hex[i..i + 2], 16).unwrap()) / 255.0;
            if value <= 0.03928 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5)
    };
    let (la, lb) = (lum(a), lum(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

#[test]
fn svg_keeps_segments_apart_in_both_themes() {
    let art = plot(&["1/x", "x^3"], -1.0, 1.0);
    let segments_total: usize = art.series.iter().map(|one| one.segments.len()).sum();
    for theme in [TOKENS_LIGHT, TOKENS_DARK] {
        let svg = render_svg(&art, &theme, 800, 500);
        assert!(svg.starts_with("<svg ") && svg.ends_with("</svg>"));
        assert_eq!(svg.matches("<path ").count(), segments_total);
        assert!(svg.contains("<title") && svg.contains("<desc"));
        assert!(!svg.contains("NaN") && !svg.contains("inf"));
        assert!(contrast(theme.text, theme.surface) >= 7.0);
        assert!(contrast(theme.muted, theme.surface) >= 4.5);
    }
    assert_ne!(
        render_svg(&art, &TOKENS_LIGHT, 800, 500),
        render_svg(&art, &TOKENS_DARK, 800, 500)
    );
}

#[test]
fn svg_escapes_labels() {
    let mut req = request(&["x"], json!([0, 1]));
    req.series[0].label = Some("a<b & \"c\"".into());
    let svg = render_svg(
        &build(&req, &Limits::default()).unwrap(),
        &TOKENS_LIGHT,
        800,
        500,
    );
    assert!(svg.contains("a&lt;b &amp; &quot;c&quot;"));
    assert!(!svg.contains("a<b"));
}

#[test]
fn ticks_are_clean() {
    assert_eq!(
        katu_plot::nice_ticks(-1.0, 1.0),
        [-1.0, -0.5, 0.0, 0.5, 1.0]
    );
    assert_eq!(
        katu_plot::nice_ticks(0.0, 1.0),
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    );
    assert_eq!(
        katu_plot::nice_ticks(0.1, 0.35),
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    );
    assert_eq!(
        katu_plot::nice_ticks(0.0, 2000.0),
        [0.0, 500.0, 1000.0, 1500.0, 2000.0]
    );
}

#[test]
fn figure_builds_the_same_as_json() {
    let mut figure = Figure::new();
    figure.x_range(-1.0, 1.0).y_range(-20.0, 20.0).x_label("x");
    figure.function("1/x", "x").function("x^3", "x");
    let art = figure.build(&Limits::default()).unwrap();
    assert_eq!(art.series.len(), 2);
    assert_eq!(art.axes.y.range, [-20.0, 20.0]);
    assert_eq!(art.axes.y.y_auto, None);
    assert_eq!(art.series[0].label_latex, "\\frac{1}{x}");
    assert_eq!(kinds(&art, 0), ["pole_suspected"]);
}
