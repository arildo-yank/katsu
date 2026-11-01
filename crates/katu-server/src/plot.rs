//! Rota `POST /v1/plot`: o katu-plot no envelope `katu.result/1`. Recusa é resultado e volta com HTTP 200; 400 fica
//! para pedido malformado. Divide as vagas e o prazo com `/v1/compute`.
use crate::compute::{base, take_slot, CancelOnDrop, SLOTS, TIME_LIMIT};
use crate::math::failure;
use axum::{extract::rejection::JsonRejection, http::StatusCode, Json};
use katu_math::budget::with_budget;
use katu_math::ErrorCode;
use katu_plot::{build, render_svg, Limits, PlotError, PlotRequest, TOKENS_DARK, TOKENS_LIGHT};
use serde_json::{json, Map, Value};
use std::{
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};

fn refused(request: &Value, why: &PlotError, elapsed_us: u64) -> Value {
    let mut out = base(request, &why.status, elapsed_us);
    out.insert(
        "refusal".into(),
        json!({"code": why.code, "message": why.message}),
    );
    out.insert("provenance".into(), provenance());
    Value::Object(out)
}

fn provenance() -> Value {
    json!({"engine": "katu-plot", "engine_version": katu_plot::VERSION})
}

fn answer(parsed: &PlotRequest, request: &Value) -> Value {
    let started = Instant::now();
    let elapsed = || started.elapsed().as_micros() as u64;
    if let Some(format) = parsed
        .formats
        .iter()
        .find(|format| !matches!(format.as_str(), "svg" | "csv"))
    {
        let why = PlotError {
            status: "invalid_input".into(),
            code: ErrorCode::InvalidArgument.as_str().into(),
            message: format!("formato desconhecido: {format}. Há svg e csv."),
        };
        return refused(request, &why, 0);
    }
    let artifact = match build(parsed, &Limits::default()) {
        Ok(artifact) => artifact,
        Err(why) => return refused(request, &why, elapsed()),
    };
    let mut content = Map::new();
    if parsed.formats.iter().any(|format| format == "svg") {
        let theme = if parsed.theme == "dark" {
            &TOKENS_DARK
        } else {
            &TOKENS_LIGHT
        };
        content.insert(
            "svg".into(),
            json!(render_svg(
                &artifact,
                theme,
                parsed.viewport.width_px,
                parsed.viewport.height_px
            )),
        );
    }
    if parsed.formats.iter().any(|format| format == "csv") {
        content.insert("csv".into(), json!(artifact.to_csv()));
    }
    let warnings = json!(artifact.warnings);
    content.insert("value".into(), json!(artifact));
    let mut out = base(request, "success", elapsed());
    out.insert("nature".into(), json!("numeric"));
    out.insert("content".into(), Value::Object(content));
    out.insert(
        "evidence".into(),
        json!({"method": "Amostragem adaptativa", "checks": []}),
    );
    out.insert("warnings".into(), warnings);
    out.insert("provenance".into(), provenance());
    Value::Object(out)
}

pub(super) async fn handle(
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Json(request) = body.map_err(|why| failure(StatusCode::BAD_REQUEST, &why.body_text()))?;
    let parsed: PlotRequest = serde_json::from_value(request.clone())
        .map_err(|why| failure(StatusCode::BAD_REQUEST, &why.to_string()))?;
    let Some(slot) = take_slot(&SLOTS) else {
        let why = PlotError {
            status: ErrorCode::LimitConcurrency.status().into(),
            code: ErrorCode::LimitConcurrency.as_str().into(),
            message: "muitas contas ao mesmo tempo; tente de novo em instantes".into(),
        };
        return Ok(Json(refused(&request, &why, 0)));
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let _cancel_on_drop = CancelOnDrop(cancel.clone());
    let out = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        with_budget(TIME_LIMIT, Some(cancel), || answer(&parsed, &request))
    })
    .await
    .map_err(|_| failure(StatusCode::INTERNAL_SERVER_ERROR, "plot worker failed"))?;
    Ok(Json(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(body: Value) -> Value {
        let parsed: PlotRequest = serde_json::from_value(body.clone()).unwrap();
        answer(&parsed, &body)
    }

    fn request(expr: &str, range: Value) -> Value {
        json!({
            "schema": "katu.plot.request/1",
            "axes": {"x": {"range": range}},
            "series": [{"id": "a", "kind": "function", "expr": expr}],
        })
    }

    #[test]
    fn success_has_the_envelope_and_the_artifact() {
        let out = run(request("1/x", json!([-1, 1])));
        assert_eq!(out["schema"], "katu.result/1");
        assert_eq!(out["status"], "success");
        assert_eq!(out["nature"], "numeric");
        assert_eq!(out["provenance"]["engine"], "katu-plot");
        let value = &out["content"]["value"];
        assert_eq!(value["schema"], "katu.plot/1");
        assert_eq!(value["series"][0]["breaks"][0]["kind"], "pole_suspected");
        assert_eq!(out["warnings"][0]["code"], "finite_sampling");
        assert!(out["content"].get("svg").is_none());
    }

    #[test]
    fn refusal_is_a_result_without_artifact() {
        let out = run(request("ln(x)", json!([-1, 0])));
        assert_eq!(out["status"], "invalid_input");
        assert_eq!(out["refusal"]["code"], "domain_error");
        assert!(out.get("content").is_none());
        assert_eq!(
            run(request("foo(x)", json!([0, 1])))["refusal"]["code"],
            "parse_error"
        );
        assert_eq!(
            run(request("x + a", json!([0, 1])))["status"],
            "data_unavailable"
        );
    }

    #[test]
    fn formats_come_from_the_artifact() {
        let mut body = request("1/x", json!([-1, 1]));
        body["formats"] = json!(["svg", "csv"]);
        body["theme"] = json!("dark");
        let out = run(body);
        let svg = out["content"]["svg"].as_str().unwrap();
        assert!(svg.contains(TOKENS_DARK.surface));
        assert_eq!(svg.matches("<path ").count(), 2);
        assert!(out["content"]["csv"]
            .as_str()
            .unwrap()
            .starts_with("series,segment,x,y\n"));
        let mut bad = request("x", json!([0, 1]));
        bad["formats"] = json!(["bmp"]);
        assert_eq!(run(bad)["status"], "invalid_input");
    }

    #[test]
    fn same_request_same_artifact_bytes() {
        let body = request("tan(x)", json!([-5, 5]));
        let once = serde_json::to_string(&run(body.clone())["content"]["value"]).unwrap();
        assert_eq!(
            once,
            serde_json::to_string(&run(body)["content"]["value"]).unwrap()
        );
    }

    #[test]
    fn expired_budget_refuses() {
        let body = request("sin(x)", json!([0, 1]));
        let parsed: PlotRequest = serde_json::from_value(body.clone()).unwrap();
        let out = with_budget(std::time::Duration::ZERO, None, || answer(&parsed, &body));
        assert_eq!(out["status"], "resource_limit");
    }
}
