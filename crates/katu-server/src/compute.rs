//! Rota `POST /v1/compute`: as mesmas contas de `/math`, no envelope `katu.result/1` de
//! `docs/KATU_CONTRATO_RESULTADOS.md`. Recusa é resultado e volta com HTTP 200; 400 fica para pedido
//! malformado. A conta é sempre a de `math::solve`: esta camada só descreve o que aconteceu.
use crate::math::{add_latex, failure, solve, Request};
use axum::{extract::rejection::JsonRejection, http::StatusCode, Json};
use katu_math::linalg::{multiply, Matrix};
use katu_math::{eval, parse, ErrorCode, Vars};
use serde_json::{json, Map, Value};
use std::time::Instant;

const SCHEMA: &str = "katu.result/1";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Auto,
    Numeric,
    Exact,
}

fn mode_of(body: &Value) -> Result<Mode, String> {
    match body.get("mode") {
        None => Ok(Mode::Auto),
        Some(mode) => match mode.as_str() {
            Some("auto") => Ok(Mode::Auto),
            Some("numeric") => Ok(Mode::Numeric),
            Some("exact") => Ok(Mode::Exact),
            _ => Err("mode precisa ser auto, numeric ou exact".into()),
        },
    }
}

fn base(request: &Value, status: &str, elapsed_us: u64) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert("schema".into(), json!(SCHEMA));
    out.insert("status".into(), json!(status));
    out.insert("request".into(), request.clone());
    out.insert(
        "provenance".into(),
        json!({"engine": "katu-math", "engine_version": env!("CARGO_PKG_VERSION")}),
    );
    out.insert("steps".into(), json!([]));
    out.insert("warnings".into(), json!([]));
    out.insert("usage".into(), json!({"elapsed_us": elapsed_us}));
    out
}

fn refused(request: &Value, code: ErrorCode, message: &str, elapsed_us: u64) -> Value {
    let mut out = base(request, code.status(), elapsed_us);
    out.insert(
        "refusal".into(),
        json!({"code": code.as_str(), "message": message}),
    );
    Value::Object(out)
}

/// O que o pedido fixou: variável, intervalo, ponto de partida. Expressão e matrizes ficam em `request`.
fn parameters(op: &str, request: &Value) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(fields) = request.as_object() {
        for (key, value) in fields {
            if !matches!(key.as_str(), "op" | "mode" | "expr" | "a" | "b" | "u" | "v") {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    if matches!(op, "derive" | "integrate" | "antiderivative" | "root") {
        out.entry("var").or_insert(json!("x"));
    }
    out
}

/// |f(x)| na raiz devolvida. Resíduo, não erro em x.
fn root_residual(request: &Value, x: f64) -> Option<f64> {
    let expr = parse(request["expr"].as_str()?).ok()?;
    let var = request["var"].as_str().unwrap_or("x").to_owned();
    eval(&expr, &Vars::from([(var, x)])).ok().map(f64::abs)
}

/// Maior entrada de |A⁻¹A - I|.
fn inverse_residual(request: &Value, inverse: &Value) -> Option<f64> {
    let a: Matrix = serde_json::from_value(request["a"].clone()).ok()?;
    let inverse: Matrix = serde_json::from_value(inverse.clone()).ok()?;
    let product = multiply(&inverse, &a).ok()?;
    let mut worst = 0.0f64;
    for (i, row) in product.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            worst = worst.max((v - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    Some(worst)
}

fn warning(code: &str, message: &Value) -> Value {
    json!({"code": code, "message": message})
}

fn describe(request: &Value, solved: &Value, elapsed_us: u64) -> Value {
    let op = solved["op"].as_str().unwrap_or("");
    let approximate = solved["approximate"].as_bool().unwrap_or(true);
    let note = &solved["note"];
    let result = &solved["result"];
    let nature = match op {
        "derive" | "antiderivative" => "symbolic",
        "eval" if !approximate => "exact",
        _ => "numeric",
    };
    let value = match op {
        "derive" | "antiderivative" => json!({"kind": "expression", "text": result}),
        "inv" | "matmul" => json!({"kind": "matrix", "rows": result}),
        "cross" => json!({"kind": "vector", "items": result}),
        _ => json!({"kind": "number", "value": result}),
    };

    let mut content = Map::new();
    content.insert("interpreted".into(), solved["input"].clone());
    if !solved["input_latex"].is_null() {
        content.insert("interpreted_latex".into(), solved["input_latex"].clone());
    }
    content.insert("value".into(), value);
    if !solved["result_latex"].is_null() {
        content.insert("value_latex".into(), solved["result_latex"].clone());
    }
    let params = parameters(op, request);
    if !params.is_empty() {
        content.insert("parameters".into(), Value::Object(params));
    }

    let mut evidence = Map::new();
    evidence.insert("method".into(), solved["method"].clone());
    let mut checks = Vec::new();
    let mut warnings = Vec::new();
    match op {
        "eval" if !approximate => {
            checks.push(json!({"kind": "exact_arithmetic", "detail": "só inteiros abaixo de 2^53, sem arredondamento"}));
        }
        "eval" | "det" | "matmul" | "dot" | "cross" => warnings.push(warning("rounding", note)),
        "derive" if !solved["value_at"].is_null() => {
            let rounded = solved["value_at_approximate"].as_bool().unwrap_or(true);
            content.insert(
                "value_at".into(),
                json!({
                    "value": solved["value_at"],
                    "nature": if rounded { "numeric" } else { "exact" },
                }),
            );
            checks.push(json!({"kind": "domain_check", "detail": "a função original está definida no ponto"}));
        }
        "antiderivative" => {
            checks.push(json!({"kind": "substitution", "detail": "a derivada da primitiva foi comparada com a função em até 7 pontos, com folga relativa de 1e-7"}));
        }
        "integrate" => {
            evidence.insert(
                "tolerance_requested".into(),
                request.get("tol").cloned().unwrap_or(json!(1e-9)),
            );
            warnings.push(warning("tolerance_not_guaranteed", note));
        }
        "root" => {
            if solved["method"] == "Bisseção" {
                checks.push(json!({"kind": "sign_change_and_decay", "detail": "troca de sinal, com |f| caindo junto com o intervalo"}));
                warnings.push(warning("other_roots_possible", note));
            } else {
                checks.push(json!({"kind": "numeric_residual", "detail": "|f| e o passo de Newton pequenos no ponto final"}));
                warnings.push(warning("depends_on_start", note));
            }
            if let Some(residual) = result.as_f64().and_then(|x| root_residual(request, x)) {
                evidence.insert(
                    "error_observed".into(),
                    json!({"kind": "residual", "value": residual}),
                );
            }
        }
        "inv" => {
            warnings.push(warning("rounding", note));
            if let Some(residual) = inverse_residual(request, result) {
                checks.push(
                    json!({"kind": "numeric_residual", "detail": "maior entrada de |A⁻¹A - I|"}),
                );
                evidence.insert(
                    "error_observed".into(),
                    json!({"kind": "residual", "value": residual}),
                );
            }
        }
        "ode" => warnings.push(warning("fixed_step_no_error_estimate", note)),
        _ => {}
    }
    evidence.insert("checks".into(), Value::Array(checks));

    let mut out = base(request, "success", elapsed_us);
    out.insert("nature".into(), json!(nature));
    out.insert("content".into(), Value::Object(content));
    out.insert("evidence".into(), Value::Object(evidence));
    out.insert("warnings".into(), Value::Array(warnings));
    if let Some(steps) = solved["steps"].as_array() {
        let steps: Vec<Value> = steps
            .iter()
            .map(|step| json!({"rule": "term_by_term", "latex": step}))
            .collect();
        out.insert("steps".into(), Value::Array(steps));
    }
    Value::Object(out)
}

fn compute(mode: Mode, request: &Value, parsed: Request) -> Value {
    let started = Instant::now();
    let elapsed = || started.elapsed().as_micros() as u64;
    let mut solved = match solve(parsed) {
        Ok(solved) => solved,
        Err(why) => return refused(request, why.code, &why.message, elapsed()),
    };
    add_latex(&mut solved);
    // Determinante zero em ponto flutuante quer dizer pivô indistinguível do arredondamento, não prova.
    if solved["op"] == "det" && solved["result"].as_f64() == Some(0.0) {
        let message =
            "o determinante não se distingue de zero em ponto flutuante: a matriz é singular ou quase";
        return refused(request, ErrorCode::NearSingularMatrix, message, elapsed());
    }
    let answer = describe(request, &solved, elapsed());
    if mode == Mode::Exact && answer["nature"] != "exact" {
        let message = "esta conta ainda não tem resultado exato; peça mode auto ou numeric para a aproximação";
        return refused(request, ErrorCode::ExactModeUnavailable, message, elapsed());
    }
    answer
}

pub(super) async fn handle(
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Json(request) = body.map_err(|why| failure(StatusCode::BAD_REQUEST, &why.body_text()))?;
    let mode = mode_of(&request).map_err(|why| failure(StatusCode::BAD_REQUEST, &why))?;
    let mut fields = request.clone();
    if let Some(object) = fields.as_object_mut() {
        object.remove("mode");
    }
    let parsed: Request = serde_json::from_value(fields)
        .map_err(|why| failure(StatusCode::BAD_REQUEST, &why.to_string()))?;
    let out = tokio::task::spawn_blocking(move || compute(mode, &request, parsed))
        .await
        .map_err(|_| failure(StatusCode::INTERNAL_SERVER_ERROR, "math worker failed"))?;
    Ok(Json(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(body: Value) -> Value {
        let mode = mode_of(&body).unwrap();
        let mut fields = body.clone();
        fields.as_object_mut().unwrap().remove("mode");
        compute(mode, &body, serde_json::from_value(fields).unwrap())
    }

    fn refusal(body: Value) -> (String, String) {
        let out = run(body);
        (
            out["status"].as_str().unwrap().to_owned(),
            out["refusal"]["code"].as_str().unwrap().to_owned(),
        )
    }

    fn checks(out: &Value) -> Vec<&str> {
        out["evidence"]["checks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["kind"].as_str().unwrap())
            .collect()
    }

    #[test]
    fn every_answer_has_the_envelope() {
        for body in [
            json!({"op":"eval","expr":"2 + 2"}),
            json!({"op":"eval","expr":"sin(1)"}),
            json!({"op":"derive","expr":"x^2","at":3}),
            json!({"op":"derive","expr":"x^2"}),
            json!({"op":"antiderivative","expr":"x^3 - 2*x"}),
            json!({"op":"integrate","expr":"x^2","from":0,"to":1}),
            json!({"op":"root","expr":"x^2 - 2","from":0,"to":2}),
            json!({"op":"root","expr":"x^2 - 2","guess":1}),
            json!({"op":"det","a":[[1,2],[3,4]]}),
            json!({"op":"inv","a":[[2,0],[0,4]]}),
            json!({"op":"matmul","a":[[1,2]],"b":[[3],[4]]}),
            json!({"op":"dot","u":[1,2],"v":[3,4]}),
            json!({"op":"cross","u":[1,0,0],"v":[0,1,0]}),
            json!({"op":"ode","expr":"-2 * y","t0":0,"y0":1,"t1":1,"steps":100}),
        ] {
            let out = run(body.clone());
            assert_eq!(out["schema"], SCHEMA, "{body}");
            assert_eq!(out["status"], "success", "{body}: {out}");
            assert_eq!(out["request"], body);
            assert!(out["nature"].is_string(), "{body}");
            assert!(out["content"]["value"]["kind"].is_string(), "{body}");
            assert!(out["evidence"]["method"].is_string(), "{body}");
            assert!(out["evidence"]["checks"].is_array(), "{body}");
            assert!(out["provenance"]["engine_version"].is_string(), "{body}");
            assert!(out["usage"]["elapsed_us"].is_u64(), "{body}");
            assert!(
                out["steps"].is_array() && out["warnings"].is_array(),
                "{body}"
            );
            assert!(out.get("refusal").is_none(), "{body}");
        }
    }

    /// A conta é a mesma de /math: o envelope só muda a forma.
    #[test]
    fn values_match_the_old_route() {
        for (body, read) in [
            (
                json!({"op":"eval","expr":"(12 + 8) * 3"}),
                "/content/value/value",
            ),
            (json!({"op":"derive","expr":"x^2"}), "/content/value/text"),
            (
                json!({"op":"antiderivative","expr":"x^3"}),
                "/content/value/text",
            ),
            (
                json!({"op":"integrate","expr":"x^2","from":0,"to":1}),
                "/content/value/value",
            ),
            (
                json!({"op":"root","expr":"x^2 - 2","from":0,"to":2}),
                "/content/value/value",
            ),
            (
                json!({"op":"det","a":[[1,2],[3,4]]}),
                "/content/value/value",
            ),
            (json!({"op":"inv","a":[[2,0],[0,4]]}), "/content/value/rows"),
            (
                json!({"op":"cross","u":[1,0,0],"v":[0,1,0]}),
                "/content/value/items",
            ),
        ] {
            let old = solve(serde_json::from_value(body.clone()).unwrap()).unwrap();
            assert_eq!(
                run(body.clone()).pointer(read),
                Some(&old["result"]),
                "{body}"
            );
        }
    }

    #[test]
    fn nature_does_not_promote_an_approximation() {
        assert_eq!(
            run(json!({"op":"eval","expr":"(12 + 8) * 3"}))["nature"],
            "exact"
        );
        for text in ["0.1 + 0.2", "1 / 3", "sqrt(4)", "pi", "2^-1"] {
            let out = run(json!({"op":"eval","expr":text}));
            assert_eq!(out["nature"], "numeric", "{text}");
            assert_eq!(out["warnings"][0]["code"], "rounding", "{text}");
            assert!(checks(&out).is_empty(), "{text}");
        }
        assert_eq!(
            run(json!({"op":"derive","expr":"x^2"}))["nature"],
            "symbolic"
        );
        assert_eq!(
            run(json!({"op":"det","a":[[1,0],[0,1]]}))["nature"],
            "numeric"
        );
    }

    #[test]
    fn exact_mode_refuses_instead_of_approximating() {
        let ok = run(json!({"op":"eval","expr":"2^10","mode":"exact"}));
        assert_eq!(ok["nature"], "exact");
        assert_eq!(ok["content"]["value"]["value"], 1024.0);
        for body in [
            json!({"op":"eval","expr":"0.1 + 0.2","mode":"exact"}),
            json!({"op":"eval","expr":"sqrt(2)","mode":"exact"}),
            json!({"op":"integrate","expr":"x","from":0,"to":1,"mode":"exact"}),
            json!({"op":"derive","expr":"x^2","mode":"exact"}),
        ] {
            assert_eq!(
                refusal(body.clone()),
                (
                    "unsupported".to_owned(),
                    "exact_mode_unavailable".to_owned()
                ),
                "{body}"
            );
        }
        // Pedido errado continua errado, em qualquer modo.
        assert_eq!(
            refusal(json!({"op":"eval","expr":"1/0","mode":"exact"})),
            ("invalid_input".to_owned(), "division_by_zero".to_owned())
        );
    }

    #[test]
    fn refusals_are_results_with_a_code() {
        let cases = [
            (
                json!({"op":"eval","expr":"1/0"}),
                "invalid_input",
                "division_by_zero",
            ),
            (
                json!({"op":"eval","expr":"ln(-1)"}),
                "invalid_input",
                "domain_error",
            ),
            (
                json!({"op":"eval","expr":"2 +"}),
                "invalid_input",
                "parse_error",
            ),
            (
                json!({"op":"eval","expr":"x + 1"}),
                "data_unavailable",
                "missing_variable",
            ),
            (
                json!({"op":"eval","expr":"9^9^9"}),
                "inconclusive",
                "not_representable",
            ),
            (
                json!({"op":"derive","expr":"ln(x)","at":-1}),
                "invalid_input",
                "domain_error",
            ),
            (
                json!({"op":"antiderivative","expr":"exp(-x^2)"}),
                "unsupported",
                "no_closed_form",
            ),
            (
                json!({"op":"root","expr":"1/(x-0.1)","from":0,"to":1}),
                "inconclusive",
                "pole_or_jump",
            ),
            (
                json!({"op":"root","expr":"x + 5","from":0,"to":1}),
                "inconclusive",
                "no_sign_change",
            ),
            (
                json!({"op":"root","expr":"1/x","guess":1}),
                "inconclusive",
                "not_converged",
            ),
            (
                json!({"op":"root","expr":"x"}),
                "invalid_input",
                "invalid_argument",
            ),
            (
                json!({"op":"integrate","expr":"1/x","from":0,"to":1}),
                "invalid_input",
                "division_by_zero",
            ),
            (
                json!({"op":"integrate","expr":"x","from":0,"to":1,"tol":0}),
                "invalid_input",
                "invalid_argument",
            ),
            (
                json!({"op":"det","a":[[1,2]]}),
                "invalid_input",
                "invalid_argument",
            ),
            (
                json!({"op":"det","a":[[1,2],[2,4]]}),
                "inconclusive",
                "near_singular_matrix",
            ),
            (
                json!({"op":"inv","a":[[1,2],[2,4]]}),
                "invalid_input",
                "singular_matrix",
            ),
            (
                json!({"op":"ode","expr":"y","t0":0,"y0":1,"t1":1,"steps":100001}),
                "resource_limit",
                "limit_size",
            ),
        ];
        for (body, status, code) in cases {
            let out = run(body.clone());
            assert_eq!(out["status"], status, "{body}: {out}");
            assert_eq!(out["refusal"]["code"], code, "{body}: {out}");
            assert!(!out["refusal"]["message"].as_str().unwrap().is_empty());
            assert!(
                out.get("content").is_none(),
                "nenhum número numa recusa: {body}"
            );
            assert!(out.get("nature").is_none(), "{body}");
        }
    }

    #[test]
    fn evidence_says_what_was_done() {
        let out = run(json!({"op":"root","expr":"x^2 - 2","from":0,"to":2}));
        assert_eq!(checks(&out), ["sign_change_and_decay"]);
        assert_eq!(out["evidence"]["error_observed"]["kind"], "residual");
        assert!(out["evidence"]["error_observed"]["value"].as_f64().unwrap() < 1e-9);
        let out = run(json!({"op":"root","expr":"x^2 - 2","guess":1}));
        assert_eq!(checks(&out), ["numeric_residual"]);
        let out = run(json!({"op":"antiderivative","expr":"x"}));
        assert_eq!(checks(&out), ["substitution"]);
        let out = run(json!({"op":"inv","a":[[4,7],[2,6]]}));
        assert_eq!(checks(&out), ["numeric_residual"]);
        assert!(out["evidence"]["error_observed"]["value"].as_f64().unwrap() < 1e-12);
        let out = run(json!({"op":"integrate","expr":"x","from":0,"to":1,"tol":1e-6}));
        assert_eq!(out["evidence"]["tolerance_requested"], 1e-6);
        assert!(
            checks(&out).is_empty(),
            "Simpson não confere nada por conta própria"
        );
        assert_eq!(out["warnings"][0]["code"], "tolerance_not_guaranteed");
        let out = run(json!({"op":"derive","expr":"x^2","at":3}));
        assert_eq!(
            out["content"]["value_at"],
            json!({"value": 6.0, "nature": "exact"})
        );
        assert_eq!(checks(&out), ["domain_check"]);
        assert_eq!(out["content"]["parameters"], json!({"at": 3, "var": "x"}));
        let out = run(json!({"op":"derive","expr":"sin(x)","at":3}));
        assert_eq!(out["content"]["value_at"]["nature"], "numeric");
        let out = run(json!({"op":"antiderivative","expr":"x^3 - 2 * x"}));
        assert_eq!(out["steps"][0]["rule"], "term_by_term");
        assert_eq!(out["steps"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn malformed_requests_are_the_only_http_errors() {
        let call = |body: Value| handle(Ok(Json(body)));
        for bad in [
            json!({"op":"apagar","expr":"1"}),
            json!({"op":"eval","expr":"1","extra":1}),
            json!({"op":"eval"}),
            json!({"op":"eval","expr":"1","mode":"magico"}),
            json!({"op":"eval","expr":"1","mode":1}),
            json!([1, 2]),
        ] {
            let got = call(bad.clone()).await;
            assert_eq!(
                got.err().map(|e| e.0),
                Some(StatusCode::BAD_REQUEST),
                "{bad}"
            );
        }
        let Json(out) = call(json!({"op":"eval","expr":"1/0"})).await.unwrap();
        assert_eq!(out["status"], "invalid_input");
    }
}
