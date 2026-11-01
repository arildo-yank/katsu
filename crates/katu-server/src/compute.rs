//! Rota `POST /v1/compute`: as mesmas contas de `/math`, no envelope `katu.result/1` de
//! `docs/KATU_CONTRATO_RESULTADOS.md`. Recusa é resultado e volta com HTTP 200; 400 fica para pedido
//! malformado. Em `exact` e `auto`, uma conta de `eval` tenta os racionais de `katu_math::exact`; o resto é
//! de `math::solve`, e esta camada só descreve o que aconteceu. Toda conta tem prazo e vaga limitada.
use crate::math::{add_latex, failure, solve, tex_rational, Request};
use axum::{extract::rejection::JsonRejection, http::StatusCode, Json};
use katu_math::budget::with_budget;
use katu_math::linalg::{multiply, Matrix};
use katu_math::{eval, exact_eval_with, parse, ErrorCode, MathError, Rational, Vars};
use serde_json::{json, Map, Value};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{Semaphore, SemaphorePermit};

const SCHEMA: &str = "katu.result/1";
/// Prazo de uma conta. As partes sem conferência são limitadas pelos tamanhos máximos e levam até ~100 ms.
pub(crate) const TIME_LIMIT: Duration = Duration::from_secs(2);
/// Contas simultâneas: o pool de threads de bloqueio aceitaria centenas, e cada uma pode usar um núcleo.
pub(crate) static SLOTS: Semaphore = Semaphore::const_new(4);
const PREVIEW_PLACES: u32 = 30;
/// Literal mais longo que isto perderia dígitos no LaTeX, que passa por f64.
const LATEX_MAX_DIGITS: usize = 15;

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

pub(crate) fn base(request: &Value, status: &str, elapsed_us: u64) -> Map<String, Value> {
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
    if let Some(limit) = code.limit() {
        out["usage"]["limit_hit"] = json!(limit);
    }
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

/// Pedido já validado. Valor de variável que não é número fica para virar recusa, não erro de transporte.
struct Prepared {
    mode: Mode,
    parsed: Request,
    bad_variable: Option<MathError>,
}

/// `vars` aceita texto, que preserva o valor exato: "1/3", "0.1". Para a conta em f64, vira o f64 mais próximo.
fn prepare(request: &Value) -> Result<Prepared, String> {
    let mode = mode_of(request)?;
    let mut fields = request.clone();
    let mut bad_variable = None;
    if let Some(object) = fields.as_object_mut() {
        object.remove("mode");
        if let Some(vars) = object.get_mut("vars").and_then(Value::as_object_mut) {
            for value in vars.values_mut() {
                let Some(text) = value.as_str() else { continue };
                match Rational::parse(text).and_then(|r| r.to_f64()) {
                    Ok(number) => *value = json!(number),
                    Err(why) => {
                        bad_variable = Some(why);
                        *value = json!(0.0);
                    }
                }
            }
        }
    }
    let parsed = serde_json::from_value(fields).map_err(|why| why.to_string())?;
    Ok(Prepared {
        mode,
        parsed,
        bad_variable,
    })
}

/// Valores exatos das variáveis. Inteiro em JSON é exato; decimal em JSON já chegou binário e não pode
/// passar por exato: `None` manda para o caminho numérico, ou para a recusa em `exact`.
fn exact_vars(request: &Value) -> Result<Option<HashMap<String, Rational>>, MathError> {
    let mut out = HashMap::new();
    let Some(vars) = request.get("vars").and_then(Value::as_object) else {
        return Ok(Some(out));
    };
    for (name, value) in vars {
        let exact = if let Some(text) = value.as_str() {
            Rational::parse(text)?
        } else if let Some(whole) = value.as_i64() {
            Rational::from_int(whole.into())
        } else {
            return Ok(None);
        };
        out.insert(name.clone(), exact);
    }
    Ok(Some(out))
}

fn has_long_literal(text: &str) -> bool {
    text.split(|c: char| !c.is_ascii_digit())
        .any(|digits| digits.len() > LATEX_MAX_DIGITS)
}

fn describe_exact(request: &Value, text: &str, value: &Rational, elapsed_us: u64) -> Value {
    let shown = if value.is_integer() {
        json!({"kind": "integer", "text": value.numerator().to_string()})
    } else {
        json!({
            "kind": "rational",
            "num": value.numerator().to_string(),
            "den": value.denominator().to_string(),
        })
    };
    let mut content = Map::new();
    content.insert("interpreted".into(), json!(text.trim()));
    if !has_long_literal(text) {
        if let Ok(expr) = parse(text) {
            content.insert("interpreted_latex".into(), json!(katu_math::latex(&expr)));
        }
    }
    content.insert("value".into(), shown);
    content.insert("value_latex".into(), json!(tex_rational(value)));
    if let Ok((preview, complete)) = value.decimal_preview(PREVIEW_PLACES) {
        content.insert(
            "decimal_preview".into(),
            json!({"text": preview, "exact": complete}),
        );
    }
    let params = parameters("eval", request);
    if !params.is_empty() {
        content.insert("parameters".into(), Value::Object(params));
    }
    let mut out = base(request, "success", elapsed_us);
    out.insert("nature".into(), json!("exact"));
    out.insert("content".into(), Value::Object(content));
    out.insert(
        "evidence".into(),
        json!({
            "method": "Aritmética racional exata",
            "checks": [{"kind": "exact_arithmetic", "detail": "racionais de tamanho arbitrário, sem arredondamento"}],
        }),
    );
    Value::Object(out)
}

/// A aproximação pode entrar no lugar da conta exata só nestes casos, e sempre com aviso. Erro de verdade,
/// como 1/0 ou 0^0, não vira número só porque f64 teria uma resposta.
fn may_fall_back(code: ErrorCode) -> bool {
    matches!(
        code,
        ErrorCode::ExactModeUnavailable | ErrorCode::LimitSize | ErrorCode::LimitTime
    )
}

fn compute(prepared: Prepared, request: &Value) -> Value {
    let started = Instant::now();
    let elapsed = || started.elapsed().as_micros() as u64;
    let Prepared {
        mode,
        parsed,
        bad_variable,
    } = prepared;
    if let Some(why) = bad_variable {
        return refused(request, why.code, &why.message, elapsed());
    }
    let mut fallback = None;
    if mode != Mode::Numeric && matches!(parsed, Request::Eval { .. }) {
        let text = request["expr"].as_str().unwrap_or("");
        let attempt = exact_vars(request).and_then(|vars| match vars {
            Some(vars) => exact_eval_with(text, &vars),
            None => Err(MathError::new(
                ErrorCode::ExactModeUnavailable,
                "variável decimal enviada como número já chegou em binário; envie como texto, por exemplo \"0.1\"",
            )),
        });
        match attempt {
            Ok(value) => return describe_exact(request, text, &value, elapsed()),
            Err(why) if mode == Mode::Auto && may_fall_back(why.code) => fallback = Some(why),
            Err(why) => return refused(request, why.code, &why.message, elapsed()),
        }
    }
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
    let mut answer = describe(request, &solved, elapsed());
    if mode == Mode::Exact && answer["nature"] != "exact" {
        let message = "esta conta ainda não tem resultado exato; peça mode auto ou numeric para a aproximação";
        return refused(request, ErrorCode::ExactModeUnavailable, message, elapsed());
    }
    if let Some(why) = fallback {
        if let Some(warnings) = answer["warnings"].as_array_mut() {
            warnings.insert(0, warning("exact_unavailable", &json!(why.message)));
        }
    }
    answer
}

/// Marca a conta como cancelada se o pedido sumir antes do fim, como quando o cliente desiste.
pub(crate) struct CancelOnDrop(pub(crate) Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

pub(crate) fn take_slot(slots: &'static Semaphore) -> Option<SemaphorePermit<'static>> {
    slots.try_acquire().ok()
}

pub(super) async fn handle(
    body: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Json(request) = body.map_err(|why| failure(StatusCode::BAD_REQUEST, &why.body_text()))?;
    let prepared = prepare(&request).map_err(|why| failure(StatusCode::BAD_REQUEST, &why))?;
    let Some(slot) = take_slot(&SLOTS) else {
        let message = "muitas contas ao mesmo tempo; tente de novo em instantes";
        return Ok(Json(refused(
            &request,
            ErrorCode::LimitConcurrency,
            message,
            0,
        )));
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let _cancel_on_drop = CancelOnDrop(cancel.clone());
    // A vaga vai junto para a thread: só volta quando a conta termina de verdade.
    let out = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        with_budget(TIME_LIMIT, Some(cancel), || compute(prepared, &request))
    })
    .await
    .map_err(|_| failure(StatusCode::INTERNAL_SERVER_ERROR, "math worker failed"))?;
    Ok(Json(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(body: Value) -> Value {
        compute(prepare(&body).unwrap(), &body)
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
            (json!({"op":"eval","expr":"sin(1)"}), "/content/value/value"),
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
            let out = run(json!({"op":"eval","expr":text,"mode":"numeric"}));
            assert_eq!(out["nature"], "numeric", "{text}");
            assert_eq!(out["warnings"][0]["code"], "rounding", "{text}");
            assert!(checks(&out).is_empty(), "{text}");
        }
        // Em auto, a aproximação só entra com o motivo à vista.
        for text in ["sqrt(4)", "pi", "sin(1)", "2^0.5"] {
            let out = run(json!({"op":"eval","expr":text}));
            assert_eq!(out["nature"], "numeric", "{text}");
            assert_eq!(out["warnings"][0]["code"], "exact_unavailable", "{text}");
            assert_eq!(out["warnings"][1]["code"], "rounding", "{text}");
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
        assert_eq!(
            ok["content"]["value"],
            json!({"kind":"integer","text":"1024"})
        );
        for body in [
            json!({"op":"eval","expr":"2^0.5","mode":"exact"}),
            json!({"op":"eval","expr":"x + 1","vars":{"x":0.1},"mode":"exact"}),
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

    /// Gabaritos da seção 8 do contrato, pela rota.
    #[test]
    fn exact_values_match_the_contract_table() {
        for (text, kind, value) in [
            ("0.1 + 0.2", "rational", json!({"num":"3","den":"10"})),
            ("1/3 + 1/6", "rational", json!({"num":"1","den":"2"})),
            ("2^-3", "rational", json!({"num":"1","den":"8"})),
            ("-1/3", "rational", json!({"num":"-1","den":"3"})),
            ("(1/3) * 3", "integer", json!({"text":"1"})),
            (
                "2^100",
                "integer",
                json!({"text":"1267650600228229401496703205376"}),
            ),
        ] {
            for mode in ["exact", "auto"] {
                let out = run(json!({"op":"eval","expr":text,"mode":mode}));
                assert_eq!(out["nature"], "exact", "{text}");
                assert_eq!(out["content"]["value"]["kind"], kind, "{text}");
                for (key, want) in value.as_object().unwrap() {
                    assert_eq!(&out["content"]["value"][key], want, "{text}");
                }
                assert_eq!(checks(&out), ["exact_arithmetic"], "{text}");
                assert!(out["warnings"].as_array().unwrap().is_empty(), "{text}");
            }
        }
        for (text, status, code) in [
            ("1/0", "invalid_input", "division_by_zero"),
            ("0^0", "invalid_input", "domain_error"),
            ("sqrt(2)", "unsupported", "exact_mode_unavailable"),
            ("2^(10^9)", "resource_limit", "limit_size"),
            ("x + 1", "data_unavailable", "missing_variable"),
        ] {
            let got = refusal(json!({"op":"eval","expr":text,"mode":"exact"}));
            assert_eq!(got, (status.to_owned(), code.to_owned()), "{text}");
        }
    }

    #[test]
    fn exact_answers_show_a_decimal_that_says_if_it_is_complete() {
        let third = run(json!({"op":"eval","expr":"1/3"}));
        assert_eq!(third["content"]["value_latex"], "\\frac{1}{3}");
        assert_eq!(third["content"]["decimal_preview"]["exact"], false);
        assert!(third["content"]["decimal_preview"]["text"]
            .as_str()
            .unwrap()
            .starts_with("0.3333"));
        let half = run(json!({"op":"eval","expr":"-1/2"}));
        assert_eq!(half["content"]["value_latex"], "-\\frac{1}{2}");
        assert_eq!(
            half["content"]["decimal_preview"],
            json!({"text":"-0.5","exact":true})
        );
        // Literal longo não passa pelo LaTeX, que o arredondaria em f64.
        let long = run(json!({"op":"eval","expr":"1.000000000000000001 + 1"}));
        assert!(long["content"].get("interpreted_latex").is_none());
        assert_eq!(long["content"]["value"]["num"], "2000000000000000001");
    }

    #[test]
    fn auto_does_not_turn_a_real_error_into_a_number() {
        // Em f64, 0^0 daria 1 e o texto de auto poderia cair nele.
        assert_eq!(
            refusal(json!({"op":"eval","expr":"0^0"})),
            ("invalid_input".to_owned(), "domain_error".to_owned())
        );
        // Grande demais para o exato: tenta f64, e o aviso diz por quê.
        let big = run(json!({"op":"eval","expr":"1.0000001^10000000"}));
        assert_eq!(big["nature"], "numeric", "{big}");
        assert_eq!(big["warnings"][0]["code"], "exact_unavailable");
        assert_eq!(
            run(json!({"op":"eval","expr":"9^9^9"}))["usage"].get("limit_hit"),
            None
        );
        let exact = run(json!({"op":"eval","expr":"9^9^9","mode":"exact"}));
        assert_eq!(exact["usage"]["limit_hit"], "size");
    }

    #[test]
    fn variables_decide_whether_the_value_can_be_exact() {
        let body =
            |vars: Value, mode: &str| json!({"op":"eval","expr":"x + 1","vars":vars,"mode":mode});
        let text = run(body(json!({"x":"1/3"}), "exact"));
        assert_eq!(text["content"]["value"]["num"], "4");
        let whole = run(body(json!({"x":2}), "exact"));
        assert_eq!(whole["content"]["value"]["text"], "3");
        let binary = run(body(json!({"x":0.1}), "auto"));
        assert_eq!(binary["nature"], "numeric");
        assert_eq!(binary["warnings"][0]["code"], "exact_unavailable");
        let numeric = run(body(json!({"x":"0.5"}), "numeric"));
        assert_eq!(numeric["content"]["value"]["value"], 1.5);
        assert_eq!(
            numeric["request"]["vars"]["x"], "0.5",
            "o pedido fica como veio"
        );
        let invalid = run(body(json!({"x":"abc"}), "auto"));
        assert_eq!(invalid["status"], "invalid_input");
        assert_eq!(invalid["refusal"]["code"], "parse_error");
    }

    #[test]
    fn the_time_budget_becomes_a_refusal() {
        let heavy =
            json!({"op":"eval","expr":"3^165000 / 7^93000 + 5^112000 / 11^75000","mode":"exact"});
        let out = with_budget(Duration::ZERO, None, || run(heavy.clone()));
        assert_eq!(out["status"], "resource_limit", "{out}");
        assert_eq!(out["refusal"]["code"], "limit_time");
        assert_eq!(out["usage"]["limit_hit"], "time");
        let hard =
            json!({"op":"integrate","expr":"sin(1 / x) / x^3","from":1e-6,"to":1,"tol":1e-12});
        let out = with_budget(Duration::ZERO, None, || run(hard.clone()));
        assert_eq!(out["refusal"]["code"], "limit_time", "{out}");
        let flag = Arc::new(AtomicBool::new(true));
        let out = with_budget(Duration::from_secs(60), Some(flag), || run(heavy.clone()));
        assert_eq!(out["refusal"]["code"], "cancelled", "{out}");
        assert!(out["usage"].get("limit_hit").is_none());
    }

    #[test]
    fn only_a_fixed_number_of_slots_exist() {
        let slots: &'static Semaphore = Box::leak(Box::new(Semaphore::new(2)));
        let first = take_slot(slots).unwrap();
        let second = take_slot(slots).unwrap();
        assert!(take_slot(slots).is_none());
        drop(first);
        assert!(take_slot(slots).is_some());
        drop(second);
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
