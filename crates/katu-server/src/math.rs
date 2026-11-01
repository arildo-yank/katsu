//! Rota `POST /math`: entrada matemática direta para o `katu-math`, sem o modelo. A resposta diz o método
//! e se o resultado é aproximado; o que o motor recusa volta como erro 400, nunca como número.
use axum::{extract::rejection::JsonRejection, http::StatusCode, Json};
use katu_math::linalg::{self, Matrix};
use katu_math::{
    antiderivative, bisect, derivative_at, diff, eval, exact_value, integrate, latex, newton,
    parse, rk4, understand, Expr, Intent, Vars,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::HashMap, time::Instant};

const MAX_ODE_STEPS: usize = 100_000;
const DEFAULT_TOL: f64 = 1e-9;
const MIN_TOL: f64 = 1e-12;
const ROUNDING: &str =
    "Calculado em ponto flutuante; as últimas casas podem ter erro de arredondamento.";

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase", deny_unknown_fields)]
pub(super) enum Request {
    Eval {
        expr: String,
        #[serde(default)]
        vars: HashMap<String, f64>,
    },
    Derive {
        expr: String,
        #[serde(default = "default_var")]
        var: String,
        at: Option<f64>,
    },
    Integrate {
        expr: String,
        #[serde(default = "default_var")]
        var: String,
        from: f64,
        to: f64,
        #[serde(default = "default_tol")]
        tol: f64,
    },
    Antiderivative {
        expr: String,
        #[serde(default = "default_var")]
        var: String,
    },
    Root {
        expr: String,
        #[serde(default = "default_var")]
        var: String,
        from: Option<f64>,
        to: Option<f64>,
        guess: Option<f64>,
    },
    Det {
        a: Matrix,
    },
    Inv {
        a: Matrix,
    },
    Matmul {
        a: Matrix,
        b: Matrix,
    },
    Dot {
        u: Vec<f64>,
        v: Vec<f64>,
    },
    Cross {
        u: Vec<f64>,
        v: Vec<f64>,
    },
    Ode {
        expr: String,
        t0: f64,
        y0: f64,
        t1: f64,
        steps: usize,
    },
}

fn default_var() -> String {
    "x".into()
}
fn default_tol() -> f64 {
    DEFAULT_TOL
}

fn expression(text: &str) -> Result<Expr, String> {
    parse(text)
}

fn answer(
    input: &Expr,
    op: &str,
    result: Value,
    method: &str,
    approximate: bool,
    note: String,
) -> Value {
    json!({"op": op, "input": input.to_string(), "result": result, "method": method, "approximate": approximate, "note": note})
}

/// Termos de uma soma, cada um com o sinal: é o que vira passo a passo.
fn terms(e: &Expr) -> Vec<(bool, &Expr)> {
    fn walk<'a>(e: &'a Expr, minus: bool, out: &mut Vec<(bool, &'a Expr)>) {
        match e {
            Expr::Add(a, c) => {
                walk(a, minus, out);
                walk(c, minus, out);
            }
            Expr::Sub(a, c) => {
                walk(a, minus, out);
                walk(c, !minus, out);
            }
            Expr::Neg(a) => walk(a, !minus, out),
            term => out.push((minus, term)),
        }
    }
    let mut out = Vec::new();
    walk(e, false, &mut out);
    out
}

fn group_latex(e: &Expr, minus: bool) -> String {
    if minus && matches!(e, Expr::Add(..) | Expr::Sub(..)) {
        format!("\\left({}\\right)", latex(e))
    } else {
        latex(e)
    }
}

/// Parênteses em volta de soma, para o símbolo da integral alcançar a expressão inteira.
fn wrapped(e: &Expr) -> String {
    if matches!(e, Expr::Add(..) | Expr::Sub(..)) {
        format!("\\left({}\\right)", latex(e))
    } else {
        latex(e)
    }
}

fn number_latex(value: f64) -> String {
    let rounded = format!("{value:.10}");
    let trimmed = rounded.trim_end_matches('0').trim_end_matches('.');
    if value != 0.0 && (value.abs() >= 1e15 || value.abs() < 1e-6) {
        latex(&Expr::Num(value))
    } else {
        trimmed.replace('.', "{,}")
    }
}

fn tex_vector(v: &[f64]) -> String {
    let rows: Vec<String> = v.iter().map(|x| number_latex(*x)).collect();
    format!(
        "\\begin{{bmatrix}} {} \\end{{bmatrix}}",
        rows.join(" \\\\ ")
    )
}

fn tex_matrix(m: &Matrix) -> String {
    let rows: Vec<String> = m
        .iter()
        .map(|row| {
            row.iter()
                .map(|x| number_latex(*x))
                .collect::<Vec<_>>()
                .join(" & ")
        })
        .collect();
    format!(
        "\\begin{{bmatrix}} {} \\end{{bmatrix}}",
        rows.join(" \\\\ ")
    )
}

/// Resultado de álgebra linear: ponto flutuante, por isso marcado como aproximado.
fn linear(op: &str, input: String, result: Value, tex: String, method: &str) -> Value {
    json!({"op": op, "input": input.clone(), "input_latex": input, "result": result, "result_latex": tex, "method": method, "approximate": true, "note": ROUNDING})
}

fn solve(req: Request) -> Result<Value, String> {
    let none = Vars::new();
    match req {
        Request::Det { a } => {
            let value = linalg::det(&a)?;
            Ok(linear(
                "det",
                format!("\\det {}", tex_matrix(&a)),
                json!(value),
                number_latex(value),
                "Eliminação de Gauss",
            ))
        }
        Request::Inv { a } => {
            let out = linalg::inverse(&a)?;
            Ok(linear(
                "inv",
                format!("{}^{{-1}}", tex_matrix(&a)),
                json!(out),
                tex_matrix(&out),
                "Gauss-Jordan",
            ))
        }
        Request::Matmul { a, b } => {
            let out = linalg::multiply(&a, &b)?;
            Ok(linear(
                "matmul",
                format!("{} {}", tex_matrix(&a), tex_matrix(&b)),
                json!(out),
                tex_matrix(&out),
                "Produto de matrizes",
            ))
        }
        Request::Dot { u, v } => {
            let value = linalg::dot(&u, &v)?;
            Ok(linear(
                "dot",
                format!("{} \\cdot {}", tex_vector(&u), tex_vector(&v)),
                json!(value),
                number_latex(value),
                "Produto escalar",
            ))
        }
        Request::Cross { u, v } => {
            let out = linalg::cross(&u, &v)?;
            Ok(linear(
                "cross",
                format!("{} \\times {}", tex_vector(&u), tex_vector(&v)),
                json!(out),
                tex_vector(&out),
                "Produto vetorial",
            ))
        }
        Request::Eval { expr, vars } => {
            let e = expression(&expr)?;
            let vars: Vars = vars.into_iter().collect();
            let value = eval(&e, &vars)?;
            let exact = exact_value(&e, &vars);
            Ok(answer(
                &e,
                "eval",
                json!(value),
                "Avaliação direta",
                !exact,
                if exact {
                    String::new()
                } else {
                    ROUNDING.to_owned()
                },
            ))
        }
        Request::Derive { expr, var, at } => {
            let e = expression(&expr)?;
            let d = diff(&e, &var)?;
            let value = at.map(|x| derivative_at(&e, &var, x, &none)).transpose()?;
            let mut out = answer(
                &e,
                "derive",
                json!(d.to_string()),
                "Derivada simbólica",
                false,
                format!("Em relação a {var}."),
            );
            out["value_at"] = json!(value);
            // `approximate` fala da derivada simbólica; o valor no ponto é outra conta.
            if let Some(x) = at {
                let env = Vars::from([(var.clone(), x)]);
                out["value_at_approximate"] = json!(!exact_value(&d, &env));
            }
            let mut steps = Vec::new();
            for (minus, term) in terms(&e) {
                let sign = if minus { "-" } else { "" };
                steps.push(format!(
                    "{sign}\\frac{{d}}{{d{var}}}\\left({}\\right) = {sign}{}",
                    latex(term),
                    group_latex(&diff(term, &var)?, minus)
                ));
            }
            out["steps"] = json!(steps);
            Ok(out)
        }
        Request::Integrate {
            expr,
            var,
            from,
            to,
            tol,
        } => {
            if !(MIN_TOL..=1.0).contains(&tol) {
                return Err(format!("a tolerância precisa estar entre {MIN_TOL} e 1"));
            }
            let e = expression(&expr)?;
            let value = integrate(&e, &var, from, to, tol, &none)?;
            Ok(answer(
                &e,
                "integrate",
                json!(value),
                "Simpson adaptativo",
                true,
                format!("Aproximação numérica, tolerância pedida de {tol:e}."),
            ))
        }
        Request::Antiderivative { expr, var } => {
            let e = expression(&expr)?;
            let p = antiderivative(&e, &var)?;
            let mut out = answer(
                &e,
                "antiderivative",
                json!(format!("{p} + C")),
                "Primitiva em forma fechada",
                false,
                "Conferida derivando o resultado em vários pontos.".into(),
            );
            let mut steps = Vec::new();
            for (minus, term) in terms(&e) {
                let part = antiderivative(term, &var)?;
                let sign = if minus { "-" } else { "" };
                steps.push(format!(
                    "{sign}\\int {}\\,d{var} = {sign}{}",
                    wrapped(term),
                    group_latex(&part, minus)
                ));
            }
            out["steps"] = json!(steps);
            out["input_latex"] = json!(format!("\\int {}\\,d{var}", wrapped(&e)));
            out["result_latex"] = json!(format!("{} + C", latex(&p)));
            Ok(out)
        }
        Request::Root {
            expr,
            var,
            from,
            to,
            guess,
        } => {
            let e = expression(&expr)?;
            match (from, to, guess) {
                (Some(a), Some(c), None) => {
                    let value = bisect(&e, &var, a, c, 1e-12, &none)?;
                    Ok(answer(
                        &e,
                        "root",
                        json!(value),
                        "Bisseção",
                        true,
                        "Uma raiz do intervalo; pode haver outras.".into(),
                    ))
                }
                (None, None, Some(start)) => {
                    let value = newton(&e, &var, start, 1e-12, &none)?;
                    Ok(answer(
                        &e,
                        "root",
                        json!(value),
                        "Newton-Raphson",
                        true,
                        "A raiz encontrada depende do ponto de partida.".into(),
                    ))
                }
                _ => Err("informe um intervalo (from e to) ou um ponto de partida (guess)".into()),
            }
        }
        Request::Ode {
            expr,
            t0,
            y0,
            t1,
            steps,
        } => {
            if steps == 0 || steps > MAX_ODE_STEPS {
                return Err(format!(
                    "o número de passos precisa estar entre 1 e {MAX_ODE_STEPS}"
                ));
            }
            let e = expression(&expr)?;
            let value = rk4(&e, "t", "y", t0, y0, t1, steps, &none)?;
            Ok(answer(
                &e,
                "ode",
                json!(value),
                "Runge-Kutta de 4.ª ordem",
                true,
                format!("y({t1}), com y' = f(t, y), em {steps} passos."),
            ))
        }
    }
}

/// Só para as operações sem LaTeX próprio: entrada e resultado do jeito que o motor calculou.
fn add_latex(out: &mut Value) {
    if out.get("result_latex").is_some() {
        return;
    }
    if let Some(e) = out["input"].as_str().and_then(|text| parse(text).ok()) {
        let tex = latex(&e);
        out["input_latex"] = json!(match out["op"].as_str() {
            Some("derive") => format!("\\frac{{d}}{{dx}}\\left({tex}\\right)"),
            _ => tex,
        });
    }
    out["result_latex"] = match &out["result"] {
        Value::Number(n) => json!(number_latex(n.as_f64().unwrap_or(0.0))),
        Value::String(text) => parse(text).map(|e| json!(latex(&e))).unwrap_or(Value::Null),
        _ => Value::Null,
    };
}

const FOOTER: &str = "Calculado pelo motor do Katu neste computador, sem usar um modelo de IA.";

fn latex_block(tex: &str) -> String {
    format!("$$\n{tex}\n$$")
}

/// Resposta do pedido escrito, em Markdown com LaTeX, sempre mostrando o que foi entendido.
fn explain(intent: Intent) -> Value {
    let (understood, result): (String, Result<(String, bool, String), String>) = match intent {
        Intent::Calculate(e) => (
            latex(&e),
            eval(&e, &Vars::new()).map(|v| {
                let exact = exact_value(&e, &Vars::new());
                let note = if exact { "" } else { ROUNDING };
                (number_latex(v), !exact, note.to_owned())
            }),
        ),
        Intent::Antiderivative { expr, var } => (
            format!("\\int {}\\,d{var}", wrapped(&expr)),
            antiderivative(&expr, &var).map(|p| {
                (
                    format!("{} + C", latex(&p)),
                    false,
                    "A primitiva foi conferida derivando o resultado.".to_owned(),
                )
            }),
        ),
        Intent::Definite {
            expr,
            var,
            from,
            to,
        } => (
            format!(
                "\\int_{{{}}}^{{{}}} {}\\,d{var}",
                latex(&Expr::Num(from)),
                latex(&Expr::Num(to)),
                wrapped(&expr)
            ),
            integrate(&expr, &var, from, to, DEFAULT_TOL, &Vars::new()).map(|v| {
                (
                    format!("\\approx {}", number_latex(v)),
                    true,
                    format!("Aproximação numérica, tolerância de {DEFAULT_TOL:e}."),
                )
            }),
        ),
        Intent::Derivative { expr, var, at } => (
            format!("\\frac{{d}}{{d{var}}}\\left({}\\right)", latex(&expr)),
            diff(&expr, &var).and_then(|d| {
                let base = latex(&d);
                match at {
                    None => Ok((base, false, String::new())),
                    Some(x) => derivative_at(&expr, &var, x, &Vars::new()).map(|v| {
                        (
                            format!(
                                "{base} \\quad\\text{{; em }} {var} = {} \\text{{: }} {}",
                                number_latex(x),
                                number_latex(v)
                            ),
                            false,
                            String::new(),
                        )
                    }),
                }
            }),
        ),
        Intent::Root { expr, var } => (
            format!("{} = 0", latex(&expr)),
            root_search(&expr, &var).map(|v| {
                (
                    format!("{var} \\approx {}", number_latex(v)),
                    true,
                    "Uma raiz encontrada numericamente; pode haver outras.".to_owned(),
                )
            }),
        ),
    };
    match result {
        Ok((tex, approximate, note)) => {
            let mut markdown = format!(
                "Entendi assim:\n\n{}\n\nResultado:\n\n{}\n\nEm LaTeX: `{tex}`\n\n",
                latex_block(&understood),
                latex_block(&tex)
            );
            if !note.is_empty() {
                markdown.push_str(&format!("{note} "));
            }
            markdown.push_str(FOOTER);
            json!({"understood": true, "ok": true, "markdown": markdown, "input_latex": understood, "result_latex": tex, "approximate": approximate})
        }
        Err(why) => {
            let markdown = format!(
                "Entendi assim:\n\n{}\n\nNão consegui calcular: {why}.\n\n{FOOTER}",
                latex_block(&understood)
            );
            json!({"understood": true, "ok": false, "markdown": markdown, "input_latex": understood, "error": why})
        }
    }
}

/// Procura uma troca de sinal em janelas crescentes em volta de zero e refina por bisseção.
fn root_search(e: &Expr, var: &str) -> Result<f64, String> {
    let none = Vars::new();
    let value = |x: f64| eval(e, &Vars::from([(var.to_owned(), x)]));
    let mut last: Option<(f64, f64)> = None;
    for i in 0..=2000 {
        let x = f64::from(i) * 0.05;
        for point in [x, -x] {
            if let Ok(v) = value(point) {
                if v == 0.0 {
                    return Ok(point);
                }
            }
        }
        if let (Ok(here), Some((px, pv))) = (value(x), last) {
            if here.signum() != pv.signum() {
                if let Ok(root) = bisect(e, var, px, x, 1e-12, &none) {
                    return Ok(root);
                }
            }
        }
        if let (Ok(a), Ok(c)) = (value(-x), value(-x - 0.05)) {
            if a.signum() != c.signum() {
                if let Ok(root) = bisect(e, var, -x - 0.05, -x, 1e-12, &none) {
                    return Ok(root);
                }
            }
        }
        last = value(x).ok().map(|v| (x, v));
    }
    Err("não encontrei raiz real entre -100 e 100".into())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Ask {
    text: String,
}

/// `POST /math/ask`: pedido escrito. `understood: false` manda o texto seguir como conversa comum.
pub(super) async fn ask(
    body: Result<Json<Ask>, JsonRejection>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Json(req) = body.map_err(|why| failure(StatusCode::BAD_REQUEST, &why.body_text()))?;
    let out = tokio::task::spawn_blocking(move || match understand(&req.text) {
        Some(intent) => explain(intent),
        None => json!({"understood": false}),
    })
    .await
    .map_err(|_| failure(StatusCode::INTERNAL_SERVER_ERROR, "math worker failed"))?;
    Ok(Json(out))
}

fn failure(code: StatusCode, message: &str) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"error": message})))
}

pub(super) async fn handle(
    body: Result<Json<Request>, JsonRejection>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Json(req) = body.map_err(|why| failure(StatusCode::BAD_REQUEST, &why.body_text()))?;
    let started = Instant::now();
    let solved = tokio::task::spawn_blocking(move || solve(req))
        .await
        .map_err(|_| failure(StatusCode::INTERNAL_SERVER_ERROR, "math worker failed"))?;
    let mut out = solved.map_err(|why| failure(StatusCode::BAD_REQUEST, &why))?;
    add_latex(&mut out);
    out["elapsed_us"] = json!(started.elapsed().as_micros() as u64);
    Ok(Json(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(body: Value) -> Result<Value, String> {
        solve(serde_json::from_value(body).map_err(|e| e.to_string())?)
    }

    #[test]
    fn solves_each_operation() {
        assert_eq!(
            run(json!({"op":"eval","expr":"(12 + 8) * 3"})).unwrap()["result"],
            60.0
        );
        assert_eq!(
            run(json!({"op":"eval","expr":"g * t","vars":{"g":10,"t":5}})).unwrap()["result"],
            50.0
        );
        let d = run(json!({"op":"derive","expr":"x^2","at":3})).unwrap();
        assert_eq!(d["result"], "2 * x");
        assert_eq!(d["value_at"], 6.0);
        assert_eq!(d["approximate"], false);
        let i = run(json!({"op":"integrate","expr":"x^2","from":0,"to":1})).unwrap();
        assert!((i["result"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-8);
        assert_eq!(i["approximate"], true);
        let r = run(json!({"op":"root","expr":"x^2 - 2","from":0,"to":2})).unwrap();
        assert!((r["result"].as_f64().unwrap() - 2f64.sqrt()).abs() < 1e-9);
        let n = run(json!({"op":"root","expr":"x^2 - 2","guess":1})).unwrap();
        assert_eq!(n["method"], "Newton-Raphson");
        let o = run(json!({"op":"ode","expr":"-2 * y","t0":0,"y0":1,"t1":1,"steps":100})).unwrap();
        assert!((o["result"].as_f64().unwrap() - (-2f64).exp()).abs() < 1e-7);
    }

    #[test]
    fn solves_linear_algebra_and_writes_latex() {
        let d = run(json!({"op":"det","a":[[1,2],[3,4]]})).unwrap();
        assert!((d["result"].as_f64().unwrap() + 2.0).abs() < 1e-9);
        let i = run(json!({"op":"inv","a":[[2,0],[0,4]]})).unwrap();
        assert_eq!(i["result"], json!([[0.5, 0.0], [0.0, 0.25]]));
        assert!(i["result_latex"]
            .as_str()
            .unwrap()
            .starts_with("\\begin{bmatrix}"));
        assert_eq!(
            run(json!({"op":"matmul","a":[[1,2]],"b":[[3],[4]]})).unwrap()["result"],
            json!([[11.0]])
        );
        assert_eq!(
            run(json!({"op":"dot","u":[1,2,3],"v":[4,5,6]})).unwrap()["result"],
            32.0
        );
        assert_eq!(
            run(json!({"op":"cross","u":[1,0,0],"v":[0,1,0]})).unwrap()["result"],
            json!([0.0, 0.0, 1.0])
        );
        assert!(run(json!({"op":"inv","a":[[1,2],[2,4]]})).is_err());
    }

    #[test]
    fn breaks_sums_into_steps() {
        let out = run(json!({"op":"antiderivative","expr":"x^3 - 2 * x"})).unwrap();
        assert_eq!(out["result_latex"], "\\frac{x^{4}}{4} - x^{2} + C");
        assert_eq!(out["steps"][0], "\\int x^{3}\\,dx = \\frac{x^{4}}{4}");
        assert_eq!(out["steps"][1], "-\\int 2 x\\,dx = -x^{2}");
        let d = run(json!({"op":"derive","expr":"x^2 + sin(x)"})).unwrap();
        assert_eq!(d["steps"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn answers_written_requests_in_latex() {
        let out = explain(understand("faz a integral de X3 -2x").unwrap());
        assert_eq!(out["ok"], true);
        assert_eq!(out["input_latex"], "\\int \\left(x^{3} - 2 x\\right)\\,dx");
        assert_eq!(out["result_latex"], "\\frac{x^{4}}{4} - x^{2} + C");
        assert!(out["markdown"].as_str().unwrap().contains("$$"));
        let hard = explain(understand("integral de exp(-x^2)").unwrap());
        assert_eq!(hard["ok"], false);
        assert!(understand("qual é a capital do Brasil?").is_none());
        let root = explain(understand("resolva x^2 - 2 = 0").unwrap());
        assert!(
            root["result_latex"]
                .as_str()
                .unwrap()
                .contains("1{,}4142135624"),
            "{root}"
        );
        let none = explain(understand("raizes de x^2 + 1").unwrap());
        assert_eq!(none["ok"], false);
    }

    #[test]
    fn refuses_what_it_cannot_do() {
        for bad in [
            json!({"op":"eval","expr":"1/0"}),
            json!({"op":"eval","expr":"x + 1"}),
            json!({"op":"eval","expr":"import os"}),
            json!({"op":"root","expr":"x"}),
            json!({"op":"root","expr":"x","from":0,"to":1,"guess":1}),
            json!({"op":"integrate","expr":"x","from":0,"to":1,"tol":0}),
            json!({"op":"ode","expr":"y","t0":0,"y0":1,"t1":1,"steps":0}),
            json!({"op":"ode","expr":"y","t0":0,"y0":1,"t1":1,"steps":100001}),
            json!({"op":"apagar","expr":"1"}),
            json!({"op":"eval","expr":"1","extra":1}),
        ] {
            assert!(run(bad.clone()).is_err(), "{bad} deveria falhar");
        }
    }

    /// Determinístico não é exato: o rótulo só diz exato quando nada arredondou.
    #[test]
    fn labels_say_what_was_rounded() {
        let exact = run(json!({"op":"eval","expr":"(12 + 8) * 3"})).unwrap();
        assert_eq!(exact["approximate"], false);
        assert_eq!(exact["note"], "");
        let rounded = run(json!({"op":"eval","expr":"0.1 + 0.2"})).unwrap();
        assert_eq!(rounded["approximate"], true);
        assert_eq!(rounded["note"], ROUNDING);
        let derive = run(json!({"op":"derive","expr":"x^2","at":3})).unwrap();
        assert_eq!(derive["approximate"], false);
        assert_eq!(derive["value_at_approximate"], false);
        let derive = run(json!({"op":"derive","expr":"sin(x)","at":3})).unwrap();
        assert_eq!(derive["approximate"], false);
        assert_eq!(derive["value_at_approximate"], true);
        assert!(run(json!({"op":"derive","expr":"x^2"}))
            .unwrap()
            .get("value_at_approximate")
            .is_none());
        assert_eq!(
            explain(understand("calcule 1/3").unwrap())["approximate"],
            true
        );
    }

    /// As reproduções da auditoria, pelo mesmo caminho que o Arandu usa.
    #[test]
    fn audit_reproductions_stay_fixed() {
        assert!(run(json!({"op":"root","expr":"1/(x-0.1)","from":0,"to":1})).is_err());
        assert!(run(json!({"op":"derive","expr":"ln(x)","at":-1})).is_err());
        let ok = run(json!({"op":"derive","expr":"ln(x)","at":2})).unwrap();
        assert_eq!(ok["value_at"], 0.5);
        let free = run(json!({"op":"derive","expr":"x^2 + y","at":3})).unwrap();
        assert_eq!(free["value_at"], 6.0);
        assert_eq!(
            run(json!({"op":"derive","expr":"ln(x)"})).unwrap()["result"],
            "1 / x"
        );
        let det = run(json!({"op":"det","a":[[1e13,0],[0,1]]})).unwrap();
        assert_eq!(det["result"], 1e13);
        let small = run(json!({"op":"antiderivative","expr":"0.0000000001*x"})).unwrap();
        assert_ne!(small["result"], "0 + C");
        let written = explain(understand("derivada de ln(x) em x = -1").unwrap());
        assert_eq!(written["ok"], false, "{written}");
        assert!(explain(understand("resolva 1/(x-0.12) = 0").unwrap())["ok"] == false);
        // O polo em 0,12 vem antes na busca; a raiz verdadeira fica em -0,88.
        let zero = explain(understand("resolva 1/(x-0.12) + 1 = 0").unwrap());
        assert!(
            zero["result_latex"].as_str().unwrap().contains("-0{,}88"),
            "{zero}"
        );
    }
}
