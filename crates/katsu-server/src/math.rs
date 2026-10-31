//! Rota `POST /math`: entrada matemática direta para o `katsu-math`, sem o modelo. A resposta diz o método
//! e se o resultado é aproximado; o que o motor recusa volta como erro 400, nunca como número.
use axum::{extract::rejection::JsonRejection, http::StatusCode, Json};
use katsu_math::{bisect, diff, eval, integrate, newton, parse, rk4, Expr, Vars};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{collections::HashMap, time::Instant};

const MAX_ODE_STEPS: usize = 100_000;
const DEFAULT_TOL: f64 = 1e-9;
const MIN_TOL: f64 = 1e-12;

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
    Root {
        expr: String,
        #[serde(default = "default_var")]
        var: String,
        from: Option<f64>,
        to: Option<f64>,
        guess: Option<f64>,
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

fn solve(req: Request) -> Result<Value, String> {
    let none = Vars::new();
    match req {
        Request::Eval { expr, vars } => {
            let e = expression(&expr)?;
            let value = eval(&e, &vars.into_iter().collect())?;
            Ok(answer(
                &e,
                "eval",
                json!(value),
                "Avaliação direta",
                false,
                String::new(),
            ))
        }
        Request::Derive { expr, var, at } => {
            let e = expression(&expr)?;
            let d = diff(&e, &var)?;
            let value = at
                .map(|x| eval(&d, &Vars::from([(var.clone(), x)])))
                .transpose()?;
            let mut out = answer(
                &e,
                "derive",
                json!(d.to_string()),
                "Derivada simbólica",
                false,
                format!("Em relação a {var}."),
            );
            out["value_at"] = json!(value);
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
}
