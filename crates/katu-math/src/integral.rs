//! Primitiva em forma fechada para as classes de função listadas em `integ`. O que não está nelas é
//! recusado. Toda primitiva é conferida antes de sair: derivar o resultado precisa devolver a função.
use super::{add, b, diff, div, eval, mul, neg, pow, uses, Expr, Func, Vars};
use Expr::{Add, Call, Div, Mul, Neg, Num, Pow, Sub, Var};

const UNKNOWN: &str = "ainda não sei escrever essa primitiva em forma fechada; com limites, a integral numérica funciona";

/// Fração simples, para escrever x^4 / 4 em vez de 0.25 * x^4.
/// A folga é relativa e cobre só o arredondamento: coeficiente pequeno não vira zero, nem 0.3000000001 vira 3/10.
fn fraction(c: f64) -> Option<(i64, i64)> {
    if c == 0.0 {
        return Some((0, 1));
    }
    (1..=1000i64).find_map(|q| {
        let p = c * q as f64;
        let r = p.round();
        (r != 0.0 && (p - r).abs() <= 1e-14 * r.abs() && r.abs() < 1e12).then_some((r as i64, q))
    })
}

fn scaled(c: f64, e: Expr) -> Expr {
    match fraction(c) {
        Some((p, 1)) => {
            if p < 0 {
                neg(mul(Num(-p as f64), e))
            } else {
                mul(Num(p as f64), e)
            }
        }
        Some((p, q)) if p < 0 => neg(div(mul(Num(-p as f64), e), Num(q as f64))),
        Some((p, q)) => div(mul(Num(p as f64), e), Num(q as f64)),
        None => mul(Num(c), e),
    }
}

fn constant(e: &Expr) -> Option<f64> {
    eval(e, &Vars::new()).ok()
}

/// O coeficiente `a` de um argumento linear a*x + b, sem o qual a regra da cadeia não se inverte.
fn linear(g: &Expr, var: &str) -> Result<f64, String> {
    let slope = diff(g, var)?;
    if uses(&slope, var) {
        return Err(UNKNOWN.into());
    }
    match constant(&slope) {
        Some(a) if a != 0.0 => Ok(a),
        _ => Err(UNKNOWN.into()),
    }
}

fn power(c: f64, n: f64, x: &Expr) -> Expr {
    if n == -1.0 {
        scaled(c, Call(Func::Ln, b(Call(Func::Abs, b(x.clone())))))
    } else {
        scaled(c / (n + 1.0), pow(x.clone(), Num(n + 1.0)))
    }
}

/// Soma, constante vezes função, x^n, 1/x, sqrt(x), a^x, ln(x), e sin, cos e exp de argumento linear.
fn integ(e: &Expr, var: &str, c: f64) -> Result<Expr, String> {
    let x = Var(var.to_owned());
    if !uses(e, var) {
        return Ok(match constant(e) {
            Some(k) => scaled(c * k, x),
            None => scaled(c, mul(e.clone(), x)),
        });
    }
    Ok(match e {
        Var(_) => power(c, 1.0, &x),
        Neg(a) => integ(a, var, -c)?,
        Add(a, d) => add(integ(a, var, c)?, integ(d, var, c)?),
        Sub(a, d) => add(integ(a, var, c)?, integ(d, var, -c)?),
        Mul(a, d) if !uses(a, var) => match constant(a) {
            Some(k) => integ(d, var, c * k)?,
            None => mul((**a).clone(), integ(d, var, c)?),
        },
        Mul(a, d) if !uses(d, var) => match constant(d) {
            Some(k) => integ(a, var, c * k)?,
            None => mul((**d).clone(), integ(a, var, c)?),
        },
        Div(a, d) if !uses(d, var) => match constant(d) {
            Some(k) if k != 0.0 => integ(a, var, c / k)?,
            _ => return Err(UNKNOWN.into()),
        },
        Div(a, d) if !uses(a, var) => {
            let k = constant(a).ok_or(UNKNOWN)?;
            match &**d {
                Var(_) => power(c * k, -1.0, &x),
                Pow(base, n) if **base == x && !uses(n, var) => {
                    power(c * k, -constant(n).ok_or(UNKNOWN)?, &x)
                }
                _ => return Err(UNKNOWN.into()),
            }
        }
        Pow(base, n) if **base == x && !uses(n, var) => power(c, constant(n).ok_or(UNKNOWN)?, &x),
        Pow(base, n) if !uses(base, var) => {
            let k = constant(base)
                .filter(|k| *k > 0.0 && *k != 1.0)
                .ok_or(UNKNOWN)?;
            let slope = linear(n, var)?;
            scaled(c / (slope * k.ln()), e.clone())
        }
        Call(Func::Sqrt, a) if **a == x => power(c, 0.5, &x),
        Call(Func::Ln, a) if **a == x => scaled(
            c,
            Sub(b(mul(x.clone(), Call(Func::Ln, b(x.clone())))), b(x)),
        ),
        Call(f @ (Func::Sin | Func::Cos | Func::Exp), g) => {
            let a = linear(g, var)?;
            match f {
                Func::Sin => scaled(-c / a, Call(Func::Cos, g.clone())),
                Func::Cos => scaled(c / a, Call(Func::Sin, g.clone())),
                _ => scaled(c / a, Call(Func::Exp, g.clone())),
            }
        }
        _ => return Err(UNKNOWN.into()),
    })
}

pub(super) fn free_names(e: &Expr, out: &mut Vec<String>) {
    match e {
        Num(_) => {}
        Var(v) => {
            if !out.contains(v) {
                out.push(v.clone());
            }
        }
        Neg(a) | Call(_, a) => free_names(a, out),
        Add(a, c) | Sub(a, c) | Mul(a, c) | Div(a, c) | Pow(a, c) => {
            free_names(a, out);
            free_names(c, out);
        }
    }
}

/// Maior soma de valores absolutos dos termos: a escala do arredondamento quando os termos se cancelam.
fn magnitude(e: &Expr, vars: &Vars) -> Result<f64, String> {
    match e {
        Add(a, c) | Sub(a, c) => Ok(magnitude(a, vars)? + magnitude(c, vars)?),
        Neg(a) => magnitude(a, vars),
        other => eval(other, vars).map(f64::abs),
    }
}

/// Deriva a primitiva e compara com a função em vários pontos. Sem isto, um erro de regra sairia como
/// resposta. A folga é relativa à escala da própria função, sem parte absoluta: 1e-10 * x não passa por zero.
fn confirm(e: &Expr, primitive: &Expr, var: &str) -> Result<(), String> {
    let d = diff(primitive, var)?;
    let mut names = Vec::new();
    free_names(e, &mut names);
    let mut checked = 0;
    for x in [0.37, 0.91, 1.63, 2.29, 3.17, -0.53, -1.41] {
        let vars: Vars = names
            .iter()
            .map(|n| (n.clone(), if n == var { x } else { 1.3 }))
            .collect();
        if let (Ok(want), Ok(got)) = (eval(e, &vars), eval(&d, &vars)) {
            let scale = magnitude(e, &vars)?.max(magnitude(&d, &vars)?);
            if (want - got).abs() > 1e-7 * scale {
                return Err("a primitiva não passou na conferência; prefiro não responder".into());
            }
            checked += 1;
        }
    }
    if checked < 3 {
        return Err("não foi possível conferir a primitiva; prefiro não responder".into());
    }
    Ok(())
}

/// Primitiva de `e` em relação a `var`, sem a constante de integração.
pub fn antiderivative(e: &Expr, var: &str) -> Result<Expr, String> {
    let primitive = integ(e, var, 1.0)?;
    confirm(e, &primitive, var)?;
    Ok(primitive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{latex::latex, parse};

    fn primitive(text: &str) -> Result<String, String> {
        antiderivative(&parse(text).unwrap(), "x").map(|p| p.to_string())
    }

    #[test]
    fn writes_the_expected_forms() {
        assert_eq!(primitive("x^3 - 2*x").unwrap(), "x^4 / 4 - x^2");
        assert_eq!(primitive("3").unwrap(), "3 * x");
        assert_eq!(primitive("x").unwrap(), "x^2 / 2");
        assert_eq!(primitive("1 / x").unwrap(), "ln(abs(x))");
        assert_eq!(primitive("cos(x)").unwrap(), "sin(x)");
        assert_eq!(primitive("sin(x)").unwrap(), "-cos(x)");
        assert_eq!(primitive("exp(2*x)").unwrap(), "exp(2 * x) / 2");
        assert_eq!(primitive("ln(x)").unwrap(), "x * ln(x) - x");
    }

    #[test]
    fn writes_latex_for_the_requested_example() {
        let p = antiderivative(&parse("x^3 - 2*x").unwrap(), "x").unwrap();
        assert_eq!(latex(&p), "\\frac{x^{4}}{4} - x^{2}");
    }

    /// Cada caso passa pela conferência interna; aqui, ela é testada também contra a integral numérica.
    #[test]
    fn matches_numeric_integrals() {
        for text in [
            "x^3 - 2*x",
            "5*x^4 - 3/x + 2",
            "sqrt(x)",
            "2^x",
            "sin(3*x + 1)",
            "4*cos(x / 2)",
            "exp(-x)",
            "ln(x)",
            "1 / x^3",
            "x^-2 + 7",
            "-x + pi",
        ] {
            let e = parse(text).unwrap();
            let p = antiderivative(&e, "x").unwrap_or_else(|why| panic!("{text}: {why}"));
            let (a, c) = (1.0, 2.5);
            let at = |v: f64| eval(&p, &Vars::from([("x".to_owned(), v)])).unwrap();
            let numeric = crate::integrate(&e, "x", a, c, 1e-9, &Vars::new()).unwrap();
            assert!((at(c) - at(a) - numeric).abs() < 1e-6, "{text}");
        }
    }

    #[test]
    fn keeps_other_variables_as_constants() {
        assert_eq!(primitive("y * x^2").unwrap(), "y * (x^3 / 3)");
    }

    #[test]
    fn refuses_what_it_does_not_know() {
        for text in [
            "exp(-x^2)",
            "x * sin(x)",
            "sin(x^2)",
            "x^x",
            "tan(x)",
            "1 / (x + 1)",
            "x * (x + 1)",
        ] {
            assert!(primitive(text).is_err(), "{text} deveria ser recusada");
        }
    }

    /// Coeficiente pequeno não é zero: nem na fração, nem na conferência.
    #[test]
    fn small_coefficients_survive() {
        for (text, c) in [
            ("0.0000000001*x", 1e-10),
            ("0.000000000000001*x", 1e-15),
            ("0.0000000001*x + 0.0000000002*x", 3e-10),
            ("x / 10000000000", 1e-10),
        ] {
            let e = parse(text).unwrap();
            let p = antiderivative(&e, "x").unwrap_or_else(|why| panic!("{text}: {why}"));
            let got = eval(&p, &Vars::from([("x".to_owned(), 2.0)])).unwrap();
            let want = c * 2.0;
            assert!((got - want).abs() <= 1e-12 * want, "{text} -> {p} = {got}");
        }
        assert_eq!(primitive("0.25*x^3").unwrap(), "x^4 / 16");
        assert_eq!(primitive("0*x").unwrap(), "0");
    }

    /// Valor quase igual a uma fração simples não é arredondado para ela.
    #[test]
    fn near_fractions_keep_their_value() {
        assert_eq!(fraction(0.3000000001), None);
        assert_eq!(fraction(0.1 + 0.2), Some((3, 10)));
        assert_eq!(fraction(1.0 / 3.0), Some((1, 3)));
        assert_eq!(fraction(0.0), Some((0, 1)));
        assert_eq!(fraction(1e-10), None);
    }

    #[test]
    fn confirmation_catches_a_wrong_primitive_at_any_scale() {
        let e = parse("0.0000000001*x").unwrap();
        assert!(confirm(&e, &Num(0.0), "x").is_err());
        assert!(confirm(&e, &parse("0.00000000011*x^2/2").unwrap(), "x").is_err());
        assert!(confirm(&e, &parse("0.0000000001*x^2/2").unwrap(), "x").is_ok());
        // Cancelamento: a função zera num ponto de teste e a conferência não pode recusar por arredondamento.
        assert!(antiderivative(&parse("x - 0.37").unwrap(), "x").is_ok());
        assert!(antiderivative(&parse("1000000*x - 370000").unwrap(), "x").is_ok());
    }
}
