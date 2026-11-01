//! Escreve uma expressão em LaTeX, com só os parênteses necessários.
use super::{rank, Expr, Func};

fn group(e: &Expr, min: u8) -> String {
    if rank(e) < min {
        format!("\\left({}\\right)", latex(e))
    } else {
        latex(e)
    }
}

fn name(var: &str) -> String {
    if var.chars().count() == 1 {
        var.to_owned()
    } else {
        format!("\\mathrm{{{}}}", var.replace('_', "\\_"))
    }
}

/// Número sem notação científica, que o LaTeX mostraria como texto cru.
fn number(n: f64) -> String {
    if n == std::f64::consts::PI {
        "\\pi".into()
    } else if n == std::f64::consts::E {
        "e".into()
    } else if n.abs() >= 1e15 {
        let text = format!("{n:e}");
        let (mantissa, exp) = text.split_once('e').unwrap_or((&text, "0"));
        format!("{mantissa} \\times 10^{{{exp}}}")
    } else {
        format!("{n}")
    }
}

/// O produto dispensa o ponto quando a leitura é óbvia: 2x, 3\sin(x), x^2 \cos(x).
fn implicit(a: &Expr, c: &Expr) -> bool {
    matches!(a, Expr::Num(n) if *n >= 0.0)
        && matches!(c, Expr::Var(_) | Expr::Pow(..) | Expr::Call(..))
        && !matches!(c, Expr::Pow(base, _) if matches!(**base, Expr::Num(_)))
        || matches!(a, Expr::Var(_) | Expr::Pow(..)) && matches!(c, Expr::Call(..))
}

pub fn latex(e: &Expr) -> String {
    match e {
        Expr::Num(n) => number(*n),
        Expr::Var(v) => name(v),
        Expr::Neg(a) => format!("-{}", group(a, 3)),
        Expr::Add(a, c) => format!("{} + {}", group(a, 1), group(c, 2)),
        Expr::Sub(a, c) => format!("{} - {}", group(a, 1), group(c, 2)),
        Expr::Mul(a, c) if implicit(a, c) => format!("{} {}", group(a, 2), group(c, 3)),
        Expr::Mul(a, c) => format!("{} \\cdot {}", group(a, 2), group(c, 3)),
        Expr::Div(a, c) => format!("\\frac{{{}}}{{{}}}", latex(a), latex(c)),
        Expr::Pow(a, c) => format!("{}^{{{}}}", group(a, 5), latex(c)),
        Expr::Call(Func::Sqrt, a) => format!("\\sqrt{{{}}}", latex(a)),
        Expr::Call(Func::Abs, a) => format!("\\left|{}\\right|", latex(a)),
        Expr::Call(Func::Exp, a) => format!("e^{{{}}}", latex(a)),
        Expr::Call(f, a) => format!("\\{}\\left({}\\right)", f.name(), latex(a)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn tex(text: &str) -> String {
        latex(&parse(text).unwrap())
    }

    #[test]
    fn writes_common_forms() {
        assert_eq!(tex("x^3 - 2*x"), "x^{3} - 2 x");
        assert_eq!(tex("x^4 / 4"), "\\frac{x^{4}}{4}");
        assert_eq!(tex("sqrt(x + 1)"), "\\sqrt{x + 1}");
        assert_eq!(tex("sin(x)^2"), "\\sin\\left(x\\right)^{2}");
        assert_eq!(tex("(x + 1) * 2"), "\\left(x + 1\\right) \\cdot 2");
        assert_eq!(tex("exp(-x^2)"), "e^{-x^{2}}");
        assert_eq!(tex("abs(x) + pi"), "\\left|x\\right| + \\pi");
        assert_eq!(tex("x^2 * cos(x)"), "x^{2} \\cos\\left(x\\right)");
        assert_eq!(tex("2 * 3"), "2 \\cdot 3");
    }
}
