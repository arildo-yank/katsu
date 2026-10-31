//! Motor de cálculo determinístico do Katsu, sem dependências externas: expressões com variáveis,
//! derivada simbólica, integral numérica, raízes e EDO por Runge-Kutta. Plano em
//! `docs/KATSU_MATH_PLAN.md`. Não executa código, não lê arquivos nem acessa a rede: o que está fora
//! da gramática é recusado. Ângulos em radianos.

#![allow(clippy::redundant_guards)]

use std::{collections::HashMap, fmt};

const MAX_LEN: usize = 500;
const MAX_DEPTH: usize = 48;
const MAX_INTEGRAL_CALLS: usize = 2_000_000;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
    Abs,
}

impl Func {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "sin" | "sen" => Func::Sin,
            "cos" => Func::Cos,
            "tan" | "tg" => Func::Tan,
            "exp" => Func::Exp,
            "ln" => Func::Ln,
            "sqrt" | "raiz" => Func::Sqrt,
            "abs" => Func::Abs,
            _ => return None,
        })
    }
    fn name(self) -> &'static str {
        match self {
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Exp => "exp",
            Func::Ln => "ln",
            Func::Sqrt => "sqrt",
            Func::Abs => "abs",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Num(f64),
    Var(String),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Call(Func, Box<Expr>),
}
use Expr::{Add, Call, Div, Mul, Neg, Num, Pow, Sub, Var};

fn b(e: Expr) -> Box<Expr> {
    Box::new(e)
}

pub fn parse(text: &str) -> Result<Expr, String> {
    if text.trim().is_empty() || text.len() > MAX_LEN {
        return Err("expressão vazia ou longa demais".into());
    }
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
        depth: 0,
    };
    let e = p.sum()?;
    p.space();
    if p.i != p.s.len() {
        return Err(format!("símbolo inesperado na posição {}", p.i + 1));
    }
    Ok(e)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn space(&mut self) {
        while self.s.get(self.i) == Some(&b' ') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.space();
        let hit = self.s.get(self.i) == Some(&c);
        if hit {
            self.i += 1;
        }
        hit
    }
    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err("expressão aninhada demais".into());
        }
        Ok(())
    }
    fn sum(&mut self) -> Result<Expr, String> {
        let mut e = self.product()?;
        loop {
            if self.eat(b'+') {
                e = Add(b(e), b(self.product()?));
            } else if self.eat(b'-') {
                e = Sub(b(e), b(self.product()?));
            } else {
                return Ok(e);
            }
        }
    }
    fn product(&mut self) -> Result<Expr, String> {
        let mut e = self.unary()?;
        loop {
            if self.eat(b'*') {
                e = Mul(b(e), b(self.unary()?));
            } else if self.eat(b'/') {
                e = Div(b(e), b(self.unary()?));
            } else {
                return Ok(e);
            }
        }
    }
    /// O sinal vale menos que a potência: -x^2 é -(x^2), como na matemática.
    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat(b'-') {
            self.enter()?;
            let e = Neg(b(self.unary()?));
            self.depth -= 1;
            return Ok(e);
        }
        self.power()
    }
    /// `^` associa à direita: 2^3^2 = 2^9.
    fn power(&mut self) -> Result<Expr, String> {
        let base = self.atom()?;
        if self.eat(b'^') {
            self.enter()?;
            let exp = self.unary()?;
            self.depth -= 1;
            return Ok(Pow(b(base), b(exp)));
        }
        Ok(base)
    }
    fn atom(&mut self) -> Result<Expr, String> {
        self.enter()?;
        self.space();
        let e = if self.eat(b'(') {
            let e = self.sum()?;
            if !self.eat(b')') {
                return Err("parêntese não fechado".into());
            }
            e
        } else if self.s.get(self.i).is_some_and(u8::is_ascii_alphabetic) {
            self.word()?
        } else {
            self.number()?
        };
        self.depth -= 1;
        Ok(e)
    }
    fn word(&mut self) -> Result<Expr, String> {
        let start = self.i;
        while self
            .s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
        {
            self.i += 1;
        }
        let name = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        self.space();
        if self.s.get(self.i) == Some(&b'(') {
            let func = Func::parse(name).ok_or_else(|| format!("função desconhecida: {name}"))?;
            self.i += 1;
            let arg = self.sum()?;
            if !self.eat(b')') {
                return Err("parêntese não fechado".into());
            }
            return Ok(Call(func, b(arg)));
        }
        Ok(match name {
            "pi" => Num(std::f64::consts::PI),
            "e" => Num(std::f64::consts::E),
            _ => Var(name.to_owned()),
        })
    }
    fn number(&mut self) -> Result<Expr, String> {
        let start = self.i;
        while self
            .s
            .get(self.i)
            .is_some_and(|c| c.is_ascii_digit() || *c == b'.')
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.s[start..self.i])
            .unwrap_or("")
            .parse::<f64>()
            .map(Num)
            .map_err(|_| format!("número esperado na posição {}", start + 1))
    }
}

pub type Vars = HashMap<String, f64>;

/// Valor da expressão. Resultado não finito, divisão por zero e domínio inválido são erro, nunca número.
pub fn eval(e: &Expr, vars: &Vars) -> Result<f64, String> {
    let v = match e {
        Num(n) => *n,
        Var(name) => *vars
            .get(name)
            .ok_or_else(|| format!("variável sem valor: {name}"))?,
        Neg(a) => -eval(a, vars)?,
        Add(a, c) => eval(a, vars)? + eval(c, vars)?,
        Sub(a, c) => eval(a, vars)? - eval(c, vars)?,
        Mul(a, c) => eval(a, vars)? * eval(c, vars)?,
        Div(a, c) => {
            let d = eval(c, vars)?;
            if d == 0.0 {
                return Err("divisão por zero".into());
            }
            eval(a, vars)? / d
        }
        Pow(a, c) => eval(a, vars)?.powf(eval(c, vars)?),
        Call(f, a) => {
            let x = eval(a, vars)?;
            match f {
                Func::Sin => x.sin(),
                Func::Cos => x.cos(),
                Func::Tan => x.tan(),
                Func::Exp => x.exp(),
                Func::Abs => x.abs(),
                Func::Ln if x <= 0.0 => return Err("logaritmo de número não positivo".into()),
                Func::Ln => x.ln(),
                Func::Sqrt if x < 0.0 => return Err("raiz de número negativo".into()),
                Func::Sqrt => x.sqrt(),
            }
        }
    };
    if !v.is_finite() {
        return Err("resultado não finito".into());
    }
    Ok(v)
}

fn add(a: Expr, c: Expr) -> Expr {
    match (a, c) {
        (Num(x), Num(y)) => Num(x + y),
        (Num(z), e) | (e, Num(z)) if z == 0.0 => e,
        (a, c) => Add(b(a), b(c)),
    }
}
fn sub(a: Expr, c: Expr) -> Expr {
    match (a, c) {
        (Num(x), Num(y)) => Num(x - y),
        (e, Num(z)) if z == 0.0 => e,
        (Num(z), e) if z == 0.0 => neg(e),
        (a, c) => Sub(b(a), b(c)),
    }
}
fn neg(e: Expr) -> Expr {
    match e {
        Num(x) => Num(-x),
        Neg(a) => *a,
        e => Neg(b(e)),
    }
}
fn mul(a: Expr, c: Expr) -> Expr {
    match (a, c) {
        (Num(x), Num(y)) => Num(x * y),
        (Num(z), _) | (_, Num(z)) if z == 0.0 => Num(0.0),
        (Num(o), e) | (e, Num(o)) if o == 1.0 => e,
        (a, c) => Mul(b(a), b(c)),
    }
}
fn div(a: Expr, c: Expr) -> Expr {
    match (a, c) {
        (Num(z), _) if z == 0.0 => Num(0.0),
        (e, Num(o)) if o == 1.0 => e,
        (a, c) => Div(b(a), b(c)),
    }
}
fn pow(a: Expr, c: Expr) -> Expr {
    match (a, c) {
        (_, Num(z)) if z == 0.0 => Num(1.0),
        (e, Num(o)) if o == 1.0 => e,
        (a, c) => Pow(b(a), b(c)),
    }
}

fn uses(e: &Expr, var: &str) -> bool {
    match e {
        Num(_) => false,
        Var(n) => n == var,
        Neg(a) | Call(_, a) => uses(a, var),
        Add(a, c) | Sub(a, c) | Mul(a, c) | Div(a, c) | Pow(a, c) => uses(a, var) || uses(c, var),
    }
}

/// Derivada simbólica. Serve também como derivada parcial: as outras variáveis são constantes.
pub fn diff(e: &Expr, var: &str) -> Result<Expr, String> {
    Ok(match e {
        Num(_) => Num(0.0),
        Var(n) => Num(if n == var { 1.0 } else { 0.0 }),
        Neg(a) => neg(diff(a, var)?),
        Add(a, c) => add(diff(a, var)?, diff(c, var)?),
        Sub(a, c) => sub(diff(a, var)?, diff(c, var)?),
        Mul(a, c) => add(
            mul(diff(a, var)?, (**c).clone()),
            mul((**a).clone(), diff(c, var)?),
        ),
        Div(a, c) => div(
            sub(
                mul(diff(a, var)?, (**c).clone()),
                mul((**a).clone(), diff(c, var)?),
            ),
            pow((**c).clone(), Num(2.0)),
        ),
        Pow(a, c) if !uses(c, var) => mul(
            mul(
                (**c).clone(),
                pow((**a).clone(), sub((**c).clone(), Num(1.0))),
            ),
            diff(a, var)?,
        ),
        Pow(a, c) => mul(
            e.clone(),
            add(
                mul(diff(c, var)?, Call(Func::Ln, a.clone())),
                mul((**c).clone(), div(diff(a, var)?, (**a).clone())),
            ),
        ),
        Call(f, a) => {
            let x = (**a).clone();
            let outer = match f {
                Func::Sin => Call(Func::Cos, b(x)),
                Func::Cos => neg(Call(Func::Sin, b(x))),
                Func::Tan => div(Num(1.0), pow(Call(Func::Cos, b(x)), Num(2.0))),
                Func::Exp => Call(Func::Exp, b(x)),
                Func::Ln => div(Num(1.0), x),
                Func::Sqrt => div(Num(1.0), mul(Num(2.0), Call(Func::Sqrt, b(x)))),
                Func::Abs => div(x.clone(), Call(Func::Abs, b(x))),
            };
            mul(outer, diff(a, var)?)
        }
    })
}

/// Precedência para escrever só os parênteses necessários.
fn rank(e: &Expr) -> u8 {
    match e {
        Add(..) | Sub(..) => 1,
        Mul(..) | Div(..) => 2,
        Neg(..) => 3,
        Num(n) if *n < 0.0 => 3,
        Pow(..) => 4,
        _ => 5,
    }
}

fn side(f: &mut fmt::Formatter<'_>, e: &Expr, min: u8) -> fmt::Result {
    if rank(e) < min {
        write!(f, "({e})")
    } else {
        write!(f, "{e}")
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let binary =
            |f: &mut fmt::Formatter<'_>, a: &Expr, op: &str, c: &Expr, left: u8, right: u8| {
                side(f, a, left)?;
                write!(f, " {op} ")?;
                side(f, c, right)
            };
        match self {
            Num(n) if *n == std::f64::consts::PI => write!(f, "pi"),
            Num(n) if *n == std::f64::consts::E => write!(f, "e"),
            Num(n) => write!(f, "{n}"),
            Var(v) => write!(f, "{v}"),
            Neg(a) => {
                write!(f, "-")?;
                side(f, a, 3)
            }
            Add(a, c) => binary(f, a, "+", c, 1, 2),
            Sub(a, c) => binary(f, a, "-", c, 1, 2),
            Mul(a, c) => binary(f, a, "*", c, 2, 3),
            Div(a, c) => binary(f, a, "/", c, 2, 3),
            Pow(a, c) => {
                side(f, a, 5)?;
                write!(f, "^")?;
                side(f, c, 3)
            }
            Call(func, a) => write!(f, "{}({a})", func.name()),
        }
    }
}

fn at(e: &Expr, var: &str, x: f64, base: &Vars) -> Result<f64, String> {
    let mut vars = base.clone();
    vars.insert(var.to_owned(), x);
    eval(e, &vars)
}

/// Integral definida por Simpson adaptativo. Numérica: devolve aproximação dentro de `tol`, não forma fechada.
pub fn integrate(
    e: &Expr,
    var: &str,
    a: f64,
    c: f64,
    tol: f64,
    vars: &Vars,
) -> Result<f64, String> {
    fn simpson(fa: f64, fm: f64, fc: f64, h: f64) -> f64 {
        h / 6.0 * (fa + 4.0 * fm + fc)
    }
    #[allow(clippy::too_many_arguments)]
    fn rec(
        f: &dyn Fn(f64) -> Result<f64, String>,
        a: f64,
        c: f64,
        fa: f64,
        fm: f64,
        fc: f64,
        whole: f64,
        tol: f64,
        depth: u32,
    ) -> Result<f64, String> {
        let m = (a + c) / 2.0;
        let (lm, rm) = ((a + m) / 2.0, (m + c) / 2.0);
        let (flm, frm) = (f(lm)?, f(rm)?);
        let (left, right) = (simpson(fa, flm, fm, m - a), simpson(fm, frm, fc, c - m));
        let delta = left + right - whole;
        if delta.abs() <= 15.0 * tol {
            return Ok(left + right + delta / 15.0);
        }
        if depth == 0 {
            return Err("a integral não convergiu na tolerância pedida".into());
        }
        Ok(rec(
            f,
            a,
            m,
            fa,
            flm,
            fm,
            left,
            (tol / 2.0).max(1e-14),
            depth - 1,
        )? + rec(
            f,
            m,
            c,
            fm,
            frm,
            fc,
            right,
            (tol / 2.0).max(1e-14),
            depth - 1,
        )?)
    }
    if !a.is_finite() || !c.is_finite() || tol <= 0.0 {
        return Err("limites e tolerância precisam ser finitos, com tolerância positiva".into());
    }
    let calls = std::cell::Cell::new(0usize);
    let f = |x: f64| {
        calls.set(calls.get() + 1);
        if calls.get() > MAX_INTEGRAL_CALLS {
            return Err("a integral não convergiu dentro do limite de avaliações".to_owned());
        }
        at(e, var, x, vars)
    };
    let (fa, fm, fc) = (f(a)?, f((a + c) / 2.0)?, f(c)?);
    rec(&f, a, c, fa, fm, fc, simpson(fa, fm, fc, c - a), tol, 40)
}

/// Raiz por bisseção num intervalo em que a função troca de sinal.
pub fn bisect(
    e: &Expr,
    var: &str,
    mut a: f64,
    mut c: f64,
    tol: f64,
    vars: &Vars,
) -> Result<f64, String> {
    let (mut fa, fc) = (at(e, var, a, vars)?, at(e, var, c, vars)?);
    if fa == 0.0 {
        return Ok(a);
    }
    if fc == 0.0 {
        return Ok(c);
    }
    if fa.signum() == fc.signum() {
        return Err("a função não troca de sinal nesse intervalo".into());
    }
    for _ in 0..200 {
        let m = (a + c) / 2.0;
        let fm = at(e, var, m, vars)?;
        if fm == 0.0 || (c - a).abs() / 2.0 < tol {
            return Ok(m);
        }
        if fa.signum() == fm.signum() {
            (a, fa) = (m, fm);
        } else {
            c = m;
        }
    }
    Err("a bisseção não convergiu".into())
}

/// Raiz por Newton-Raphson, com a derivada simbólica. Recusa quando não converge, sem devolver palpite.
pub fn newton(e: &Expr, var: &str, start: f64, tol: f64, vars: &Vars) -> Result<f64, String> {
    let d = diff(e, var)?;
    let mut x = start;
    for _ in 0..100 {
        let fx = at(e, var, x, vars)?;
        if fx.abs() < tol {
            return Ok(x);
        }
        let slope = at(&d, var, x, vars)?;
        if slope == 0.0 {
            return Err("derivada nula: o método de Newton não segue daqui".into());
        }
        x -= fx / slope;
        if !x.is_finite() {
            return Err("o método de Newton divergiu".into());
        }
    }
    Err("o método de Newton não convergiu".into())
}

#[allow(clippy::too_many_arguments)]
/// y' = f(t, y) por Runge-Kutta de 4.ª ordem, de `t0` a `t1` em `steps` passos. Devolve y(t1).
pub fn rk4(
    f: &Expr,
    t: &str,
    y: &str,
    t0: f64,
    y0: f64,
    t1: f64,
    steps: usize,
    vars: &Vars,
) -> Result<f64, String> {
    if steps == 0 || steps > 10_000_000 {
        return Err("número de passos fora do limite".into());
    }
    let mut env = vars.clone();
    let mut call = |tv: f64, yv: f64| {
        env.insert(t.to_owned(), tv);
        env.insert(y.to_owned(), yv);
        eval(f, &env)
    };
    let h = (t1 - t0) / steps as f64;
    let (mut tv, mut yv) = (t0, y0);
    for i in 0..steps {
        let k1 = call(tv, yv)?;
        let k2 = call(tv + h / 2.0, yv + h * k1 / 2.0)?;
        let k3 = call(tv + h / 2.0, yv + h * k2 / 2.0)?;
        let k4 = call(tv + h, yv + h * k3)?;
        yv += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        tv = t0 + h * (i + 1) as f64;
        if !yv.is_finite() {
            return Err("a solução divergiu".into());
        }
    }
    Ok(yv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: f64, c: f64, tol: f64) {
        assert!((a - c).abs() <= tol, "{a} != {c}");
    }
    fn none() -> Vars {
        Vars::new()
    }
    fn value(text: &str) -> Result<f64, String> {
        eval(&parse(text)?, &none())
    }
    fn at_x(e: &Expr, x: f64) -> f64 {
        at(e, "x", x, &none()).unwrap()
    }

    #[test]
    fn evaluates_with_mathematical_precedence() {
        assert_eq!(value("(12 + 8) * 3"), Ok(60.0));
        assert_eq!(value("2 + 3 * 4"), Ok(14.0));
        assert_eq!(value("2^3^2"), Ok(512.0));
        assert_eq!(value("-2^2"), Ok(-4.0));
        assert_eq!(value("2^-1"), Ok(0.5));
        assert_eq!(value("15 / 100 * 2300"), Ok(345.0));
        close(value("sen(30 * pi / 180)").unwrap(), 0.5, 1e-12);
        close(value("exp(ln(5))").unwrap(), 5.0, 1e-12);
    }

    #[test]
    fn refuses_instead_of_guessing() {
        for bad in [
            "",
            "1/0",
            "sqrt(-1)",
            "ln(0)",
            "9^9^9",
            "2 +",
            "(1",
            "1.2.3",
            "foo(2)",
            "__import__('os')",
            "x y",
            "2 2",
        ] {
            assert!(value(bad).is_err(), "{bad} deveria falhar");
        }
        assert!(value("x + 1").is_err(), "variável sem valor");
        assert!(parse(&"(".repeat(200)).is_err());
        assert!(parse(&"-".repeat(200)).is_err());
        assert!(parse(&"2^".repeat(100)).is_err());
        assert!(parse(&"1+".repeat(400)).is_err(), "longa demais");
    }

    /// Cada derivada simbólica é conferida contra a diferença central, em vários pontos.
    #[test]
    fn derivatives_match_finite_differences() {
        for text in [
            "x^2",
            "3*x^3 - 2*x + 7",
            "sin(x)*cos(x)",
            "exp(x^2)",
            "ln(x^2 + 1)",
            "sqrt(x + 2)",
            "x^x",
            "(x + 1) / (x^2 + 1)",
            "tan(x / 3)",
            "abs(x)",
            "2^x",
        ] {
            let e = parse(text).unwrap();
            let d = diff(&e, "x").unwrap();
            for x in [0.3, 0.9, 1.7, 2.4] {
                let h = 1e-6;
                let numeric = (at_x(&e, x + h) - at_x(&e, x - h)) / (2.0 * h);
                close(at_x(&d, x), numeric, 1e-5 * (1.0 + numeric.abs()));
            }
        }
    }

    #[test]
    fn partial_derivative_treats_other_variables_as_constants() {
        let d = diff(&parse("x^2 * y + y^3").unwrap(), "y").unwrap();
        let vars = Vars::from([("x".to_owned(), 3.0), ("y".to_owned(), 2.0)]);
        assert_eq!(eval(&d, &vars), Ok(21.0));
    }

    #[test]
    fn integrals_match_known_answers() {
        let cases = [
            ("x^2", 0.0, 1.0, 1.0 / 3.0),
            ("sin(x)", 0.0, PI, 2.0),
            ("exp(x)", 0.0, 1.0, std::f64::consts::E - 1.0),
            ("1 / x", 1.0, std::f64::consts::E, 1.0),
            ("sqrt(1 - x^2)", 0.0, 1.0, PI / 4.0),
        ];
        for (text, a, c, expected) in cases {
            close(
                integrate(&parse(text).unwrap(), "x", a, c, 1e-9, &none()).unwrap(),
                expected,
                1e-6,
            );
        }
        assert!(
            integrate(&parse("1 / x").unwrap(), "x", 0.0, 1.0, 1e-9, &none()).is_err(),
            "singular em 0"
        );
    }

    #[test]
    fn roots_match_known_answers_or_refuse() {
        let two = parse("x^2 - 2").unwrap();
        close(
            bisect(&two, "x", 0.0, 2.0, 1e-12, &none()).unwrap(),
            2f64.sqrt(),
            1e-10,
        );
        close(
            newton(&two, "x", 1.0, 1e-12, &none()).unwrap(),
            2f64.sqrt(),
            1e-10,
        );
        assert!(bisect(&two, "x", 2.0, 3.0, 1e-9, &none()).is_err());
        assert!(newton(&parse("x^2 + 1").unwrap(), "x", 0.0, 1e-9, &none()).is_err());
    }

    #[test]
    fn rk4_matches_the_exact_solution_with_fourth_order_error() {
        let f = parse("-2 * y").unwrap();
        let exact = (-10f64).exp();
        let coarse = rk4(&f, "t", "y", 0.0, 1.0, 5.0, 100, &none()).unwrap();
        let fine = rk4(&f, "t", "y", 0.0, 1.0, 5.0, 200, &none()).unwrap();
        close(fine, exact, 1e-7);
        let order = ((coarse - exact).abs() / (fine - exact).abs()).log2();
        assert!((3.5..4.5).contains(&order), "ordem medida {order}");
    }

    #[test]
    fn hard_integrals_stop_instead_of_hanging() {
        let start = std::time::Instant::now();
        let e = parse("sin(1 / x) / x^3").unwrap();
        let _ = integrate(&e, "x", 1e-6, 1.0, 1e-12, &none());
        assert!(start.elapsed().as_secs() < 30);
    }

    #[test]
    fn printed_expressions_parse_back_to_the_same_value() {
        for text in [
            "-x^2",
            "(-x)^2",
            "x - (x - 1)",
            "x / (x / 2)",
            "2^(x + 1)",
            "-(x + 1) * 3",
            "x^-1",
            "2^3^x",
            "sin(x)^2 - cos(-x)",
            "(x * 2) / 3 - 1",
        ] {
            let e = parse(text).unwrap();
            let again = parse(&e.to_string()).unwrap_or_else(|why| panic!("{text} -> {e}: {why}"));
            close(at_x(&e, 1.3), at_x(&again, 1.3), 1e-12);
        }
        assert_eq!(
            diff(&parse("x^2").unwrap(), "x").unwrap().to_string(),
            "2 * x"
        );
        assert_eq!(parse("(x + 1) * 2").unwrap().to_string(), "(x + 1) * 2");
        assert_eq!(
            parse("sin(30 * pi / 180) + e").unwrap().to_string(),
            "sin(30 * pi / 180) + e"
        );
    }

    #[test]
    fn free_fall_uses_given_values() {
        let vars = Vars::from([("g".to_owned(), 9.81), ("t".to_owned(), 5.0)]);
        close(eval(&parse("g * t").unwrap(), &vars).unwrap(), 49.05, 1e-12);
        assert!(value("g * t").is_err(), "sem os dados, recusa");
    }
}
