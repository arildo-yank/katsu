//! Ferramenta de cálculo do Arandu: o modelo responde `[CALC: expressão]` e o servidor faz a conta.
//! Avaliador próprio (descida recursiva) de números, `+ - * / ^`, parênteses e `sqrt(...)`. Não
//! executa código, não lê arquivos nem acessa a rede: expressões fora dessa gramática são recusadas.

const MAX_LEN: usize = 200;
const MAX_DEPTH: usize = 32;

/// Se `text` é uma chamada `[CALC: …]` (sozinha, com espaços em volta), devolve a expressão.
pub fn call(text: &str) -> Option<&str> {
    text.trim()
        .strip_prefix("[CALC:")?
        .strip_suffix(']')
        .map(str::trim)
}

/// Valor da expressão, finito.
pub fn eval(expr: &str) -> Result<f64, String> {
    if expr.is_empty() || expr.len() > MAX_LEN {
        return Err("expressão vazia ou longa demais".into());
    }
    let mut p = Parser {
        s: expr.as_bytes(),
        i: 0,
        depth: 0,
    };
    let v = p.sum()?;
    p.space();
    if p.i != p.s.len() {
        return Err(format!("símbolo inesperado na posição {}", p.i + 1));
    }
    if !v.is_finite() {
        return Err("resultado não finito (divisão por zero ou número grande demais)".into());
    }
    Ok(v)
}

/// Número no formato brasileiro, sem casas desnecessárias: 12, 3,5, 0,333333.
pub fn format(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = format!("{v:.6}");
    s.trim_end_matches('0')
        .trim_end_matches('.')
        .replace('.', ",")
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn space(&mut self) {
        while self.i < self.s.len() && self.s[self.i] == b' ' {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> bool {
        self.space();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn sum(&mut self) -> Result<f64, String> {
        let mut v = self.product()?;
        loop {
            if self.eat(b'+') {
                v += self.product()?;
            } else if self.eat(b'-') {
                v -= self.product()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn product(&mut self) -> Result<f64, String> {
        let mut v = self.power()?;
        loop {
            if self.eat(b'*') {
                v *= self.power()?;
            } else if self.eat(b'/') {
                let d = self.power()?;
                if d == 0.0 {
                    return Err("divisão por zero".into());
                }
                v /= d;
            } else {
                return Ok(v);
            }
        }
    }
    /// `^` associa à direita: 2^3^2 = 2^9.
    fn power(&mut self) -> Result<f64, String> {
        let base = self.unary()?;
        if self.eat(b'^') {
            let exp = self.power()?;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }
    fn unary(&mut self) -> Result<f64, String> {
        if self.eat(b'-') {
            return Ok(-self.unary()?);
        }
        self.atom()
    }
    fn atom(&mut self) -> Result<f64, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err("expressão aninhada demais".into());
        }
        self.space();
        let v = if self.s[self.i..].starts_with(b"sqrt") {
            self.i += 4;
            if !self.eat(b'(') {
                return Err("sqrt precisa de parênteses".into());
            }
            let x = self.sum()?;
            if !self.eat(b')') {
                return Err("parêntese não fechado".into());
            }
            if x < 0.0 {
                return Err("raiz de número negativo".into());
            }
            x.sqrt()
        } else if self.eat(b'(') {
            let x = self.sum()?;
            if !self.eat(b')') {
                return Err("parêntese não fechado".into());
            }
            x
        } else {
            self.number()?
        };
        self.depth -= 1;
        Ok(v)
    }
    /// Aceita vírgula ou ponto decimal (3,5 ou 3.5); não aceita separador de milhar.
    fn number(&mut self) -> Result<f64, String> {
        self.space();
        let start = self.i;
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_digit() || matches!(self.s[self.i], b'.' | b','))
        {
            self.i += 1;
        }
        let text = std::str::from_utf8(&self.s[start..self.i])
            .unwrap_or("")
            .replace(',', ".");
        text.parse::<f64>()
            .map_err(|_| format!("número esperado na posição {}", start + 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_with_precedence_and_functions() {
        assert_eq!(eval("sqrt(144)"), Ok(12.0));
        assert_eq!(eval("2 + 3 * 4"), Ok(14.0));
        assert_eq!(eval("(2 + 3) * 4"), Ok(20.0));
        assert_eq!(eval("2^3^2"), Ok(512.0));
        assert_eq!(eval("-3 + 10 / 4"), Ok(-0.5));
        assert_eq!(eval("0,15 * 300"), Ok(45.0));
        assert_eq!(
            eval("sqrt(2) * sqrt(2)").map(|v| (v - 2.0).abs() < 1e-12),
            Ok(true)
        );
    }

    #[test]
    fn refuses_anything_outside_the_grammar() {
        for bad in [
            "",
            "1/0",
            "sqrt(-1)",
            "import os",
            "__import__('os')",
            "2 +",
            "(1",
            "1.2.3",
            "9^9^9",
        ] {
            assert!(eval(bad).is_err(), "{bad} deveria falhar");
        }
        assert!(eval(&"(".repeat(40)).is_err());
        assert!(eval(&"1+".repeat(150)).is_err(), "longa demais");
    }

    #[test]
    fn detects_calls_and_formats_results() {
        assert_eq!(call("  [CALC: sqrt(144)] "), Some("sqrt(144)"));
        assert_eq!(
            call("O resultado é [CALC: 1+1]"),
            None,
            "só a chamada sozinha"
        );
        assert_eq!(format(12.0), "12");
        assert_eq!(format(3.5), "3,5");
        assert_eq!(format(1.0 / 3.0), "0,333333");
    }
}
