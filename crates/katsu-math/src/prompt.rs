//! Entende pedidos de matemática escritos, sem modelo: palavra-chave + expressão que o parser aceita.
//! Se a expressão não fecha, devolve `None` e o texto segue como conversa comum: melhor não pegar
//! o pedido do que responder com exatidão à pergunta errada. A resposta sempre mostra o que foi entendido.
use super::{parse, Expr, Func};

#[derive(Clone, Debug, PartialEq)]
pub enum Intent {
    Calculate(Expr),
    Antiderivative {
        expr: Expr,
        var: String,
    },
    Definite {
        expr: Expr,
        var: String,
        from: f64,
        to: f64,
    },
    Derivative {
        expr: Expr,
        var: String,
        at: Option<f64>,
    },
    Root {
        expr: Expr,
        var: String,
    },
}

/// Minúsculas, sem acento, com os símbolos que aparecem em texto trocados pelos da gramática.
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.trim().chars() {
        match ch {
            'á' | 'à' | 'â' | 'ã' | 'Á' | 'À' | 'Â' | 'Ã' => out.push('a'),
            'é' | 'ê' | 'É' | 'Ê' => out.push('e'),
            'í' | 'Í' => out.push('i'),
            'ó' | 'ô' | 'õ' | 'Ó' | 'Ô' | 'Õ' => out.push('o'),
            'ú' | 'Ú' => out.push('u'),
            'ç' | 'Ç' => out.push('c'),
            '−' | '–' | '—' => out.push('-'),
            '×' | '·' => out.push('*'),
            '÷' => out.push('/'),
            '²' => out.push_str("^2"),
            '³' => out.push_str("^3"),
            'π' => out.push_str("pi"),
            '√' => out.push_str("sqrt"),
            c => out.extend(c.to_lowercase()),
        }
    }
    out
}

const WORDS: [(&str, &str); 8] = [
    (" ao quadrado", "^2"),
    (" ao cubo", "^3"),
    (" elevado a ", "^"),
    (" dividido por ", "/"),
    (" vezes ", "*"),
    (" mais ", "+"),
    (" menos ", "-"),
    (" sobre ", "/"),
];

/// Só nomes de uma letra viram variável. Uma palavra desconhecida derruba a expressão.
fn known(word: &str) -> bool {
    word.len() == 1 || word == "pi" || Func::parse(word).is_some()
}

/// Do jeito que se escreve à mão para a gramática: x3 → x^3, 2x → 2*x, 2(x+1) → 2*(x+1), x(x+1) → x*(x+1).
fn normalize(text: &str) -> Option<String> {
    let mut joined = format!(" {text}");
    for (word, symbol) in WORDS {
        joined = joined.replace(word, symbol);
    }
    let chars: Vec<char> = joined.trim().chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_alphabetic() {
            let start = i;
            while chars.get(i).is_some_and(char::is_ascii_alphabetic) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            if !known(&word) {
                return None;
            }
            if out
                .trim_end()
                .chars()
                .last()
                .is_some_and(|p| p.is_ascii_digit() || p == ')')
            {
                out.push('*');
            }
            out.push_str(&word);
            let is_func = Func::parse(&word).is_some();
            if word.len() == 1 && chars.get(i).is_some_and(char::is_ascii_digit) {
                out.push('^');
            }
            if !is_func && chars.get(i) == Some(&'(') {
                out.push('*');
            }
            continue;
        }
        if c == '('
            && out
                .chars()
                .last()
                .is_some_and(|p| p.is_ascii_digit() || p == ')')
        {
            out.push('*');
        }
        if c == ','
            && out.chars().last().is_some_and(|p| p.is_ascii_digit())
            && chars.get(i + 1).is_some_and(char::is_ascii_digit)
        {
            out.push('.');
        } else {
            out.push(c);
        }
        i += 1;
    }
    Some(out)
}

fn expression(text: &str) -> Option<Expr> {
    let mut body = text.trim().trim_end_matches(['?', '.', '!', ':', ' ']);
    for prefix in [
        "de ",
        "da ",
        "do ",
        "funcao ",
        "a funcao ",
        "o valor de ",
        "indefinida de ",
        "definida de ",
        "f(x) =",
        "y =",
        "=",
        ":",
    ] {
        body = body.trim_start();
        body = body.strip_prefix(prefix).unwrap_or(body);
    }
    for prefix in ["funcao ", "de "] {
        body = body.trim_start();
        body = body.strip_prefix(prefix).unwrap_or(body);
    }
    if body.trim().is_empty() {
        return None;
    }
    parse(&normalize(body)?).ok()
}

fn number(text: &str) -> Option<f64> {
    expression(text).and_then(|e| super::eval(&e, &super::Vars::new()).ok())
}

fn split_at<'a>(text: &'a str, marker: &str) -> Option<(&'a str, &'a str)> {
    text.rfind(marker)
        .map(|i| (&text[..i], &text[i + marker.len()..]))
}

/// Variável pedida ("em relação a y", "dy"); sem pedido, a única da expressão, ou `x`.
fn variable(expr: &Expr, asked: Option<String>) -> String {
    asked.unwrap_or_else(|| {
        let mut names = Vec::new();
        super::integral::free_names(expr, &mut names);
        if names.len() == 1 || !names.iter().any(|n| n == "x") {
            names.into_iter().next().unwrap_or_else(|| "x".into())
        } else {
            "x".into()
        }
    })
}

/// Tira "em relação a y" ou o "dy" final.
fn take_variable(text: &str) -> (String, Option<String>) {
    let t = text.trim_end_matches(['?', '.', '!', ' ']);
    for marker in [
        " em relacao a ",
        " com relacao a ",
        " em relacao ao ",
        " with respect to ",
    ] {
        if let Some((head, tail)) = split_at(t, marker) {
            let name = tail.trim();
            if name.len() == 1 && name.chars().all(|c| c.is_ascii_alphabetic()) {
                return (head.to_owned(), Some(name.to_owned()));
            }
        }
    }
    let bytes = t.as_bytes();
    if bytes.len() > 3
        && bytes[bytes.len() - 2] == b'd'
        && bytes[bytes.len() - 1].is_ascii_alphabetic()
        && bytes[bytes.len() - 3] == b' '
    {
        return (
            t[..t.len() - 2].to_owned(),
            Some(t[t.len() - 1..].to_owned()),
        );
    }
    (t.to_owned(), None)
}

fn limits(text: &str) -> Option<(String, f64, f64)> {
    for (from, to) in [
        (" de ", " a "),
        (" de ", " ate "),
        (" entre ", " e "),
        (" from ", " to "),
    ] {
        if let Some((head, range)) = split_at(text, from) {
            if let Some((low, high)) = range.split_once(to) {
                if let (Some(a), Some(c)) = (number(low), number(high)) {
                    return Some((head.to_owned(), a, c));
                }
            }
        }
    }
    None
}

fn after<'a>(text: &'a str, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| text.find(key).map(|i| &text[i + key.len()..]))
}

/// O texto é um pedido de matemática que dá para atender? `None` manda seguir como conversa comum.
pub fn understand(text: &str) -> Option<Intent> {
    let t = clean(text);
    if t.is_empty() || t.len() > 300 {
        return None;
    }
    if let Some(rest) = after(
        &t,
        &[
            "antiderivada",
            "primitiva",
            "integral",
            "integre",
            "integrar",
            "integrate",
        ],
    ) {
        let (rest, asked) = take_variable(rest);
        if let Some((head, from, to)) = limits(&rest) {
            let (head, asked2) = take_variable(&head);
            let expr = expression(&head)?;
            let var = variable(&expr, asked.or(asked2));
            return Some(Intent::Definite {
                expr,
                var,
                from,
                to,
            });
        }
        let expr = expression(&rest)?;
        let var = variable(&expr, asked);
        return Some(Intent::Antiderivative { expr, var });
    }
    if let Some(rest) = after(&t, &["derivada", "derive", "derivar", "derivative of"]) {
        let (rest, asked) = take_variable(rest);
        let (head, at) = match split_at(&rest, " no ponto ")
            .or_else(|| split_at(&rest, " em x ="))
            .or_else(|| split_at(&rest, " at x ="))
        {
            Some((head, point)) => (head.to_owned(), Some(number(point)?)),
            None => (rest.clone(), None),
        };
        let expr = expression(&head)?;
        let var = variable(&expr, asked);
        return Some(Intent::Derivative { expr, var, at });
    }
    if let Some(rest) = after(
        &t,
        &[
            "raizes de ",
            "raizes da ",
            "raizes do ",
            "raiz de ",
            "raiz da ",
            "raiz do ",
            "zero de ",
        ],
    ) {
        let expr = expression(rest)?;
        return Some(if has_variable(&expr) {
            let var = variable(&expr, None);
            Intent::Root { expr, var }
        } else {
            Intent::Calculate(Expr::Call(Func::Sqrt, Box::new(expr)))
        });
    }
    if let Some(rest) = after(
        &t,
        &[
            "quanto e ",
            "quanto da ",
            "calcule ",
            "calcular ",
            "calcula ",
            "resolva ",
            "resolver ",
            "resultado de ",
            "what is ",
            "compute ",
        ],
    ) {
        let body = rest.trim().trim_end_matches(['?', '.', '!', ' ']);
        if let Some(left) = body.strip_suffix("= 0").or_else(|| body.strip_suffix("=0")) {
            let expr = expression(left)?;
            let var = variable(&expr, None);
            return Some(Intent::Root { expr, var });
        }
        let expr = expression(body)?;
        return (!has_variable(&expr)).then_some(Intent::Calculate(expr));
    }
    // Só a conta, sem palavras: precisa ter dígito e operador, para não pegar "x" ou "2026".
    let plain = t.trim_end_matches(['?', '=', ' ']);
    if plain.chars().any(|c| c.is_ascii_digit()) && plain.chars().any(|c| "+-*/^".contains(c)) {
        let expr = expression(plain)?;
        return (!has_variable(&expr)).then_some(Intent::Calculate(expr));
    }
    None
}

fn has_variable(e: &Expr) -> bool {
    let mut names = Vec::new();
    super::integral::free_names(e, &mut names);
    !names.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(text: &str) -> String {
        match understand(text) {
            None => "nada".into(),
            Some(Intent::Calculate(e)) => format!("conta {e}"),
            Some(Intent::Antiderivative { expr, var }) => format!("primitiva {expr} d{var}"),
            Some(Intent::Definite {
                expr,
                var,
                from,
                to,
            }) => format!("integral {expr} d{var} {from}..{to}"),
            Some(Intent::Derivative { expr, var, at }) => format!("derivada {expr} d{var} {at:?}"),
            Some(Intent::Root { expr, var }) => format!("raiz {expr} {var}"),
        }
    }

    #[test]
    fn understands_the_way_people_write() {
        assert_eq!(
            shown("faz a integral de X3 -2x"),
            "primitiva x^3 - 2 * x dx"
        );
        assert_eq!(shown("Integral de x² - 2x dx"), "primitiva x^2 - 2 * x dx");
        assert_eq!(
            shown("integre sen(2x) de 0 a pi"),
            "integral sin(2 * x) dx 0..3.141592653589793"
        );
        assert_eq!(
            shown("qual a integral de x^2 entre 0 e 1?"),
            "integral x^2 dx 0..1"
        );
        assert_eq!(
            shown("calcule a derivada de x^2 sen(x)"),
            "derivada x^2 * sin(x) dx None"
        );
        assert_eq!(
            shown("derivada de 3x^2 + 1 no ponto 2"),
            "derivada 3 * x^2 + 1 dx Some(2.0)"
        );
        assert_eq!(
            shown("derive y^3 + x em relação a y"),
            "derivada y^3 + x dy None"
        );
        assert_eq!(shown("raiz de 144"), "conta sqrt(144)");
        assert_eq!(shown("raízes de x^2 - 2"), "raiz x^2 - 2 x");
        assert_eq!(shown("resolva x^2 - 2 = 0"), "raiz x^2 - 2 x");
        assert_eq!(shown("quanto é 2 mais 2?"), "conta 2 + 2");
        assert_eq!(shown("quanto é 3,5 vezes 2"), "conta 3.5 * 2");
        assert_eq!(shown("(12 + 8) * 3"), "conta (12 + 8) * 3");
        assert_eq!(shown("2(3 + 1)"), "conta 2 * (3 + 1)");
    }

    #[test]
    fn leaves_ordinary_conversation_alone() {
        for text in [
            "qual é a capital do Brasil?",
            "calcule o custo do projeto",
            "me fale sobre integral de linha",
            "resolva esse problema para mim",
            "a raiz do problema é outra",
            "quanto é o salário mínimo?",
            "a derivada do petróleo subiu",
            "integral de Riemann, o que é?",
            "em 2026 teve 3-1 no jogo",
            "CPF 123.456.789-09",
            "2026",
        ] {
            assert_eq!(shown(text), "nada", "{text}");
        }
    }
}
