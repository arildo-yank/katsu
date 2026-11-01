//! Leitura com fonte: duas checagens contra o trecho e a pergunta.
//!
//! - `NumberGuard`, durante a geração: todo número que o modelo escrever tem de aparecer na fonte.
//!   Corrige erros de cópia de dígitos ("1003" virando "1010", "60" virando "6"); não impede o
//!   modelo de escolher outro número que esteja no trecho.
//! - `support`, depois da geração: que parte das palavras de conteúdo da resposta está no trecho.
//!   Resposta com palavras que não vêm da fonte (nome trocado, país trocado, frase inventada) vira
//!   recusa no servidor.
//!
//! Um número é uma sequência de dígitos ASCII, com `.` ou `,` entre dígitos: "6.000" e "59,8" são um
//! número só. A checagem é feita no texto acumulado: cada número terminado precisa estar na fonte, e
//! o número ainda aberto no fim precisa ser começo de algum número da fonte.
use std::collections::HashSet;

pub struct NumberGuard {
    numbers: HashSet<String>,
    prefixes: HashSet<String>,
}

impl NumberGuard {
    pub fn new(source: &str) -> Self {
        let numbers: HashSet<String> = runs(source)
            .into_iter()
            .map(|(a, b)| source[a..b].to_owned())
            .collect();
        let prefixes = numbers
            .iter()
            .flat_map(|n| (1..=n.len()).map(move |i| n[..i].to_owned()))
            .collect();
        Self { numbers, prefixes }
    }

    /// `text` é aceitável como saída (completa ou parcial)?
    pub fn allows(&self, text: &str) -> bool {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let all = runs(text);
        // Número aberto: o texto termina nele, ou num separador logo depois dele ("6." pode virar
        // "6.000" ou ser o ponto final de "6").
        let open = match all.last() {
            Some(&(a, b)) if b == n => Some((a, b, false)),
            Some(&(a, b)) if b + 1 == n && is_sep(bytes[b]) => Some((a, b, true)),
            _ => None,
        };
        let done = &all[..all.len() - usize::from(open.is_some())];
        done.iter()
            .all(|&(a, b)| self.numbers.contains(&text[a..b]))
            && open.is_none_or(|(a, b, sep)| {
                if sep {
                    self.prefixes.contains(&text[a..=b]) || self.numbers.contains(&text[a..b])
                } else {
                    self.prefixes.contains(&text[a..b])
                }
            })
    }
}

fn is_sep(c: u8) -> bool {
    c == b'.' || c == b','
}

/// Intervalos (início, fim) em bytes de cada número.
fn runs(text: &str) -> Vec<(usize, usize)> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        loop {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < b.len() && is_sep(b[i]) && b[i + 1].is_ascii_digit() {
                i += 1;
                continue;
            }
            break;
        }
        out.push((start, i));
    }
    out
}

/// Palavras de moldura das respostas ("Ele nasceu em…", "Fica no…"): não contam como conteúdo.
const FRAME: &[&str] = &[
    "ele",
    "ela",
    "eles",
    "elas",
    "este",
    "esta",
    "estes",
    "estas",
    "esse",
    "essa",
    "isso",
    "isto",
    "tem",
    "possui",
    "fica",
    "ficam",
    "localiza",
    "localizada",
    "localizado",
    "situada",
    "situado",
    "sao",
    "trecho",
    "segundo",
    "chamado",
    "chamada",
    "conhecido",
    "conhecida",
    "nome",
    "tambem",
    "nasceu",
    "nascido",
    "nascida",
    "morreu",
    "faleceu",
    "realizado",
    "realizada",
    "ocorreu",
];

/// Fração das palavras de conteúdo da resposta (fora as da pergunta e as de moldura) que aparecem
/// no trecho; 1 quando não sobra nenhuma.
pub fn support(passage: &str, question: &str, answer: &str) -> f32 {
    let source: HashSet<String> = katu_rag::terms(passage).into_iter().collect();
    let asked: HashSet<String> = katu_rag::terms(question).into_iter().collect();
    let words: Vec<String> = katu_rag::terms(answer)
        .into_iter()
        .filter(|w| !asked.contains(w) && !FRAME.contains(&w.as_str()))
        .collect();
    if words.is_empty() {
        return 1.0;
    }
    words.iter().filter(|w| source.contains(*w)).count() as f32 / words.len() as f32
}

/// Trecho e pergunta de um prompt no formato de leitura ("Trecho: …\nPergunta: …").
pub fn reading_parts(prompt: &str) -> Option<(&str, &str)> {
    let rest = &prompt[prompt.find("Trecho: ")? + "Trecho: ".len()..];
    let (passage, rest) = rest.split_once("\nPergunta: ")?;
    let question = rest.split("\n<|assistant|>").next()?;
    Some((passage, question))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_must_come_from_the_source() {
        let g = NumberGuard::new("Roma, 12 de maio de 1003. Pergunta: Em que ano morreu?");
        assert!(g.allows("Morreu em 1003."));
        assert!(g.allows("Morreu em 10"), "começo de 1003, ainda aberto");
        assert!(
            !g.allows("Morreu em 101"),
            "nenhum número da fonte começa com 101"
        );
        assert!(!g.allows("Morreu em 1010."));
        assert!(!g.allows("Morreu em 10."), "10 terminado não está na fonte");
        assert!(g.allows("Não tenho dados suficientes."));
    }

    #[test]
    fn a_shorter_number_is_not_a_copy() {
        let g = NumberGuard::new("estabeleceu um total de 60 empresas em mais de 100 escritórios");
        assert!(!g.allows("A TAG-Org estabeleceu 6 empresas."));
        assert!(g.allows("A TAG-Org estabeleceu 6"), "pode ainda virar 60");
        assert!(g.allows("A TAG-Org estabeleceu 60 empresas."));
    }

    #[test]
    fn numbers_with_separators_are_one_number() {
        let g = NumberGuard::new("vistos a 6.000 metros, numa área de 59,8 km²");
        assert!(g.allows("A 6.000 metros."));
        assert!(g.allows("A 6."), "6. ainda pode virar 6.000");
        assert!(!g.allows("A 6. 000 metros."), "6 sozinho não está na fonte");
        assert!(!g.allows("A 6 metros."));
        assert!(!g.allows("A 7.000 metros."));
        assert!(g.allows("Tem 59,8 km²."));
        assert!(!g.allows("Tem 59 km²."));
    }

    #[test]
    fn a_number_can_end_a_sentence() {
        let g = NumberGuard::new("Morreu em 1003");
        assert!(g.allows("Morreu em 1003."));
        assert!(g.allows("Morreu em 1003, em Roma."));
    }

    #[test]
    fn support_counts_content_words_from_the_passage() {
        let passage =
            "Silvestre II (Auvérnia, 946 — Roma, 1003), nome de batismo Gerbert d'Aurillac.";
        let q = "Qual era o nome de batismo de Silvestre II?";
        assert_eq!(
            support(passage, q, "O nome de batismo era Gerbert d'Aurillac."),
            1.0
        );
        assert!(support(passage, q, "Era Helena Lobo, uma monja francesa.") < 0.5);
        assert_eq!(
            support(passage, "Onde nasceu?", "Ele nasceu em Auvérnia."),
            1.0
        );
    }

    #[test]
    fn splits_reading_prompts() {
        let p = "<|system|>S\n<|user|>Trecho: Um texto.\nPergunta: Qual?\n<|assistant|>";
        assert_eq!(reading_parts(p), Some(("Um texto.", "Qual?")));
        assert_eq!(reading_parts("<|user|>Oi\n<|assistant|>"), None);
    }
}
