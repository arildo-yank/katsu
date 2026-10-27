//! Decodificação restrita para leitura: todo número que o modelo escrever tem de aparecer na fonte
//! (o trecho e a pergunta). Corrige erros de cópia de dígitos ("1003" virando "1010", "60" virando
//! "6"); não impede o modelo de escolher outro número que esteja no trecho.
//!
//! Um número é uma sequência de dígitos ASCII; "6.000" são os números "6" e "000". A checagem é
//! feita no texto acumulado: cada número terminado precisa estar na fonte, e o número ainda aberto
//! no fim precisa ser começo de algum número da fonte.
use std::collections::HashSet;

pub struct NumberGuard {
    numbers: HashSet<String>,
    prefixes: HashSet<String>,
}

impl NumberGuard {
    pub fn new(source: &str) -> Self {
        let numbers: HashSet<String> = runs(source).into_iter().map(str::to_owned).collect();
        let prefixes = numbers
            .iter()
            .flat_map(|n| (1..=n.len()).map(move |i| n[..i].to_owned()))
            .collect();
        Self { numbers, prefixes }
    }

    /// `text` é aceitável como saída (completa ou parcial)?
    pub fn allows(&self, text: &str) -> bool {
        let open = text.ends_with(|c: char| c.is_ascii_digit());
        let all = runs(text);
        let (last, done) = match (open, all.split_last()) {
            (true, Some((last, done))) => (Some(*last), done),
            _ => (None, all.as_slice()),
        };
        done.iter().all(|n| self.numbers.contains(*n))
            && last.is_none_or(|n| self.prefixes.contains(n))
    }
}

fn runs(text: &str) -> Vec<&str> {
    text.split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .collect()
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
    fn thousands_separators_split_numbers() {
        let g = NumberGuard::new("vistos a 6.000 metros");
        assert!(g.allows("A 6.000 metros."));
        assert!(!g.allows("A 7.000 metros."));
    }
}
