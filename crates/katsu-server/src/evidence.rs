//! Conservative document mode: preserve the complete source, not just its numbers.
//! This guarantees a verbatim quotation, not relevance or factual correctness.
pub fn quote_or_verified_copy<'a>(raw: &str, source: &'a str) -> (&'a str, bool) {
    let normalize = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    (source, normalize(raw) != normalize(source))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn altered_numbers_and_claims_return_the_complete_source() {
        let source = "Prazo de cinco anos, até dois anos após a extinção. Art. 7º, XXIX.";
        for raw in [
            "O prazo é de oito anos.",
            "O prazo é de dez anos.",
            "O prazo é de 8 anos.",
            "Art. 7º, XVII.",
            "O prazo após a extinção é de cinco anos.",
            "dois anos",
            "Não tenho dados suficientes.",
            "",
            "Sem limite.",
        ] {
            assert_eq!(quote_or_verified_copy(raw, source), (source, true));
        }
    }

    #[test]
    fn exact_copy_is_not_a_model_correction() {
        let source = "Prazo de dois anos.";
        assert_eq!(quote_or_verified_copy(source, source), (source, false));
        assert_eq!(
            quote_or_verified_copy(" Prazo de  dois\nanos. ", source),
            (source, false)
        );
    }
}
