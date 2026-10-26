use katsu_core::{BpeTokenizer, GenerationOptions, KatsuLmConfig, Tensor, INSTRUCTION_TOKENS};
#[test]
fn ids_are_atomic_fixed_roundtrip_and_legacy_is_preserved() {
    let mut tok =
        BpeTokenizer::train("Olá jogador brasileiro! Olá jogador brasileiro!", 300).unwrap();
    let base = tok.vocab_size();
    let before = tok.encode("Olá jogador!");
    let mut legacy = Vec::new();
    tok.save(&mut legacy).unwrap();
    assert_eq!(&legacy[..8], b"KATSUBP1");
    tok.reserve_instruction_tokens().unwrap();
    tok.reserve_instruction_tokens().unwrap();
    assert_eq!(tok.vocab_size(), base + 4);
    for (i, t) in INSTRUCTION_TOKENS.iter().enumerate() {
        assert_eq!(tok.encode(t), vec![base + i]);
        assert_eq!(tok.special_token_id(t), Some(base + i));
    }
    assert_eq!(before, tok.encode("Olá jogador!"));
    assert_ne!(tok.encode_ordinary("<|end|>"), tok.encode("<|end|>"));
    let text = "<|system|>Olá<|user|>{\"x\":2}<|assistant|>2<|end|>";
    assert_eq!(tok.decode(&tok.encode(text)).unwrap(), text);
    let mut bytes = Vec::new();
    tok.save(&mut bytes).unwrap();
    assert_eq!(&bytes[..8], b"KATSUBP2");
    assert_eq!(tok, BpeTokenizer::load(&mut bytes.as_slice()).unwrap());
    assert_eq!(
        BpeTokenizer::load(&mut legacy.as_slice())
            .unwrap()
            .vocab_size(),
        base
    );
}
#[test]
fn vocab_migration_preserves_head_strides_and_old_logits() {
    let c = KatsuLmConfig {
        vocab_size: 8,
        block_size: 4,
        embed_dim: 4,
        layers: 1,
        heads: 1,
    };
    let p = c.init(1).unwrap();
    let (n, q) = c.extend_vocab(&p, 12, 2).unwrap();
    assert_eq!(q[0].shape(), &[12, 4]);
    assert_eq!(q.last().unwrap().shape(), &[4, 12]);
    let old = c.forward(&p, &[1, 2]).unwrap();
    let new = n.forward(&q, &[1, 2]).unwrap();
    for i in 0..2 {
        assert_eq!(
            &old.data()[i * 8..i * 8 + 8],
            &new.data()[i * 12..i * 12 + 8]
        );
    }
    n.loss(&q, &[&[8, 1, 11]]).unwrap().backward().unwrap();
    assert!(q[0].grad().unwrap().as_ref().unwrap()[32..36]
        .iter()
        .any(|v| *v != 0.));
    assert!(c.extend_vocab(&p, 7, 1).is_err());
}
#[test]
fn real_end_id_stops_on_first_generated_token() {
    let mut tok = BpeTokenizer::train("abc abc", 256).unwrap();
    tok.reserve_instruction_tokens().unwrap();
    let end = tok.special_token_id("<|end|>").unwrap();
    let c = KatsuLmConfig {
        vocab_size: tok.vocab_size(),
        block_size: 4,
        embed_dim: 4,
        layers: 0,
        heads: 1,
    };
    let mut p = c.init(1).unwrap();
    // Force end as the unique argmax while using the real tokenizer and generation loop.
    p[2] = Tensor::zeros(&[4]).unwrap();
    p[3] = Tensor::new(vec![1.; 4], &[4]).unwrap();
    let last = p.len() - 1;
    let mut weights = vec![0.; 4 * c.vocab_size];
    for d in 0..4 {
        weights[d * c.vocab_size + end] = 1.;
    }
    p[last] = Tensor::new(weights, &[4, c.vocab_size]).unwrap();
    let ids = c
        .generate_with_options(
            &p,
            &[1],
            20,
            &GenerationOptions {
                temperature: 0.,
                repetition_penalty: 1.2,
                seed: 1,
                stop_tokens: vec![end],
            },
        )
        .unwrap();
    assert_eq!(ids, vec![1, end]);
}
#[test]
fn batch_encoder_matches_encode_ordinary_and_keeps_markers_literal() {
    let mut tok = BpeTokenizer::train(
        "O impedimento ocorre quando o atacante está à frente. O atacante recua.\n",
        340,
    )
    .unwrap();
    tok.reserve_instruction_tokens().unwrap();
    let mut encoder = tok.ordinary_encoder();
    for text in [
        "O impedimento ocorre quando o atacante está à frente.",
        "O atacante recua, e o impedimento ocorre de novo.\n\nOutro parágrafo.",
        "texto com <|end|> literal e emoji 🇧🇷",
        "",
    ] {
        // duas vezes: a segunda passa pelo cache de palavras
        assert_eq!(encoder.encode(text), tok.encode_ordinary(text));
        assert_eq!(encoder.encode(text), tok.encode_ordinary(text));
    }
    let end = tok.special_token_id("<|end|>").unwrap();
    assert!(!encoder.encode("a <|end|> b").contains(&end));
}
