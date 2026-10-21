use katsu_core::{fit, Adam, CharTokenizer, KatsuLmConfig, SplitMix64, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

// Texto original, escrito para os testes.
const TEXT: &str = "o sabiá canta na palmeira. a palmeira balança ao vento. \
o vento leva o canto do sabiá até o rio.";

fn tiny(vocab_size: usize) -> KatsuLmConfig {
    KatsuLmConfig {
        vocab_size,
        block_size: 8,
        embed_dim: 8,
        layers: 1,
        heads: 1,
    }
}

#[test]
fn tokenizer_round_trip_with_accents() -> Result<()> {
    let tok = CharTokenizer::from_text(TEXT);
    let ids = tok.encode("sabiá")?;
    assert_eq!(tok.decode(&ids), "sabiá");
    assert!(ids.iter().all(|&i| i < tok.vocab_size()));
    assert!(matches!(
        tok.encode("z"),
        Err(TensorError::IndexOutOfRange { .. })
    ));
    Ok(())
}

#[test]
fn shapes_and_parameter_count() -> Result<()> {
    let config = tiny(10);
    let params = config.init(7)?;
    assert_eq!(params.len(), config.param_tensors());
    let total: usize = params.iter().map(Tensor::numel).sum();
    assert_eq!(total, config.param_count());
    let logits = config.forward(&params, &[1, 2, 3])?;
    assert_eq!(logits.shape(), &[3, 10]);
    assert!(matches!(
        config.forward(&params, &[0; 9]),
        Err(TensorError::InvalidShape(_))
    ));
    assert!(matches!(
        config.forward(&params[1..], &[0]),
        Err(TensorError::InvalidShape(_))
    ));
    Ok(())
}

#[test]
fn initial_loss_is_close_to_uniform() -> Result<()> {
    let tok = CharTokenizer::from_text(TEXT);
    let ids = tok.encode(TEXT)?;
    let config = tiny(tok.vocab_size());
    let loss = config.loss(&config.init(1)?, &[&ids[..9], &ids[20..29]])?;
    let uniform = (tok.vocab_size() as f32).ln();
    assert!(
        (loss.data()[0] - uniform).abs() < 0.05,
        "{} vs {uniform}",
        loss.data()[0]
    );
    Ok(())
}

#[test]
fn future_tokens_do_not_change_past_logits() -> Result<()> {
    let config = tiny(10);
    let params = config.init(3)?;
    let a = config.forward(&params, &[1, 2, 3, 4])?;
    let b = config.forward(&params, &[1, 2, 9, 0])?;
    assert_eq!(a.data()[..20], b.data()[..20]); // posições 0 e 1 idênticas
    assert_ne!(a.data()[20..30], b.data()[20..30]);
    Ok(())
}

#[test]
fn gradient_of_one_weight_matches_finite_difference() -> Result<()> {
    let config = tiny(6);
    let params = config.init(5)?;
    let window: &[usize] = &[0, 3, 1, 4, 1, 5];
    config.loss(&params, &[window])?.backward()?;
    // Confere alguns pesos de WQ (índice 2+2) e da cabeça (último tensor).
    for (tensor, element) in [(4, 3), (4, 17), (params.len() - 1, 5)] {
        let analytical = params[tensor].grad()?.as_ref().unwrap()[element];
        let eval = |delta: f32| -> Result<f32> {
            let mut changed = params.clone();
            let mut data = params[tensor].data().to_vec();
            data[element] += delta;
            changed[tensor] = Tensor::new(data, params[tensor].shape())?;
            Ok(config.loss(&changed, &[window])?.data()[0])
        };
        let eps = 1e-2;
        let numerical = (eval(eps)? - eval(-eps)?) / (2.0 * eps);
        assert!(
            (analytical - numerical).abs() < 2e-3 + 2e-2 * numerical.abs(),
            "tensor {tensor}[{element}]: {analytical} vs {numerical}"
        );
    }
    Ok(())
}

#[test]
fn learns_to_continue_a_portuguese_sentence() -> Result<()> {
    let tok = CharTokenizer::from_text(TEXT);
    let ids = tok.encode(TEXT)?;
    let config = KatsuLmConfig {
        vocab_size: tok.vocab_size(),
        block_size: 16,
        embed_dim: 24,
        layers: 1,
        heads: 1,
    };
    let mut rng = SplitMix64(11);
    let mut losses = Vec::new();
    let params = fit(
        config.init(2)?,
        &mut Adam::new(1e-2)?,
        150,
        |p| {
            let windows: Vec<&[usize]> = (0..4)
                .map(|_| {
                    let start = rng.below(ids.len() - 17);
                    &ids[start..start + 17]
                })
                .collect();
            config.loss(p, &windows)
        },
        |m| losses.push(m.loss),
    )?;
    let first = losses[..10].iter().sum::<f32>() / 10.0;
    let last = losses[140..].iter().sum::<f32>() / 10.0;
    assert!(last < first * 0.35, "loss {first} -> {last}");
    // Prefixo sem ambiguidade no texto: só pode continuar com "ança".
    let prompt = tok.encode("a palmeira bal")?;
    let text = tok.decode(&config.generate(&params, &prompt, 4, 0.0, 0)?);
    assert!(text == "a palmeira balança", "{text:?}");
    Ok(())
}

#[test]
fn generation_is_reproducible_and_in_vocabulary() -> Result<()> {
    let config = tiny(10);
    let params = config.init(9)?;
    let a = config.generate(&params, &[1], 20, 1.0, 42)?;
    let b = config.generate(&params, &[1], 20, 1.0, 42)?;
    assert_eq!(a, b);
    assert_eq!(a.len(), 21);
    assert!(a.iter().all(|&t| t < 10));
    assert!(matches!(
        config.generate(&params, &[], 1, 1.0, 0),
        Err(TensorError::InvalidShape(_))
    ));
    Ok(())
}

#[test]
fn checkpoint_round_trip_and_rejects_garbage() -> Result<()> {
    let config = tiny(10);
    let params = config.init(4)?;
    let mut bytes = Vec::new();
    katsu_core::save_params(&params, &mut bytes).unwrap();
    let loaded = katsu_core::load_params(&mut bytes.as_slice()).unwrap();
    assert_eq!(loaded.len(), params.len());
    for (a, b) in params.iter().zip(&loaded) {
        assert_eq!((a.shape(), a.data()), (b.shape(), b.data()));
    }
    assert_eq!(
        config.forward(&params, &[1, 2])?.data(),
        config.forward(&loaded, &[1, 2])?.data()
    );
    assert!(katsu_core::load_params(&mut &bytes[..bytes.len() - 1]).is_err());
    assert!(katsu_core::load_params(&mut &b"NOTKATSU"[..]).is_err());
    Ok(())
}

fn multi_head(vocab_size: usize) -> KatsuLmConfig {
    KatsuLmConfig {
        vocab_size,
        block_size: 8,
        embed_dim: 8,
        layers: 2,
        heads: 4,
    }
}

#[test]
fn heads_must_divide_embed_dim() {
    let bad = KatsuLmConfig {
        heads: 3,
        ..multi_head(10)
    };
    assert!(matches!(bad.init(0), Err(TensorError::InvalidShape(_))));
    let zero = KatsuLmConfig {
        heads: 0,
        ..multi_head(10)
    };
    assert!(matches!(zero.init(0), Err(TensorError::InvalidShape(_))));
}

#[test]
fn multi_head_layout_count_and_causality() -> Result<()> {
    let config = multi_head(10);
    let params = config.init(8)?;
    assert_eq!(params.len(), config.param_tensors());
    // Mesmo total de números que uma cabeça: as cabeças dividem a largura.
    let total: usize = params.iter().map(Tensor::numel).sum();
    assert_eq!(total, config.param_count());
    assert_eq!(total, KatsuLmConfig { heads: 1, ..config }.param_count());
    assert_eq!(params[2 + 2].shape(), &[8, 2]); // wq da 1ª cabeça: d -> d/4
    assert_eq!(params[2 + 5].shape(), &[2, 8]); // wo da 1ª cabeça: d/4 -> d
    let a = config.forward(&params, &[1, 2, 3, 4])?;
    let b = config.forward(&params, &[1, 2, 3, 7])?;
    assert_eq!(a.data()[..30], b.data()[..30]);
    Ok(())
}

#[test]
fn multi_head_gradients_match_finite_differences() -> Result<()> {
    let config = multi_head(6);
    let params = config.init(13)?;
    let window: &[usize] = &[0, 3, 1, 4, 1, 5, 2];
    config.loss(&params, &[window])?.backward()?;
    let per_layer = 2 + 4 * 4 + 6;
    // wq da 3ª cabeça da 2ª camada, wo da 4ª cabeça da 1ª camada e W1 da 2ª camada.
    let wq = 2 + per_layer + 2 + 2 * 4;
    let wo = 2 + 2 + 3 * 4 + 3;
    let w1 = 2 + per_layer + 2 + 16 + 2;
    for (tensor, element) in [(wq, 5), (wo, 9), (w1, 30)] {
        let analytical = params[tensor].grad()?.as_ref().unwrap()[element];
        let eval = |delta: f32| -> Result<f32> {
            let mut changed = params.clone();
            let mut data = params[tensor].data().to_vec();
            data[element] += delta;
            changed[tensor] = Tensor::new(data, params[tensor].shape())?;
            Ok(config.loss(&changed, &[window])?.data()[0])
        };
        let eps = 1e-2;
        let numerical = (eval(eps)? - eval(-eps)?) / (2.0 * eps);
        assert!(
            (analytical - numerical).abs() < 2e-3 + 2e-2 * numerical.abs(),
            "tensor {tensor}[{element}]: {analytical} vs {numerical}"
        );
    }
    Ok(())
}
