//! Arandu-Ve, etapa 0 (CPU): entrada por vetores, máscara de perda nas posições do prefixo,
//! congelamento de parâmetros no Adam e projetor. A paridade com a GPU fica em gpu_train.rs.
use katsu_core::{
    step_partial, Adam, CpuBackend, KatsuLmConfig, Optimizer, Projector, SftBatch, SplitMix64,
    Tensor, TensorError,
};
use std::sync::Arc;

type Result<T> = std::result::Result<T, TensorError>;

fn tiny() -> KatsuLmConfig {
    KatsuLmConfig {
        vocab_size: 11,
        block_size: 12,
        embed_dim: 8,
        layers: 1,
        heads: 2,
    }
}

fn cpu() -> Arc<dyn katsu_core::Backend> {
    Arc::new(CpuBackend::default())
}

fn random(rows: usize, cols: usize, seed: u64) -> Result<Tensor> {
    let mut rng = SplitMix64(seed);
    Tensor::new(
        (0..rows * cols).map(|_| rng.uniform() - 0.5).collect(),
        &[rows, cols],
    )
}

fn close(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol * (1.0 + y.abs()))
}

#[test]
fn prefix_equal_to_token_embeddings_gives_the_same_logits() -> Result<()> {
    let c = tiny();
    let p = c.init(1)?;
    // O prefixo é exatamente o embedding dos 3 primeiros tokens: os logits têm de bater com
    // o forward comum sobre os mesmos tokens, nas duas sequências do lote.
    let tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    let (b, n, m) = (2, 3, 2);
    let mut prefix_ids = Vec::new();
    let mut rest = Vec::new();
    for s in tokens.chunks(n + m) {
        prefix_ids.extend_from_slice(&s[..n]);
        rest.extend_from_slice(&s[n..]);
    }
    let prefix = p[0].gather_rows(&prefix_ids)?;
    let with_prefix = c.forward_with_prefix(&p, &prefix, n, &rest, b, cpu())?;
    let plain = c.forward_batch(&p, &tokens, b)?;
    assert_eq!(with_prefix.shape(), plain.shape());
    assert!(close(with_prefix.data(), plain.data(), 1e-6));
    Ok(())
}

#[test]
fn prefix_gradient_matches_finite_difference() -> Result<()> {
    let c = tiny();
    let p = c.init(2)?;
    let n = 2;
    let examples: Vec<(&[usize], usize)> = vec![(&[1, 2, 3, 4, 5], 2), (&[6, 7, 8], 1)];
    let prefix = random(examples.len() * n, c.embed_dim, 3)?;
    c.prefix_loss(&p, &prefix, n, &examples, cpu())?
        .backward()?;
    let grad = prefix.grad()?.as_ref().unwrap().clone();
    for element in [0, 5, 13, 30] {
        let eval = |delta: f32| -> Result<f32> {
            let mut data = prefix.data().to_vec();
            data[element] += delta;
            let shifted = Tensor::new(data, prefix.shape())?;
            Ok(c.prefix_loss(&p, &shifted, n, &examples, cpu())?.data()[0])
        };
        let eps = 1e-2;
        let numerical = (eval(eps)? - eval(-eps)?) / (2.0 * eps);
        assert!(
            (numerical - grad[element]).abs() < 2e-3,
            "prefix[{element}]: numérico {numerical}, analítico {}",
            grad[element]
        );
    }
    Ok(())
}

#[test]
fn loss_counts_only_answer_tokens_never_the_prefix() -> Result<()> {
    let c = tiny();
    let p = c.init(4)?;
    let n = 3;
    let ids: &[usize] = &[2, 7, 1, 8, 2];
    let start = 3;
    let prefix = random(n, c.embed_dim, 5)?;
    let loss = c
        .prefix_loss(&p, &prefix, n, &[(ids, start)], cpu())?
        .data()[0];
    // A mesma conta à mão: linhas n+start-1 .. n+len-2 dos logits, alvos ids[start..].
    let logits = c
        .forward_with_prefix(&p, &prefix, n, &ids[..ids.len() - 1], 1, cpu())?
        .reshape(&[n + ids.len() - 1, c.vocab_size])?;
    let rows: Vec<usize> = (n + start - 1..n + ids.len() - 1).collect();
    let manual = logits
        .gather_rows(&rows)?
        .cross_entropy(&ids[start..])?
        .data()[0];
    assert!((loss - manual).abs() < 1e-6);
    // Mexer só na resposta muda a perda; o prefixo não tem alvo nenhum na conta.
    let other: &[usize] = &[2, 7, 1, 9, 9];
    let changed = c
        .prefix_loss(&p, &prefix, n, &[(other, start)], cpu())?
        .data()[0];
    assert!((loss - changed).abs() > 1e-4);
    Ok(())
}

#[test]
fn frozen_parameters_stay_bit_identical_and_the_rest_moves_like_adam() -> Result<()> {
    let c = tiny();
    let p = c.init(6)?;
    let n = 2;
    let examples: Vec<(&[usize], usize)> = vec![(&[1, 2, 3, 4], 2)];
    let prefix = random(n, c.embed_dim, 7)?;
    c.prefix_loss(&p, &prefix, n, &examples, cpu())?
        .backward()?;
    // Congela tudo menos a cabeça de saída (último tensor) e o embedding de posição.
    let frozen: Vec<bool> = (0..p.len()).map(|i| i != p.len() - 1 && i != 1).collect();
    let mut adam = Adam::new(0.01)?;
    let stepped = step_partial(&mut adam, &p, &frozen)?;
    for (i, (before, after)) in p.iter().zip(&stepped).enumerate() {
        if frozen[i] {
            assert_eq!(before.data(), after.data(), "tensor congelado {i} mudou");
        } else {
            assert_ne!(
                before.data(),
                after.data(),
                "tensor treinável {i} não mudou"
            );
        }
    }
    // Os treináveis saem iguais a um Adam rodado só sobre eles.
    let mut alone = Adam::new(0.01)?;
    let reference = alone.step(&[p[1].clone(), p[p.len() - 1].clone()])?;
    assert_eq!(stepped[1].data(), reference[0].data());
    assert_eq!(stepped[p.len() - 1].data(), reference[1].data());
    assert_eq!(adam.moments().len(), 2, "momentos só dos treináveis");
    assert!(step_partial(&mut adam, &p, &frozen[1..]).is_err());
    Ok(())
}

#[test]
fn projector_learns_to_drive_a_frozen_language_model() -> Result<()> {
    let c = tiny();
    // Um modelo aleatório responde igual a qualquer entrada; então primeiro ele aprende uma
    // tarefa em que o primeiro token (1 ou 2) decide a continuação (4 ou 9)...
    let mut lm = c.init(8)?;
    let mut adam = Adam::new(0.05)?;
    let (a, b): (&[usize], &[usize]) = (&[1, 0, 4, 4, 4], &[2, 0, 9, 9, 9]);
    for _ in 0..200 {
        let loss = c.loss(&lm, &[a, b])?;
        loss.backward()?;
        lm = adam.step(&lm)?;
    }
    let frozen_lm: Vec<Vec<f32>> = lm.iter().map(|t| t.data().to_vec()).collect();
    // ...e depois, congelado, recebe no lugar desse token o vetor projetado de uma "imagem":
    // o projetor tem de aprender, sozinho, a levar cada imagem ao sinal da sua continuação.
    let n = 1;
    let features = random(2, 6, 9)?;
    let examples: Vec<(&[usize], usize)> = vec![(&[0, 4, 4, 4], 1), (&[0, 9, 9, 9], 1)];
    let mut projector = Projector::new(6, 16, c.embed_dim, 10)?;
    let mut adam = Adam::new(0.05)?;
    let mut losses = Vec::new();
    for _ in 0..120 {
        let prefix = projector.forward(&features)?;
        let loss = c.prefix_loss(&lm, &prefix, n, &examples, cpu())?;
        losses.push(loss.data()[0]);
        loss.backward()?;
        projector = Projector::from_params(adam.step(&projector.params())?)?;
    }
    // medido em 2026-10-27: 6,29 -> 0,0001
    assert!(
        losses[0] > 1.0 && *losses.last().unwrap() < 0.05,
        "perdas {:?}",
        (losses[0], losses.last())
    );
    for (before, after) in frozen_lm.iter().zip(&lm) {
        assert_eq!(
            before.as_slice(),
            after.data(),
            "o modelo de linguagem mudou"
        );
    }
    assert!(Projector::from_params(vec![projector.w1.clone()]).is_err());
    Ok(())
}

#[test]
fn prefix_inputs_are_validated() -> Result<()> {
    let c = tiny();
    let p = c.init(11)?;
    let prefix = random(2, c.embed_dim, 12)?;
    assert!(c
        .forward_with_prefix(&p, &prefix, 0, &[1, 2], 1, cpu())
        .is_err());
    assert!(c
        .forward_with_prefix(&p, &random(3, c.embed_dim, 1)?, 2, &[1, 2], 1, cpu())
        .is_err());
    assert!(
        c.forward_with_prefix(&p, &prefix, 2, &[1; 11], 1, cpu())
            .is_err(),
        "passa do block_size"
    );
    assert!(c
        .prefix_loss(&p, &prefix, 2, &[(&[1, 2], 0)], cpu())
        .is_err());
    Ok(())
}

#[test]
fn prefix_batch_puts_vectors_first_and_weights_only_the_answer() -> Result<()> {
    let (n, d) = (2, 3);
    let examples: Vec<(&[usize], usize)> = vec![(&[5, 6, 7, 8], 2), (&[9, 4, 3], 1)];
    let prefix: Vec<f32> = (0..examples.len() * n * d).map(|i| i as f32).collect();
    let b = SftBatch::with_prefix(&examples, &prefix, n, d)?;
    assert_eq!((b.batch, b.width, b.prefix_len), (2, n + 3, n));
    // linha 0: [pre, pre, 5, 6, 7]; alvos só nas posições da resposta (ids[2..] = 7, 8)
    assert_eq!(&b.tokens[..5], &[0, 0, 5, 6, 7]);
    assert_eq!(&b.targets[..5], &[0, 0, 0, 7, 8]);
    assert_eq!(&b.tokens[5..], &[0, 0, 9, 4, 0]);
    assert!(
        b.weights
            .iter()
            .enumerate()
            .all(|(i, &w)| i % 5 >= n || w == 0.0),
        "prefixo com peso"
    );
    assert!(
        (b.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6,
        "pesos somam 1"
    );
    assert_eq!(b.prefix, prefix);
    assert!(SftBatch::with_prefix(&examples, &prefix[1..], n, d).is_err());
    assert!(SftBatch::with_prefix(&examples, &prefix, 0, d).is_err());
    Ok(())
}
