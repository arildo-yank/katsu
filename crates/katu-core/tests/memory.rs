//! Operações que economizam memória no treino: precisam dar exatamente as mesmas contas
//! que o caminho sem economia.
use katu_core::{Adam, CpuBackend, KatuLmConfig, Optimizer, SplitMix64, Tensor, TensorError};
use std::sync::Arc;

type Result<T> = std::result::Result<T, TensorError>;

fn grad(t: &Tensor) -> Vec<f32> {
    t.grad().unwrap().as_ref().unwrap().clone()
}

fn random(rng: &mut SplitMix64, shape: &[usize]) -> Result<Tensor> {
    let count = shape.iter().product::<usize>();
    let data = (0..count)
        .map(|_| (rng.next_u64() % 2001) as f32 / 1000.0 - 1.0)
        .collect();
    Tensor::new(data, shape)
}

#[test]
fn fused_causal_attention_matches_the_unfused_graph_bit_for_bit() -> Result<()> {
    let (batch, t, dh) = (2, 5, 3);
    let mut rng = SplitMix64(11);
    let q = random(&mut rng, &[batch, t, dh])?;
    let k = random(&mut rng, &[batch, t, dh])?;
    let weights = random(&mut rng, &[batch, t, t])?;
    let backend: Arc<dyn katu_core::Backend> = Arc::new(CpuBackend::default());

    let fused = q.causal_attention_with_backend(&k, backend.clone())?;
    let reference = q
        .bmm_with_backend(&k.transpose()?, backend.clone())?
        .mul(&Tensor::scalar(1.0 / (dh as f32).sqrt())?)?
        .add_with_backend(&Tensor::causal_mask(t)?, backend.clone())?
        .softmax()?;
    assert_eq!(fused.shape(), reference.shape());
    assert_eq!(fused.data(), reference.data());

    fused.mul(&weights)?.sum()?.backward()?;
    let (fused_q, fused_k) = (grad(&q), grad(&k));
    q.zero_grad()?;
    k.zero_grad()?;
    reference.mul(&weights)?.sum()?.backward()?;
    assert_eq!(fused_q, grad(&q));
    assert_eq!(fused_k, grad(&k));
    assert!(q
        .causal_attention_with_backend(&random(&mut rng, &[batch, t, 4])?, backend)
        .is_err());
    Ok(())
}

#[test]
fn concat_routes_each_gradient_back_to_its_part() -> Result<()> {
    let a = Tensor::new(vec![1., 2., 3., 4.], &[2, 2])?;
    let b = Tensor::new(vec![5., 6.], &[2, 1])?;
    let side = Tensor::concat(&[a.clone(), b.clone()], 1)?;
    assert_eq!(side.shape(), &[2, 3]);
    assert_eq!(side.data(), &[1., 2., 5., 3., 4., 6.]);
    let weights = Tensor::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3])?;
    side.mul(&weights)?.sum()?.backward()?;
    assert_eq!(grad(&a), vec![1., 2., 4., 5.]);
    assert_eq!(grad(&b), vec![3., 6.]);

    let c = Tensor::new(vec![7., 8.], &[1, 2])?;
    a.zero_grad()?;
    let stacked = Tensor::concat(&[a.clone(), c.clone(), a.clone()], 0)?;
    assert_eq!(stacked.shape(), &[5, 2]);
    stacked.sum()?.backward()?;
    assert_eq!(grad(&a), vec![2.; 4]); // usada duas vezes
    assert_eq!(grad(&c), vec![1.; 2]);
    assert!(Tensor::concat(&[a.clone(), b], 0).is_err());
    assert!(Tensor::concat(&[], 1).is_err());
    Ok(())
}

fn model() -> Result<(KatuLmConfig, Vec<Tensor>, Vec<Vec<usize>>)> {
    let config = KatuLmConfig {
        vocab_size: 17,
        block_size: 6,
        embed_dim: 8,
        layers: 2,
        heads: 2,
    };
    let params = config.init(3)?;
    let windows = vec![vec![1, 4, 2, 9, 3, 7], vec![5, 5, 0, 16, 8, 2]];
    Ok((config, params, windows))
}

#[test]
fn backward_release_matches_backward_and_keeps_held_nodes() -> Result<()> {
    let (config, params, windows) = model()?;
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let backend: Arc<dyn katu_core::Backend> = Arc::new(CpuBackend::default());
    config
        .loss_batch_with_backend(&params, &refs, backend.clone())?
        .backward()?;
    let expected: Vec<Vec<f32>> = params.iter().map(grad).collect();
    for p in &params {
        p.zero_grad()?;
    }
    let loss = config.loss_batch_with_backend(&params, &refs, backend)?;
    // Um nó que o chamador ainda segura continua válido depois do backward.
    let held = loss.parents()[0].clone();
    let value = loss.data()[0];
    loss.backward_release()?;
    assert!(value.is_finite());
    assert!(held.data().iter().all(|x| x.is_finite()));
    let actual: Vec<Vec<f32>> = params.iter().map(grad).collect();
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn step_owned_matches_step_and_respects_shared_params() -> Result<()> {
    let (config, params, windows) = model()?;
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let (mut copying, mut owned) = (Adam::new(0.01)?, Adam::new(0.01)?);
    // Cópias independentes: clone() de Tensor compartilharia nós e gradientes.
    let copy = |p: &Tensor| Tensor::new(p.data().to_vec(), p.shape());
    let mut a = params.iter().map(copy).collect::<Result<Vec<_>>>()?;
    let mut b = params.iter().map(copy).collect::<Result<Vec<_>>>()?;
    drop(params);
    for _ in 0..3 {
        config.loss(&a, &refs)?.backward()?;
        a = copying.step(&a)?;
        config.loss(&b, &refs)?.backward()?;
        // Mantém um handle para o primeiro peso: ele não pode ser alterado no lugar.
        let shared = b[0].clone();
        let before = shared.data().to_vec();
        b = owned.step_owned(b)?;
        assert_eq!(shared.data(), before.as_slice());
    }
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.data(), y.data());
        assert_eq!(x.shape(), y.shape());
        assert!(y.grad()?.is_none());
    }
    // Sem gradiente, o passo é recusado antes de mexer nos momentos.
    assert!(matches!(
        owned.step_owned(b),
        Err(TensorError::MissingGradient { index: 0 })
    ));
    Ok(())
}

#[test]
fn fused_linear_and_layer_norm_match_the_unfused_graph_bit_for_bit() -> Result<()> {
    let mut rng = SplitMix64(29);
    let x = random(&mut rng, &[4, 6])?;
    let w = random(&mut rng, &[6, 5])?;
    let b = random(&mut rng, &[5])?;
    let (gamma, beta) = (random(&mut rng, &[6])?, random(&mut rng, &[6])?);
    let out_weights = random(&mut rng, &[4, 5])?;
    let norm_weights = random(&mut rng, &[4, 6])?;
    let backend: Arc<dyn katu_core::Backend> = Arc::new(CpuBackend::default());
    let leaves = [&x, &w, &b, &gamma, &beta];
    let grads = |leaves: &[&Tensor]| -> Vec<Vec<f32>> {
        leaves
            .iter()
            .map(|t| t.grad().unwrap().clone().unwrap_or_default())
            .collect()
    };

    let linear = x.linear_with_backend(&w, &b, backend.clone())?;
    let norm = x.layer_norm_affine(&gamma, &beta)?;
    linear
        .mul(&out_weights)?
        .sum()?
        .add(&norm.mul(&norm_weights)?.sum()?)?
        .backward()?;
    let fused = grads(&leaves);
    for leaf in leaves {
        leaf.zero_grad()?;
    }

    let linear_ref = x
        .matmul_with_backend(&w, backend.clone())?
        .add_with_backend(&b, backend.clone())?;
    let norm_ref = x
        .layer_norm()?
        .mul(&gamma)?
        .add_with_backend(&beta, backend)?;
    assert_eq!(linear.data(), linear_ref.data());
    assert_eq!(norm.data(), norm_ref.data());
    linear_ref
        .mul(&out_weights)?
        .sum()?
        .add(&norm_ref.mul(&norm_weights)?.sum()?)?
        .backward()?;
    assert_eq!(fused, grads(&leaves));
    assert!(x.layer_norm_affine(&b, &beta).is_err());
    Ok(())
}
