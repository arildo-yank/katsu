use katsu_core::{
    Backend, BpeTokenizer, CpuBackend, GpuBackend, KatsuLmConfig, Tensor, TensorError,
};
use std::sync::Arc;
type Result<T> = std::result::Result<T, TensorError>;
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!((a - b).abs() < 2e-4 + 2e-4 * b.abs(), "{a} vs {b}");
    }
}
#[test]
fn parallel_simd_matches_scalar_with_tails_and_gradients() -> Result<()> {
    let (m, k, n) = (33, 37, 31);
    let a: Vec<_> = (0..m * k).map(|i| ((i % 19) as f32 - 9.) / 19.).collect();
    let b: Vec<_> = (0..k * n).map(|i| ((i % 23) as f32 - 11.) / 23.).collect();
    let g: Vec<_> = (0..m * n).map(|i| (i % 7) as f32 / 7.).collect();
    let reference = CpuBackend::serial();
    let accelerated = CpuBackend::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    pool.install(|| -> Result<()> {
        close(
            &reference.matmul(&a, &b, m, k, n)?,
            &accelerated.matmul(&a, &b, m, k, n)?,
        );
        let (da, db) = reference.matmul_backward(&a, &b, &g, m, k, n)?;
        let (pa, pb) = accelerated.matmul_backward(&a, &b, &g, m, k, n)?;
        close(&da, &pa);
        close(&db, &pb);
        Ok(())
    })
}
#[test]
fn kernels_validate_and_gpu_does_not_fallback() -> Result<()> {
    assert!(CpuBackend::default().matmul(&[1.], &[1.], 2, 1, 1).is_err());
    assert_eq!(
        CpuBackend::default().matmul(&[], &[], 2, 0, 3)?,
        vec![0.; 6]
    );
    assert!(matches!(
        GpuBackend.matmul(&[1.], &[1.], 1, 1, 1),
        Err(TensorError::BackendUnavailable(_))
    ));
    let a = Tensor::new(vec![1.], &[1, 1])?;
    assert!(a.matmul_with_backend(&a, Arc::new(GpuBackend)).is_err());
    Ok(())
}
#[test]
fn graph_is_send_sync_and_concurrent_backward_accumulates() -> Result<()> {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Tensor>();
    let x = Tensor::new(vec![2., 3.], &[2])?;
    let y = x.mul(&x)?.sum()?;
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let y = y.clone();
            std::thread::spawn(move || y.backward())
        })
        .collect();
    for h in handles {
        h.join().unwrap()?;
    }
    close(x.grad()?.as_ref().unwrap(), &[32., 48.]);
    Ok(())
}
#[test]
fn gather_scatter_adds_repeated_indices() -> Result<()> {
    let table = Tensor::new(vec![1., 2., 3., 4., 5., 6.], &[3, 2])?;
    let selected = table.gather_rows(&[2, 0, 2])?;
    assert_eq!(selected.data(), &[5., 6., 1., 2., 5., 6.]);
    selected.sum()?.backward()?;
    assert_eq!(table.grad()?.as_ref().unwrap(), &[1., 1., 0., 0., 2., 2.]);
    assert!(table.gather_rows(&[3]).is_err());
    Ok(())
}
#[test]
fn bmm_isolates_batches_and_differentiates() -> Result<()> {
    let a = Tensor::new((0..12).map(|i| i as f32 / 10.).collect(), &[2, 2, 3])?;
    let b = Tensor::new((0..12).map(|i| i as f32 / 7.).collect(), &[2, 3, 2])?;
    let y = a.bmm(&b)?;
    y.sum()?.backward()?;
    for batch in 0..2 {
        let aa = Tensor::new(a.data()[batch * 6..batch * 6 + 6].to_vec(), &[2, 3])?;
        let bb = Tensor::new(b.data()[batch * 6..batch * 6 + 6].to_vec(), &[3, 2])?;
        let out = aa.matmul(&bb)?;
        out.sum()?.backward()?;
        close(&y.data()[batch * 4..batch * 4 + 4], out.data());
        close(
            &a.grad()?.as_ref().unwrap()[batch * 6..batch * 6 + 6],
            aa.grad()?.as_ref().unwrap(),
        );
        close(
            &b.grad()?.as_ref().unwrap()[batch * 6..batch * 6 + 6],
            bb.grad()?.as_ref().unwrap(),
        );
    }
    assert!(a.bmm(&Tensor::zeros(&[1, 3, 2])?).is_err());
    Ok(())
}
#[test]
fn batched_lm_matches_separate_windows_and_gradients() -> Result<()> {
    let cfg = KatsuLmConfig {
        vocab_size: 12,
        block_size: 4,
        embed_dim: 8,
        layers: 1,
        heads: 2,
    };
    let batch = cfg.init(7)?;
    let serial = cfg.init(7)?;
    let w1 = &[1, 2, 3, 4][..];
    let w2 = &[7, 8, 9, 10][..];
    let loss = cfg.loss(&batch, &[w1, w2])?;
    let a = cfg.forward(&serial, &w1[..3])?.cross_entropy(&w1[1..])?;
    let b = cfg.forward(&serial, &w2[..3])?.cross_entropy(&w2[1..])?;
    let reference = a.add(&b)?.mul(&Tensor::scalar(0.5)?)?;
    close(loss.data(), reference.data());
    loss.backward()?;
    reference.backward()?;
    for (p, q) in batch.iter().zip(&serial) {
        close(p.grad()?.as_ref().unwrap(), q.grad()?.as_ref().unwrap());
    }
    let logits = cfg.forward_batch(&batch, &[1, 2, 3, 7, 8, 9], 2)?;
    assert_eq!(logits.shape(), &[2, 3, 12]);
    let changed = cfg.forward_batch(&batch, &[1, 2, 3, 0, 0, 0], 2)?;
    close(&logits.data()[..36], &changed.data()[..36]);
    assert!(cfg.forward_batch(&batch, &[1, 2, 3], 2).is_err());
    Ok(())
}
#[test]
fn profiles_count_without_allocating_weights() {
    let small = KatsuLmConfig::native_139m(30_000);
    let large = KatsuLmConfig::native_349m(34_816);
    assert!((138_000_000..140_000_000).contains(&small.param_count()));
    assert!((348_000_000..350_000_000).contains(&large.param_count()));
    assert_eq!(small.embed_dim % small.heads, 0);
    assert_eq!(large.embed_dim % large.heads, 0);
}
#[test]
fn bpe_is_deterministic_and_handles_unseen_utf8() -> std::io::Result<()> {
    let corpus = "ação brasileira futebol prontidão ação brasileira futebol prontidão ".repeat(20);
    let tok = BpeTokenizer::train(&corpus, 320)?;
    assert!(tok.vocab_size() > 256);
    assert!(tok.encode(&corpus).len() < corpus.len());
    assert_eq!(tok, BpeTokenizer::train(&corpus, 320)?);
    for text in [&corpus, "⚽ amanhã: café! 新\n\t", ""] {
        assert_eq!(tok.decode(&tok.encode(text))?, text);
    }
    let mut bytes = Vec::new();
    tok.save(&mut bytes)?;
    assert_eq!(BpeTokenizer::load(&mut bytes.as_slice())?, tok);
    assert!(BpeTokenizer::load(&mut &bytes[..bytes.len() - 1]).is_err());
    assert!(BpeTokenizer::train("texto", 255).is_err());
    assert!(tok.decode(&[tok.vocab_size()]).is_err());
    Ok(())
}
#[test]
fn bpe_rejects_forward_merge_references() {
    let mut bytes = b"KATSUBP1".to_vec();
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(999u32.to_le_bytes());
    bytes.extend(1u32.to_le_bytes());
    assert!(BpeTokenizer::load(&mut bytes.as_slice()).is_err());
}

#[test]
fn fused_cross_entropy_matches_composite_and_gradient() -> Result<()> {
    let data = vec![1000., 999., 998., -3., 2., 4.];
    let fast = Tensor::new(data.clone(), &[2, 3])?;
    let reference = Tensor::new(data, &[2, 3])?;
    let loss = fast.cross_entropy(&[1, 2])?;
    let old = reference
        .log_softmax()?
        .mul(&Tensor::one_hot(&[1, 2], 3)?)?
        .sum()?
        .mul(&Tensor::scalar(-0.5)?)?;
    close(loss.data(), old.data());
    loss.backward()?;
    old.backward()?;
    close(
        fast.grad()?.as_ref().unwrap(),
        reference.grad()?.as_ref().unwrap(),
    );
    assert!(fast.cross_entropy(&[3, 0]).is_err());
    let empty = Tensor::zeros(&[0, 3])?;
    let empty_loss = empty.cross_entropy(&[])?;
    assert_eq!(empty_loss.data(), &[0.]);
    empty_loss.backward()?;
    Ok(())
}
