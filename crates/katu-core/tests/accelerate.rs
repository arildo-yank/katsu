//! O backend Accelerate só muda a ordem das somas: resultados próximos aos do CpuBackend.
#![cfg(target_os = "macos")]
use katu_core::{AccelerateBackend, Backend, CpuBackend, KatuLmConfig, TensorError};
use std::sync::Arc;

type Result<T> = std::result::Result<T, TensorError>;

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 1e-4 + 1e-4 * b.abs(), "{a} vs {b}");
    }
}

#[test]
fn matmul_and_gradients_match_the_cpu_reference() -> Result<()> {
    let (accelerate, cpu) = (AccelerateBackend::default(), CpuBackend::default());
    for (m, k, n) in [(33, 37, 31), (1, 5, 1), (64, 8, 129)] {
        let a: Vec<_> = (0..m * k).map(|i| ((i % 19) as f32 - 9.) / 19.).collect();
        let b: Vec<_> = (0..k * n).map(|i| ((i % 23) as f32 - 11.) / 23.).collect();
        let g: Vec<_> = (0..m * n).map(|i| (i % 7) as f32 / 7.).collect();
        close(
            &accelerate.matmul(&a, &b, m, k, n)?,
            &cpu.matmul(&a, &b, m, k, n)?,
        );
        let (da, db) = accelerate.matmul_backward(&a, &b, &g, m, k, n)?;
        let (ra, rb) = cpu.matmul_backward(&a, &b, &g, m, k, n)?;
        close(&da, &ra);
        close(&db, &rb);
    }
    assert_eq!(accelerate.matmul(&[], &[], 2, 0, 3)?, vec![0.; 6]);
    assert!(accelerate.matmul(&[1.], &[1.], 2, 1, 1).is_err());
    Ok(())
}

#[test]
fn a_training_step_matches_the_cpu_reference() -> Result<()> {
    let config = KatuLmConfig {
        vocab_size: 31,
        block_size: 8,
        embed_dim: 16,
        layers: 2,
        heads: 4,
    };
    let windows = [
        vec![1, 4, 2, 9, 3, 7, 30, 0],
        vec![5, 5, 0, 16, 8, 2, 11, 12],
    ];
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let mut results = Vec::new();
    for backend in [
        Arc::new(AccelerateBackend::default()) as Arc<dyn Backend>,
        Arc::new(CpuBackend::default()),
    ] {
        let params = config.init(5)?;
        let loss = config.loss_batch_with_backend(&params, &refs, backend)?;
        let value = loss.data()[0];
        loss.backward_release()?;
        let grads: Vec<f32> = params
            .iter()
            .flat_map(|p| p.grad().unwrap().clone().unwrap())
            .collect();
        results.push((value, grads));
    }
    assert!((results[0].0 - results[1].0).abs() < 1e-5);
    close(&results[0].1, &results[1].1);
    Ok(())
}
