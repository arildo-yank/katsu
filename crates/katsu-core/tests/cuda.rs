#![cfg(feature = "cuda")]
use katsu_core::{Adam, Backend, CpuBackend, CudaBackend, KatsuLmConfig, Optimizer, Tensor};
use std::sync::Arc;
fn close(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let error = (a - b).abs();
        assert!(
            a.is_finite() && b.is_finite() && error <= atol + rtol * b.abs(),
            "index {i}: {a} vs {b}"
        );
        max = max.max(error);
    }
    max
}
#[test]
#[cfg(not(target_os = "linux"))]
fn cuda_unavailable_is_an_error() {
    assert!(CudaBackend::new(0).is_err());
}

#[test]
#[ignore = "requires a real Linux NVIDIA GPU and NVRTC; never silently skips"]
fn cuda_kernels_match_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let cpu = CpuBackend::serial();
    let gpu = CudaBackend::new(0)?;
    let (m, k, n) = (19, 23, 17);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 / 13.).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 17) as f32 / 17.).collect();
    let g = vec![0.25; m * n];
    close(
        &gpu.matmul(&a, &b, m, k, n)?,
        &cpu.matmul(&a, &b, m, k, n)?,
        2e-5,
        2e-5,
    );
    let (da, db) = gpu.matmul_backward(&a, &b, &g, m, k, n)?;
    let (ra, rb) = cpu.matmul_backward(&a, &b, &g, m, k, n)?;
    close(&da, &ra, 2e-5, 2e-5);
    close(&db, &rb, 2e-5, 2e-5);
    close(&gpu.add(&a, &a)?, &cpu.add(&a, &a)?, 1e-6, 1e-6);
    let left: Vec<_> = (0..g.len()).map(|i| i % 3).collect();
    let right: Vec<_> = (0..g.len()).map(|i| i % 7).collect();
    let (ga, gb) = gpu.add_backward(&g, &left, &right, 3, 7)?;
    let (ca, cb) = cpu.add_backward(&g, &left, &right, 3, 7)?;
    close(&ga, &ca, 2e-5, 2e-5);
    close(&gb, &cb, 2e-5, 2e-5);
    let logits = [1000., 999., 998., -5., 2., 4.];
    let labels = [1, 2];
    close(
        &[gpu.cross_entropy(&logits, &labels, 3)?],
        &[cpu.cross_entropy(&logits, &labels, 3)?],
        2e-5,
        2e-5,
    );
    close(
        &gpu.cross_entropy_backward(&logits, &labels, 3, 1.5)?,
        &cpu.cross_entropy_backward(&logits, &labels, 3, 1.5)?,
        2e-5,
        2e-5,
    );
    assert!(gpu.matmul(&[1.], &[1.], 2, 1, 1).is_err());
    assert!(gpu.kernel_launches() >= 7);
    Ok(())
}

#[test]
#[ignore = "requires NVIDIA; compares the same 2.06M architecture for ten Adam steps"]
fn cuda_ten_steps_match_cpu_gradients_and_weights() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = KatsuLmConfig {
        vocab_size: 32000,
        block_size: 16,
        embed_dim: 32,
        layers: 1,
        heads: 4,
    };
    assert_eq!(cfg.param_count(), 2_061_152);
    let mut cpu_params = cfg.init(1)?;
    let mut gpu_params = cfg.init(1)?;
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::default());
    let gpu = Arc::new(CudaBackend::new(0)?);
    let mut co = Adam::new(0.001)?;
    let mut go = Adam::new(0.001)?;
    for step in 0..10 {
        let w1: Vec<_> = (0..17).map(|i| (i * 29 + step * 7) % 32000).collect();
        let w2: Vec<_> = (0..17).map(|i| (i * 47 + step * 11) % 32000).collect();
        let windows = [w1.as_slice(), w2.as_slice()];
        let c = cfg.loss_batch_with_backend(&cpu_params, &windows, cpu.clone())?;
        let g = cfg.loss_batch_with_backend(&gpu_params, &windows, gpu.clone())?;
        close(g.data(), c.data(), 3e-4, 3e-4);
        c.backward()?;
        g.backward()?;
        let mut max_grad = 0.0f32;
        for (a, b) in gpu_params.iter().zip(&cpu_params) {
            max_grad = max_grad.max(close(
                a.grad()?.as_ref().unwrap(),
                b.grad()?.as_ref().unwrap(),
                3e-4,
                3e-3,
            ));
        }
        gpu_params = go.step(&gpu_params)?;
        cpu_params = co.step(&cpu_params)?;
        let mut max_weight = 0.0f32;
        for (a, b) in gpu_params.iter().zip(&cpu_params) {
            max_weight = max_weight.max(close(a.data(), b.data(), 5e-4, 5e-3));
        }
        println!("{{\"step\":{},\"loss_cpu\":{},\"loss_cuda\":{},\"max_abs_gradient_error\":{},\"max_abs_weight_error\":{}}}",step+1,c.data()[0],g.data()[0],max_grad,max_weight);
    }
    assert!(gpu.kernel_launches() > 100);
    Ok(())
}

// Check that host API shapes still reject malformed batches before launch.
#[test]
fn cpu_broadcast_contract_remains_valid() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::new(vec![1., 2., 3., 4.], &[2, 2])?;
    let b = Tensor::new(vec![5., 6.], &[2])?;
    a.add(&b)?.sum()?.backward()?;
    assert_eq!(b.grad()?.as_ref().unwrap(), &[2., 2.]);
    Ok(())
}
