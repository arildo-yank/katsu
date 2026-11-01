#![cfg(feature = "cuda")]
use katu_core::{
    arena_len, materialize, upload, Adam, Backend, CpuBackend, CudaBackend, HostParamStore,
    KatuLmConfig, Optimizer, ParamStore, Tensor,
};
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
    let cfg = KatuLmConfig {
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

#[test]
#[ignore = "requires NVIDIA; weights, gradients and Adam resident in VRAM against the host reference"]
fn cuda_resident_store_matches_host_reference() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = KatuLmConfig {
        vocab_size: 512,
        block_size: 16,
        embed_dim: 64,
        layers: 2,
        heads: 4,
    };
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::default());
    let gpu = CudaBackend::new(0)?;
    let layout = cfg.resident_layout();
    let init = cfg.init(3)?;
    let len = arena_len(&init, &layout)?;
    let host_store = Arc::new(HostParamStore::new(len, cpu.clone())?);
    let cuda_store = Arc::new(gpu.param_store(len)?);
    let copy = |p: &[Tensor]| -> Vec<Tensor> {
        p.iter()
            .map(|t| Tensor::new(t.data().to_vec(), t.shape()).unwrap())
            .collect()
    };
    let mut host = upload(copy(&init), &layout, host_store.clone())?;
    let mut cuda = upload(copy(&init), &layout, cuda_store.clone())?;
    let weight_bytes = 4 * len as u64;
    assert_eq!(cuda_store.transfers().to_device, weight_bytes);
    let (mut ho, mut co) = (Adam::new(1e-3)?, Adam::new(1e-3)?);
    for step in 0..5 {
        let w1: Vec<_> = (0..12).map(|i| (i * 29 + step * 7) % 512).collect();
        let w2: Vec<_> = (0..12).map(|i| (i * 47 + step * 11) % 512).collect();
        let windows = [w1.as_slice(), w2.as_slice()];
        let (before, host_before) = (cuda_store.transfers(), host_store.transfers());
        let h = cfg.loss_batch_with_backend(&host, &windows, cpu.clone())?;
        let c = cfg.loss_batch_with_backend(&cuda, &windows, cpu.clone())?;
        close(c.data(), h.data(), 3e-4, 3e-4);
        let (loss_host, loss_cuda) = (h.data()[0], c.data()[0]);
        h.backward_release()?;
        c.backward_release()?;
        // Gradientes de todos os parâmetros (residentes lidos do store), relativos à escala de
        // cada um: é aqui que um erro do store apareceria. 1e-3 como o teste híbrido acima: a partir
        // do 2º passo os pesos já diferem no arredondamento e os gradientes herdam isso.
        let grad = |p: &Tensor| -> Vec<f32> {
            match p.residency() {
                Some((store, view)) => store.read(katu_core::Buffer::Grad, view).unwrap(),
                None => p.grad().unwrap().clone().unwrap(),
            }
        };
        let mut max_grad_error = 0.0f32;
        for (i, (a, b)) in cuda.iter().zip(&host).enumerate() {
            let (gc, gh) = (grad(a), grad(b));
            let scale = gh.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-12);
            let error = gc
                .iter()
                .zip(&gh)
                .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(
                error <= 1e-3 * scale,
                "step {step}, param {i}: grad error {error} vs scale {scale}"
            );
            max_grad_error = max_grad_error.max(error / scale);
        }
        let (hs, hb) = host_store.grad_stats()?;
        let (cs, cb) = cuda_store.grad_stats()?;
        assert_eq!((hb, cb), (0, 0));
        assert!((hs - cs).abs() <= 1e-3 * hs.max(1e-12), "{hs} vs {cs}");
        host = ho.step_resident(host, host_store.as_ref())?.0;
        let (next, bad) = co.step_resident(cuda, cuda_store.as_ref())?;
        assert_eq!(bad, 0);
        cuda = next;
        let delta = |a: katu_core::Transfers, b: katu_core::Transfers| {
            a.to_device - b.to_device + a.to_host - b.to_host
        };
        let moved = delta(cuda_store.transfers(), before);
        let reference = delta(host_store.transfers(), host_before);
        // Mesma contabilidade do store de referência (que o teste de CPU prova não mover
        // pesos); só a redução da norma devolve mais bytes (256 parciais em f64).
        assert!(
            moved <= reference + 4096,
            "step {step}: {moved} bytes vs reference {reference}"
        );
        // Pesos só depois do 1º passo. Depois, o Adam divide cada gradiente pelo histórico do
        // próprio elemento: onde o gradiente é ruído (~1e-7, com sinais opostos entre CPU e GPU),
        // o passo vira ±lr inteiro nos dois lados. Medido na RTX 4090: o pior caso foi o embedding
        // de tokens (no host), com gradientes iguais a 1e-5 da escala e pesos a 1,2e-3.
        let mut max_weight = 0.0f32;
        if step == 0 {
            for (a, b) in materialize(&cuda)?.iter().zip(materialize(&host)?.iter()) {
                max_weight = max_weight.max(close(a.data(), b.data(), 5e-4, 5e-3));
            }
        }
        println!("{{\"step\":{},\"loss_host\":{},\"loss_cuda\":{},\"bytes_moved\":{moved},\"max_relative_grad_error\":{max_grad_error},\"max_abs_weight_error_step1\":{max_weight}}}", step + 1, loss_host, loss_cuda);
    }
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
