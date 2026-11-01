//! GEMM em blocos (`sgemm`) contra o GEMM atual do backend de CPU, em matrizes quadradas.
//!   cargo run -p katu-core --release --example bench_gemm
//! Imprime a mediana de cada tamanho e os GFLOPS (2·n³ operações).
use std::time::Instant;

use katu_core::{sgemm, Backend, CpuBackend, Micro};

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.total_cmp(b));
    times[times.len() / 2]
}

fn time(runs: usize, mut f: impl FnMut()) -> f64 {
    f();
    median(
        (0..runs)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .collect(),
    )
}

fn main() {
    let backend = CpuBackend::default();
    println!(
        "microkernel: {}, threads: {}",
        Micro::detect().name(),
        rayon::current_num_threads()
    );
    println!(
        "{:>6} {:>12} {:>12} {:>10} {:>10}",
        "n", "atual (ms)", "sgemm (ms)", "GFLOPS", "ganho"
    );
    for n in [256usize, 512, 1024, 2048] {
        let a: Vec<f32> = (0..n * n)
            .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let b: Vec<f32> = (0..n * n)
            .map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5)
            .collect();
        let runs = if n >= 2048 { 5 } else { 15 };
        let old = time(runs, || {
            std::hint::black_box(backend.matmul(&a, &b, n, n, n).unwrap());
        });
        let mut c = vec![0.0f32; n * n];
        let new = time(runs, || {
            sgemm(&a, &b, &mut c, n, n, n);
            std::hint::black_box(&c);
        });
        let flops = 2.0 * (n as f64).powi(3);
        println!(
            "{n:>6} {:>12.2} {:>12.2} {:>10.0} {:>9.2}x",
            old * 1e3,
            new * 1e3,
            flops / new / 1e9,
            old / new
        );
    }
}
