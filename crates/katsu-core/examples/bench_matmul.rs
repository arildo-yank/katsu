use katsu_core::{Backend, CpuBackend};
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(256);
    let repeats = 5;
    let a: Vec<_> = (0..size * size).map(|i| (i % 17) as f32 / 17.).collect();
    let b: Vec<_> = (0..size * size).map(|i| (i % 13) as f32 / 13.).collect();
    for cpu in [
        CpuBackend::serial(),
        CpuBackend {
            parallel: false,
            simd: true,
        },
        CpuBackend::default(),
    ] {
        black_box(cpu.matmul(&a, &b, size, size, size)?);
        let start = Instant::now();
        for _ in 0..repeats {
            black_box(cpu.matmul(black_box(&a), black_box(&b), size, size, size)?);
        }
        println!(
            "shape={size}x{size} dtype=f32 kernel={} parallel={} pool={} mean_ms={:.3}",
            cpu.kernel_name(),
            cpu.parallel,
            rayon::current_num_threads(),
            start.elapsed().as_secs_f64() * 1000. / repeats as f64
        );
    }
    Ok(())
}
