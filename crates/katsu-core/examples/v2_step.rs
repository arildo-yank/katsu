//! Passos de treino medidos para a V2: tempo por passo, pico de memória e perda.
//!
//!   cargo run --release -p katsu-core --example v2_step -- \
//!       [precisao] [d] [camadas] [cabecas] [vocab] [contexto] [batch] [passos]
//!   precisao: fp32 (padrão) ou bf16 (precisão mista: operandos das matmuls em BF16,
//!   acumulação, pesos-mestre e Adam em FP32)
//!   padrão: o 134M (768, 12 camadas, 12 cabeças, vocab 32004), contexto 256, batch 1, 5 passos
//!
//! Repete sempre o mesmo lote (tokens aleatórios, semente fixa): a perda precisa cair, e duas
//! execuções com precisões diferentes partem da mesma inicialização e veem os mesmos dados.
//! Rode cada precisão num processo próprio: o pico de memória é o do processo inteiro.
//! A última linha é um JSON para scripts.
use std::sync::Arc;
use std::time::Instant;

use katsu_core::{Adam, Backend, Bf16Backend, KatsuLmConfig, Optimizer, SplitMix64};

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
fn peak_footprint_mb() -> Option<f64> {
    #[repr(C)]
    struct RusageInfoV4 {
        uuid: [u8; 16],
        fields: [u64; 35],
    }
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
    }
    let mut info = RusageInfoV4 {
        uuid: [0; 16],
        fields: [0; 35],
    };
    // SAFETY: flavor 4 (RUSAGE_INFO_V4) preenche exatamente esta struct; o pid é o próprio processo.
    let status = unsafe { proc_pid_rusage(std::process::id() as i32, 4, &mut info) };
    // fields[28] = ri_lifetime_max_phys_footprint (inclui memória comprimida e swap).
    (status == 0).then(|| info.fields[28] as f64 / 1e6)
}

#[cfg(not(target_os = "macos"))]
fn peak_footprint_mb() -> Option<f64> {
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize, default: usize| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(args.get(i).map_or(Ok(default), |a| a.parse())?)
    };
    let precision = args.first().map_or("fp32", String::as_str).to_string();
    let config = KatsuLmConfig {
        embed_dim: arg(1, 768)?,
        layers: arg(2, 12)?,
        heads: arg(3, 12)?,
        vocab_size: arg(4, 32004)?,
        block_size: arg(5, 256)?,
    };
    let (batch, steps) = (arg(6, 1)?, arg(7, 5)?);
    #[cfg(target_os = "macos")]
    let inner: Arc<dyn Backend> = Arc::new(katsu_core::AccelerateBackend::default());
    #[cfg(not(target_os = "macos"))]
    let inner: Arc<dyn Backend> = Arc::new(katsu_core::CpuBackend::default());
    let backend: Arc<dyn Backend> = match precision.as_str() {
        "fp32" => inner,
        "bf16" => Arc::new(Bf16Backend::new(inner)),
        other => return Err(format!("precisão desconhecida: {other}").into()),
    };
    let mut rng = SplitMix64(7);
    let windows: Vec<Vec<usize>> = (0..batch)
        .map(|_| {
            (0..config.block_size + 1)
                .map(|_| (rng.next_u64() % config.vocab_size as u64) as usize)
                .collect()
        })
        .collect();
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let mut params = config.init(1)?;
    let mut adam = Adam::new(5e-4)?;
    println!(
        "{} parâmetros, {precision}, backend {}, batch {batch} × {} tokens",
        config.param_count(),
        backend.name(),
        config.block_size
    );
    let (mut losses, mut seconds) = (Vec::new(), Vec::new());
    for step in 1..=steps {
        let started = Instant::now();
        let loss = config.loss_batch_with_backend(&params, &refs, backend.clone())?;
        let value = loss.data()[0];
        loss.backward_release()?;
        params = adam.step_owned(std::mem::take(&mut params))?;
        let elapsed = started.elapsed().as_secs_f64();
        println!("passo {step}: perda {value:.4}, {elapsed:.2} s");
        losses.push(value);
        seconds.push(elapsed);
    }
    // O primeiro passo cria os momentos do Adam: a média de tempo usa os demais.
    let warm = &seconds[seconds.len().min(1)..];
    let mean = warm.iter().sum::<f64>() / warm.len().max(1) as f64;
    let peak = peak_footprint_mb().unwrap_or(f64::NAN);
    println!("pico de memória do processo: {peak:.0} MB; média por passo (sem o 1º): {mean:.2} s");
    println!(
        "{{\"precision\":\"{precision}\",\"params\":{},\"batch\":{batch},\"context\":{},\"losses\":{losses:?},\"seconds\":{seconds:?},\"mean_step_s\":{mean:.3},\"peak_mb\":{peak:.0}}}",
        config.param_count(),
        config.block_size
    );
    Ok(())
}
