//! Perfil de um passo de pré-treino do `train_sft`, fase a fase, para planejar a aceleração.
//!
//!   cargo run --release -p katu-core --example profile_step -- [batch] [perfil] [passos] [backend]
//!   perfil: 134m (padrão; 12×768, 12 cabeças, contexto 256, vocab 32004) ou small
//!   backend: cpu (padrão, kernel próprio) ou accelerate (macOS, matmul pelo Accelerate)
//!
//! Repete as etapas de `update` em train_sft: forward + cross-entropy, backward, norma do
//! gradiente, Adam e checagem de pesos. O backend é a CPU embrulhada num contador que registra,
//! por método do `Backend`, chamadas, tempo, FLOPs e bytes de entrada/saída. Como o CudaBackend
//! recebe e devolve `&[f32]`/`Vec<f32>` do host, esses bytes são exatamente o que ele copiaria
//! entre CPU e GPU a cada passo.
use katu_core::{Adam, Backend, CpuBackend, KatuLmConfig, Optimizer, SplitMix64, TensorError};

type Result<T> = std::result::Result<T, TensorError>;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Default, Clone, Copy)]
struct Usage {
    calls: u64,
    bytes_in: u64,
    bytes_out: u64,
    flops: f64,
    time: Duration,
}

#[derive(Debug)]
struct Counting {
    inner: Arc<dyn Backend>,
    usage: Mutex<BTreeMap<&'static str, Usage>>,
}

impl Counting {
    fn record<T>(
        &self,
        name: &'static str,
        bytes_in: usize,
        flops: f64,
        run: impl FnOnce() -> Result<T>,
        out: impl Fn(&T) -> usize,
    ) -> Result<T> {
        let started = Instant::now();
        let value = run()?;
        let elapsed = started.elapsed();
        let mut usage = self.usage.lock().unwrap();
        let entry = usage.entry(name).or_default();
        entry.calls += 1;
        entry.bytes_in += bytes_in as u64;
        entry.bytes_out += out(&value) as u64;
        entry.flops += flops;
        entry.time += elapsed;
        Ok(value)
    }
    fn take(&self) -> BTreeMap<&'static str, Usage> {
        std::mem::take(&mut *self.usage.lock().unwrap())
    }
}

const F: usize = std::mem::size_of::<f32>();

impl Backend for Counting {
    fn name(&self) -> &'static str {
        "counting"
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        self.record(
            "matmul",
            (a.len() + b.len()) * F,
            2.0 * (m * k * n) as f64,
            || self.inner.matmul(a, b, m, k, n),
            |o| o.len() * F,
        )
    }
    fn matmul_backward(
        &self,
        a: &[f32],
        b: &[f32],
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.record(
            "matmul_backward",
            (a.len() + b.len() + g.len()) * F,
            4.0 * (m * k * n) as f64,
            || self.inner.matmul_backward(a, b, g, m, k, n),
            |(x, y)| (x.len() + y.len()) * F,
        )
    }
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.record(
            "add",
            (a.len() + b.len()) * F,
            a.len() as f64,
            || self.inner.add(a, b),
            |o| o.len() * F,
        )
    }
    fn add_backward(
        &self,
        g: &[f32],
        l: &[usize],
        r: &[usize],
        ll: usize,
        rl: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // O CudaBackend também sobe os dois mapas de índices (u64 por elemento).
        self.record(
            "add_backward",
            g.len() * F + (l.len() + r.len()) * 8,
            g.len() as f64,
            || self.inner.add_backward(g, l, r, ll, rl),
            |(x, y)| (x.len() + y.len()) * F,
        )
    }
    fn cross_entropy(&self, logits: &[f32], targets: &[usize], classes: usize) -> Result<f32> {
        self.record(
            "cross_entropy",
            logits.len() * F + targets.len() * 8,
            3.0 * logits.len() as f64,
            || self.inner.cross_entropy(logits, targets, classes),
            |_| F,
        )
    }
    fn cross_entropy_backward(
        &self,
        logits: &[f32],
        targets: &[usize],
        classes: usize,
        seed: f32,
    ) -> Result<Vec<f32>> {
        self.record(
            "cross_entropy_backward",
            logits.len() * F + targets.len() * 8,
            3.0 * logits.len() as f64,
            || {
                self.inner
                    .cross_entropy_backward(logits, targets, classes, seed)
            },
            |o| o.len() * F,
        )
    }
}

/// Memória física do processo, em MB: (atual, pico desde o início). É o "peak memory footprint"
/// do `/usr/bin/time -l` e inclui o que foi para o swap. Fora do macOS devolve None.
#[cfg(target_os = "macos")]
fn footprint() -> Option<(f64, f64)> {
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
    // fields[7] = ri_phys_footprint, fields[28] = ri_lifetime_max_phys_footprint.
    (status == 0).then(|| (info.fields[7] as f64 / 1e6, info.fields[28] as f64 / 1e6))
}

#[cfg(not(target_os = "macos"))]
fn footprint() -> Option<(f64, f64)> {
    None
}

fn mb(sample: Option<(f64, f64)>) -> String {
    sample.map_or_else(String::new, |(now, _)| format!("{now:.0} MB"))
}

fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let batch: usize = args.next().map_or(Ok(1), |a| a.parse())?;
    let profile = args.next().unwrap_or_else(|| "134m".into());
    let steps: usize = args.next().map_or(Ok(2), |a| a.parse())?;
    let inner: Arc<dyn Backend> = match args.next().as_deref().unwrap_or("cpu") {
        "cpu" => Arc::new(CpuBackend::default()),
        #[cfg(target_os = "macos")]
        "accelerate" => Arc::new(katu_core::AccelerateBackend::default()),
        other => return Err(format!("backend desconhecido: {other}").into()),
    };
    let config = match profile.as_str() {
        "134m" => KatuLmConfig {
            vocab_size: 32004,
            block_size: 256,
            embed_dim: 768,
            layers: 12,
            heads: 12,
        },
        "small" => KatuLmConfig {
            vocab_size: 32004,
            block_size: 256,
            embed_dim: 32,
            layers: 1,
            heads: 4,
        },
        other => return Err(format!("perfil desconhecido: {other}").into()),
    };
    let window = config.block_size + 1;
    let started = Instant::now();
    let mut params = config.init(1)?;
    println!(
        "perfil {profile}: {} parâmetros, batch {batch} × {} tokens, backend {}, init {:.1}s, threads rayon {}",
        config.param_count(),
        window - 1,
        inner.name(),
        secs(started.elapsed()),
        rayon::current_num_threads()
    );
    let counting = Arc::new(Counting {
        inner,
        usage: Mutex::default(),
    });
    let backend: Arc<dyn Backend> = counting.clone();
    let mut adam = Adam::new(1e-4)?;
    let mut rng = SplitMix64(7);
    // O primeiro passo inclui a criação dos momentos do Adam; o perfil usa o último.
    for step in 1..=steps {
        let windows: Vec<Vec<usize>> = (0..batch)
            .map(|_| {
                (0..window)
                    .map(|_| (rng.next_u64() % config.vocab_size as u64) as usize)
                    .collect()
            })
            .collect();
        let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
        let t0 = Instant::now();
        let loss = config.loss_batch_with_backend(&params, &refs, backend.clone())?;
        let forward = t0.elapsed();
        let after_forward = footprint();
        let forward_usage = counting.take();
        let t1 = Instant::now();
        // Como no train_sft: lê a perda e consome o grafo no backward, liberando as
        // ativações antes do Adam.
        let loss_value = loss.data()[0];
        loss.backward_release()?;
        let backward = t1.elapsed();
        let after_backward = footprint();
        let backward_usage = counting.take();
        let t2 = Instant::now();
        let mut norm = 0.0f64;
        for p in &params {
            let guard = p.grad()?;
            for &g in guard.as_ref().ok_or("missing gradient")? {
                norm += f64::from(g).powi(2);
            }
        }
        let grad_norm = t2.elapsed();
        let t3 = Instant::now();
        params = adam.step_owned(std::mem::take(&mut params))?;
        let adam_time = t3.elapsed();
        let after_adam = footprint();
        let t4 = Instant::now();
        let finite = params
            .iter()
            .all(|p| p.data().iter().all(|x| x.is_finite()));
        let check = t4.elapsed();
        let total = t0.elapsed();
        if step < steps {
            println!("passo {step} (aquecimento): {:.2}s", secs(total));
            continue;
        }
        let tokens = (batch * (window - 1)) as f64;
        println!(
            "\npasso {step}: loss {:.3}, |g| {:.3}, pesos finitos {finite}",
            loss_value,
            norm.sqrt()
        );
        println!("{:<26}{:>9}{:>8}{:>16}", "fase", "s", "%", "memória ao fim");
        for (name, d, memory) in [
            ("forward + cross-entropy", forward, mb(after_forward)),
            ("backward", backward, mb(after_backward)),
            ("norma do gradiente", grad_norm, String::new()),
            ("Adam (host)", adam_time, mb(after_adam)),
            ("checagem de pesos", check, String::new()),
        ] {
            println!(
                "{name:<26}{:>9.3}{:>7.1}%{memory:>16}",
                secs(d),
                100.0 * secs(d) / secs(total)
            );
        }
        println!(
            "{:<26}{:>9.3}   {:.0} tokens/s na CPU deste computador",
            "total",
            secs(total),
            tokens / secs(total)
        );
        println!(
            "\n{:<24}{:>7}{:>10}{:>10}{:>10}{:>9}",
            "chamada do Backend", "vezes", "GFLOP", "sobe MB", "desce MB", "s (CPU)"
        );
        let mut total_in = 0.0;
        let mut total_out = 0.0;
        let mut total_flops = 0.0;
        let mut total_calls = 0;
        let mut backend_time = Duration::ZERO;
        for (phase, usage) in [("fwd", forward_usage), ("bwd", backward_usage)] {
            for (name, u) in usage {
                println!(
                    "{:<24}{:>7}{:>10.1}{:>10.1}{:>10.1}{:>9.3}",
                    format!("{phase} {name}"),
                    u.calls,
                    u.flops / 1e9,
                    u.bytes_in as f64 / 1e6,
                    u.bytes_out as f64 / 1e6,
                    secs(u.time)
                );
                total_in += u.bytes_in as f64;
                total_out += u.bytes_out as f64;
                total_flops += u.flops;
                total_calls += u.calls;
                backend_time += u.time;
            }
        }
        let host_ops = secs(forward + backward) - secs(backend_time);
        println!(
            "{:<24}{:>7}{:>10.1}{:>10.1}{:>10.1}{:>9.3}",
            "total",
            total_calls,
            total_flops / 1e9,
            total_in / 1e6,
            total_out / 1e6,
            secs(backend_time)
        );
        println!("\nfora do Backend no forward+backward (LayerNorm, GELU, softmax, reshape, grafo): {host_ops:.3}s");
        if let Some((_, peak)) = footprint() {
            println!("pico de memória do processo (inclui swap): {peak:.0} MB");
        }
        println!("por token: {:.2} GFLOP, {:.1} MB copiados entre host e GPU, {:.0} chamadas de kernel por passo",
            total_flops / 1e9 / tokens, (total_in + total_out) / 1e6 / tokens, total_calls as f64);
    }
    Ok(())
}
