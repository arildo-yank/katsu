//! Treino Rust CPU ou CUDA híbrido experimental, com BPE e batches.
mod support;
use katu_core::{
    create_cuda_backend, save_params, Adam, Backend, BpeTokenizer, CpuBackend, KatuLmConfig,
    Optimizer, SplitMix64,
};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut options = HashMap::new();
    let mut describe = false;
    while let Some(flag) = args.next() {
        if flag == "--describe" {
            describe = true;
            continue;
        }
        if ![
            "--corpus",
            "--tokenizer",
            "--profile",
            "--batch-size",
            "--steps",
            "--backend",
            "--output",
            "--sequence-length",
            "--memory-budget-mib",
        ]
        .contains(&flag.as_str())
        {
            return Err(format!("unknown option {flag}").into());
        }
        options.insert(flag, args.next().ok_or("option needs a value")?);
    }
    let get =
        |key: &str, default: &str| options.get(key).cloned().unwrap_or_else(|| default.into());
    let backend: Arc<dyn Backend> = match get("--backend", "cpu").as_str() {
        "cpu" => Arc::new(CpuBackend::default()),
        "cuda" => create_cuda_backend(0)?,
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let tokenizer_path = options
        .get("--tokenizer")
        .ok_or("--tokenizer file.bpe is required")?;
    let tok = BpeTokenizer::load(&mut File::open(tokenizer_path)?)?;
    let profile = get("--profile", "tiny");
    let config = match profile.as_str() {
        "tiny" => KatuLmConfig {
            vocab_size: tok.vocab_size(),
            block_size: 16,
            embed_dim: 32,
            layers: 1,
            heads: 4,
        },
        "139m" => KatuLmConfig::native_139m(tok.vocab_size()),
        "349m" => KatuLmConfig::native_349m(tok.vocab_size()),
        _ => return Err("profile must be tiny, 139m or 349m".into()),
    };
    let batch: usize = get("--batch-size", "2").parse()?;
    let steps: usize = get("--steps", "10").parse()?;
    if batch == 0 || steps == 0 {
        return Err("batch and steps must be positive".into());
    }
    eprintln!(
        "backend={} cpu_kernel={} parameters={} weight_bytes={} batch={batch} context={}",
        backend.name(),
        CpuBackend::default().kernel_name(),
        config.param_count(),
        config.param_count() * 4,
        config.block_size
    );
    let sequence: usize = get("--sequence-length", &config.block_size.to_string()).parse()?;
    if sequence == 0 || sequence > config.block_size {
        return Err("sequence-length must be in 1..=block_size".into());
    }
    // Estimativa de planejamento: pesos/gradientes/Adam/cópias + ativações e logits.
    // Não é uma garantia de pico; não conta bibliotecas CUDA e memória do processo.
    let estimated = config.param_count() as u128 * 32
        + batch as u128 * sequence as u128 * config.embed_dim as u128 * config.layers as u128 * 64
        + batch as u128
            * config.heads as u128
            * sequence as u128
            * sequence as u128
            * config.layers as u128
            * 16
        + batch as u128 * sequence as u128 * config.vocab_size as u128 * 16;
    let budget: u128 = get("--memory-budget-mib", "1024").parse()?;
    eprintln!("estimated_host_bytes={estimated} memory_budget_mib={budget} train_sequence_length={sequence}");
    if describe {
        return Ok(());
    }
    if estimated > budget.saturating_mul(1024 * 1024) {
        return Err("estimated training memory exceeds --memory-budget-mib; use --describe and provision sufficient RAM".into());
    }
    let corpus = options.get("--corpus").ok_or("--corpus is required")?;
    let output = options
        .get("--output")
        .ok_or("--output must be a new directory")?;
    if Path::new(output).exists() {
        return Err("output exists; choose a new run directory".into());
    }
    let (train_text, valid_text, files) = support::corpus(Path::new(corpus))?;
    let train = tok.encode(&train_text);
    let valid = tok.encode(&valid_text);
    let window = sequence + 1;
    if train.len() < window || valid.len() < window {
        return Err("each split needs at least block_size+1 tokens".into());
    }
    fs::create_dir_all(output)?;
    fs::copy(tokenizer_path, Path::new(output).join("tokenizer.bpe"))?;
    let mut metrics = BufWriter::new(File::create(Path::new(output).join("metrics.jsonl"))?);
    let mut rng = SplitMix64(2026);
    let mut params = config.init(1)?;
    let mut adam = Adam::new(0.001)?;
    let started = std::time::Instant::now();
    let mut ema = 0.0;
    for step in 1..=steps {
        let windows: Vec<&[usize]> = (0..batch)
            .map(|_| {
                let i = rng.below(train.len() - window + 1);
                &train[i..i + window]
            })
            .collect();
        let loss = config.loss_batch_with_backend(&params, &windows, backend.clone())?;
        let value = loss.data()[0];
        if !value.is_finite() {
            return Err(format!("nonfinite loss at {step}").into());
        }
        loss.backward()?;
        let mut norm2 = 0.0_f64;
        let mut max_grad = 0.0_f32;
        for p in &params {
            let guard = p.grad()?;
            let grad = guard.as_ref().ok_or("missing gradient")?;
            for g in grad {
                if !g.is_finite() {
                    return Err(format!("nonfinite gradient at {step}").into());
                }
                norm2 += f64::from(*g).powi(2);
                max_grad = max_grad.max(g.abs());
            }
        }
        params = adam.step(&params)?;
        if params
            .iter()
            .any(|p| p.data().iter().any(|v| !v.is_finite()))
        {
            return Err(format!("nonfinite parameter at {step}").into());
        }
        ema = if step == 1 {
            value
        } else {
            0.98 * ema + 0.02 * value
        };
        writeln!(metrics, "{{\"epoch\":{step},\"loss\":{value},\"loss_ema\":{ema},\"grad_l2\":{},\"grad_max\":{max_grad},\"learning_rate\":0.001,\"elapsed_seconds\":{}}}", norm2.sqrt(), started.elapsed().as_secs_f64())?;
        metrics.flush()?;
        if step % 100 == 0 {
            eprintln!(
                "step={step} loss={value:.4} ema={ema:.4} grad_l2={:.4}",
                norm2.sqrt()
            );
        }
        if step % 500 == 0 {
            let path = Path::new(output).join("latest.ckpt.tmp");
            let mut file = BufWriter::new(File::create(&path)?);
            save_params(&params, &mut file)?;
            file.flush()?;
            fs::rename(path, Path::new(output).join("latest.ckpt"))?;
        }
    }
    let validation = config
        .loss_batch_with_backend(&params, &[&valid[..window]], backend)?
        .data()[0];
    let mut checkpoint = BufWriter::new(File::create(Path::new(output).join("model.ckpt"))?);
    save_params(&params, &mut checkpoint)?;
    checkpoint.flush()?;
    let metadata=format!("{{\"format\":\"Katu-native-rust-v1\",\"vocab_size\":{},\"block_size\":{},\"embed_dim\":{},\"layers\":{},\"heads\":{},\"parameters\":{},\"batch_size\":{},\"steps\":{},\"train_sequence_length\":{},\"seed\":1,\"sampling_seed\":2026,\"books\":{},\"validation_loss\":{}}}\n",config.vocab_size,config.block_size,config.embed_dim,config.layers,config.heads,config.param_count(),batch,steps,sequence,files.len(),validation);
    fs::write(Path::new(output).join("config.json"), metadata)?;
    eprintln!("saved {output}; validation_loss={validation:.5}");
    Ok(())
}
