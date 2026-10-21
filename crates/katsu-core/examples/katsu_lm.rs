//! Treina o Katsu LM, modelo de linguagem de caracteres, em um texto em português.
//!
//!   cargo run --release -p katsu-core --example katsu_lm -- <corpus.txt> [metrics.jsonl] [passos]
//!
//! - Cada passo grava uma linha JSON (contrato do monitor arandu) com flush imediato.
//! - A cada 100 passos imprime a perda de validação e uma amostra de texto em stderr.
//! - Ao final salva os pesos em `<metrics>.ckpt` (ou `katsu-lm-v0.ckpt`).
//! - Textos do Project Gutenberg têm o cabeçalho/rodapé da licença removidos.
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    time::Instant,
};

use katsu_core::{fit, save_params, Adam, CharTokenizer, KatsuLmConfig, SplitMix64};

const BATCH: usize = 8;

/// Mantém só o texto da obra e normaliza espaços.
fn clean(raw: &str) -> String {
    let start = raw
        .find("*** START")
        .and_then(|i| raw[i..].find('\n').map(|j| i + j + 1))
        .unwrap_or(0);
    let end = raw.find("*** END").unwrap_or(raw.len());
    raw[start..end]
        .replace("\r\n", "\n")
        .split("\n\n")
        .map(|paragraph| paragraph.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|paragraph| !paragraph.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let corpus_path = args
        .next()
        .ok_or("uso: katsu_lm <corpus.txt> [metrics.jsonl] [passos]")?;
    let metrics_path = args.next();
    let steps: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(2000);

    let text = clean(&fs::read_to_string(&corpus_path)?);
    let tok = CharTokenizer::from_text(&text);
    let ids = tok.encode(&text)?;
    let split = ids.len() * 9 / 10;
    let (train, valid) = ids.split_at(split);
    let config = KatsuLmConfig {
        vocab_size: tok.vocab_size(),
        block_size: 64,
        embed_dim: 64,
        layers: 2,
    };
    let window = config.block_size + 1;
    if valid.len() <= window {
        return Err("corpus pequeno demais".into());
    }
    eprintln!(
        "corpus: {} caracteres, vocabulário {}, {} parâmetros, {steps} passos",
        ids.len(),
        tok.vocab_size(),
        config.param_count()
    );

    let mut out: Box<dyn Write> = match &metrics_path {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(io::stdout().lock()),
    };
    let mut rng = SplitMix64(2026);
    let mut step = 0;
    let mut write_error = None;
    let started = Instant::now();
    let params = fit(
        config.init(1)?,
        &mut Adam::new(3e-3)?,
        steps,
        |p| {
            let windows: Vec<&[usize]> = (0..BATCH)
                .map(|_| {
                    let start = rng.below(train.len() - window);
                    &train[start..start + window]
                })
                .collect();
            step += 1;
            // Avaliação e amostra periódicas, com os parâmetros desta época.
            if step % 100 == 0 || step == 1 {
                let val_windows: Vec<&[usize]> = (0..BATCH)
                    .map(|i| &valid[i * (valid.len() - window) / BATCH..][..window])
                    .collect();
                let val = config.loss(p, &val_windows)?.data()[0];
                let prompt = tok.encode("Capitu")?;
                let sample = config
                    .generate(p, &prompt, 80, 0.8, step as u64)
                    .map(|t| tok.decode(&t))
                    .unwrap_or_default();
                eprintln!(
                    "[passo {step:>5}] validação {val:.4} · {:.1}s\n  {sample:?}",
                    started.elapsed().as_secs_f32()
                );
            }
            config.loss(p, &windows)
        },
        |metrics| {
            if write_error.is_none() {
                write_error = writeln!(out, "{metrics}").and_then(|_| out.flush()).err();
            }
        },
    )?;
    if let Some(error) = write_error {
        return Err(error.into());
    }
    let checkpoint = metrics_path.map_or("katsu-lm-v0.ckpt".to_string(), |m| format!("{m}.ckpt"));
    save_params(&params, &mut BufWriter::new(File::create(&checkpoint)?))?;
    eprintln!(
        "pesos salvos em {checkpoint} · {:.1}s no total",
        started.elapsed().as_secs_f32()
    );
    Ok(())
}
