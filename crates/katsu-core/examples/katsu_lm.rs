//! Treina o Katsu LM, modelo de linguagem de caracteres, em textos em português.
//!
//!   cargo run --release -p katsu-core --example katsu_lm -- <corpus> [metrics.jsonl] [passos] [v0|v1]
//!
//! - `<corpus>`: um arquivo .txt ou uma pasta; numa pasta, todos os .txt em ordem alfabética.
//! - Validação: os últimos 10% de cada obra, para todas as obras estarem representadas.
//! - Cada passo grava uma linha JSON (contrato do monitor arandu) com flush imediato.
//! - A cada 100 passos imprime a perda de validação e uma amostra de texto em stderr.
//! - Ao final salva os pesos em `<metrics>.ckpt` (ou `katsu-lm-<versão>.ckpt`) e o
//!   vocabulário em `<checkpoint>.vocab.txt`, necessário para gerar texto depois.
//! - Textos do Project Gutenberg têm o cabeçalho/rodapé da licença removidos.
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use katsu_core::{fit, save_params, Adam, CharTokenizer, KatsuLmConfig, SplitMix64};

const BATCH: usize = 8;
const VALID_WINDOWS: usize = 16;

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

fn corpus_files(path: &Path) -> io::Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = fs::read_dir(path)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<io::Result<_>>()?;
    files.retain(|f| f.extension().is_some_and(|e| e == "txt"));
    files.sort();
    Ok(files)
}

/// v0: o primeiro modelo (~116 mil parâmetros). v1: 4 camadas × 4 cabeças, contexto 96.
fn config(version: &str, vocab_size: usize) -> Result<KatsuLmConfig, String> {
    let (block_size, embed_dim, layers, heads) = match version {
        "v0" => (64, 64, 2, 1),
        "v1" => (96, 96, 4, 4),
        other => return Err(format!("versão desconhecida: {other} (use v0 ou v1)")),
    };
    Ok(KatsuLmConfig {
        vocab_size,
        block_size,
        embed_dim,
        layers,
        heads,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let corpus_path = args
        .next()
        .ok_or("uso: katsu_lm <corpus> [metrics.jsonl] [passos] [v0|v1]")?;
    let metrics_path = args.next();
    let steps: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(2000);
    let version = args.next().unwrap_or_else(|| "v1".to_string());

    let files = corpus_files(Path::new(&corpus_path))?;
    let books = files
        .iter()
        .map(|f| fs::read_to_string(f).map(|raw| clean(&raw)))
        .collect::<io::Result<Vec<_>>>()?;
    let tok = CharTokenizer::from_text(&books.concat());
    let (mut train, mut valid) = (Vec::new(), Vec::new());
    for book in &books {
        let ids = tok.encode(book)?;
        let split = ids.len() * 9 / 10;
        train.extend_from_slice(&ids[..split]);
        train.push(tok.encode("\n")?[0]);
        valid.extend_from_slice(&ids[split..]);
    }
    let config = config(&version, tok.vocab_size())?;
    let window = config.block_size + 1;
    if valid.len() <= window * VALID_WINDOWS || train.len() <= window {
        return Err("corpus pequeno demais".into());
    }
    eprintln!(
        "Katsu LM {version}: {} obras, {} caracteres, vocabulário {}, {} parâmetros \
         ({} camadas × {} cabeças, contexto {}), {steps} passos",
        books.len(),
        train.len() + valid.len(),
        tok.vocab_size(),
        config.param_count(),
        config.layers,
        config.heads,
        config.block_size,
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
                let val_windows: Vec<&[usize]> = (0..VALID_WINDOWS)
                    .map(|i| &valid[i * (valid.len() - window) / VALID_WINDOWS..][..window])
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
    let checkpoint =
        metrics_path.map_or(format!("katsu-lm-{version}.ckpt"), |m| format!("{m}.ckpt"));
    save_params(&params, &mut BufWriter::new(File::create(&checkpoint)?))?;
    // O vocabulário ordenado reconstrói o tokenizer: CharTokenizer::from_text(vocab).
    let vocab: Vec<usize> = (0..tok.vocab_size()).collect();
    fs::write(format!("{checkpoint}.vocab.txt"), tok.decode(&vocab))?;
    eprintln!(
        "pesos salvos em {checkpoint} · {:.1}s no total",
        started.elapsed().as_secs_f32()
    );
    Ok(())
}
