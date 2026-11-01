//! cargo run --release --example train_bpe -- corpus output.bpe 32000
mod support;
use katu_core::BpeTokenizer;
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("usage: train_bpe <corpus> <output.bpe> [vocab=32000]".into());
    }
    let (train, valid, files) = support::corpus(Path::new(&args[0]))?;
    let target: usize = args.get(2).map(|v| v.parse()).transpose()?.unwrap_or(32000);
    // create_new protege um vocabulário associado a checkpoints existentes.
    let mut out = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[1])?;
    let start = Instant::now();
    let tok = BpeTokenizer::train(&train, target)?;
    tok.save(&mut out)?;
    out.flush()?;
    eprintln!(
        "books={} train_bytes={} validation_bytes={} vocabulary={} elapsed={:.2}s",
        files.len(),
        train.len(),
        valid.len(),
        tok.vocab_size(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
