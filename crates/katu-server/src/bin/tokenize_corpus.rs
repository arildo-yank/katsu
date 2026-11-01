//! Tokeniza um corpus de pré-treino (JSONL, um documento por linha com o campo "text") num
//! arquivo binário de IDs u16 little-endian, com <|end|> depois de cada documento. É o formato
//! que o train_sft lê quando o corpus termina em .bin.
//!
//!   tokenize_corpus TOKENIZER.bpe INPUT.jsonl OUTPUT.bin
//!
//! O texto é literal (encode_ordinary): um "<|end|>" escrito num artigo não vira marcador.
use katu_core::BpeTokenizer;
use serde_json::{json, Value};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

const BATCH: usize = 4096;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        return Err("TOKENIZER INPUT.jsonl OUTPUT.bin".into());
    }
    let tok = BpeTokenizer::load(&mut File::open(&a[1])?)?;
    if tok.vocab_size() > 65_536 {
        return Err("vocabulary does not fit in u16".into());
    }
    let end = tok
        .special_token_id("<|end|>")
        .ok_or("tokenizer without instruction tokens (KATSUBP2)")? as u16;
    let output = Path::new(&a[3]);
    let temporary = output.with_extension("bin.tmp");
    let mut writer = BufWriter::with_capacity(1 << 24, File::create(&temporary)?);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let mut encoders: Vec<_> = (0..threads).map(|_| tok.ordinary_encoder()).collect();
    let mut lines = BufReader::with_capacity(1 << 24, File::open(&a[2])?).lines();
    let (mut documents, mut tokens, mut bytes) = (0u64, 0u64, 0u64);
    let started = Instant::now();
    loop {
        let mut texts = Vec::with_capacity(BATCH);
        for line in lines.by_ref().take(BATCH) {
            let row: Value = serde_json::from_str(&line?)?;
            texts.push(row["text"].as_str().ok_or("line without text")?.to_owned());
        }
        if texts.is_empty() {
            break;
        }
        // Cada thread codifica uma fatia contígua; a escrita segue a ordem de entrada.
        let size = texts.len().div_ceil(threads);
        let encoded: Vec<Vec<u16>> = std::thread::scope(|scope| {
            let handles: Vec<_> = texts
                .chunks(size)
                .zip(encoders.iter_mut())
                .map(|(chunk, encoder)| {
                    scope.spawn(move || {
                        chunk
                            .iter()
                            .map(|text| {
                                let mut ids: Vec<u16> = encoder
                                    .encode(text)
                                    .into_iter()
                                    .map(|id| id as u16)
                                    .collect();
                                ids.push(end);
                                ids
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        for (text, ids) in texts.iter().zip(&encoded) {
            bytes += text.len() as u64;
            tokens += ids.len() as u64;
            for id in ids {
                writer.write_all(&id.to_le_bytes())?;
            }
        }
        documents += texts.len() as u64;
        if documents % (BATCH as u64 * 25) == 0 {
            eprintln!(
                "{documents} documentos, {tokens} tokens, {:.0} tokens/s",
                tokens as f64 / started.elapsed().as_secs_f64()
            );
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(&temporary, output)?;
    let manifest = json!({"format":"u16le token ids, <|end|> after each document","input":a[2],
        "tokenizer":a[1],"vocab_size":tok.vocab_size(),"end_token":end,"documents":documents,
        "tokens":tokens,"utf8_bytes":bytes,"seconds":started.elapsed().as_secs_f64()});
    fs::write(
        output.with_extension("json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!("{manifest}");
    Ok(())
}
