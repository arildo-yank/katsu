//! Train on an already separated bilingual train.txt; never on validation.
use katu_core::{BpeTokenizer, INSTRUCTION_TOKENS};
use serde_json::json;
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::Path,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("CORPUS_DIR OUTPUT_DIR SFT_DIR".into());
    }
    let corpus = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    fs::create_dir(out)?;
    let mut tok = BpeTokenizer::train(&fs::read_to_string(corpus.join("train.txt"))?, 31996)?;
    tok.reserve_instruction_tokens()?;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("tokenizer.bpe"))?;
    tok.save(&mut f)?;
    f.flush()?;
    let loaded = BpeTokenizer::load(&mut File::open(out.join("tokenizer.bpe"))?)?;
    let mut probes = Vec::new();
    for text in [
        "ACWR",
        "biomechanics",
        "ligament",
        "attention",
        "algorithm",
        "recovery",
        "prontidão",
        "recuperação",
        "O atleta está em recuperação.",
        "The athlete is recovering from training.",
    ] {
        let ids = loaded.encode(text);
        assert_eq!(loaded.decode(&ids)?, text);
        probes.push(json!({"text":text,"ids":ids,"tokens":ids.len()}));
    }
    for t in INSTRUCTION_TOKENS {
        assert_eq!(loaded.encode(t), vec![loaded.special_token_id(t).unwrap()]);
    }
    let mut sft = Vec::new();
    for name in ["arandu_sft.jsonl", "arandu_eval.jsonl"] {
        let mut count = 0;
        let mut max = 0;
        for line in fs::read_to_string(Path::new(&a[3]).join(name))?.lines() {
            let r: serde_json::Value = serde_json::from_str(line)?;
            let n = loaded
                .encode(r["text"].as_str().ok_or("missing text")?)
                .len();
            if n > 257 {
                return Err(format!("SFT row {} exceeds context: {n}", r["id"]).into());
            }
            max = max.max(n);
            count += 1;
        }
        sft.push(json!({"file":name,"examples":count,"max_tokens":max}));
    }
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(corpus.join("manifest.json"))?)?;
    let mut compression = Vec::new();
    for lang in ["pt", "en"] {
        let mut chars = 0;
        let mut tokens = 0;
        for doc in manifest["documents"]
            .as_array()
            .ok_or("documents")?
            .iter()
            .filter(|d| d["language"] == lang && d["split"] == "validation")
            .take(3)
        {
            let sample: String =
                fs::read_to_string(corpus.join(doc["path"].as_str().ok_or("path")?))?
                    .chars()
                    .take(4096)
                    .collect();
            let ids = loaded.encode(&sample);
            assert_eq!(loaded.decode(&ids)?, sample);
            chars += sample.chars().count();
            tokens += ids.len();
        }
        compression.push(json!({"language":lang,"characters":chars,"tokens":tokens,"characters_per_token":chars as f64/tokens as f64}));
    }
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"vocab_size":loaded.vocab_size(),"requested_base_vocab":31996,"special_ids":INSTRUCTION_TOKENS.iter().map(|t|(*t,loaded.special_token_id(t))).collect::<std::collections::BTreeMap<_,_>>(),"probes":probes,"sft":sft,"heldout_compression":compression,"old_checkpoints_compatible":false}),
        )?,
    )?;
    println!(
        "vocabulary={} special tokens and round trips passed",
        loaded.vocab_size()
    );
    Ok(())
}
