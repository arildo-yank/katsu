use katsu_core::{
    arena_len, load_params, materialize, save_params, upload, Adam, BpeTokenizer, KatsuLmConfig,
    Optimizer, ParamStore, SplitMix64, Tensor,
};
use katsu_core::{Backend, CpuBackend};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Write},
    path::Path,
};
#[derive(Deserialize)]
struct Record {
    prompt: String,
    response: String,
}
struct Example {
    ids: Vec<usize>,
    answer_start: usize,
}
fn examples(
    path: &str,
    tok: &BpeTokenizer,
    context: usize,
) -> Result<Vec<Example>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let r: Record = serde_json::from_str(line)?;
        let mut ids = tok.encode(&r.prompt);
        let answer_start = ids.len();
        ids.extend(tok.encode(&(r.response + "<|end|>")));
        if answer_start == 0 || ids.len() <= answer_start || ids.len() - 1 > context {
            return Err("SFT example exceeds context or is empty".into());
        }
        rows.push(Example { ids, answer_start });
    }
    if rows.is_empty() {
        return Err("empty dataset".into());
    }
    Ok(rows)
}
fn loss(
    c: &KatsuLmConfig,
    p: &[Tensor],
    e: &Example,
    backend: Arc<dyn Backend>,
) -> Result<Tensor, katsu_core::TensorError> {
    let logits = c
        .forward_batch_with_backend(p, &e.ids[..e.ids.len() - 1], 1, backend.clone())?
        .reshape(&[e.ids.len() - 1, c.vocab_size])?;
    masked_loss(&logits, e, backend)
}

/// Média das perdas mascaradas de vários exemplos num forward só. Cada exemplo é completado
/// no fim (token 0) até o maior do lote: com atenção causal, o padding não afeta os tokens
/// reais, e as posições continuam as mesmas do batch 1. Com um exemplo, é igual a `loss`.
fn batch_loss(
    c: &KatsuLmConfig,
    p: &[Tensor],
    batch: &[&Example],
    backend: Arc<dyn Backend>,
) -> Result<Tensor, katsu_core::TensorError> {
    if let [e] = batch {
        return loss(c, p, e, backend);
    }
    let width = batch.iter().map(|e| e.ids.len() - 1).max().unwrap_or(0);
    let mut inputs = Vec::with_capacity(batch.len() * width);
    for e in batch {
        inputs.extend_from_slice(&e.ids[..e.ids.len() - 1]);
        inputs.resize(inputs.len() + width + 1 - e.ids.len(), 0);
    }
    let logits = c
        .forward_batch_with_backend(p, &inputs, batch.len(), backend.clone())?
        .reshape(&[batch.len() * width, c.vocab_size])?;
    let mut total: Option<Tensor> = None;
    for (i, e) in batch.iter().enumerate() {
        let rows: Vec<usize> = (e.answer_start - 1..e.ids.len() - 1)
            .map(|r| i * width + r)
            .collect();
        let example = logits
            .gather_rows(&rows)?
            .cross_entropy_with_backend(&e.ids[e.answer_start..], backend.clone())?;
        total = Some(match total {
            Some(sum) => sum.add_with_backend(&example, backend.clone())?,
            None => example,
        });
    }
    total
        .ok_or(katsu_core::TensorError::InvalidShape("empty batch"))?
        .mul(&Tensor::scalar(1.0 / batch.len() as f32)?)
}
fn masked_loss(
    logits: &Tensor,
    e: &Example,
    backend: Arc<dyn Backend>,
) -> Result<Tensor, katsu_core::TensorError> {
    let rows: Vec<usize> = (e.answer_start - 1..e.ids.len() - 1).collect();
    logits
        .gather_rows(&rows)?
        .cross_entropy_with_backend(&e.ids[e.answer_start..], backend)
}

fn measure(
    c: &KatsuLmConfig,
    p: &[Tensor],
    data: &[Example],
    backend: Arc<dyn Backend>,
) -> Result<f64, katsu_core::TensorError> {
    let mut total = 0.;
    for e in data {
        total += f64::from(loss(c, p, e, backend.clone())?.data()[0]);
    }
    Ok(total / data.len() as f64)
}
/// Consome a perda e os pesos: o backward libera as ativações à medida que avança e o
/// Adam atualiza no próprio buffer os pesos sem outro dono, sem alocar um conjunto novo.
fn update(
    params: Vec<Tensor>,
    loss: Tensor,
    opt: &mut Adam,
    store: Option<&dyn ParamStore>,
) -> Result<(Vec<Tensor>, f32, f64), Box<dyn std::error::Error>> {
    let value = loss.data()[0];
    if !value.is_finite() {
        return Err("nonfinite loss".into());
    }
    loss.backward_release()?;
    let mut norm = 0.;
    for p in params.iter().filter(|p| !p.is_resident()) {
        let guard = p.grad()?;
        for &g in guard.as_ref().ok_or("missing gradient")? {
            if !g.is_finite() {
                return Err("nonfinite gradient".into());
            }
            norm += f64::from(g).powi(2);
        }
    }
    let next = match store {
        // Pesos residentes: norma e checagens rodam no store, sem trazer o gradiente.
        Some(store) => {
            let (sum, nonfinite) = store.grad_stats()?;
            if nonfinite > 0 {
                return Err("nonfinite gradient".into());
            }
            norm += sum;
            let (next, nonfinite) = opt.step_resident(params, store)?;
            if nonfinite > 0 {
                return Err("nonfinite weights".into());
            }
            next
        }
        None => opt.step_owned(params)?,
    };
    if next
        .iter()
        .filter(|p| !p.is_resident())
        .any(|p| p.data().iter().any(|x| !x.is_finite()))
    {
        return Err("nonfinite weights".into());
    }
    Ok((next, value, norm.sqrt()))
}
/// Backend do host para as operações que não usam pesos residentes.
fn host_backend() -> Arc<dyn Backend> {
    #[cfg(target_os = "macos")]
    {
        Arc::new(katsu_core::AccelerateBackend::default())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Arc::new(CpuBackend::default())
    }
}
fn save(p: &[Tensor], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = BufWriter::new(File::create(path)?);
    save_params(&materialize(p)?, &mut f)?;
    f.flush()?;
    Ok(())
}

// Step numbers are one-based. Warmup reaches the peak, cosine reaches the floor.
fn scheduled_lr(peak: f32, step: usize, total: usize, warmup_ratio: f64, floor: f64) -> f32 {
    let warmup = ((total as f64 * warmup_ratio).ceil() as usize).min(total.saturating_sub(1));
    let scale = if warmup > 0 && step <= warmup {
        step as f64 / warmup as f64
    } else {
        let progress = (step - warmup) as f64 / (total - warmup).max(1) as f64;
        floor + (1.0 - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
    };
    peak * scale as f32
}

fn checkpoint(
    params: &[Tensor],
    adam: &Adam,
    state: &Value,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let temporary = path.with_extension("ckpt.tmp");
    let mut writer = BufWriter::new(File::create(&temporary)?);
    // Mesmo formato com ou sem pesos residentes: os do store são lidos de volta.
    save_params(&materialize(params)?, &mut writer)?;
    adam.save_state_resident(&mut writer, params)?;
    serde_json::to_writer(&mut writer, state)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// Corpus da fase de pré-treino/adaptação como IDs u16. `.bin` é a saída do tokenize_corpus
/// (u16 little-endian, já com <|end|> entre documentos); qualquer outro arquivo é texto.
fn load_corpus(path: &Path, tok: &BpeTokenizer) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
    let ids: Vec<u16> = if path.extension().is_some_and(|e| e == "bin") {
        let bytes = fs::read(path)?;
        if bytes.len() % 2 != 0 {
            return Err("corpus .bin with odd byte count".into());
        }
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    } else {
        let ids = tok.encode(&fs::read_to_string(path)?);
        ids.into_iter()
            .map(|id| u16::try_from(id).map_err(|_| "token id exceeds u16"))
            .collect::<Result<_, _>>()?
    };
    if ids.iter().any(|&id| id as usize >= tok.vocab_size()) {
        return Err("corpus token outside the tokenizer vocabulary".into());
    }
    Ok(ids)
}

/// Janela de 257 tokens (256 de entrada e o alvo deslocado) como IDs do modelo.
fn window(corpus: &[u16], start: usize) -> Vec<usize> {
    corpus[start..start + 257]
        .iter()
        .map(|&id| id as usize)
        .collect()
}

// Stable content fingerprint to reject accidental data changes on resume.
// SHA-256 provenance is independently enforced by the launcher data gate.
fn fingerprint(paths: &[&Path]) -> Result<String, Box<dyn std::error::Error>> {
    let mut h = 0xcbf29ce484222325_u64;
    for path in paths {
        for byte in fs::read(path)? {
            h = (h ^ byte as u64).wrapping_mul(0x100000001b3);
        }
        h = (h ^ 255).wrapping_mul(0x100000001b3);
    }
    Ok(format!("{h:016x}"))
}

// Largura máxima: 2.048 cobre o 1B da V2 (docs/ARANDU_V2_PLAN.md). Os kernels percorrem a largura em
// laços (sem tamanho fixo); a norma de camada usa 2 × d floats de memória compartilhada (16 KB em 2.048).
const MAX_EMBED_DIM: usize = 2048;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wall_started = std::time::Instant::now();
    let benchmark_only = std::env::var("KATSU_BENCHMARK_ONLY").as_deref() == Ok("1");
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 8 && a.len() != 9 {
        return Err("BASE TRAIN EVAL CORPUS OUTPUT STEPS ADAPT_STEPS [LEARNING_RATE]".into());
    }
    let checkpoint_every: usize = std::env::var("KATSU_CHECKPOINT_EVERY")
        .unwrap_or_else(|_| "100".into())
        .parse()?;
    if checkpoint_every == 0 {
        return Err("checkpoint interval must be positive".into());
    }
    let base = Path::new(&a[1]);
    let output = Path::new(&a[5]);
    let steps: usize = a[6].parse()?;
    let adapt: usize = a[7].parse()?;
    let learning_rate: f32 = a.get(8).map_or("0.0001", String::as_str).parse()?;
    Adam::new(learning_rate)?;
    let corpus_lr: f32 = std::env::var("KATSU_CORPUS_LR")
        .unwrap_or_else(|_| "0.0001".into())
        .parse()?;
    Adam::new(corpus_lr)?;
    let warmup_ratio: f64 = std::env::var("KATSU_WARMUP_RATIO")
        .unwrap_or_else(|_| "0.05".into())
        .parse()?;
    let min_lr_ratio: f64 = std::env::var("KATSU_MIN_LR_RATIO")
        .unwrap_or_else(|_| "0.1".into())
        .parse()?;
    if !(0.0..1.0).contains(&warmup_ratio)
        || warmup_ratio == 0.0
        || !(0.0..=1.0).contains(&min_lr_ratio)
        || min_lr_ratio == 0.0
    {
        return Err("warmup must be (0,1), min LR ratio (0,1]".into());
    }
    let eval_samples: usize = std::env::var("KATSU_EVAL_SAMPLES")
        .unwrap_or_else(|_| "25".into())
        .parse()?;
    if eval_samples == 0 {
        return Err("evaluation sample count must be positive".into());
    }
    if output.exists() || (steps == 0 && adapt == 0) {
        return Err("new output and at least one training step required".into());
    }
    let mut metadata: Value = serde_json::from_reader(File::open(base.join("config.json"))?)?;
    let f = |k: &str| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(usize::try_from(
            metadata[k].as_u64().ok_or("config field missing")?,
        )?)
    };
    let old = KatsuLmConfig {
        vocab_size: f("vocab_size")?,
        block_size: f("block_size")?,
        embed_dim: f("embed_dim")?,
        layers: f("layers")?,
        heads: f("heads")?,
    };
    if old.vocab_size == 0
        || old.vocab_size > 65536
        || old.block_size == 0
        || old.block_size > 512
        || old.embed_dim == 0
        || old.embed_dim > MAX_EMBED_DIM
        || old.layers == 0
        || old.layers > 24
        || old.heads == 0
        || old.embed_dim % old.heads != 0
    {
        return Err("invalid SFT configuration".into());
    }
    let backend_name = std::env::var("KATSU_BACKEND").unwrap_or_else(|_| "cpu".into());
    let mut backend: Arc<dyn Backend> = match backend_name.as_str() {
        "cpu" => Arc::new(CpuBackend::default()),
        #[cfg(feature = "cuda")]
        "cuda" => Arc::new(katsu_core::CudaBackend::new(0)?),
        #[cfg(target_os = "macos")]
        "accelerate" => Arc::new(katsu_core::AccelerateBackend::default()),
        _ => return Err("unsupported backend or missing cuda feature".into()),
    };
    let batch: usize = std::env::var("KATSU_BATCH")
        .unwrap_or_else(|_| "1".into())
        .parse()?;
    if !(1..=64).contains(&batch) {
        return Err("KATSU_BATCH must be 1..=64".into());
    }
    let large = old.embed_dim > 128 || old.layers > 4;
    let budget: u64 = std::env::var("KATSU_MEMORY_BUDGET_MIB")
        .unwrap_or_else(|_| "1024".into())
        .parse()?;
    // Conservative planning estimate, not an OOM guarantee. CUDA keeps Adam on host. On the
    // Mac (Accelerate), measured 2026-10-25 on the 134M: weights + gradients + Adam moments
    // = 16 bytes/param, ~1.1 MB/token of activations (≈300 MiB per 256-token window).
    let estimate = if backend_name == "cuda" {
        (old.param_count() as u64 * 48).div_ceil(1024 * 1024) + 2048
    } else {
        (old.param_count() as u64 * 16).div_ceil(1024 * 1024) + 300 * batch as u64 + 512
    };
    if large && (!matches!(backend_name.as_str(), "cuda" | "accelerate") || estimate > budget) {
        return Err(format!(
            "large SFT requires CUDA or Accelerate and KATSU_MEMORY_BUDGET_MIB >= {estimate}"
        )
        .into());
    }
    if old.embed_dim > MAX_EMBED_DIM || old.layers > 24 || old.vocab_size > 65536 {
        return Err("SFT architecture exceeds supported bounds".into());
    }
    let mut tok = BpeTokenizer::load(&mut File::open(base.join("tokenizer.bpe"))?)?;
    if tok.vocab_size() != old.vocab_size {
        return Err("vocab mismatch".into());
    }
    let fresh = std::env::var("KATSU_INIT_RANDOM").as_deref() == Ok("1");
    let source = if fresh {
        old.init(2026)?
    } else {
        load_params(&mut File::open(base.join("model.ckpt"))?)?
    };
    let (context, expanded) = old.extend_context(&source, 256, 2026)?;
    tok.reserve_instruction_tokens()?;
    let (c, mut params) = context.extend_vocab(&expanded, tok.vocab_size(), 2027)?;
    drop(expanded);
    drop(source);
    let train = examples(&a[2], &tok, 256)?;
    let eval = examples(&a[3], &tok, 256)?;
    let corpus = if adapt > 0 {
        load_corpus(Path::new(&a[4]), &tok)?
    } else {
        Vec::new()
    };
    // KATSU_CORPUS_EVAL: corpus de validação (.bin ou texto), medido em janelas fixas antes,
    // a cada checkpoint e no fim da fase de corpus. Só no motor GPU.
    let corpus_eval = match std::env::var("KATSU_CORPUS_EVAL") {
        Ok(path) if !path.is_empty() && adapt > 0 => load_corpus(Path::new(&path), &tok)?,
        _ => Vec::new(),
    };
    let corpus_eval_windows: usize = std::env::var("KATSU_CORPUS_EVAL_WINDOWS")
        .unwrap_or_else(|_| "64".into())
        .parse()?;
    if !corpus_eval.is_empty() && (corpus_eval.len() < 257 || corpus_eval_windows == 0) {
        return Err("corpus eval too short or zero windows".into());
    }
    // KATSU_DEADLINE_SECONDS: teto de tempo da fase de corpus. Ao estourar, grava o checkpoint e
    // para com erro, sem SFT: retoma depois com KATSU_RESUME (teto de custo no pod).
    let deadline: Option<f64> = match std::env::var("KATSU_DEADLINE_SECONDS") {
        Ok(v) if !v.is_empty() => Some(v.parse()?),
        _ => None,
    };
    if std::env::var("KATSU_ENGINE").as_deref() != Ok("gpu")
        && (!corpus_eval.is_empty() || deadline.is_some())
    {
        return Err("KATSU_CORPUS_EVAL and KATSU_DEADLINE_SECONDS need KATSU_ENGINE=gpu".into());
    }
    if adapt > 0 && corpus.len() < 257 {
        return Err("corpus too short".into());
    }
    let signature = json!({"data":fingerprint(&[base.join("tokenizer.bpe").as_path(), Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4])])?,
        "steps":steps,"adapt":adapt,"peak_sft_lr":learning_rate,"peak_corpus_lr":corpus_lr,
        "warmup_ratio":warmup_ratio,"min_lr_ratio":min_lr_ratio,"config":[c.vocab_size,c.block_size,c.embed_dim,c.layers,c.heads],
        "batch":batch});
    let mut resume_adam = None;
    let mut resume_phase = String::new();
    let mut resume_step = 0;
    let mut resume_rng = 2026;
    if let Ok(path) = std::env::var("KATSU_RESUME") {
        if !path.is_empty() {
            let mut reader = BufReader::new(File::open(path)?);
            let loaded = load_params(&mut reader)?;
            if loaded.len() != params.len()
                || loaded
                    .iter()
                    .zip(&params)
                    .any(|(a, b)| a.shape() != b.shape())
            {
                return Err("resume shape mismatch".into());
            }
            let adam = Adam::load_state(&mut reader, &loaded)?;
            let state: Value = serde_json::from_reader(reader)?;
            if state["signature"] != signature {
                return Err("resume data/schedule mismatch".into());
            }
            resume_phase = state["phase"].as_str().ok_or("resume phase")?.to_owned();
            resume_step = state["step"].as_u64().ok_or("resume step")? as usize;
            resume_rng = state["rng"].as_u64().ok_or("resume rng")?;
            if !matches!(resume_phase.as_str(), "pretrain" | "sft")
                || resume_step > if resume_phase == "sft" { steps } else { adapt }
            {
                return Err("resume phase/step mismatch".into());
            }
            params = loaded;
            resume_adam = Some(adam);
        }
    }
    fs::create_dir_all(output)?;
    tok.save(&mut File::create(output.join("tokenizer.bpe"))?)?;
    metadata["vocab_size"] = json!(tok.vocab_size());
    metadata["special_token_ids"] = json!(katsu_core::INSTRUCTION_TOKENS
        .iter()
        .map(|t| (*t, tok.special_token_id(t).unwrap()))
        .collect::<std::collections::BTreeMap<_, _>>());
    metadata["block_size"] = json!(256);
    metadata["parameters"] = json!(c.param_count());
    metadata["format"] = json!("Katsu-native-rust-sft-v1");
    metadata["phase"] = json!("sft");
    metadata["backend"] = json!(backend_name);
    metadata["random_initialization"] = json!(fresh);
    metadata["base_checkpoint"] = json!(a[1]);
    metadata["adam_state_resumed"] = json!(resume_adam.is_some());
    metadata["steps"] = json!(steps);
    metadata["learning_rate"] = json!(learning_rate);
    metadata["adapt_steps"] = json!(adapt);
    metadata["corpus_phase"] = json!(if fresh { "pretrain" } else { "adapt" });
    metadata["corpus_learning_rate"] = json!(corpus_lr);
    metadata["scheduler"] = json!({"type":"linear-warmup-cosine","warmup_ratio":warmup_ratio,"min_lr_ratio":min_lr_ratio});
    metadata["eval_samples"] = json!(eval_samples);
    metadata["checkpoint_contents"] =
        json!("weights + Adam moments/step + RNG + phase/step + data/schedule signature");
    metadata["checkpoint_every"] = json!(checkpoint_every);
    metadata["benchmark_only"] = json!(benchmark_only);
    metadata["corpus_tokens"] = json!(corpus.len());
    metadata["corpus_eval_tokens"] = json!(corpus_eval.len());
    metadata["corpus_eval_windows"] = json!(corpus_eval_windows);
    metadata["deadline_seconds"] = json!(deadline);
    metadata["batch_size"] = json!(batch);
    metadata["train_sequence_length"] = json!(256);
    metadata["loss_mask"] = json!("assistant-only");
    metadata["markers"] = json!("KATSUBP2 reserved instruction IDs");
    metadata.as_object_mut().unwrap().remove("validation_loss");
    metadata.as_object_mut().unwrap().remove("books");
    fs::write(
        output.join("config.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    save(&params, &output.join("migrated.ckpt"))?;
    // KATSU_ENGINE=gpu: forward, backward e Adam inteiros na GPU (katsu_core::GpuTrainer),
    // mesmas fases, métricas, checkpoints e relatório do caminho abaixo.
    if std::env::var("KATSU_ENGINE").as_deref() == Ok("gpu") {
        #[cfg(feature = "cuda")]
        return gpu::run(gpu::Run {
            c,
            params,
            resume_adam,
            resume_phase,
            resume_step,
            resume_rng,
            train: &train,
            eval: &eval,
            corpus: &corpus,
            corpus_eval: &corpus_eval,
            corpus_eval_windows,
            deadline,
            steps,
            adapt,
            learning_rate,
            corpus_lr,
            warmup_ratio,
            min_lr_ratio,
            eval_samples,
            checkpoint_every,
            benchmark_only,
            signature: &signature,
            batch,
            fresh,
            output,
            metadata,
            wall_started,
        });
        #[cfg(not(feature = "cuda"))]
        return Err("KATSU_ENGINE=gpu requires --features cuda".into());
    }
    // KATSU_RESIDENT=1: pesos de matmul, gradientes e Adam ficam no store (na VRAM com
    // KATSU_BACKEND=cuda); as demais operações rodam no host, sem cópia
    // (KATSU_RESIDENT_REST=device as mantém no backend escolhido).
    let store: Option<Arc<dyn ParamStore>> =
        if std::env::var("KATSU_RESIDENT").as_deref() == Ok("1") {
            let layout = c.resident_layout();
            let len = arena_len(&params, &layout)?;
            let store: Arc<dyn ParamStore> = match backend_name.as_str() {
                #[cfg(feature = "cuda")]
                "cuda" => Arc::new(katsu_core::CudaBackend::new(0)?.param_store(len)?),
                _ => Arc::new(katsu_core::HostParamStore::new(len, backend.clone())?),
            };
            params = upload(std::mem::take(&mut params), &layout, store.clone())?;
            if let Some(adam) = resume_adam.as_mut() {
                adam.offload_moments(&params)?;
            }
            if std::env::var("KATSU_RESIDENT_REST").as_deref() != Ok("device") {
                backend = host_backend();
            }
            metadata["resident_store"] = json!(store.name());
            metadata["resident_parameters"] = json!(len);
            metadata["rest_backend"] = json!(backend.name());
            fs::write(
                output.join("config.json"),
                serde_json::to_vec_pretty(&metadata)?,
            )?;
            Some(store)
        } else {
            None
        };
    let moved = || {
        store.as_ref().map(|s| {
            let t = s.transfers();
            t.to_device + t.to_host
        })
    };
    let mut log = BufWriter::new(File::create(output.join("metrics.jsonl"))?);
    let mut rng = SplitMix64(resume_rng);
    let mut adam = if resume_phase == "pretrain" {
        resume_adam.take().unwrap()
    } else {
        Adam::new(corpus_lr)?
    };
    let train_sample = &train[..train.len().min(eval_samples)];
    let eval_sample = &eval[..eval.len().min(eval_samples)];
    let before = if benchmark_only {
        None
    } else {
        Some(measure(&c, &params, eval_sample, backend.clone())?)
    };
    let pretrain_start = if resume_phase == "sft" {
        adapt
    } else {
        resume_step
    };
    for step in pretrain_start + 1..=adapt {
        let step_started = std::time::Instant::now();
        let lr = scheduled_lr(corpus_lr, step, adapt, warmup_ratio, min_lr_ratio);
        adam.set_learning_rate(lr)?;
        let i = rng.below(corpus.len() - 256);
        let loss = c.loss_batch_with_backend(&params, &[&window(&corpus, i)], backend.clone())?;
        let before = moved();
        let (next, value, norm) = update(
            std::mem::take(&mut params),
            loss,
            &mut adam,
            store.as_deref(),
        )?;
        let transfer_bytes = moved().zip(before).map(|(a, b)| a - b);
        params = next;
        if step % checkpoint_every == 0 {
            checkpoint(
                &params,
                &adam,
                &json!({"signature":signature,"phase":"pretrain","step":step,"rng":rng.0}),
                &output.join("latest.ckpt"),
            )?;
        }
        writeln!(
            log,
            "{}",
            json!({"phase":if fresh {"pretrain"} else {"adapt"},"step":step,"epoch":step,"learning_rate":lr,"loss":value,"grad_l2":norm,"step_seconds":step_started.elapsed().as_secs_f64(),"input_tokens":256,"target_tokens":256,"transfer_bytes":transfer_bytes})
        )?;
        log.flush()?;
    }
    save(&params, &output.join("adapted.ckpt"))?;
    if resume_phase != "sft" && adapt > 0 {
        checkpoint(
            &params,
            &adam,
            &json!({"signature":signature,"phase":"pretrain","step":adapt,"rng":rng.0}),
            &output.join("pretrain-final.ckpt"),
        )?;
    }
    let initial_train = if benchmark_only {
        None
    } else {
        Some(measure(&c, &params, train_sample, backend.clone())?)
    };
    let initial_eval = if benchmark_only {
        None
    } else {
        Some(measure(&c, &params, eval_sample, backend.clone())?)
    };
    let mut adam = if resume_phase == "sft" {
        resume_adam.take().unwrap()
    } else {
        Adam::new(learning_rate)?
    };
    let started = std::time::Instant::now();
    let sft_start = if resume_phase == "sft" {
        resume_step
    } else {
        0
    };
    for step in sft_start + 1..=steps {
        let step_started = std::time::Instant::now();
        let lr = scheduled_lr(learning_rate, step, steps, warmup_ratio, min_lr_ratio);
        adam.set_learning_rate(lr)?;
        let examples: Vec<&Example> = (0..batch).map(|_| &train[rng.below(train.len())]).collect();
        let step_loss = batch_loss(&c, &params, &examples, backend.clone())?;
        let before = moved();
        let (next, value, norm) = update(
            std::mem::take(&mut params),
            step_loss,
            &mut adam,
            store.as_deref(),
        )?;
        let transfer_bytes = moved().zip(before).map(|(a, b)| a - b);
        params = next;
        if step % checkpoint_every == 0 {
            checkpoint(
                &params,
                &adam,
                &json!({"signature":signature,"phase":"sft","step":step,"rng":rng.0}),
                &output.join("latest.ckpt"),
            )?;
        }
        writeln!(
            log,
            "{}",
            json!({"phase":"sft","step":step,"epoch":adapt+step,"learning_rate":lr,"loss":value,"grad_l2":norm,"elapsed_seconds":started.elapsed().as_secs_f64(),"step_seconds":step_started.elapsed().as_secs_f64(),"batch":batch,"input_tokens":examples.iter().map(|e|e.ids.len()-1).sum::<usize>(),"target_tokens":examples.iter().map(|e|e.ids.len()-e.answer_start).sum::<usize>(),"transfer_bytes":transfer_bytes})
        )?;
        log.flush()?;
        if step % 100 == 0 {
            eprintln!("SFT step={step} loss={value:.4} grad_l2={norm:.4}");
        }
    }
    if steps > 0 {
        checkpoint(
            &params,
            &adam,
            &json!({"signature":signature,"phase":"sft","step":steps,"rng":rng.0}),
            &output.join("training-final.ckpt"),
        )?;
    }
    let final_train = if benchmark_only {
        None
    } else {
        Some(measure(&c, &params, train_sample, backend.clone())?)
    };
    let final_eval = if benchmark_only {
        None
    } else {
        Some(measure(&c, &params, eval_sample, backend.clone())?)
    };
    save(&params, &output.join("model.ckpt"))?;
    let report = json!({"steps":steps,"adapt_steps":adapt,"learning_rate":learning_rate,"train_measured_examples":train_sample.len(),"eval_measured_examples":eval_sample.len(),"train_examples":train.len(),"heldout_examples":eval.len(),"max_train_tokens":train.iter().map(|e|e.ids.len()).max(),"base_eval_loss":before,"initial_train_loss":initial_train,"final_train_loss":final_train,"initial_eval_loss":initial_eval,"final_eval_loss":final_eval,"elapsed_sft_seconds":started.elapsed().as_secs_f64(),"finite_checks_passed":true,"benchmark_only":benchmark_only,"corpus_tokens":corpus.len(),"wall_seconds":wall_started.elapsed().as_secs_f64()});
    fs::write(
        output.join("sft-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    eprintln!("{report}");
    Ok(())
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use katsu_core::{GemmKernel, GpuTrainOptions, GpuTrainer, SftBatch};

    pub(super) struct Run<'a> {
        pub c: KatsuLmConfig,
        pub params: Vec<Tensor>,
        pub resume_adam: Option<Adam>,
        pub resume_phase: String,
        pub resume_step: usize,
        pub resume_rng: u64,
        pub train: &'a [Example],
        pub eval: &'a [Example],
        pub corpus: &'a [u16],
        pub corpus_eval: &'a [u16],
        pub corpus_eval_windows: usize,
        pub deadline: Option<f64>,
        pub steps: usize,
        pub adapt: usize,
        pub learning_rate: f32,
        pub corpus_lr: f32,
        pub warmup_ratio: f64,
        pub min_lr_ratio: f64,
        pub eval_samples: usize,
        pub checkpoint_every: usize,
        pub benchmark_only: bool,
        pub signature: &'a Value,
        pub batch: usize,
        pub fresh: bool,
        pub output: &'a Path,
        pub metadata: Value,
        pub wall_started: std::time::Instant,
    }

    fn one(e: &Example) -> Result<SftBatch, katsu_core::TensorError> {
        SftBatch::new(&[(&e.ids, e.answer_start)])
    }

    /// Média da perda por exemplo, um forward por exemplo como o `measure` do host.
    fn measure(t: &mut GpuTrainer, data: &[Example]) -> Result<f64, Box<dyn std::error::Error>> {
        let mut total = 0.;
        for e in data {
            total += t.loss(&one(e)?)?;
        }
        Ok(total / data.len() as f64)
    }

    /// Perda média em janelas fixas, igualmente espaçadas, do corpus de validação. Todas têm 256
    /// alvos, então a média dos lotes é a média por token.
    fn corpus_loss(t: &mut GpuTrainer, r: &Run) -> Result<f64, Box<dyn std::error::Error>> {
        let span = r.corpus_eval.len() - 257;
        let starts: Vec<usize> = (0..r.corpus_eval_windows)
            .map(|k| k * span / r.corpus_eval_windows.max(1))
            .collect();
        let mut total = 0.;
        for chunk in starts.chunks(r.batch) {
            let windows: Vec<Vec<usize>> =
                chunk.iter().map(|&i| window(r.corpus_eval, i)).collect();
            let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
            total += t.loss(&SftBatch::windows(&refs)?)? * chunk.len() as f64;
        }
        Ok(total / starts.len() as f64)
    }

    fn log_corpus_eval(
        t: &mut GpuTrainer,
        r: &Run,
        log: &mut impl Write,
        step: usize,
    ) -> Result<Option<f64>, Box<dyn std::error::Error>> {
        if r.corpus_eval.is_empty() {
            return Ok(None);
        }
        let loss = corpus_loss(t, r)?;
        writeln!(
            log,
            "{}",
            json!({"step":step,"loss":loss,"windows":r.corpus_eval_windows})
        )?;
        log.flush()?;
        eprintln!("corpus eval step={step} loss={loss:.4}");
        Ok(Some(loss))
    }

    /// Passo do Adam com as mesmas recusas do `update` do host; devolve a norma.
    fn step(t: &mut GpuTrainer, loss: f64) -> Result<f64, Box<dyn std::error::Error>> {
        if !loss.is_finite() {
            return Err("nonfinite loss".into());
        }
        let (norm, nonfinite) = t.grad_stats()?;
        if nonfinite > 0 {
            return Err("nonfinite gradient".into());
        }
        if t.adam_step()? > 0 {
            return Err("nonfinite weights".into());
        }
        Ok(norm)
    }

    fn save_checkpoint(
        t: &GpuTrainer,
        state: &Value,
        path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        checkpoint(&t.params()?, &t.adam()?, state, path)
    }

    pub(super) fn run(mut r: Run) -> Result<(), Box<dyn std::error::Error>> {
        let gemm = match std::env::var("KATSU_GEMM").as_deref() {
            Ok("reference16") => GemmKernel::Reference16,
            Err(_) | Ok("tiled64") => GemmKernel::Tiled64,
            Ok("tf32") => GemmKernel::Tf32,
            _ => return Err("KATSU_GEMM must be reference16, tiled64 or tf32".into()),
        };
        // KATSU_FUSED=0 desliga as fusões da Etapa 2 (para medir o ganho delas)
        let fused = std::env::var("KATSU_FUSED").as_deref() != Ok("0");
        let options = GpuTrainOptions { gemm, fused };
        let pretrain_adam = if r.resume_phase == "pretrain" {
            r.resume_adam.take().ok_or("resume Adam")?
        } else {
            Adam::new(r.corpus_lr)?
        };
        let mut t = GpuTrainer::new(
            r.c,
            &r.params,
            &pretrain_adam,
            r.batch,
            r.c.block_size,
            0,
            options,
        )?;
        drop(std::mem::take(&mut r.params));
        let (free, total) = t.memory()?;
        r.metadata["engine"] = json!("gpu");
        r.metadata["gpu_options"] = json!(format!("{options:?}"));
        r.metadata["vram_used_after_alloc_mib"] = json!((total - free) / (1 << 20));
        r.metadata["rest_backend"] = json!("cuda (all ops on device)");
        fs::write(
            r.output.join("config.json"),
            serde_json::to_vec_pretty(&r.metadata)?,
        )?;

        let mut log = BufWriter::new(File::create(r.output.join("metrics.jsonl"))?);
        let mut rng = SplitMix64(r.resume_rng);
        let train_sample = &r.train[..r.train.len().min(r.eval_samples)];
        let eval_sample = &r.eval[..r.eval.len().min(r.eval_samples)];
        let before = if r.benchmark_only {
            None
        } else {
            Some(measure(&mut t, eval_sample)?)
        };
        let pretrain_start = if r.resume_phase == "sft" {
            r.adapt
        } else {
            r.resume_step
        };
        let mut eval_log = if r.corpus_eval.is_empty() || pretrain_start >= r.adapt {
            None
        } else {
            Some(BufWriter::new(
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(r.output.join("corpus-eval.jsonl"))?,
            ))
        };
        let corpus_eval_initial = match eval_log.as_mut() {
            Some(out) => log_corpus_eval(&mut t, &r, out, pretrain_start)?,
            None => None,
        };
        let mut corpus_eval_final = corpus_eval_initial;
        for s in pretrain_start + 1..=r.adapt {
            let started = std::time::Instant::now();
            let lr = scheduled_lr(r.corpus_lr, s, r.adapt, r.warmup_ratio, r.min_lr_ratio);
            t.set_learning_rate(lr)?;
            // r.batch janelas aleatórias de 256 tokens; com batch 1, o mesmo sorteio de antes.
            let windows: Vec<Vec<usize>> = (0..r.batch)
                .map(|_| window(r.corpus, rng.below(r.corpus.len() - 256)))
                .collect();
            let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
            let loss = t.loss_and_grad(&SftBatch::windows(&refs)?)?;
            let norm = step(&mut t, loss)?;
            let out_of_time = r
                .deadline
                .is_some_and(|limit| r.wall_started.elapsed().as_secs_f64() > limit);
            if s % r.checkpoint_every == 0 || out_of_time {
                save_checkpoint(
                    &t,
                    &json!({"signature":r.signature,"phase":"pretrain","step":s,"rng":rng.0}),
                    &r.output.join("latest.ckpt"),
                )?;
                if let Some(out) = eval_log.as_mut() {
                    corpus_eval_final = log_corpus_eval(&mut t, &r, out, s)?;
                }
            }
            writeln!(
                log,
                "{}",
                json!({"phase":if r.fresh {"pretrain"} else {"adapt"},"step":s,"epoch":s,"learning_rate":lr,"loss":loss,"grad_l2":norm,"step_seconds":started.elapsed().as_secs_f64(),"batch":r.batch,"input_tokens":256*r.batch,"target_tokens":256*r.batch,"transfer_bytes":null})
            )?;
            log.flush()?;
            if s % 100 == 0 {
                eprintln!("pretrain step={s} loss={loss:.4} grad_l2={norm:.4}");
            }
            if out_of_time {
                return Err(format!(
                    "KATSU_DEADLINE_SECONDS reached at pretrain step {s}/{}; latest.ckpt saved, resume with KATSU_RESUME",
                    r.adapt
                )
                .into());
            }
        }
        if let Some(out) = eval_log.as_mut() {
            if r.adapt % r.checkpoint_every != 0 {
                corpus_eval_final = log_corpus_eval(&mut t, &r, out, r.adapt)?;
            }
        }
        save(&t.params()?, &r.output.join("adapted.ckpt"))?;
        if r.resume_phase != "sft" && r.adapt > 0 {
            save_checkpoint(
                &t,
                &json!({"signature":r.signature,"phase":"pretrain","step":r.adapt,"rng":rng.0}),
                &r.output.join("pretrain-final.ckpt"),
            )?;
        }
        let initial_train = if r.benchmark_only {
            None
        } else {
            Some(measure(&mut t, train_sample)?)
        };
        let initial_eval = if r.benchmark_only {
            None
        } else {
            Some(measure(&mut t, eval_sample)?)
        };
        let sft_adam = if r.resume_phase == "sft" {
            r.resume_adam.take().ok_or("resume Adam")?
        } else {
            Adam::new(r.learning_rate)?
        };
        t.load_adam(&sft_adam)?;
        // KATSU_DUMP_BATCHES=arquivo: grava os lotes exatos de cada passo (para treinar os mesmos
        // tokens em outro framework e comparar velocidade com os mesmos dados).
        let mut dump = match std::env::var("KATSU_DUMP_BATCHES") {
            Ok(path) if !path.is_empty() => Some(BufWriter::new(File::create(path)?)),
            _ => None,
        };
        let started = std::time::Instant::now();
        let sft_start = if r.resume_phase == "sft" {
            r.resume_step
        } else {
            0
        };
        for s in sft_start + 1..=r.steps {
            let step_started = std::time::Instant::now();
            let lr = scheduled_lr(r.learning_rate, s, r.steps, r.warmup_ratio, r.min_lr_ratio);
            t.set_learning_rate(lr)?;
            let examples: Vec<&Example> = (0..r.batch)
                .map(|_| &r.train[rng.below(r.train.len())])
                .collect();
            let pairs: Vec<(&[usize], usize)> = examples
                .iter()
                .map(|e| (e.ids.as_slice(), e.answer_start))
                .collect();
            let batch = SftBatch::new(&pairs)?;
            if let Some(out) = dump.as_mut() {
                writeln!(
                    out,
                    "{}",
                    json!({"step":s,"batch":batch.batch,"width":batch.width,"input_tokens":batch.input_tokens,"tokens":batch.tokens,"targets":batch.targets,"weights":batch.weights})
                )?;
            }
            let loss = t.loss_and_grad(&batch)?;
            let norm = step(&mut t, loss)?;
            if s % r.checkpoint_every == 0 {
                save_checkpoint(
                    &t,
                    &json!({"signature":r.signature,"phase":"sft","step":s,"rng":rng.0}),
                    &r.output.join("latest.ckpt"),
                )?;
            }
            writeln!(
                log,
                "{}",
                json!({"phase":"sft","step":s,"epoch":r.adapt+s,"learning_rate":lr,"loss":loss,"grad_l2":norm,"elapsed_seconds":started.elapsed().as_secs_f64(),"step_seconds":step_started.elapsed().as_secs_f64(),"batch":r.batch,"input_tokens":batch.input_tokens,"padded_tokens":batch.batch*batch.width,"target_tokens":examples.iter().map(|e|e.ids.len()-e.answer_start).sum::<usize>(),"transfer_bytes":null})
            )?;
            log.flush()?;
            if s % 100 == 0 {
                eprintln!("SFT step={s} loss={loss:.4} grad_l2={norm:.4}");
            }
        }
        if r.steps > 0 {
            save_checkpoint(
                &t,
                &json!({"signature":r.signature,"phase":"sft","step":r.steps,"rng":rng.0}),
                &r.output.join("training-final.ckpt"),
            )?;
        }
        let final_train = if r.benchmark_only {
            None
        } else {
            Some(measure(&mut t, train_sample)?)
        };
        let final_eval = if r.benchmark_only {
            None
        } else {
            Some(measure(&mut t, eval_sample)?)
        };
        save(&t.params()?, &r.output.join("model.ckpt"))?;
        let (free, total) = t.memory()?;
        let report = json!({"steps":r.steps,"adapt_steps":r.adapt,"learning_rate":r.learning_rate,"train_measured_examples":train_sample.len(),"eval_measured_examples":eval_sample.len(),"train_examples":r.train.len(),"heldout_examples":r.eval.len(),"max_train_tokens":r.train.iter().map(|e|e.ids.len()).max(),"base_eval_loss":before,"initial_train_loss":initial_train,"final_train_loss":final_train,"initial_eval_loss":initial_eval,"final_eval_loss":final_eval,"elapsed_sft_seconds":started.elapsed().as_secs_f64(),"finite_checks_passed":true,"benchmark_only":r.benchmark_only,"corpus_tokens":r.corpus.len(),"corpus_eval_windows":if r.corpus_eval.is_empty() {None} else {Some(r.corpus_eval_windows)},"corpus_eval_initial_loss":corpus_eval_initial,"corpus_eval_final_loss":corpus_eval_final,"wall_seconds":r.wall_started.elapsed().as_secs_f64(),"engine":"gpu","vram_used_mib":(total-free)/(1<<20),"vram_total_mib":total/(1<<20),"kernel_launches":t.kernel_launches()});
        fs::write(
            r.output.join("sft-report.json"),
            serde_json::to_vec_pretty(&report)?,
        )?;
        eprintln!("{report}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corpus_bin_and_text_load_the_same_ids() {
        let tok = BpeTokenizer::train("o atacante recua e o zagueiro sobe ", 300).unwrap();
        let text = "o atacante recua, o zagueiro sobe";
        let dir = std::env::temp_dir().join(format!("katsu-corpus-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (txt, bin) = (dir.join("c.txt"), dir.join("c.bin"));
        fs::write(&txt, text).unwrap();
        let ids = tok.encode(text);
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| (id as u16).to_le_bytes())
            .collect();
        fs::write(&bin, &bytes).unwrap();
        let from_text = load_corpus(&txt, &tok).unwrap();
        assert_eq!(from_text, load_corpus(&bin, &tok).unwrap());
        assert_eq!(
            from_text.iter().map(|&id| id as usize).collect::<Vec<_>>(),
            ids
        );
        fs::write(&bin, &bytes[1..]).unwrap();
        assert!(load_corpus(&bin, &tok).is_err(), "odd byte count");
        fs::write(&bin, 60_000u16.to_le_bytes()).unwrap();
        assert!(
            load_corpus(&bin, &tok).is_err(),
            "id outside the vocabulary"
        );
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn window_is_257_ids_from_start() {
        let corpus: Vec<u16> = (0..300).collect();
        let w = window(&corpus, 40);
        assert_eq!(w.len(), 257);
        assert_eq!((w[0], w[256]), (40, 296));
    }
    #[test]
    fn warmup_cosine_endpoints() {
        let rates: Vec<_> = (1..=100)
            .map(|step| scheduled_lr(1.0, step, 100, 0.1, 0.1))
            .collect();
        assert!((rates[0] - 0.1).abs() < 1e-6);
        assert!((rates[9] - 1.0).abs() < 1e-6);
        assert!((rates[99] - 0.1).abs() < 1e-6);
        assert!(rates[..10].windows(2).all(|p| p[1] > p[0]));
        assert!(rates[9..].windows(2).all(|p| p[1] <= p[0]));
    }
    #[test]
    fn adam_checkpoint_replays_next_update_exactly() {
        let c = KatsuLmConfig {
            vocab_size: 16,
            block_size: 8,
            embed_dim: 8,
            layers: 1,
            heads: 2,
        };
        let p = c.init(2026).unwrap();
        let e = Example {
            ids: vec![1, 2, 3, 4, 5],
            answer_start: 3,
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::default());
        let mut adam = Adam::new(0.001).unwrap();
        let (p, _, _) = update(
            p.clone(),
            loss(&c, &p, &e, backend.clone()).unwrap(),
            &mut adam,
            None,
        )
        .unwrap();
        let mut bytes = Vec::new();
        save_params(&p, &mut bytes).unwrap();
        adam.save_state(&mut bytes).unwrap();
        let mut reader = std::io::Cursor::new(bytes);
        let loaded = load_params(&mut reader).unwrap();
        let mut restored = Adam::load_state(&mut reader, &loaded).unwrap();
        adam.set_learning_rate(0.0005).unwrap();
        restored.set_learning_rate(0.0005).unwrap();
        let (expected, _, _) = update(
            p.clone(),
            loss(&c, &p, &e, backend.clone()).unwrap(),
            &mut adam,
            None,
        )
        .unwrap();
        let (actual, _, _) = update(
            loaded.clone(),
            loss(&c, &loaded, &e, backend).unwrap(),
            &mut restored,
            None,
        )
        .unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.data(), b.data());
        }
    }

    #[test]
    fn backend_forward_supports_masked_sft_and_backward() {
        let c = KatsuLmConfig {
            vocab_size: 16,
            block_size: 8,
            embed_dim: 8,
            layers: 1,
            heads: 2,
        };
        let p = c.init(2026).unwrap();
        let e = Example {
            ids: vec![1, 2, 3, 4, 5],
            answer_start: 3,
        };
        let l = loss(&c, &p, &e, Arc::new(CpuBackend::default())).unwrap();
        assert!(l.data()[0].is_finite());
        l.backward().unwrap();
        for t in p {
            assert!(t
                .grad()
                .unwrap()
                .as_ref()
                .unwrap()
                .iter()
                .all(|x| x.is_finite()));
        }
    }
    #[test]
    fn prompt_rows_are_excluded_and_answer_targets_are_shifted() {
        let logits = Tensor::new(
            vec![100., 0., 0., 0., 100., 0., 4., 0., 0., 0., 4., 0.],
            &[4, 3],
        )
        .unwrap();
        let e = Example {
            ids: vec![0, 1, 2, 0, 1],
            answer_start: 3,
        };
        let l = masked_loss(&logits, &e, Arc::new(CpuBackend::default())).unwrap();
        assert!((l.data()[0] - (1.0_f32 + 2.0 * (-4.0_f32).exp()).ln()).abs() < 1e-6);
        l.backward().unwrap();
        let g = logits.grad().unwrap();
        let g = g.as_ref().unwrap();
        assert_eq!(&g[..6], &[0.; 6]);
        assert!(g[6] < 0. && g[10] < 0.);
    }

    #[test]
    fn batch_loss_is_the_mean_of_single_example_losses() {
        let c = KatsuLmConfig {
            vocab_size: 260,
            block_size: 16,
            embed_dim: 8,
            layers: 1,
            heads: 2,
        };
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::default());
        let short = Example {
            ids: vec![1, 2, 3, 4, 5],
            answer_start: 3,
        };
        let long = Example {
            ids: vec![9, 8, 7, 6, 5, 4, 3, 2],
            answer_start: 5,
        };
        let grads = |p: &[Tensor]| -> Vec<Vec<f32>> {
            p.iter()
                .map(|t| {
                    let g = t.grad().unwrap().clone().unwrap();
                    t.zero_grad().unwrap();
                    g
                })
                .collect()
        };
        let p = c.init(7).unwrap();

        // Um exemplo: idêntico ao caminho antigo, bit a bit.
        let single = loss(&c, &p, &short, backend.clone()).unwrap();
        let batched = batch_loss(&c, &p, &[&short], backend.clone()).unwrap();
        assert_eq!(single.data(), batched.data());

        // Dois exemplos de tamanhos diferentes: média das perdas e dos gradientes.
        let mut expected_loss = 0.;
        let mut expected: Vec<Vec<f32>> = p.iter().map(|t| vec![0.; t.numel()]).collect();
        for e in [&short, &long] {
            let l = loss(&c, &p, e, backend.clone()).unwrap();
            expected_loss += l.data()[0] / 2.;
            l.backward().unwrap();
            for (acc, g) in expected.iter_mut().zip(grads(&p)) {
                for (a, v) in acc.iter_mut().zip(g) {
                    *a += v / 2.;
                }
            }
        }
        let l = batch_loss(&c, &p, &[&short, &long], backend).unwrap();
        assert!((l.data()[0] - expected_loss).abs() < 1e-5);
        l.backward().unwrap();
        for (acc, g) in expected.iter().zip(grads(&p)) {
            for (a, b) in acc.iter().zip(g) {
                assert!((a - b).abs() < 1e-5, "{a} vs {b}");
            }
        }
    }
}
