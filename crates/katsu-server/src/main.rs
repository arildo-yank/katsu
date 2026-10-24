use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use katsu_core::{load_params, BpeTokenizer, GenerationOptions, KatsuLmConfig, Tensor};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{fs::File, path::PathBuf, sync::Arc};
use tokio::sync::Semaphore;
/// Mesmos tetos do treinador (train_sft); cobrem o perfil 134M (12×768, 12 heads, contexto 256).
const MAX_EMBED_DIM: usize = 1024;
const MAX_LAYERS: usize = 24;
const MAX_BLOCK_SIZE: usize = 512;
const MAX_VOCAB: usize = 65536;
/// ~800 MB em f32: protege um Mac de 8 GB de carregar um checkpoint maior por engano.
const MAX_PARAMETERS: usize = 200_000_000;
const MAX_NEW_TOKENS: usize = 512;
struct Model {
    config: KatsuLmConfig,
    params: Vec<Tensor>,
    tokenizer: BpeTokenizer,
    gate: Arc<Semaphore>,
}
#[derive(Deserialize)]
struct Generate {
    prompt: String,
    #[serde(default = "default_count")]
    max_new_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_penalty")]
    repetition_penalty: f32,
    #[serde(default)]
    seed: u64,
    /// false gera exatamente max_new_tokens, ignorando <|end|>: medição de velocidade.
    #[serde(default = "default_stop_at_end")]
    stop_at_end: bool,
}
fn default_stop_at_end() -> bool {
    true
}
fn default_count() -> usize {
    64
}
fn default_temperature() -> f32 {
    0.7
}
fn default_penalty() -> f32 {
    1.2
}
type Error = (StatusCode, Json<Value>);
fn error(code: StatusCode, message: &str) -> Error {
    (code, Json(json!({"error":message})))
}
async fn generate(
    State(model): State<Arc<Model>>,
    Json(req): Json<Generate>,
) -> Result<Json<Value>, Error> {
    if req.prompt.trim().is_empty()
        || req.prompt.len() > 16384
        || req.max_new_tokens == 0
        || req.max_new_tokens > MAX_NEW_TOKENS
        || !req.temperature.is_finite()
        || !(0.0..=2.0).contains(&req.temperature)
        || !req.repetition_penalty.is_finite()
        || !(1.0..=2.0).contains(&req.repetition_penalty)
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            &format!("invalid prompt, max_new_tokens (1..{MAX_NEW_TOKENS}), temperature (0..2), or repetition_penalty (1..2)"),
        ));
    }
    let permit = model
        .gate
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "model busy"))?;
    tokio::task::spawn_blocking(move || {
  let _permit=permit;
  let ids=model.tokenizer.encode(&req.prompt);let n=ids.len();
  let stop=model.tokenizer.special_token_id("<|end|>").map(|id|vec![id]).unwrap_or_else(||model.tokenizer.encode("\n<|end|>"));
  let options=GenerationOptions{temperature:req.temperature,repetition_penalty:req.repetition_penalty,seed:req.seed,stop_tokens:if req.stop_at_end {stop.clone()} else {Vec::new()}};
  let tokens=model.config.generate_with_options(&model.params,&ids,req.max_new_tokens,&options).map_err(|e|error(StatusCode::INTERNAL_SERVER_ERROR,&e.to_string()))?;
  let stopped=req.stop_at_end&&tokens[n..].ends_with(&stop);
  let end=if stopped {tokens.len()-stop.len()} else {tokens.len()};
  let bytes=model.tokenizer.decode_bytes(&tokens[n..end]).map_err(|e|error(StatusCode::INTERNAL_SERVER_ERROR,&e.to_string()))?;
  let valid_utf8=std::str::from_utf8(&bytes).is_ok();
  Ok(Json(json!({"text":String::from_utf8_lossy(&bytes),"token_ids":&tokens[n..end],"prompt_tokens":n,"context_tokens":model.config.block_size,"prompt_truncated":n>model.config.block_size,"generated_tokens":end-n,"valid_utf8":valid_utf8,"model":"katsu-native","grounded":false,"temperature":req.temperature,"repetition_penalty":req.repetition_penalty,"finish_reason":if stopped {"stop"} else {"length"}})))
 }).await.map_err(|_|error(StatusCode::INTERNAL_SERVER_ERROR,"generation worker failed"))?
}
async fn health(State(m): State<Arc<Model>>) -> Json<Value> {
    Json(
        json!({"status":"ok","backend":"cpu","parameters":m.config.param_count(),"context_tokens":m.config.block_size,"vocab_size":m.config.vocab_size,"end_token_id":m.tokenizer.special_token_id("<|end|>"),"max_new_tokens":MAX_NEW_TOKENS}),
    )
}
fn supported(c: &KatsuLmConfig) -> Result<(), String> {
    if c.heads == 0
        || c.embed_dim == 0
        || c.block_size == 0
        || c.vocab_size == 0
        || c.layers == 0
        || c.embed_dim % c.heads != 0
    {
        return Err("invalid config: zero field or heads do not divide embed_dim".into());
    }
    if c.embed_dim > MAX_EMBED_DIM
        || c.layers > MAX_LAYERS
        || c.block_size > MAX_BLOCK_SIZE
        || c.vocab_size > MAX_VOCAB
        || c.param_count() > MAX_PARAMETERS
    {
        return Err(format!(
            "unsupported config: max embed_dim {MAX_EMBED_DIM}, layers {MAX_LAYERS}, block_size {MAX_BLOCK_SIZE}, vocab {MAX_VOCAB}, parameters {MAX_PARAMETERS} (got {})",
            c.param_count()
        ));
    }
    Ok(())
}
/// Shapes na ordem de KatsuLmConfig::init (o teste abaixo garante que não divergem).
fn expected_shapes(c: &KatsuLmConfig) -> Vec<Vec<usize>> {
    let (v, t, d) = (c.vocab_size, c.block_size, c.embed_dim);
    let dh = d / c.heads;
    let mut shapes = vec![vec![v, d], vec![t, d]];
    for _ in 0..c.layers {
        shapes.extend([vec![d], vec![d]]);
        for _ in 0..c.heads {
            shapes.extend([vec![d, dh], vec![d, dh], vec![d, dh], vec![dh, d]]);
        }
        shapes.extend([
            vec![d],
            vec![d],
            vec![d, 4 * d],
            vec![4 * d],
            vec![4 * d, d],
            vec![d],
        ]);
    }
    shapes.extend([vec![d], vec![d], vec![d, v]]);
    shapes
}
/// Ctrl+C ou SIGTERM (o `docker stop` manda SIGTERM).
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let run = PathBuf::from(args.next().ok_or("usage: katsu-server RUN_DIR [PORT]")?);
    let port: u16 = args.next().unwrap_or_else(|| "8787".into()).parse()?;
    let c: Value = serde_json::from_reader(File::open(run.join("config.json"))?)?;
    let field = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(usize::try_from(
            c[key].as_u64().ok_or("invalid config field")?,
        )?)
    };
    let config = KatsuLmConfig {
        vocab_size: field("vocab_size")?,
        block_size: field("block_size")?,
        embed_dim: field("embed_dim")?,
        layers: field("layers")?,
        heads: field("heads")?,
    };
    supported(&config)?;
    let tokenizer = BpeTokenizer::load(&mut File::open(run.join("tokenizer.bpe"))?)?;
    if tokenizer.vocab_size() != config.vocab_size {
        return Err("tokenizer/config mismatch".into());
    }
    let params = load_params(&mut File::open(run.join("model.ckpt"))?)?;
    // Compara só shapes: config.init() alocaria uma segunda cópia do modelo (~540 MB no 134M).
    let expected = expected_shapes(&config);
    if params.len() != expected.len()
        || params
            .iter()
            .zip(&expected)
            .any(|(a, b)| a.shape() != b.as_slice() || a.data().iter().any(|v| !v.is_finite()))
    {
        return Err("checkpoint/config mismatch or nonfinite weights".into());
    }
    eprintln!(
        "Katsu: {} parâmetros, contexto {}",
        config.param_count(),
        config.block_size
    );
    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .layer(DefaultBodyLimit::max(32768))
        .with_state(Arc::new(Model {
            config,
            params,
            tokenizer,
            gate: Arc::new(Semaphore::new(1)),
        }));
    // KATSU_HOST=0.0.0.0 só dentro de container/rede privada: o servidor não tem autenticação.
    let host: std::net::IpAddr = std::env::var("KATSU_HOST")
        .unwrap_or_else(|_| "127.0.0.1".into())
        .parse()?;
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    eprintln!("Katsu CPU listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg(
        vocab_size: usize,
        block_size: usize,
        embed_dim: usize,
        layers: usize,
        heads: usize,
    ) -> KatsuLmConfig {
        KatsuLmConfig {
            vocab_size,
            block_size,
            embed_dim,
            layers,
            heads,
        }
    }
    #[test]
    fn stop_at_end_defaults_to_true() {
        let chat: Generate = serde_json::from_str(r#"{"prompt":"oi"}"#).unwrap();
        assert!(chat.stop_at_end);
        let bench: Generate =
            serde_json::from_str(r#"{"prompt":"oi","stop_at_end":false}"#).unwrap();
        assert!(!bench.stop_at_end);
    }
    #[test]
    fn shapes_match_init() {
        let c = cfg(40, 8, 12, 2, 3);
        let init: Vec<Vec<usize>> = c
            .init(1)
            .unwrap()
            .iter()
            .map(|t| t.shape().to_vec())
            .collect();
        assert_eq!(expected_shapes(&c), init);
    }
    #[test]
    fn accepts_134m_profile() {
        let c = cfg(32004, 256, 768, 12, 12);
        assert!(supported(&c).is_ok());
        assert!(
            (130_000_000..140_000_000).contains(&c.param_count()),
            "{}",
            c.param_count()
        );
        assert!(supported(&KatsuLmConfig::native_139m(30000)).is_ok());
    }
    #[test]
    fn rejects_oversized_or_invalid() {
        assert!(supported(&cfg(32004, 256, 768, 25, 12)).is_err());
        assert!(supported(&cfg(32004, 1024, 768, 12, 12)).is_err());
        assert!(supported(&cfg(32004, 512, 1024, 24, 16)).is_err()); // > MAX_PARAMETERS
        assert!(supported(&cfg(32004, 256, 768, 12, 7)).is_err());
        assert!(supported(&cfg(32004, 256, 768, 0, 12)).is_err());
    }
}
