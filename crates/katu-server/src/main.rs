use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use katu_core::{load_params, BpeTokenizer, GenerationOptions, KatuLmConfig, Tensor};
use katu_rag::{Document, Index, Synonyms};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    fs::File,
    io::Write,
    path::PathBuf,
    sync::{Arc, RwLock},
};
use tokio::sync::Semaphore;
mod calc;
mod evidence;
mod grounding;
mod math;
mod openai;
/// Sistema do formato de leitura (o mesmo do SFT de leitura e da régua de leitura).
const READING_SYSTEM: &str = "Você é o Arandu. Responda em português do Brasil usando só o trecho.";
const NO_DATA: &str = "Não tenho dados suficientes nos documentos para responder.";
/// Resposta de leitura com menos que isso das palavras de conteúdo no trecho vira recusa
/// (calibrado nas réguas de leitura em 28/10/2026; ARANDU docs/teste-recusas-2026-10-28.md).
const MIN_SUPPORT: f32 = 0.7;
const UNSUPPORTED: &str = "Não tenho dados suficientes no trecho para confirmar uma resposta.";
const MAX_DOCUMENT_CHARS: usize = 200_000;
/// Mesmos tetos do treinador (train_sft); cobrem o perfil 134M (12×768, 12 heads, contexto 256).
const MAX_EMBED_DIM: usize = 1024;
const MAX_LAYERS: usize = 24;
const MAX_BLOCK_SIZE: usize = 512;
const MAX_VOCAB: usize = 65536;
/// ~800 MB em f32: protege um Mac de 8 GB de carregar um checkpoint maior por engano.
const MAX_PARAMETERS: usize = 200_000_000;
const MAX_NEW_TOKENS: usize = 512;
struct Model {
    config: KatuLmConfig,
    params: Vec<Tensor>,
    tokenizer: BpeTokenizer,
    /// Texto de cada token (com perdas em UTF-8 partido), para a decodificação restrita.
    token_text: Vec<String>,
    gate: Arc<Semaphore>,
    /// Biblioteca do RAG; `rag_path` (KATU_RAG_DOCS) guarda os documentos em JSONL.
    rag: RwLock<Index>,
    rag_path: Option<PathBuf>,
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
    /// true: todo número da resposta tem de aparecer no prompt (trecho e pergunta).
    #[serde(default)]
    ground_numbers: bool,
    /// true: num prompt de leitura, resposta sem apoio no trecho vira recusa (`grounding::support`).
    #[serde(default)]
    check_support: bool,
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
fn validate(req: &Generate) -> Result<(), Error> {
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
    Ok(())
}

/// Gera a continuação e aplica a ferramenta de cálculo: se o modelo respondeu só `[CALC: …]`, o
/// texto vira "expressão = resultado" (ou o erro do cálculo) e a chamada fica em `tool`.
fn complete(model: &Model, req: &Generate) -> Result<Value, Error> {
    let ids = model.tokenizer.encode(&req.prompt);
    let n = ids.len();
    let stop = model
        .tokenizer
        .special_token_id("<|end|>")
        .map(|id| vec![id])
        .unwrap_or_else(|| model.tokenizer.encode("\n<|end|>"));
    let options = GenerationOptions {
        temperature: req.temperature,
        repetition_penalty: req.repetition_penalty,
        seed: req.seed,
        stop_tokens: if req.stop_at_end {
            stop.clone()
        } else {
            Vec::new()
        },
    };
    let guard = req
        .ground_numbers
        .then(|| grounding::NumberGuard::new(&req.prompt));
    let mut allow = |done: &[usize], next: usize| match &guard {
        None => true,
        Some(g) => {
            let mut text: String = done.iter().map(|&t| model.token_text[t].as_str()).collect();
            text.push_str(&model.token_text[next]);
            g.allows(&text)
        }
    };
    let tokens = model
        .config
        .generate_filtered(
            &model.params,
            &ids,
            req.max_new_tokens,
            &options,
            &mut allow,
        )
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    let stopped = req.stop_at_end && tokens[n..].ends_with(&stop);
    let end = if stopped {
        tokens.len() - stop.len()
    } else {
        tokens.len()
    };
    let bytes = model
        .tokenizer
        .decode_bytes(&tokens[n..end])
        .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    let valid_utf8 = std::str::from_utf8(&bytes).is_ok();
    let raw = String::from_utf8_lossy(&bytes).into_owned();
    let (text, tool) = match calc::call(&raw) {
        Some(expr) => match calc::eval(expr) {
            Ok(v) => (
                format!("{expr} = {}", calc::format(v)),
                json!({"name":"calc","input":expr,"output":calc::format(v)}),
            ),
            Err(why) => (
                format!("Não consegui calcular {expr}: {why}."),
                json!({"name":"calc","input":expr,"error":why}),
            ),
        },
        None => (raw.clone(), Value::Null),
    };
    let support = (req.check_support && tool.is_null())
        .then(|| grounding::reading_parts(&req.prompt))
        .flatten()
        .map(|(passage, question)| grounding::support(passage, question, &text));
    let unsupported = support.is_some_and(|s| s < MIN_SUPPORT);
    let text = if unsupported {
        UNSUPPORTED.to_owned()
    } else {
        text
    };
    Ok(
        json!({"text":text,"raw_text":raw,"tool":tool,"token_ids":&tokens[n..end],"prompt_tokens":n,"context_tokens":model.config.block_size,"prompt_truncated":n>model.config.block_size,"generated_tokens":end-n,"ground_numbers":req.ground_numbers,"support":support,"unsupported":unsupported,"valid_utf8":valid_utf8,"model":"katu-native","grounded":false,"temperature":req.temperature,"repetition_penalty":req.repetition_penalty,"finish_reason":if stopped {"stop"} else {"length"}}),
    )
}

async fn generate(
    State(model): State<Arc<Model>>,
    Json(req): Json<Generate>,
) -> Result<Json<Value>, Error> {
    validate(&req)?;
    let permit = model
        .gate
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "model busy"))?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        complete(&model, &req).map(Json)
    })
    .await
    .map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation worker failed",
        )
    })?
}

#[derive(Deserialize)]
struct Ask {
    question: String,
    #[serde(default = "default_ask_tokens")]
    max_new_tokens: usize,
    /// Só estes documentos da biblioteca (ids, ou prefixos das partes de um PDF); vazio = todos.
    #[serde(default)]
    docs: Vec<String>,
}
fn default_ask_tokens() -> usize {
    80
}

/// RAG: acha o melhor trecho da biblioteca e pergunta ao modelo no formato de leitura. Sem trecho
/// relevante, responde que não há dados, sem chamar o modelo (não inventa).
async fn rag_ask(
    State(model): State<Arc<Model>>,
    Json(ask): Json<Ask>,
) -> Result<Json<Value>, Error> {
    if ask.question.trim().is_empty() || ask.question.len() > 2000 || !valid_docs(&ask.docs) {
        return Err(error(StatusCode::BAD_REQUEST, "invalid question or docs"));
    }
    let hits: Vec<Value> = {
        let index = model
            .rag
            .read()
            .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "rag index poisoned"))?;
        index
            .search_in(&ask.question, 3, &ask.docs)
            .into_iter()
            .map(|h| json!({"doc":h.chunk.doc,"title":h.chunk.title,"text":h.chunk.text,"score":h.score,"coverage":h.coverage}))
            .collect()
    };
    let Some(best) = hits.first() else {
        return Ok(Json(
            json!({"answer":NO_DATA,"sources":[],"used_model":false,"fallback_used":false,"response_mode":"no_evidence"}),
        ));
    };
    let req = Generate {
        prompt: format!(
            "<|system|>{READING_SYSTEM}\n<|user|>Trecho: {}\nPergunta: {}\n<|assistant|>",
            best["text"].as_str().unwrap_or_default(),
            ask.question.trim()
        ),
        max_new_tokens: ask.max_new_tokens,
        temperature: 0.0,
        repetition_penalty: default_penalty(),
        seed: 0,
        stop_at_end: true,
        ground_numbers: true,
        check_support: true,
    };
    validate(&req)?;
    let permit = model
        .gate
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::TOO_MANY_REQUESTS, "model busy"))?;
    let out = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        complete(&model, &req)
    })
    .await
    .map_err(|_| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation worker failed",
        )
    })??;
    // Return the complete retrieved passage: even a number present in the source can
    // be attached to the wrong claim. Keep raw generation separate for audit.
    let source = hits[0]["text"].as_str().unwrap_or_default();
    let raw = out["text"].as_str().unwrap_or_default();
    let (answer, fallback_used) = evidence::quote_or_verified_copy(raw, source);
    Ok(Json(json!({
        "answer": answer, "raw_answer": raw, "sources": hits, "used_model": true,
        "fallback_used": fallback_used, "response_mode": "source_quote",
        "fallback_reason": if fallback_used { Some("generation_differs_from_complete_source") } else { None },
        "prompt_tokens": out["prompt_tokens"], "ground_numbers": true,
        "model_raw_text": out["raw_text"], "support": out["support"], "unsupported": out["unsupported"],
        "prompt_truncated": out["prompt_truncated"]
    })))
}

#[derive(Deserialize)]
struct Search {
    question: String,
    #[serde(default = "default_search_k")]
    k: usize,
    /// Só estes documentos da biblioteca (ids, ou prefixos das partes de um PDF); vazio = todos.
    #[serde(default)]
    docs: Vec<String>,
}

/// Até 50 documentos, cada id com até 128 caracteres.
fn valid_docs(docs: &[String]) -> bool {
    docs.len() <= 50 && docs.iter().all(|d| !d.trim().is_empty() && d.len() <= 128)
}
fn default_search_k() -> usize {
    3
}

/// Só a busca, sem o modelo: os melhores trechos com escore e cobertura. A API do Arandu usa para
/// decidir se a pergunta é sobre um documento da biblioteca.
async fn rag_search(
    State(model): State<Arc<Model>>,
    Json(search): Json<Search>,
) -> Result<Json<Value>, Error> {
    if search.question.trim().is_empty()
        || search.question.len() > 2000
        || !(1..=10).contains(&search.k)
        || !valid_docs(&search.docs)
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "invalid question, k (1..10) or docs",
        ));
    }
    let index = model
        .rag
        .read()
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "rag index poisoned"))?;
    let hits: Vec<Value> = index
        .search_in(&search.question, search.k, &search.docs)
        .into_iter()
        .map(|h| json!({"doc":h.chunk.doc,"title":h.chunk.title,"text":h.chunk.text,"score":h.score,"coverage":h.coverage}))
        .collect();
    Ok(Json(json!({"hits":hits,"chunks":index.len()})))
}

/// Lista os documentos da biblioteca: id, título e número de trechos.
async fn rag_list(State(model): State<Arc<Model>>) -> Result<Json<Value>, Error> {
    let index = model
        .rag
        .read()
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "rag index poisoned"))?;
    let documents: Vec<Value> = index
        .documents()
        .into_iter()
        .map(|(id, title, chunks)| json!({"id":id,"title":title,"chunks":chunks}))
        .collect();
    Ok(Json(json!({"documents":documents,"chunks":index.len()})))
}

/// Adiciona um documento à biblioteca (e ao arquivo, se KATU_RAG_DOCS estiver definido).
async fn rag_add(
    State(model): State<Arc<Model>>,
    Json(doc): Json<Document>,
) -> Result<Json<Value>, Error> {
    if doc.id.trim().is_empty()
        || doc.text.trim().is_empty()
        || doc.text.chars().count() > MAX_DOCUMENT_CHARS
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            &format!("document needs id and text (up to {MAX_DOCUMENT_CHARS} chars)"),
        ));
    }
    if let Some(path) = &model.rag_path {
        let line = serde_json::to_string(&doc)
            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
        writeln!(file, "{line}")
            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;
    }
    let mut index = model
        .rag
        .write()
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "rag index poisoned"))?;
    index.add(&doc);
    Ok(Json(json!({"chunks":index.len()})))
}

async fn health(State(m): State<Arc<Model>>) -> Json<Value> {
    Json(
        json!({"status":"ok","backend":"cpu","parameters":m.config.param_count(),"context_tokens":m.config.block_size,"vocab_size":m.config.vocab_size,"end_token_id":m.tokenizer.special_token_id("<|end|>"),"max_new_tokens":MAX_NEW_TOKENS,"rag_chunks":m.rag.read().map(|i| i.len()).unwrap_or(0),"tools":["calc"]}),
    )
}
fn supported(c: &KatuLmConfig) -> Result<(), String> {
    if c.heads == 0
        || c.embed_dim == 0
        || c.block_size == 0
        || c.vocab_size == 0
        || c.layers == 0
        || !c.embed_dim.is_multiple_of(c.heads)
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
/// Shapes na ordem de KatuLmConfig::init (o teste abaixo garante que não divergem).
fn expected_shapes(c: &KatuLmConfig) -> Vec<Vec<usize>> {
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
    let run = PathBuf::from(args.next().ok_or("usage: katu-server RUN_DIR [PORT]")?);
    let port: u16 = args.next().unwrap_or_else(|| "8787".into()).parse()?;
    let c: Value = serde_json::from_reader(File::open(run.join("config.json"))?)?;
    let field = |key: &str| -> Result<usize, Box<dyn std::error::Error>> {
        Ok(usize::try_from(
            c[key].as_u64().ok_or("invalid config field")?,
        )?)
    };
    let config = KatuLmConfig {
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
    let token_text = (0..tokenizer.vocab_size())
        .map(|id| {
            tokenizer
                .decode_bytes(&[id])
                .map(|b| String::from_utf8_lossy(&b).into_owned())
        })
        .collect::<std::io::Result<Vec<_>>>()?;
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
        "Katu: {} parâmetros, contexto {}",
        config.param_count(),
        config.block_size
    );
    // KATU_RAG_DOCS: biblioteca em JSONL ({"id","title","text"} por linha), lida aqui e acrescida
    // pelo POST /rag/documents. KATU_RAG_CHUNK: tamanho máximo do trecho em caracteres. 600 na
    // régua da NR-12 (ARANDU/docs/regua_busca_nr12.json, 31/10/2026): trecho certo em 1.º em 12/26 e
    // nos 3 primeiros em 18/26, contra 10 e 14 com 320; nenhum prompt do /rag/ask passou de 256 tokens.
    let chunk: usize = std::env::var("KATU_RAG_CHUNK")
        .unwrap_or_else(|_| "600".into())
        .parse()?;
    // KATU_RAG_SYNONYMS: todos (padrão), sem-consultor ou nenhum (katu-rag/src/synonyms.rs).
    let synonyms = Synonyms::parse(&std::env::var("KATU_RAG_SYNONYMS").unwrap_or_default())?;
    let mut rag = Index::new(chunk).with_synonyms(synonyms);
    let rag_path = std::env::var("KATU_RAG_DOCS")
        .ok()
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    if let Some(path) = rag_path.as_ref().filter(|p| p.exists()) {
        let docs = rag.load_jsonl(&std::fs::read_to_string(path)?)?;
        eprintln!("RAG: {docs} documentos, {} trechos", rag.len());
    }
    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .route("/rag/ask", post(rag_ask))
        .route("/rag/search", post(rag_search))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/models", get(openai::models))
        .route("/math", post(math::handle))
        .route("/math/ask", post(math::ask))
        .layer(DefaultBodyLimit::max(32768))
        .route(
            "/rag/documents",
            get(rag_list)
                .post(rag_add)
                .layer(DefaultBodyLimit::max(1 << 20)),
        )
        .with_state(Arc::new(Model {
            config,
            params,
            token_text,
            tokenizer,
            gate: Arc::new(Semaphore::new(1)),
            rag: RwLock::new(rag),
            rag_path,
        }));
    // KATU_HOST=0.0.0.0 só dentro de container/rede privada: o servidor não tem autenticação.
    let host: std::net::IpAddr = std::env::var("KATU_HOST")
        .unwrap_or_else(|_| "127.0.0.1".into())
        .parse()?;
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    eprintln!("Katu CPU listening on http://{}", listener.local_addr()?);
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
    ) -> KatuLmConfig {
        KatuLmConfig {
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
        assert!(supported(&KatuLmConfig::native_139m(30000)).is_ok());
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
