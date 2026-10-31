//! Rotas no formato da API da OpenAI (`/v1/chat/completions`, `/v1/models`), para um cliente existente
//! trocar só a `base_url`. O que o Katsu não sabe fazer (stream, tools, JSON mode…) é recusado com erro
//! claro, nunca ignorado: um cliente que pediu JSON não pode receber texto solto achando que é JSON.
use super::{complete, Generate, Model, MAX_NEW_TOKENS};
use axum::{extract::rejection::JsonRejection, extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

const MODEL_ID: &str = "katsu-native";
const DEFAULT_MAX_TOKENS: usize = 256;
const MAX_PROMPT_BYTES: usize = 16384;
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

type Failure = (StatusCode, Json<Value>);

fn failure(
    code: StatusCode,
    kind: &str,
    param: Option<&str>,
    tag: Option<&str>,
    message: &str,
) -> Failure {
    (
        code,
        Json(json!({"error":{"message":message,"type":kind,"param":param,"code":tag}})),
    )
}

fn bad_request(param: &str, message: &str) -> Failure {
    failure(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        Some(param),
        None,
        message,
    )
}

fn unsupported(param: &str, what: &str) -> Failure {
    failure(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        Some(param),
        Some("unsupported_parameter"),
        &format!("the Katsu server does not support {what}"),
    )
}

#[derive(Deserialize)]
pub(super) struct ChatRequest {
    #[serde(default)]
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
    n: Option<u32>,
    stop: Option<Value>,
    tools: Option<Vec<Value>>,
    functions: Option<Vec<Value>>,
    response_format: Option<Value>,
    #[serde(default)]
    logprobs: bool,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Value,
}

fn check_supported(req: &ChatRequest) -> Result<(), Failure> {
    if req.stream {
        return Err(unsupported("stream", "streaming; send stream=false"));
    }
    if req.n.is_some_and(|n| n != 1) {
        return Err(unsupported("n", "more than one choice"));
    }
    if req.stop.as_ref().is_some_and(|s| !s.is_null()) {
        return Err(unsupported("stop", "custom stop sequences"));
    }
    let has_tools = |list: &Option<Vec<Value>>| list.as_ref().is_some_and(|t| !t.is_empty());
    if has_tools(&req.tools) || has_tools(&req.functions) {
        return Err(unsupported("tools", "tool or function calling"));
    }
    if req
        .response_format
        .as_ref()
        .is_some_and(|f| !f.is_null() && f["type"] != "text")
    {
        return Err(unsupported(
            "response_format",
            "JSON mode or structured output",
        ));
    }
    if req.logprobs {
        return Err(unsupported("logprobs", "logprobs"));
    }
    if req.top_p.is_some_and(|p| p != 1.0) {
        return Err(unsupported("top_p", "top_p other than 1"));
    }
    Ok(())
}

/// Texto de uma mensagem: string ou lista de partes `{"type":"text","text":…}`. Imagem e áudio são recusados.
fn message_text(content: &Value) -> Result<String, Failure> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(
                |part| match (part["type"].as_str(), part["text"].as_str()) {
                    (Some("text"), Some(text)) => Ok(text),
                    _ => Err(unsupported("messages", "content other than text")),
                },
            )
            .collect::<Result<Vec<_>, _>>()
            .map(|texts| texts.concat()),
        _ => Err(bad_request("messages", "message content must be text")),
    }
}

/// O formato do treino: `<|system|>…\n<|user|>…\n<|assistant|>…<|end|>\n`, terminando em `<|assistant|>`.
fn chat_prompt(messages: &[ChatMessage]) -> Result<String, Failure> {
    if !messages.iter().any(|m| m.role == "user") {
        return Err(bad_request(
            "messages",
            "at least one user message is required",
        ));
    }
    let mut prompt = String::new();
    for message in messages {
        let text = message_text(&message.content)?;
        match message.role.as_str() {
            "system" | "developer" => prompt.push_str(&format!("<|system|>{text}\n")),
            "user" => prompt.push_str(&format!("<|user|>{text}\n")),
            "assistant" => prompt.push_str(&format!("<|assistant|>{text}<|end|>\n")),
            other => {
                return Err(bad_request(
                    "messages",
                    &format!("unsupported message role: {other}"),
                ))
            }
        }
    }
    prompt.push_str("<|assistant|>");
    Ok(prompt)
}

fn to_generate(req: &ChatRequest) -> Result<Generate, Failure> {
    check_supported(req)?;
    let prompt = chat_prompt(&req.messages)?;
    if prompt.len() > MAX_PROMPT_BYTES {
        return Err(context_exceeded(&format!(
            "messages exceed {MAX_PROMPT_BYTES} bytes"
        )));
    }
    let max_new_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if !(1..=MAX_NEW_TOKENS).contains(&max_new_tokens) {
        return Err(bad_request(
            "max_tokens",
            &format!("max_tokens must be between 1 and {MAX_NEW_TOKENS}"),
        ));
    }
    let temperature = req.temperature.unwrap_or_else(super::default_temperature);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(bad_request(
            "temperature",
            "temperature must be between 0 and 2",
        ));
    }
    Ok(Generate {
        prompt,
        max_new_tokens,
        temperature,
        repetition_penalty: super::default_penalty(),
        seed: req.seed.unwrap_or(0),
        stop_at_end: true,
        ground_numbers: false,
        check_support: false,
    })
}

fn context_exceeded(message: &str) -> Failure {
    failure(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        Some("messages"),
        Some("context_length_exceeded"),
        message,
    )
}

fn completion_body(out: &Value, id: u64, created: u64) -> Value {
    let prompt = out["prompt_tokens"].as_u64().unwrap_or(0);
    let generated = out["generated_tokens"].as_u64().unwrap_or(0);
    json!({
        "id": format!("chatcmpl-katsu-{created}-{id}"),
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out["text"]},
            "logprobs": null,
            "finish_reason": out["finish_reason"],
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": generated, "total_tokens": prompt + generated},
    })
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(super) async fn chat_completions(
    State(model): State<Arc<Model>>,
    body: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Json<Value>, Failure> {
    let Json(chat) = body.map_err(|why| {
        failure(
            why.status(),
            "invalid_request_error",
            None,
            None,
            &why.body_text(),
        )
    })?;
    let req = to_generate(&chat)?;
    // Sem isto o gerador só veria o fim da conversa: o sistema e o começo sumiriam sem aviso.
    let prompt_tokens = model.tokenizer.encode(&req.prompt).len();
    if prompt_tokens > model.config.block_size {
        return Err(context_exceeded(&format!(
            "messages have {prompt_tokens} tokens; this model's context is {}",
            model.config.block_size
        )));
    }
    let permit = model.gate.clone().try_acquire_owned().map_err(|_| {
        failure(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            None,
            Some("model_busy"),
            "model busy: this server generates one answer at a time",
        )
    })?;
    let out = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        complete(&model, &req)
    })
    .await
    .map_err(|_| server_error("generation worker failed"))?
    .map_err(|(_, Json(detail))| {
        server_error(detail["error"].as_str().unwrap_or("generation failed"))
    })?;
    Ok(Json(completion_body(
        &out,
        NEXT_ID.fetch_add(1, Ordering::Relaxed),
        now(),
    )))
}

fn server_error(message: &str) -> Failure {
    failure(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        None,
        None,
        message,
    )
}

pub(super) async fn models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "arandu"}],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: Value) -> ChatRequest {
        serde_json::from_value(body).unwrap()
    }

    fn rejected_param(body: Value) -> String {
        let (code, Json(detail)) = to_generate(&parse(body)).err().expect("should be refused");
        assert_eq!(code, StatusCode::BAD_REQUEST);
        detail["error"]["param"].as_str().unwrap().to_owned()
    }

    fn user(extra: Value) -> Value {
        let mut body = json!({"model":"gpt-4o","messages":[{"role":"user","content":"oi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        body
    }

    #[test]
    fn builds_the_training_format() {
        let req = parse(json!({"messages":[
            {"role":"system","content":"Seja breve."},
            {"role":"user","content":"Oi"},
            {"role":"assistant","content":"Olá"},
            {"role":"user","content":[{"type":"text","text":"Tudo "},{"type":"text","text":"bem?"}]},
        ]}));
        assert_eq!(
            to_generate(&req).unwrap().prompt,
            "<|system|>Seja breve.\n<|user|>Oi\n<|assistant|>Olá<|end|>\n<|user|>Tudo bem?\n<|assistant|>"
        );
    }

    #[test]
    fn applies_limits_and_defaults() {
        let req = to_generate(&parse(user(json!({})))).unwrap();
        assert_eq!(req.max_new_tokens, DEFAULT_MAX_TOKENS);
        assert!(req.stop_at_end);
        let newer = parse(user(
            json!({"max_tokens":9,"max_completion_tokens":5,"temperature":0}),
        ));
        let req = to_generate(&newer).unwrap();
        assert_eq!(req.max_new_tokens, 5);
        assert_eq!(req.temperature, 0.0);
        assert_eq!(rejected_param(user(json!({"max_tokens":0}))), "max_tokens");
        assert_eq!(
            rejected_param(user(json!({"max_tokens":513}))),
            "max_tokens"
        );
        assert_eq!(
            rejected_param(user(json!({"temperature":3}))),
            "temperature"
        );
    }

    #[test]
    fn refuses_what_it_cannot_honour() {
        assert_eq!(rejected_param(user(json!({"stream":true}))), "stream");
        assert_eq!(rejected_param(user(json!({"n":2}))), "n");
        assert_eq!(rejected_param(user(json!({"stop":["\n"]}))), "stop");
        assert_eq!(
            rejected_param(user(json!({"tools":[{"type":"function"}]}))),
            "tools"
        );
        assert_eq!(rejected_param(user(json!({"logprobs":true}))), "logprobs");
        assert_eq!(rejected_param(user(json!({"top_p":0.5}))), "top_p");
        assert_eq!(
            rejected_param(user(json!({"response_format":{"type":"json_object"}}))),
            "response_format"
        );
    }

    #[test]
    fn accepts_the_values_clients_send_by_default() {
        let body = user(
            json!({"stream":false,"n":1,"stop":null,"tools":[],"top_p":1,
            "response_format":{"type":"text"},"frequency_penalty":0,"user":"abc"}),
        );
        assert!(to_generate(&parse(body)).is_ok());
    }

    #[test]
    fn refuses_bad_messages() {
        assert_eq!(rejected_param(json!({"messages":[]})), "messages");
        assert_eq!(
            rejected_param(json!({"messages":[{"role":"system","content":"só sistema"}]})),
            "messages"
        );
        assert_eq!(
            rejected_param(
                json!({"messages":[{"role":"user","content":"oi"},{"role":"tool","content":"x"}]})
            ),
            "messages"
        );
        assert_eq!(
            rejected_param(
                json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})
            ),
            "messages"
        );
        let huge = "x".repeat(MAX_PROMPT_BYTES);
        let (_, Json(detail)) =
            to_generate(&parse(json!({"messages":[{"role":"user","content":huge}]})))
                .err()
                .unwrap();
        assert_eq!(detail["error"]["code"], "context_length_exceeded");
    }

    #[test]
    fn response_has_the_openai_shape() {
        let out =
            json!({"text":"Olá","prompt_tokens":7,"generated_tokens":3,"finish_reason":"stop"});
        let body = completion_body(&out, 4, 1000);
        assert_eq!(body["id"], "chatcmpl-katsu-1000-4");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], MODEL_ID);
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "Olá");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["total_tokens"], 10);
    }
}
