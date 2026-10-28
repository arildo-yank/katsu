//! Busca local por trechos (RAG) para o Arandu: divide documentos em trechos por frase e
//! indexa com BM25, sem acento e sem palavras vazias. Sem modelo de embeddings: a busca é
//! lexical, roda em microssegundos e é auditável (o escore de cada trecho tem explicação).
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Documento como entra e como é gravado (JSONL, um por linha).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub text: String,
}

/// Um trecho indexado e de onde veio.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub doc: String,
    pub title: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit<'a> {
    pub score: f32,
    /// Quanto da pergunta o trecho (com o título) cobre, de 0 a 1: a soma do IDF das palavras da
    /// pergunta que estão no trecho sobre a soma do IDF de todas elas. Palavra rara (nome próprio)
    /// pesa mais que palavra comum; palavra que nenhum trecho tem pesa o máximo.
    pub coverage: f32,
    pub chunk: &'a Chunk,
}

const K1: f32 = 1.2;
const B: f32 = 0.75;
const STOPWORDS: &[&str] = &[
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e", "em", "na", "nas",
    "no", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "que", "se",
    "um", "uma", "umas", "uns", "qual", "quais", "quem", "quando", "onde", "foi", "era", "ser",
    "seu", "sua", "seus", "suas", "the", "of", "and",
];
// "sao" fica de fora de propósito: é o verbo "são", mas também o "São" de São Paulo.

/// Palavras de busca: minúsculas, sem acento, só letras e dígitos, sem palavras vazias.
pub fn terms(text: &str) -> Vec<String> {
    let folded: String = text.chars().map(fold).collect();
    folded
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !STOPWORDS.contains(w))
        .map(str::to_owned)
        .collect()
}

fn fold(c: char) -> char {
    match c.to_lowercase().next().unwrap_or(c) {
        'á' | 'à' | 'â' | 'ã' | 'ä' => 'a',
        'é' | 'ê' | 'è' | 'ë' => 'e',
        'í' | 'î' | 'ì' | 'ï' => 'i',
        'ó' | 'ô' | 'õ' | 'ò' | 'ö' => 'o',
        'ú' | 'û' | 'ù' | 'ü' => 'u',
        'ç' => 'c',
        other => other,
    }
}

/// Trechos de até `max_chars` caracteres, cortados em fim de frase (ou de palavra, se uma frase
/// sozinha passar do limite). Nenhum texto se perde: juntos, os trechos cobrem o documento.
pub fn chunks(text: &str, max_chars: usize) -> Vec<String> {
    let max_chars = max_chars.max(40);
    let mut out = Vec::new();
    let mut current = String::new();
    for sentence in sentences(text) {
        for piece in split_long(sentence, max_chars) {
            if !current.is_empty()
                && current.chars().count() + 1 + piece.chars().count() > max_chars
            {
                out.push(std::mem::take(&mut current));
            }
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(piece);
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

/// Frases: terminam em `.`, `!` ou `?` seguido de espaço (ou no fim), ou em quebra de linha. O ponto
/// de "6.000" ou de "3.5" não corta a frase.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let next = chars.peek().map(|&(_, n)| n);
        let ends =
            c == '\n' || (matches!(c, '.' | '!' | '?') && next.is_none_or(char::is_whitespace));
        if ends {
            out.push(&text[start..i + c.len_utf8()]);
            start = i + c.len_utf8();
        }
    }
    out.push(&text[start..]);
    out.into_iter()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn split_long(sentence: &str, max_chars: usize) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut rest = sentence;
    while rest.chars().count() > max_chars {
        let byte = rest
            .char_indices()
            .nth(max_chars)
            .map_or(rest.len(), |(i, _)| i);
        let cut = rest[..byte].rfind(' ').filter(|&i| i > 0).unwrap_or(byte);
        parts.push(rest[..cut].trim());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        parts.push(rest);
    }
    parts
}

/// Índice BM25 em memória.
#[derive(Debug, Default)]
pub struct Index {
    chunks: Vec<Chunk>,
    lengths: Vec<u32>,
    postings: HashMap<String, Vec<(u32, u32)>>,
    total_length: u64,
    max_chars: usize,
}

impl Index {
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Divide o documento em trechos e indexa cada um com o título junto (o título costuma ser o
    /// assunto da pergunta, e o trecho do meio do artigo não o repete).
    pub fn add(&mut self, doc: &Document) {
        for text in chunks(&doc.text, self.max_chars) {
            let id = self.chunks.len() as u32;
            let words = terms(&format!("{} {}", doc.title, text));
            let mut counts: HashMap<String, u32> = HashMap::new();
            for w in &words {
                *counts.entry(w.clone()).or_default() += 1;
            }
            for (w, n) in counts {
                self.postings.entry(w).or_default().push((id, n));
            }
            self.lengths.push(words.len() as u32);
            self.total_length += words.len() as u64;
            self.chunks.push(Chunk {
                doc: doc.id.clone(),
                title: doc.title.clone(),
                text,
            });
        }
    }

    /// Os `k` trechos de maior escore BM25 (só os com escore > 0), do maior para o menor.
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit<'_>> {
        if self.chunks.is_empty() || k == 0 {
            return Vec::new();
        }
        let n = self.chunks.len() as f32;
        let average = self.total_length as f32 / n;
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut matched: HashMap<u32, f32> = HashMap::new();
        let mut total_idf = 0.0;
        let mut seen = Vec::new();
        for term in terms(query) {
            if seen.contains(&term) {
                continue;
            }
            let Some(list) = self.postings.get(&term) else {
                total_idf += ((n + 0.5) / 0.5 + 1.0).ln();
                seen.push(term);
                continue;
            };
            let idf = ((n - list.len() as f32 + 0.5) / (list.len() as f32 + 0.5) + 1.0).ln();
            total_idf += idf;
            for &(id, tf) in list {
                let tf = tf as f32;
                let length = self.lengths[id as usize] as f32;
                let norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * length / average));
                *scores.entry(id).or_default() += idf * norm;
                *matched.entry(id).or_default() += idf;
            }
            seen.push(term);
        }
        let mut ranked: Vec<(u32, f32)> = scores.into_iter().filter(|(_, s)| *s > 0.0).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked
            .into_iter()
            .take(k)
            .map(|(id, score)| Hit {
                score,
                coverage: matched[&id] / total_idf,
                chunk: &self.chunks[id as usize],
            })
            .collect()
    }

    /// Lê documentos em JSONL (`{"id","title","text"}` por linha); linhas vazias são ignoradas.
    pub fn load_jsonl(&mut self, text: &str) -> Result<usize, serde_json::Error> {
        let mut count = 0;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            self.add(&serde_json::from_str::<Document>(line)?);
            count += 1;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str, title: &str, text: &str) -> Document {
        Document {
            id: id.into(),
            title: title.into(),
            text: text.into(),
        }
    }

    #[test]
    fn terms_fold_accents_and_drop_stopwords() {
        assert_eq!(
            terms("Qual é a Constituição de 1988?"),
            vec!["constituicao", "1988"]
        );
        assert_eq!(
            terms("São Paulo, Brasília"),
            vec!["sao", "paulo", "brasilia"]
        );
    }

    #[test]
    fn numbers_with_points_stay_whole() {
        let text = "Vive a 6.000 metros. Tem 3.5 cm de comprimento.";
        // o mínimo é 40 caracteres: cada frase vira um trecho, sem cortar o 6.000 nem o 3.5
        assert_eq!(
            chunks(text, 20),
            vec!["Vive a 6.000 metros.", "Tem 3.5 cm de comprimento."]
        );
        assert_eq!(chunks(text, 200), vec![text]);
    }

    #[test]
    fn chunks_cut_at_sentences_and_cover_the_text() {
        let text = "Primeira frase curta. Segunda frase um pouco maior que a primeira. Terceira.";
        let parts = chunks(text, 45);
        assert!(parts.iter().all(|p| p.chars().count() <= 45), "{parts:?}");
        assert_eq!(parts.join(" "), text);
        let long = "palavra ".repeat(30);
        assert!(chunks(&long, 50).iter().all(|p| p.chars().count() <= 50));
    }

    #[test]
    fn search_finds_the_right_chunk_and_ranks_by_bm25() {
        let mut index = Index::new(200);
        index.add(&doc(
            "1",
            "Descobrimento do Brasil",
            "Em 1500, a frota comandada por Pedro Álvares Cabral chegou ao litoral da Bahia.",
        ));
        index.add(&doc(
            "2",
            "Brasília",
            "Brasília é a capital federal do Brasil desde 1960, projetada por Lúcio Costa.",
        ));
        index.add(&doc(
            "3",
            "Porco-do-mar",
            "Scotoplanes é o nome científico do porco-do-mar, que vive a até 6.000 metros.",
        ));
        let hits = index.search("Quem comandou a frota que chegou ao Brasil em 1500?", 2);
        assert_eq!(hits[0].chunk.doc, "1");
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
        assert_eq!(index.search("capital do Brasil", 1)[0].chunk.doc, "2");
        assert_eq!(
            index.search("porco do mar nome cientifico", 1)[0].chunk.doc,
            "3"
        );
        assert!(
            index.search("xilofone quântico", 3).is_empty(),
            "sem termo em comum, sem resultado"
        );
    }

    #[test]
    fn coverage_is_the_share_of_question_words_found() {
        let mut index = Index::new(200);
        index.add(&doc(
            "p",
            "Porco-do-mar",
            "Scotoplanes é o nome científico do porco-do-mar.",
        ));
        index.add(&doc("b", "Brasil", "O Brasil fica na América do Sul."));
        let hit = &index.search("nome científico do porco-do-mar", 1)[0];
        assert_eq!((hit.chunk.doc.as_str(), hit.coverage), ("p", 1.0));
        let weak = &index.search("Quem descobriu o Brasil em 1500?", 1)[0];
        assert!(weak.coverage < 0.5, "{}", weak.coverage);
    }

    #[test]
    fn title_counts_for_chunks_in_the_middle_of_an_article() {
        let mut index = Index::new(60);
        index.add(&doc(
            "p",
            "Papa Silvestre II",
            "Era monge beneditino. Nasceu em Auvérnia. Morreu em Roma no ano de 1003.",
        ));
        let hits = index.search("Em que ano morreu Silvestre II?", 1);
        assert!(
            hits[0].chunk.text.contains("1003"),
            "{:?}",
            hits[0].chunk.text
        );
    }

    #[test]
    fn loads_jsonl_documents() {
        let mut index = Index::new(300);
        let n = index
            .load_jsonl("{\"id\":\"a\",\"title\":\"T\",\"text\":\"Um texto.\"}\n\n{\"id\":\"b\",\"title\":\"U\",\"text\":\"Outro.\"}\n")
            .unwrap();
        assert_eq!((n, index.len()), (2, 2));
        assert!(index.load_jsonl("{quebrado").is_err());
    }
}
