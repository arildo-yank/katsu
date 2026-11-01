//! BPE byte-level puro Rust. Treino incremental por palavras, com frequências ponderadas.
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, HashSet},
    io::{self, Read, Write},
};
pub const INSTRUCTION_TOKENS: [&str; 4] = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"];
type Pair = (usize, usize);
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BpeTokenizer {
    special: bool,
    merges: Vec<Pair>,
    tokens: Vec<Vec<u8>>,
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn segments(text: &str) -> impl Iterator<Item = &str> {
    text.split_inclusive(char::is_whitespace)
}
fn encode_word(word: &str, ranks: &HashMap<Pair, usize>) -> Vec<usize> {
    let mut ids: Vec<usize> = word.bytes().map(usize::from).collect();
    loop {
        let best = ids
            .windows(2)
            .filter_map(|p| ranks.get(&(p[0], p[1])).map(|&rank| (rank, (p[0], p[1]))))
            .min();
        let Some((rank, pair)) = best else { break };
        ids = merge(&ids, pair, rank + 256);
    }
    ids
}
/// Palavras guardadas por codificador; acima disso, as novas são codificadas sem cache.
const WORD_CACHE_LIMIT: usize = 1 << 20;
pub struct OrdinaryEncoder {
    ranks: HashMap<Pair, usize>,
    cache: HashMap<String, Vec<usize>>,
}
impl OrdinaryEncoder {
    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        let mut output = Vec::new();
        for word in segments(text) {
            if let Some(ids) = self.cache.get(word) {
                output.extend_from_slice(ids);
                continue;
            }
            let ids = encode_word(word, &self.ranks);
            output.extend_from_slice(&ids);
            if self.cache.len() < WORD_CACHE_LIMIT {
                self.cache.insert(word.to_owned(), ids);
            }
        }
        output
    }
}
fn merge(ids: &[usize], pair: Pair, token: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            out.push(token);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}
impl BpeTokenizer {
    fn bytes() -> Self {
        Self {
            special: false,
            merges: Vec::new(),
            tokens: (0..256).map(|i| vec![i as u8]).collect(),
        }
    }
    /// Para quando não há pares com frequência >=2. Não inventa tokens para atingir a meta.
    pub fn train(text: &str, vocab_size: usize) -> io::Result<Self> {
        if !(256..=65_536).contains(&vocab_size) {
            return Err(invalid("vocab must be 256..=65536"));
        }
        let mut words = BTreeMap::<Vec<usize>, u64>::new();
        for word in segments(text) {
            *words
                .entry(word.bytes().map(usize::from).collect())
                .or_default() += 1;
        }
        let mut words: Vec<_> = words.into_iter().collect();
        let mut counts = HashMap::<Pair, u64>::new();
        let mut locations = HashMap::<Pair, HashSet<usize>>::new();
        for (i, (ids, weight)) in words.iter().enumerate() {
            for pair in ids.windows(2) {
                let pair = (pair[0], pair[1]);
                *counts.entry(pair).or_default() += weight;
                locations.entry(pair).or_default().insert(i);
            }
        }
        let mut heap: BinaryHeap<_> = counts
            .iter()
            .filter(|(_, c)| **c >= 2)
            .map(|(&p, &c)| (c, Reverse(p)))
            .collect();
        let mut tokenizer = Self::bytes();
        while tokenizer.tokens.len() < vocab_size {
            let Some((count, Reverse(pair))) = heap.pop() else {
                break;
            };
            if counts.get(&pair).copied().unwrap_or(0) != count {
                continue;
            }
            let token = tokenizer.tokens.len();
            let mut bytes = tokenizer.tokens[pair.0].clone();
            bytes.extend_from_slice(&tokenizer.tokens[pair.1]);
            tokenizer.tokens.push(bytes);
            tokenizer.merges.push(pair);
            let affected: Vec<_> = locations
                .get(&pair)
                .into_iter()
                .flatten()
                .copied()
                .collect();
            let mut touched = HashSet::new();
            for i in affected {
                let (ids, weight) = &mut words[i];
                for p in ids.windows(2) {
                    let p = (p[0], p[1]);
                    *counts.get_mut(&p).expect("existing pair") -= *weight;
                    locations.get_mut(&p).expect("existing location").remove(&i);
                    touched.insert(p);
                }
                *ids = merge(ids, pair, token);
                for p in ids.windows(2) {
                    let p = (p[0], p[1]);
                    *counts.entry(p).or_default() += *weight;
                    locations.entry(p).or_default().insert(i);
                    touched.insert(p);
                }
            }
            for p in touched {
                if counts[&p] >= 2 {
                    heap.push((counts[&p], Reverse(p)));
                }
            }
        }
        Ok(tokenizer)
    }
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    /// Anexa quatro IDs em ordem fixa após o vocabulário base; nunca renumera merges.
    pub fn reserve_instruction_tokens(&mut self) -> io::Result<()> {
        if self.special {
            return Ok(());
        }
        if self.tokens.len() + 4 > 65536 {
            return Err(invalid("no room for instruction tokens"));
        }
        for token in INSTRUCTION_TOKENS {
            self.tokens.push(token.as_bytes().to_vec());
        }
        self.special = true;
        Ok(())
    }
    pub fn special_token_id(&self, token: &str) -> Option<usize> {
        if !self.special {
            return None;
        }
        INSTRUCTION_TOKENS
            .iter()
            .position(|&name| name == token)
            .map(|i| 256 + self.merges.len() + i)
    }
    /// Marcadores explícitos são estruturais. Use encode_ordinary para texto literal não confiável.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if !self.special {
            return self.encode_ordinary(text);
        }
        let mut rest = text;
        let mut out = Vec::new();
        while !rest.is_empty() {
            let next = INSTRUCTION_TOKENS
                .iter()
                .enumerate()
                .filter_map(|(i, t)| rest.find(t).map(|p| (p, i)))
                .min();
            match next {
                Some((pos, i)) => {
                    out.extend(self.encode_ordinary(&rest[..pos]));
                    out.push(256 + self.merges.len() + i);
                    rest = &rest[pos + INSTRUCTION_TOKENS[i].len()..];
                }
                None => {
                    out.extend(self.encode_ordinary(rest));
                    break;
                }
            }
        }
        out
    }
    pub fn encode_ordinary(&self, text: &str) -> Vec<usize> {
        let ranks = self.ranks();
        let mut output = Vec::new();
        for word in segments(text) {
            output.extend(encode_word(word, &ranks));
        }
        output
    }
    /// Codificador de texto literal para lotes grandes (corpus de pré-treino). Mesma saída de
    /// `encode_ordinary`, mas monta a tabela de merges uma vez só e guarda as palavras já vistas.
    pub fn ordinary_encoder(&self) -> OrdinaryEncoder {
        OrdinaryEncoder {
            ranks: self.ranks(),
            cache: HashMap::new(),
        }
    }
    fn ranks(&self) -> HashMap<Pair, usize> {
        self.merges
            .iter()
            .copied()
            .enumerate()
            .map(|(rank, p)| (p, rank))
            .collect()
    }
    pub fn decode_bytes(&self, ids: &[usize]) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(
                self.tokens
                    .get(id)
                    .ok_or_else(|| invalid("token out of vocabulary"))?,
            );
        }
        Ok(bytes)
    }
    pub fn decode(&self, ids: &[usize]) -> io::Result<String> {
        String::from_utf8(self.decode_bytes(ids)?)
            .map_err(|_| invalid("tokens do not form complete UTF-8"))
    }
    pub fn save(&self, writer: &mut impl Write) -> io::Result<()> {
        writer.write_all(if self.special {
            b"KATSUBP2"
        } else {
            b"KATSUBP1"
        })?;
        writer.write_all(&(self.merges.len() as u32).to_le_bytes())?;
        for &(a, b) in &self.merges {
            writer.write_all(&(a as u32).to_le_bytes())?;
            writer.write_all(&(b as u32).to_le_bytes())?;
        }
        Ok(())
    }
    pub fn load(reader: &mut impl Read) -> io::Result<Self> {
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"KATSUBP1" && &magic != b"KATSUBP2" {
            return Err(invalid("invalid BPE magic"));
        }
        fn word(r: &mut impl Read) -> io::Result<usize> {
            let mut b = [0; 4];
            r.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b) as usize)
        }
        let count = word(reader)?;
        if count > 65_536 - 256 {
            return Err(invalid("BPE vocabulary too large"));
        }
        let mut tokenizer = Self::bytes();
        let mut pairs = HashSet::new();
        for _ in 0..count {
            let (a, b) = (word(reader)?, word(reader)?);
            if a >= tokenizer.tokens.len() || b >= tokenizer.tokens.len() || !pairs.insert((a, b)) {
                return Err(invalid("invalid BPE merge"));
            }
            let size = tokenizer.tokens[a]
                .len()
                .checked_add(tokenizer.tokens[b].len())
                .ok_or_else(|| invalid("token overflow"))?;
            if size > 1_048_576 {
                return Err(invalid("BPE token exceeds 1 MB"));
            }
            let mut bytes = tokenizer.tokens[a].clone();
            bytes.extend_from_slice(&tokenizer.tokens[b]);
            tokenizer.tokens.push(bytes);
            tokenizer.merges.push((a, b));
        }
        if &magic == b"KATSUBP2" {
            tokenizer.reserve_instruction_tokens()?;
        }
        Ok(tokenizer)
    }
}
