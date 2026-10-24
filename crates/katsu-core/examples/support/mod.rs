use std::{
    fs, io,
    path::{Path, PathBuf},
};

/// Lê .txt em ordem, remove cabeçalhos Gutenberg e divide cada obra antes do tokenizer.
pub fn corpus(path: &Path) -> io::Result<(String, String, Vec<PathBuf>)> {
    let mut files = if path.is_file() {
        vec![path.to_path_buf()]
    } else {
        fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<io::Result<Vec<_>>>()?
    };
    files.retain(|p| p.extension().is_some_and(|e| e == "txt"));
    files.sort();
    if files.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no .txt files"));
    }
    let (mut train, mut valid) = (String::new(), String::new());
    for path in &files {
        let raw = fs::read_to_string(path)?;
        let start = raw
            .find("*** START")
            .and_then(|i| raw[i..].find('\n').map(|j| i + j + 1))
            .unwrap_or(0);
        let end = raw[start..]
            .find("*** END")
            .map_or(raw.len(), |i| start + i);
        let text = raw[start..end]
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let desired = text.len() * 9 / 10;
        let split = text
            .char_indices()
            .find(|(i, c)| *i >= desired && c.is_whitespace())
            .map_or(text.len(), |(i, _)| i);
        train.push_str(&text[..split]);
        train.push('\n');
        valid.push_str(&text[split..]);
        valid.push('\n');
    }
    Ok((train, valid, files))
}
