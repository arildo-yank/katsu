"""Monta o corpus de pré-treino em português: Wikipedia PT (dump CirrusSearch, já em texto puro)
e as obras brasileiras em domínio público do corpus-bilingual-v1. Gera JSONL (um documento por
linha) para o tokenize_corpus, com a divisão treino/validação fixa por hash.

  python3 scripts/build_pretrain_corpus.py \
      --wiki data/pretrain/raw/ptwiki-20251229-cirrussearch-content.json.gz --out data/pretrain

Saída: train.jsonl, val.jsonl e manifest.json (fontes, licenças, contagens, SHA-256 do dump).
"""
import argparse
import hashlib
import io
import itertools
import json
import multiprocessing
import os
import subprocess
from pathlib import Path

KATSU = Path(__file__).resolve().parents[1]
BOOKS = KATSU / "data/corpus-bilingual-v1"
WIKI_LICENSE = "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); autoria no histórico de cada artigo em pt.wikipedia.org"
VAL_PER_MILLE = 5  # 0,5% dos artigos ficam fora do treino, para medir a loss do corpus
MIN_CHARS = 300  # tira esboços de uma linha


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def is_val(key: str) -> bool:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000 < VAL_PER_MILLE


def article(line: str):
    """Uma linha do dump -> (page_id, título, texto) ou None. O dump alterna uma linha de índice e
    uma de documento; só o documento interessa."""
    doc = json.loads(line)
    if "index" in doc or str(doc.get("namespace")) != "0":
        return None
    text = " ".join(str(doc.get("text") or "").replace("\xa0", " ").split())
    return str(doc.get("page_id")), str(doc.get("title") or "").strip(), text


def chunk_articles(lines: list[str]) -> list:
    return [a for a in map(article, lines) if a is not None]


def wiki_articles(path: Path, workers: int):
    """Artigos na ordem do dump. A descompressão roda no gzip do sistema e o parse do JSON em
    `workers` processos (o parse é o gargalo: ~1,5 MB/s num processo só)."""
    with subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE) as gz:
        stream = io.TextIOWrapper(gz.stdout, encoding="utf-8")
        chunks = iter(lambda: list(itertools.islice(stream, 512)), [])
        with multiprocessing.Pool(workers) as pool:
            for chunk in pool.imap(chunk_articles, chunks, chunksize=4):
                yield from chunk
    if gz.returncode != 0:
        raise SystemExit(f"gzip falhou ({gz.returncode}) em {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wiki", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"wikipedia": {"train": 0, "val": 0, "skipped_short": 0, "chars": 0},
              "books": {"train": 0, "val": 0, "chars": 0}}
    temp = {split: args.out / f"{split}.jsonl.tmp" for split in ("train", "val")}
    files = {split: open(path, "w", encoding="utf-8") for split, path in temp.items()}
    try:
        for page_id, title, text in wiki_articles(args.wiki, args.workers):
            if len(text) < MIN_CHARS:
                counts["wikipedia"]["skipped_short"] += 1
                continue
            split = "val" if is_val(f"ptwiki:{page_id}") else "train"
            body = f"{title}\n\n{text}" if title and not text.startswith(title) else text
            files[split].write(json.dumps({"text": body, "source": "ptwiki", "id": page_id}, ensure_ascii=False) + "\n")
            counts["wikipedia"][split] += 1
            counts["wikipedia"]["chars"] += len(body)
            done = counts["wikipedia"]["train"] + counts["wikipedia"]["val"]
            if done % 100_000 == 0:
                print(f"wikipedia: {done} artigos", flush=True)
        manifest = json.loads((BOOKS / "manifest.json").read_text())
        books = [d for d in manifest["documents"] if d["language"] == "pt"]
        for doc in books:
            raw = (BOOKS / doc["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != doc["sha256"]:
                raise SystemExit(f"hash diferente do manifesto: {doc['path']}")
            text = raw.decode("utf-8").replace("\r\n", "\n")
            split = "val" if doc["split"] == "validation" else "train"
            files[split].write(json.dumps({"text": text, "source": "book", "id": doc["id"]}, ensure_ascii=False) + "\n")
            counts["books"][split] += 1
            counts["books"]["chars"] += len(text)
    finally:
        for file in files.values():
            file.close()
    for split, path in temp.items():
        path.replace(args.out / f"{split}.jsonl")
    record = {
        "version": "pretrain-pt-v1",
        "sources": [
            {"name": "Wikipedia em português", "file": args.wiki.name, "sha256": sha256(args.wiki),
             "url": f"https://dumps.wikimedia.org/other/cirrussearch/20251229/{args.wiki.name}",
             "license": WIKI_LICENSE, "field": "text (texto puro do CirrusSearch, sem marcação wiki)",
             "filters": f"namespace 0, pelo menos {MIN_CHARS} caracteres, espaços normalizados"},
            {"name": "Literatura brasileira em domínio público", "manifest": "data/corpus-bilingual-v1/manifest.json",
             "documents": [d["id"] for d in books], "license": "domínio público no Brasil (Lei 9.610/98, art. 41); ver docs/corpus.md"},
        ],
        "excluded": "artigos em inglês do corpus-bilingual-v1 (fora do escopo do pré-treino em português)",
        "split": f"validação = {VAL_PER_MILLE}/1000 dos artigos por SHA-256 do page_id; livros seguem o split do manifesto",
        "counts": counts,
    }
    (args.out / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
