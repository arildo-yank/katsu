"""SFT de leitura de documentos de empresa sobre o mac-reinforce-pii-v1 (v1.2.0).

  python3 scripts/build_docs_sft.py --out data/sft/mix_docs

A medida no "HD do cliente" (ARANDU agi/README.md) mostrou que o modelo quase não responde sozinho em
documentos: livros técnicos em inglês e trechos de tabela falham. Aqui:
- as perguntas de data/sft/qa_docs_train.jsonl (ARANDU scripts/distill_reading_qa.py --docs: trechos
  de documentos públicos em PT e EN, cortados como o Katsu corta), no formato de leitura, com a
  resposta no idioma do trecho;
- um caso sem resposta para cada quatro (a pergunta com um trecho de outro documento, mesmo idioma);
- para não perder o que a v1.2.0 tem: 1.500 de dado pessoal, 1.500 dos outros tipos de contexto,
  2.500 de leitura variada e 1.000 de leitura e cálculo.
Os 5% de validação saem por trecho (nenhum trecho da validação aparece no treino). Exemplos acima do
contexto (256 tokens) saem, contados com o tokenizer do Katsu.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from build_reading_sft import KATSU, NO_DATA, READING_SYSTEM, TOKENIZER, fits, row

SFT = KATSU / "data/sft"
KEEP = {"pii": 1500, "context": 1500, "variada": 2500, "reading_calc": 1000}


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def held_out(passage: str) -> bool:
    return int(hashlib.sha256(passage.encode()).hexdigest(), 16) % 100 < 5


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--qa", type=Path, default=SFT / "qa_docs_train.jsonl")
    p.add_argument("--passages", type=Path, required=True, help="passages.jsonl de build_doc_passages.py (idioma)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2042)
    a = p.parse_args(argv)
    r = random.Random(a.seed)
    language = {x["passage"]: x["lang"] for x in load(a.passages)}
    qa = load(a.qa)
    rows = [row(i, "docs", READING_SYSTEM, f"Trecho: {x['passage']}\nPergunta: {x['question']}", x["answer"])
            | {"passage": x["passage"]} for i, x in enumerate(qa)]
    by_lang = {}
    for x in sorted({x["passage"] for x in qa}):
        by_lang.setdefault(language.get(x, "pt"), []).append(x)
    for i in range(len(qa) // 4):
        x = qa[r.randrange(len(qa))]
        other = r.choice(by_lang[language.get(x["passage"], "pt")])
        if other != x["passage"]:
            rows.append(row(i, "docs_ausente", READING_SYSTEM, f"Trecho: {other}\nPergunta: {x['question']}",
                            r.choice(NO_DATA)) | {"passage": other})
    sys.path.insert(0, str(KATSU.parent / "ARANDU/scripts"))
    from katsu_bpe import KatsuBpe
    tok = KatsuBpe.load(TOKENIZER)
    before = len(rows)
    rows = [x for x in rows if fits(tok, x)]
    too_long = before - len(rows)
    val = [x for x in rows if held_out(x["passage"])]
    train = [x for x in rows if not held_out(x["passage"])]
    context = load(SFT / "context_arandu_v3.train.jsonl")
    old = {"pii": [x for x in context if x["kind"].startswith("pii")],
           "context": [x for x in context if not x["kind"].startswith("pii")],
           "variada": [x for x in load(SFT / "mix_variada.train.jsonl") if x.get("kind", "").startswith("variada")],
           "reading_calc": load(SFT / "mix_reading_calc.train.jsonl")}
    for name, n in KEEP.items():
        train += r.sample(old[name], min(n, len(old[name])))
    r.shuffle(train)
    # Validação: trechos de documento separados + um pouco do que a v1.2.0 já sabe.
    ev = load(SFT / "context_arandu_v3.eval.jsonl")
    val += r.sample(ev, min(150, len(ev))) + r.sample(load(SFT / "mix_variada.eval.jsonl"), 150)
    for name, data in (("train", train), ("eval", val)):
        Path(f"{a.out}.{name}.jsonl").write_text(
            "".join(json.dumps({k: v for k, v in x.items() if k != "passage"}, ensure_ascii=False) + "\n"
                    for x in data))
    count = lambda kind, data: sum(x.get("kind") == kind for x in data)
    manifest = {"generator": "scripts/build_docs_sft.py", "seed": a.seed, "train": len(train), "eval": len(val),
                "docs_train": count("docs", train), "docs_ausente_train": count("docs_ausente", train),
                "docs_eval": count("docs", val), "docs_ausente_eval": count("docs_ausente", val),
                "kept_old": KEEP, "dropped_over_context": too_long,
                "source": "documentos públicos (IRS, NIST, NASA, NR-12, eSocial, LGPD); perguntas pelo DeepSeek (API)"}
    Path(f"{a.out}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
