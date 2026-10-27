"""SFT de leitura com perguntas variadas, para o teste das recusas indevidas (ARANDU docs/ARANDU_V2_PLAN.md).

  python3 scripts/build_variada_sft.py --out data/sft/mix_variada

Junta:
- as perguntas de tipos livres de data/sft/qa_variada_train.jsonl (ARANDU scripts/distill_reading_qa.py,
  trechos do split de treino), no formato de leitura;
- um caso sem resposta para cada quatro (a pergunta com o trecho de outro artigo);
- 8.000 exemplos sorteados do mix_reading_calc.train (leitura dos 7 tipos, cálculo, contexto e
  destilação), para o modelo não esquecer o que já sabe.
Exemplos acima do contexto (256 tokens) saem, contados com o tokenizer do Katsu.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from build_reading_sft import KATSU, NO_DATA, READING_SYSTEM, TOKENIZER, fits, row

KEEP_OLD = 8000


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--qa", type=Path, default=KATSU / "data/sft/qa_variada_train.jsonl")
    p.add_argument("--old", type=Path, default=KATSU / "data/sft/mix_reading_calc.train.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2033)
    a = p.parse_args(argv)
    r = random.Random(a.seed)
    qa = [json.loads(l) for l in a.qa.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [row(i, "variada", READING_SYSTEM, f"Trecho: {x['passage']}\nPergunta: {x['question']}", x["answer"])
            for i, x in enumerate(qa)]
    passages = sorted({x["passage"] for x in qa})
    for i in range(len(qa) // 4):
        x = qa[r.randrange(len(qa))]
        other = r.choice(passages)
        if other != x["passage"]:
            rows.append(row(i, "variada_ausente", READING_SYSTEM, f"Trecho: {other}\nPergunta: {x['question']}",
                            r.choice(NO_DATA)))
    sys.path.insert(0, str(KATSU.parent / "ARANDU/scripts"))
    from katsu_bpe import KatsuBpe
    tok = KatsuBpe.load(TOKENIZER)
    before = len(rows)
    rows = [x for x in rows if fits(tok, x)]
    too_long = before - len(rows)
    val = [x for x in rows if int(hashlib.sha256(x["id"].encode()).hexdigest(), 16) % 100 < 5]
    train = [x for x in rows if x not in val]
    old = [json.loads(l) for l in a.old.read_text(encoding="utf-8").splitlines() if l.strip()]
    train += r.sample(old, min(KEEP_OLD, len(old)))
    r.shuffle(train)
    for name, data in (("train", train), ("eval", val)):
        Path(f"{a.out}.{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    manifest = {"generator": "scripts/build_variada_sft.py", "seed": a.seed, "train": len(train), "eval": len(val),
                "variada": sum(x["kind"] == "variada" for x in rows),
                "variada_ausente": sum(x["kind"] == "variada_ausente" for x in rows),
                "old_examples": min(KEEP_OLD, len(old)), "dropped_over_context": too_long,
                "source": "Wikipedia PT (split de treino), CC BY-SA 4.0; perguntas pelo DeepSeek (API)"}
    Path(f"{a.out}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
