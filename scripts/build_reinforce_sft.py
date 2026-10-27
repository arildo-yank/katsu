"""Reforço curto sobre o mac-variada-v1 (v1.1.0): traz de volta a recusa de dados pessoais, que o SFT
de perguntas variadas enfraqueceu (AranduBench: o e-mail pessoal deixou de ser recusado).

  python3 scripts/build_reinforce_sft.py --out data/sft/mix_reinforce_pii

No SFT variado, os exemplos de dado pessoal caíram para 1,1% da mistura (188 de 16.939). Aqui:
- 2.000 de dado pessoal e 2.000 dos outros tipos de contexto (falta de dados, instrução escondida,
  julgamento, cópia, resumo), do context_arandu_v3 (gerado sem os valores do AranduBench);
- 3.000 de leitura variada e 1.000 da mistura de leitura e cálculo, para não perder o que a v1.1.0 ganhou.
"""
import argparse
import json
import random
from pathlib import Path

KATSU = Path(__file__).resolve().parents[1]
SFT = KATSU / "data/sft"


def load(name):
    return [json.loads(l) for l in (SFT / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2034)
    a = p.parse_args(argv)
    r = random.Random(a.seed)
    context = load("context_arandu_v3.train.jsonl")
    pii = [x for x in context if x["kind"].startswith("pii")]
    other = [x for x in context if not x["kind"].startswith("pii")]
    variada = [x for x in load("mix_variada.train.jsonl") if x.get("kind", "").startswith("variada")]
    old = load("mix_reading_calc.train.jsonl")
    train = r.sample(pii, 2000) + r.sample(other, 2000) + r.sample(variada, 3000) + r.sample(old, 1000)
    r.shuffle(train)
    ev = load("context_arandu_v3.eval.jsonl")
    val = r.sample(ev, min(250, len(ev))) + r.sample(load("mix_variada.eval.jsonl"), 250)
    for name, data in (("train", train), ("eval", val)):
        Path(f"{a.out}.{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    manifest = {"generator": "scripts/build_reinforce_sft.py", "seed": a.seed, "train": len(train), "eval": len(val),
                "pii": 2000, "context_other": 2000, "variada": 3000, "reading_calc": 1000,
                "sources": ["context_arandu_v3", "mix_variada", "mix_reading_calc"]}
    Path(f"{a.out}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
