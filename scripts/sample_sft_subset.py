"""Amostra reproduzível de um dataset SFT já dividido em .train/.eval, para um treino de prova.

  python3 scripts/sample_sft_subset.py data/sft/distill_arandu_50k data/sft/distill_arandu_10k --train 10000

Sorteia --train linhas do <origem>.train.jsonl (semente fixa, ordem original preservada) e copia o
<origem>.eval.jsonl inteiro, para a validação continuar comparável com o run completo. Grava
<destino>.manifest.json com contagens, semente e SHA-256. Recusa rodar enquanto a origem não tiver
os dois arquivos (a destilação só os grava no fim).

Também aceita o arquivo bruto de uma destilação ainda em andamento (sem .train/.eval):

  python3 scripts/sample_sft_subset.py data/sft/distill_arandu_50k.jsonl data/sft/distill_arandu_10k --train 10000

Nesse caso divide pelo mesmo hash da pergunta que o distill_dataset.py usa (--eval-fraction, padrão
0.1), então a validação sai contida na validação final do dataset completo e nenhuma pergunta
aparece dos dois lados. A última linha, se estiver no meio da escrita, é ignorada.

Depois, no pod: DATA=data/sft/distill_arandu_10k EPOCHS=1 ./deploy/cuda/runpod_train.sh
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="prefixo da origem, sem .train.jsonl")
    parser.add_argument("destination", type=Path, help="prefixo do subconjunto")
    parser.add_argument("--train", type=int, default=10000, help="exemplos de treino sorteados")
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--eval-fraction", type=float, default=0.1, help="só para o arquivo bruto")
    args = parser.parse_args(argv)

    raw = args.source.suffix == ".jsonl"
    if raw:
        train_in = eval_in = args.source
        if not train_in.is_file():
            print(f"falta {train_in}", file=sys.stderr)
            return 1
        text = train_in.read_text(encoding="utf-8")
        rows = [line for line in text[:text.rfind("\n") + 1].splitlines() if line.strip()]
        cut = int(args.eval_fraction * 10_000)
        is_eval = lambda line: int(hashlib.sha256(json.loads(line)["question_id"].encode()).hexdigest(), 16) % 10_000 < cut  # noqa: E731
        lines = [line for line in rows if not is_eval(line)]
        eval_lines = [line for line in rows if is_eval(line)]
    else:
        train_in = args.source.with_name(args.source.name + ".train.jsonl")
        eval_in = args.source.with_name(args.source.name + ".eval.jsonl")
        for path in (train_in, eval_in):
            if not path.is_file():
                print(f"falta {path}: a destilação ainda não terminou?", file=sys.stderr)
                return 1
        lines = [line for line in train_in.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.train <= 0 or args.train > len(lines):
        print(f"--train precisa estar entre 1 e {len(lines)}", file=sys.stderr)
        return 2
    for number, line in enumerate(lines, 1):
        try:
            json.loads(line)
        except ValueError:
            print(f"{train_in}:{number}: linha não é JSON", file=sys.stderr)
            return 1
    keep = sorted(random.Random(args.seed).sample(range(len(lines)), args.train))

    train_out = args.destination.with_name(args.destination.name + ".train.jsonl")
    eval_out = args.destination.with_name(args.destination.name + ".eval.jsonl")
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text("".join(lines[i] + "\n" for i in keep), encoding="utf-8")
    if raw:
        eval_out.write_text("".join(line + "\n" for line in eval_lines), encoding="utf-8")
    else:
        eval_out.write_bytes(eval_in.read_bytes())
    eval_count = sum(1 for line in eval_out.read_text(encoding="utf-8").splitlines() if line.strip())

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source), "seed": args.seed,
        "eval_split": f"sha256(question_id) % 10000 < {int(args.eval_fraction * 10_000)}" if raw else "da origem",
        "split": {"train": len(keep), "eval": eval_count, "source_train": len(lines)},
        "sha256": {train_out.name: sha256(train_out), eval_out.name: sha256(eval_out)},
    }
    source_manifest = args.source.with_suffix(".manifest.json") if raw else args.source.with_name(args.source.name + ".manifest.json")
    if source_manifest.is_file():
        manifest["source_manifest"] = json.loads(source_manifest.read_text(encoding="utf-8"))
    args.destination.with_name(args.destination.name + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{train_out}: {len(keep)} de {len(lines)}; {eval_out}: {eval_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
