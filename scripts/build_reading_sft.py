"""SFT de leitura (RAG) e de cálculo para o Arandu, no formato que o Katu Server usa.

Leitura: trecho do começo de um artigo da Wikipedia (split de TREINO, nunca a validação, de onde
sai a régua de leitura), pergunta cuja resposta está no trecho e resposta curta copiada dele; e
20% de casos com o trecho de outro artigo, em que o certo é dizer que não há dados.
Tipos: estado do município, ano e lugar de nascimento, ano de morte, família botânica, população,
ano de fundação e definição ("O que é X?" → a primeira frase).

Cálculo: perguntas de conta em português → `[CALC: expressão]`, que o servidor resolve.

Exclui os assuntos da régua de conhecimento (ARANDU/katu_native/benchmarks/conhecimento.jsonl),
para o SFT não ensinar as respostas que a régua mede.

  python3 scripts/build_reading_sft.py --reading 16000 --calc 4000 --out data/sft/reading_calc
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

KATSU = Path(__file__).resolve().parents[1]
CORPUS = KATSU / "data/pretrain/train.jsonl"
RULER = KATSU.parent / "ARANDU/katu_native/benchmarks/conhecimento.jsonl"
READING_SYSTEM = "Você é o Arandu. Responda em português do Brasil usando só o trecho."
CHAT_SYSTEM = ("Você é o Arandu. Responda em português do Brasil. Nesta conversa a busca web está desligada; "
               "não alegue ter pesquisado na internet.")
NO_DATA = ["Não tenho dados suficientes no trecho para responder.",
           "O trecho não informa isso; não tenho dados suficientes para responder."]
PASSAGE_CHARS = 320
CONTEXT = 256  # o train_sft recusa o dataset inteiro se um exemplo passar do contexto
TOKENIZER = KATSU / "runs/context-v3-base/tokenizer.bpe"


def fits(tok, example: dict) -> bool:
    """Mesma conta do train_sft: prompt + resposta + <|end|>, menos o último token, cabe no contexto."""
    return len(tok.encode(example["prompt"])) + len(tok.encode(example["response"] + "<|end|>")) - 1 <= CONTEXT

MUNICIPIO = re.compile(r"^(?P<x>[^\n(]{3,40}?) é um município brasileiro (?:do estado|localizado no estado) d[aoe] "
                       r"(?P<a>[A-ZÁÉÍÓÚ][\w ]{2,25}?)[,.]")
VIDA = re.compile(r"^(?P<x>[A-ZÁÉÍÓÚ][^\n(]{3,40}?) \((?P<lugar>[A-ZÁÉÍÓÚ][^,()\d]{2,30}), "
                  r"(?:\d{1,2}º? de \w+ de )?(?P<nasc>\d{4})(?: — [^,()]{2,30}, (?:\d{1,2}º? de \w+ de )?(?P<morte>\d{4}))?")
FAMILIA = re.compile(r"^(?P<x>[A-Z][a-z]+) é um g[eé]nero (?:botânico )?pertencente à família (?P<a>[A-Z][a-z]+aceae)")
POPULACAO = re.compile(r"popula[çc][ãa]o (?:estimada |total )?(?:[^.]{0,60}?)(?:era |é |de )(?:de )?(?P<a>\d{1,3}(?:\.\d{3})+) habitantes")
FUNDACAO = re.compile(r"(?P<verbo>fundad[ao]|criad[ao]) em (?:\d{1,2} de \w+ de )?(?P<a>1[0-9]{3}|20[0-2][0-9])\b")
DEFINICAO = re.compile(r"^(?P<x>[A-ZÁÉÍÓÚ][^\n(.]{2,40}?) (?:é|foi) (?:um|uma|o|a) [^.]{10,160}\.")


def passage_of(text: str) -> tuple[str, str]:
    """(título, trecho). O corpus separa título e texto por uma linha em branco, menos quando o
    texto já começa pelo título: aí o título fica vazio (desconhecido), em vez do parágrafo."""
    title, sep, body = text.partition("\n\n")
    if not sep or len(title) > 80:
        title, body = "", text
    body = body.strip().replace("\n", " ")
    if len(body) > PASSAGE_CHARS:
        body = body[:PASSAGE_CHARS].rsplit(" ", 1)[0]
    return title.strip(), body


def qa(title: str, p: str) -> list[tuple[str, str, str]]:
    """(tipo, pergunta, resposta) cujas respostas estão em `p`."""
    out = []
    if m := MUNICIPIO.search(p):
        out.append(("municipio", f"Em que estado fica o município de {m['x']}?",
                    f"O município de {m['x']} fica no estado d{'o' if m['a'].split()[0] in ('Maranhão', 'Pará', 'Paraná', 'Piauí', 'Amazonas', 'Amapá', 'Acre', 'Ceará', 'Espírito', 'Rio', 'Mato', 'Tocantins', 'Distrito') else 'e'} {m['a']}."))
    if m := VIDA.search(p):
        out.append(("nascimento_ano", f"Em que ano nasceu {m['x']}?", f"{m['x']} nasceu em {m['nasc']}."))
        out.append(("nascimento_lugar", f"Onde nasceu {m['x']}?", f"{m['x']} nasceu em {m['lugar'].strip()}."))
        if m["morte"]:
            out.append(("morte_ano", f"Em que ano morreu {m['x']}?", f"{m['x']} morreu em {m['morte']}."))
    if m := FAMILIA.search(p):
        out.append(("familia", f"A que família pertence o gênero {m['x']}?",
                    f"O gênero {m['x']} pertence à família {m['a']}."))
    if (m := POPULACAO.search(p)) and title:
        out.append(("populacao", f"Qual é a população de {title}?",
                    f"Segundo o trecho, {title} tem {m['a']} habitantes."))
    if (m := FUNDACAO.search(p)) and title:
        out.append(("fundacao", f"Em que ano {title} foi {m['verbo'].lower()}?", f"{title} foi {m['verbo'].lower()} em {m['a']}."))
    if m := DEFINICAO.search(p):
        out.append(("definicao", f"O que é {m['x']}?", m.group(0).strip()))
    return out


def calc_examples(r: random.Random, n: int) -> list[tuple[str, str]]:
    def num(lo, hi):
        return r.randint(lo, hi)
    makers = [
        lambda a, b: (f"Quanto é {a} vezes {b}?", f"{a}*{b}"),
        lambda a, b: (f"Quanto é {a} mais {b}?", f"{a}+{b}"),
        lambda a, b: (f"Quanto é {a} menos {b}?", f"{a}-{b}"),
        lambda a, b: (f"Quanto é {a * b} dividido por {b}?", f"{a * b}/{b}"),
        lambda a, b: (f"Quanto é {a}*{b}?", f"{a}*{b}"),
        lambda a, b: (f"Calcule {a} + {b} × 3.", f"{a}+{b}*3"),
        lambda a, b: (f"Quanto é ({a} + {b}) vezes 2?", f"({a}+{b})*2"),
        lambda a, b: (f"Qual a raiz quadrada de {a * a}?", f"sqrt({a * a})"),
        lambda a, b: (f"Quanto é a raiz quadrada de {a}?", f"sqrt({a})"),
        lambda a, b: (f"Quanto é {b % 90 + 5}% de {a * 10}?", f"{b % 90 + 5}/100*{a * 10}"),
        lambda a, b: (f"Quanto é {a % 12 + 2} elevado a {b % 5 + 2}?", f"{a % 12 + 2}^{b % 5 + 2}"),
        lambda a, b: (f"Se cada caixa tem {a} itens, quantos itens há em {b} caixas?", f"{a}*{b}"),
        lambda a, b: (f"Divida {a * b} por {a}.", f"{a * b}/{a}"),
    ]
    out, seen = [], set()
    while len(out) < n:
        q, expr = r.choice(makers)(num(2, 999), num(2, 999))
        if q not in seen:
            seen.add(q)
            out.append((q, f"[CALC: {expr}]"))
    return out


def row(i, kind, system, user, response):
    prompt = f"<|system|>{system}\n<|user|>{user}\n<|assistant|>"
    return {"id": f"{kind}-{i}", "kind": kind, "prompt": prompt, "response": response, "text": prompt + response + "<|end|>"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--reading", type=int, default=16000)
    p.add_argument("--calc", type=int, default=4000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2030)
    a = p.parse_args(argv)
    r = random.Random(a.seed)
    ruler = [json.loads(l)["question"] for l in RULER.read_text(encoding="utf-8").splitlines() if l.strip()]
    banned = {w for q in ruler for w in re.findall(r"[A-ZÁÉÍÓÚ][\wáéíóúâêôãõç-]{3,}(?: [A-ZÁÉÍÓÚ][\w-]+)*", q)}
    pools: dict[str, list] = {}
    passages = []
    with open(CORPUS, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["source"] != "ptwiki":
                continue
            title, body = passage_of(doc["text"])
            if any(b in title or b in body[:80] for b in banned):
                continue
            passages.append(body)
            for kind, q, ans in qa(title, body):
                pools.setdefault(kind, []).append((q, ans, body))
    answerable = int(a.reading * 0.8)
    per_kind = answerable // len(pools)
    reading = []
    for kind, items in sorted(pools.items()):
        r.shuffle(items)
        reading += [(kind, *x) for x in items[:per_kind]]
    r.shuffle(reading)
    rows = [row(i, kind, READING_SYSTEM, f"Trecho: {body}\nPergunta: {q}", ans)
            for i, (kind, q, ans, body) in enumerate(reading)]
    # um caso sem resposta para cada quatro com resposta (20% da leitura)
    for i in range(len(rows) // 4):
        _, q, _, own = reading[r.randrange(len(reading))]
        other = r.choice(passages)
        if other == own:
            continue
        rows.append(row(i, "ausente", READING_SYSTEM, f"Trecho: {other}\nPergunta: {q}", r.choice(NO_DATA)))
    rows += [row(i, "calc", CHAT_SYSTEM, q, ans) for i, (q, ans) in enumerate(calc_examples(r, a.calc))]
    sys.path.insert(0, str(KATSU.parent / "ARANDU/scripts"))
    from katu_bpe import KatuBpe
    tok = KatuBpe.load(TOKENIZER)
    before = len(rows)
    rows = [x for x in rows if fits(tok, x)]
    too_long = before - len(rows)
    r.shuffle(rows)
    val = [x for x in rows if int(hashlib.sha256(x["id"].encode()).hexdigest(), 16) % 100 < 5]
    train = [x for x in rows if x not in val]
    for name, data in (("train", train), ("eval", val)):
        Path(f"{a.out}.{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    counts = {}
    for x in rows:
        counts[x["kind"]] = counts.get(x["kind"], 0) + 1
    manifest = {"generator": "scripts/build_reading_sft.py", "seed": a.seed, "train": len(train), "eval": len(val),
                "kinds": counts, "pool_sizes": {k: len(v) for k, v in pools.items()}, "excluded_ruler_terms": len(banned), "dropped_over_context": too_long,
                "source": "Wikipedia PT (split de treino do pré-treino), CC BY-SA 4.0"}
    Path(f"{a.out}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
