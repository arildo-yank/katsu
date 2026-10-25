#!/usr/bin/env python3
"""Dimensão efetiva dos pesos de um checkpoint Katsu (KATSUCK1), para comparar runs de tamanhos diferentes.

  python3 scripts/effective_dimension.py runs/mac-sft-special-v1 --base runs/mac-sft-special-v1/migrated.ckpt \\
      --json runs/mac-sft-special-v1/effective-dimension.json
  python3 scripts/effective_dimension.py runs/<run-134m> --compare runs/mac-sft-special-v1/effective-dimension.json

Para cada matriz W com valores singulares s (k = min das dimensões):
  erank  posto efetivo, exp(H(p)) com p = s / sum(s) (Roy & Vetterli, 2007)
  pr     razão de participação das variâncias, (sum s²)² / sum s⁴
  d90    menor número de direções com 90% de sum s²
  frac   erank / máximo possível; é o número comparável entre modelos de larguras diferentes
Cada linha traz a mesma medida numa matriz gaussiana do mesmo formato (semente fixa), como referência.
Embeddings e saída são centrados por token antes da SVD. Mede pesos, não ativações. Requer numpy.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

import numpy as np

MAGIC = b"KATSUCK1"


def load_checkpoint(path) -> list[np.ndarray]:
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: não é um checkpoint Katsu (KATSUCK1)")
    try:
        (count,) = struct.unpack_from("<Q", data, 8)
        offset, tensors = 16, []
        for _ in range(count):
            (rank,) = struct.unpack_from("<Q", data, offset)
            offset += 8
            shape = struct.unpack_from(f"<{rank}Q", data, offset)
            offset += 8 * rank
            size = int(np.prod(shape, dtype=np.int64))
            tensors.append(np.frombuffer(data, "<f4", size, offset).reshape(shape))
            offset += 4 * size
    except (struct.error, ValueError) as error:
        raise ValueError(f"{path}: checkpoint truncado ou corrompido") from error
    if offset != len(data):
        raise ValueError(f"{path}: {len(data) - offset} bytes sobrando após os tensores")
    return tensors


def named_params(params, config) -> dict[str, np.ndarray]:
    """Mesmo layout de KatsuLmConfig: wte, wpe, camadas (LN1, cabeças Q/K/V/O, LN2, MLP), LN final, head."""
    names = ["wte", "wpe"]
    for layer in range(config["layers"]):
        names += [f"L{layer}.ln1_g", f"L{layer}.ln1_b"]
        for head in range(config["heads"]):
            names += [f"L{layer}.h{head}.{w}" for w in ("Wq", "Wk", "Wv", "Wo")]
        names += [f"L{layer}.{w}" for w in ("ln2_g", "ln2_b", "W1", "b1", "W2", "b2")]
    names += ["lnf_g", "lnf_b", "head"]
    if len(names) != len(params):
        raise ValueError(f"config.json descreve {len(names)} tensores, o checkpoint tem {len(params)}")
    named = dict(zip(names, params))
    d, v = config["embed_dim"], config["vocab_size"]
    if named["wte"].shape != (v, d) or named["head"].shape != (d, v):
        raise ValueError("formas de wte/head não conferem com config.json")
    return named


def spectrum_stats(matrix, center=False, max_rank=None) -> dict:
    w = np.asarray(matrix, dtype=np.float64)
    if center:
        w = w - w.mean(axis=0, keepdims=True)
    s = np.linalg.svd(w, compute_uv=False)
    s = s[s > s.max() * 1e-9] if s.size and s.max() > 0 else s[:0]
    k = max_rank or min(w.shape)
    if s.size == 0:
        return {"k": k, "erank": 0.0, "pr": 0.0, "d90": 0, "top": 0.0, "frac": 0.0}
    p = s / s.sum()
    var = s ** 2
    energy = np.cumsum(var) / var.sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))
    return {"k": k, "erank": erank, "pr": float(var.sum() ** 2 / (var ** 2).sum()),
            "d90": int(np.searchsorted(energy, 0.90) + 1), "top": float(var[0] / var.sum()), "frac": erank / k}


def analyze(run, base_checkpoint=None, seed=0) -> dict:
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    p = named_params(load_checkpoint(run / "model.ckpt"), config)
    d, h, layers = config["embed_dim"], config["heads"], config["layers"]
    dh = d // h
    rng = np.random.default_rng(seed)
    rows = []

    def add(group, name, matrix, center=False, max_rank=None, reference=None, **extra):
        stats = spectrum_stats(matrix, center, max_rank)
        random = spectrum_stats(reference() if reference else rng.normal(0, 0.02, matrix.shape), center, max_rank)
        rows.append({"group": group, "name": name, "shape": list(matrix.shape), **stats,
                     "random_erank": random["erank"], "random_pr": random["pr"], "random_frac": random["frac"], **extra})
        return stats

    add("embeddings", "wte", p["wte"], center=True)
    touched = None
    if base_checkpoint:
        base = named_params(load_checkpoint(base_checkpoint), config)
        # Só tokens presentes nos dados recebem gradiente na tabela de entrada.
        touched = np.linalg.norm(p["wte"] - base["wte"], axis=1) > 1e-6
        if touched.sum() >= 2:
            add("embeddings", "wte_touched", p["wte"][touched], center=True, tokens=int(touched.sum()))
    add("embeddings", "wpe", p["wpe"], center=True)
    out = p["head"].T  # vocab x d
    centered = out - out.mean(axis=0)
    _, _, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
    # A direção dominante da saída costuma ser a frequência do token (o modelo não tem bias de saída).
    frequency_corr = abs(float(np.corrcoef(centered @ vt[0], out @ out.mean(axis=0))[0, 1]))
    add("output", "head", out, center=True, dominant_vs_mean_logit=frequency_corr)

    for layer in range(layers):
        group = f"L{layer}"
        for kind in ("Wq", "Wk", "Wv"):
            add(group, kind, np.concatenate([p[f"L{layer}.h{i}.{kind}"] for i in range(h)], axis=1))
        add(group, "Wo", np.concatenate([p[f"L{layer}.h{i}.Wo"] for i in range(h)], axis=0))
        for i in range(h):
            q, k = p[f"L{layer}.h{i}.Wq"], p[f"L{layer}.h{i}.Wk"]
            # Matriz bilinear da atenção: posto <= d/h; a referência é Q·Kᵀ com pesos aleatórios.
            add(group, f"h{i}.QK", q.astype(np.float64) @ k.T.astype(np.float64), max_rank=dh,
                reference=lambda: rng.normal(0, 0.02, (d, dh)) @ rng.normal(0, 0.02, (d, dh)).T)
        add(group, "MLP.W1", p[f"L{layer}.W1"])
        add(group, "MLP.W2", p[f"L{layer}.W2"])

    qk = [r for r in rows if r["name"].endswith(".QK")]
    summary = {
        "embedding_pr": next(r["pr"] for r in rows if r["name"] == ("wte_touched" if touched is not None and touched.sum() >= 2 else "wte")),
        "output_top_direction": next(r["top"] for r in rows if r["name"] == "head"),
        "attention_head_frac_mean": float(np.mean([r["frac"] for r in qk])),
        "attention_head_frac_min": float(np.min([r["frac"] for r in qk])),
        "attention_head_frac_random": float(np.mean([r["random_frac"] for r in qk])),
        "mlp_frac_mean": float(np.mean([r["frac"] for r in rows if r["name"].startswith("MLP")])),
    }
    return {"run": run.name, "config": {k: config.get(k) for k in ("vocab_size", "block_size", "embed_dim", "layers", "heads", "parameters")},
            "head_dim": dh, "sft_touched_tokens": None if touched is None else int(touched.sum()), "summary": summary, "rows": rows}


def format_report(report, compare=None) -> str:
    c, s = report["config"], report["summary"]
    lines = [f"{report['run']}: d={c['embed_dim']}, {c['layers']} camadas x {c['heads']} cabeças "
             f"(d/h={report['head_dim']}), {c['parameters']} parâmetros",
             f"{'grupo':10} {'matriz':12} {'forma':>12} {'erank':>7} {'frac':>5} {'PR':>7} {'d90':>5} {'top%':>5} | {'aleat.':>6}"]
    for r in report["rows"]:
        if r["name"].endswith(".QK") and r["group"] != "L0" and report["config"]["layers"] > 2:
            continue  # cabeças de todas as camadas entram no resumo; a tabela mostra as da primeira
        lines.append(f"{r['group']:10} {r['name']:12} {'x'.join(map(str, r['shape'])):>12} {r['erank']:7.1f} "
                     f"{r['frac']:5.2f} {r['pr']:7.1f} {r['d90']:5} {100 * r['top']:5.1f} | {r['random_frac']:6.2f}")
    lines += ["", "resumo (frac = erank / máximo; aleatório entre parênteses):",
              f"  embeddings PR {s['embedding_pr']:.1f} de {c['embed_dim']}",
              f"  saída: direção dominante com {100 * s['output_top_direction']:.1f}% da variância",
              f"  atenção por cabeça: frac média {s['attention_head_frac_mean']:.2f}, mínima {s['attention_head_frac_min']:.2f} "
              f"({s['attention_head_frac_random']:.2f})",
              f"  MLP: frac média {s['mlp_frac_mean']:.2f}"]
    if compare:
        o = compare["summary"]
        lines += ["", f"comparação com {compare['run']} (d={compare['config']['embed_dim']}):"]
        for key, label in (("attention_head_frac_mean", "atenção frac média"), ("attention_head_frac_min", "atenção frac mínima"),
                           ("mlp_frac_mean", "MLP frac média"), ("output_top_direction", "saída: direção dominante")):
            lines.append(f"  {label:26} {o[key]:6.2f} -> {s[key]:6.2f}")
        lines.append(f"  {'embeddings PR / d':26} {o['embedding_pr'] / compare['config']['embed_dim']:6.2f} -> "
                     f"{s['embedding_pr'] / c['embed_dim']:6.2f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", type=Path, help="pasta do run com config.json e model.ckpt")
    parser.add_argument("--base", type=Path, help="checkpoint de antes do SFT, do mesmo formato (ex.: migrated.ckpt)")
    parser.add_argument("--json", type=Path, help="grava o relatório completo em JSON")
    parser.add_argument("--compare", type=Path, help="relatório JSON de outro run para comparar")
    args = parser.parse_args(argv)
    try:
        report = analyze(args.run, args.base)
        compare = json.loads(args.compare.read_text()) if args.compare else None
    except (OSError, ValueError, KeyError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(format_report(report, compare))
    return 0


if __name__ == "__main__":
    sys.exit(main())
