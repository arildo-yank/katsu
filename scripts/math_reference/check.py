#!/usr/bin/env python3
"""Confere a rota /math contra NumPy e SciPy, usados só como referência independente, não como dependência.

Uso: python3 scripts/math_reference/check.py http://127.0.0.1:PORTA [--seed N]
Sai com código 1 se algum caso divergir. Passar aqui não certifica o motor: é uma amostra.
"""
import argparse
import json
import math
import random
import sys
import urllib.error
import urllib.request

import numpy as np
from scipy import integrate, optimize


def call(base, **body):
    request = urllib.request.Request(
        base.rstrip("/") + "/math",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return True, json.load(response)
    except urllib.error.HTTPError as refusal:
        if refusal.code != 400:
            raise
        return False, json.load(refusal)


class Tally:
    def __init__(self):
        self.groups, self.problems = {}, []

    def check(self, group, ok, detail):
        count = self.groups.setdefault(group, [0, 0])
        count[0] += 1
        if not ok:
            count[1] += 1
            self.problems.append(f"{group}: {detail}")


def reproductions(base, t):
    ok, _ = call(base, op="root", expr="1/(x-0.1)", **{"from": 0, "to": 1})
    t.check("auditoria", not ok, "polo aceito como raiz")
    ok, out = call(base, op="antiderivative", expr="0.0000000001*x")
    t.check("auditoria", ok and out["result"] != "0 + C", f"primitiva {out}")
    ok, out = call(base, op="det", a=[[1e13, 0], [0, 1]])
    t.check("auditoria", ok and out["result"] == 1e13, f"det {out}")
    ok, out = call(base, op="derive", expr="ln(x)", at=-1)
    t.check("auditoria", not ok, f"derivada fora do domínio {out}")


def determinants(base, t, rng, count):
    for _ in range(count):
        n = rng.randint(1, 8)
        a = np.array([[rng.uniform(-9, 9) for _ in range(n)] for _ in range(n)])
        a = a * np.array([[10.0 ** rng.randint(-13, 13)] for _ in range(n)])
        # A referência também sofre com o condicionamento: a folga acompanha o da matriz balanceada.
        rows = a / np.abs(a).max(axis=1, keepdims=True)
        cond = np.linalg.cond(rows)
        if cond > 1e8:
            continue
        want = float(np.linalg.det(a))
        ok, out = call(base, op="det", a=a.tolist())
        got = out.get("result") if ok else None
        good = ok and abs(got - want) <= 1e-13 * n * cond * abs(want)
        t.check("det x NumPy", good, f"n={n} cond={cond:.1e} {got} != {want}")
        ok, out = call(base, op="inv", a=a.tolist())
        if ok:
            residual = np.abs(np.array(out["result"]) @ a - np.eye(n)).max()
            good = residual <= 1e-13 * n * cond
        else:
            good, residual = False, out
        t.check("inv: resíduo", good, f"n={n} cond={cond:.1e} {residual}")


def singular_matrices(base, t, rng, count):
    for _ in range(count):
        n = rng.randint(2, 8)
        rank = rng.randint(1, n - 1)
        left = np.array([[rng.uniform(-9, 9) for _ in range(rank)] for _ in range(n)])
        right = np.array([[rng.uniform(-9, 9) for _ in range(n)] for _ in range(rank)])
        a = left @ right * 10.0 ** rng.randint(-13, 13)
        ok, out = call(base, op="inv", a=a.tolist())
        t.check("singular: inv recusa", not ok, f"n={n} posto={rank} {out}")
        ok, out = call(base, op="det", a=a.tolist())
        t.check("singular: det zero", ok and out["result"] == 0.0, f"n={n} posto={rank} {out}")


def roots(base, t, rng, count):
    for _ in range(count):
        r = rng.uniform(-5, 5)
        a, c = r - rng.uniform(0.1, 3), r + rng.uniform(0.1, 3)
        scale = 10.0 ** rng.randint(-12, 12)
        text = f"{scale:.20f}*(x - {abs(r):.17f})*(x^2 + 1)" if r > 0 else f"{scale:.20f}*(x + {abs(r):.17f})*(x^2 + 1)"
        f = lambda x: scale * (x - r) * (x * x + 1)
        want = optimize.brentq(f, a, c, xtol=1e-13)
        ok, out = call(base, op="root", expr=text, **{"from": a, "to": c})
        good = ok and abs(out["result"] - want) < 1e-9
        t.check("raiz x SciPy brentq", good, f"{text} [{a},{c}] {out} != {want}")
    for _ in range(count):
        p = rng.uniform(-5, 5)
        a, c = p - rng.uniform(0.1, 3), p + rng.uniform(0.1, 3)
        k = 10.0 ** rng.randint(-9, 9)
        sign = "-" if p > 0 else "+"
        text = f"{k:.12f}/(x {sign} {abs(p):.17f})"
        ok, out = call(base, op="root", expr=text, **{"from": a, "to": c})
        t.check("polo recusado", not ok, f"{text} [{a},{c}] {out}")


def primitives(base, t, rng, count):
    for _ in range(count):
        c = rng.uniform(1, 9) * 10.0 ** rng.randint(-14, 12)
        n = rng.randint(0, 5)
        text = f"{c:.30f}*x^{n}"
        ok, out = call(base, op="antiderivative", expr=text)
        if not ok:
            t.check("primitiva x SciPy quad", False, f"{text}: {out}")
            continue
        primitive = out["result"].removesuffix(" + C")
        at = lambda x: call(base, op="eval", expr=primitive, vars={"x": x})[1]["result"]
        want, _ = integrate.quad(lambda x: c * x**n, 1.0, 2.5)
        got = at(2.5) - at(1.0)
        t.check("primitiva x SciPy quad", abs(got - want) <= 1e-9 * abs(want), f"{text} -> {primitive}: {got} != {want}")


def derivatives(base, t, rng, count):
    cases = [("ln(x)", math.log, 0.0), ("sqrt(x)", math.sqrt, 0.0), ("x^3", lambda x: x**3, None), ("exp(x)", math.exp, None)]
    for _ in range(count):
        text, f, lower = rng.choice(cases)
        x = rng.uniform(-5, 5)
        if lower is not None and x <= lower + 0.01:
            ok, out = call(base, op="derive", expr=text, at=x)
            t.check("derivada: fora do domínio", not ok, f"{text} em {x}: {out}")
            continue
        h = 1e-6
        want = (f(x + h) - f(x - h)) / (2 * h)
        ok, out = call(base, op="derive", expr=text, at=x)
        t.check("derivada x diferença central", ok and abs(out["value_at"] - want) <= 1e-6 * (1 + abs(want)), f"{text} em {x}: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    rng, t = random.Random(args.seed), Tally()
    reproductions(args.base, t)
    determinants(args.base, t, rng, args.count)
    singular_matrices(args.base, t, rng, args.count // 5)
    roots(args.base, t, rng, args.count)
    primitives(args.base, t, rng, args.count)
    derivatives(args.base, t, rng, args.count)
    print(f"NumPy {np.__version__}, semente {args.seed}")
    for group, (total, bad) in t.groups.items():
        print(f"{group}: {total - bad}/{total}")
    for problem in t.problems[:25]:
        print("  -", problem[:300])
    print(f"divergências: {len(t.problems)}")
    return 1 if t.problems else 0


if __name__ == "__main__":
    sys.exit(main())
