#!/usr/bin/env python3
"""Confere a rota /math contra NumPy e SciPy, usados só como referência independente, não como dependência.

Uso: python3 scripts/math_reference/check.py http://127.0.0.1:PORTA [--seed N]
Sai com código 1 se algum caso divergir. Passar aqui não certifica o motor: é uma amostra.
"""
import argparse
import http.client
import json
import math
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from fractions import Fraction

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


def newton(base, t, rng, count):
    for _ in range(count):
        r = rng.uniform(-5, 5)
        sign = "-" if r > 0 else "+"
        text = f"(x {sign} {abs(r):.17f})*(x^2 + 1)"
        f = lambda x: (x - r) * (x * x + 1)
        fprime = lambda x: (x * x + 1) + (x - r) * 2 * x
        start = r + rng.uniform(-1, 1)
        want = optimize.newton(f, start, fprime=fprime, maxiter=200, tol=1e-13)
        ok, out = call(base, op="root", expr=text, guess=start)
        t.check("Newton x SciPy", ok and abs(out["result"] - want) < 1e-9, f"{text} de {start}: {out} != {want}")
    for text in ("1/x", "1/x^2", "exp(-x)", "1/(x^2 + 1)"):
        for _ in range(count // 20):
            ok, out = call(base, op="root", expr=text, guess=rng.uniform(1, 50))
            t.check("assíntota recusada", not ok, f"{text}: {out}")


def contract(base, t):
    """A rota nova tem de dar o mesmo número da antiga, e recusa em /v1 é HTTP 200 com código."""
    for body in (dict(op="eval", expr="sin(1)"), dict(op="det", a=[[1, 2], [3, 4]]), dict(op="root", expr="x^2 - 2", guess=1)):
        old = call(base, **body)[1]["result"]
        new = compute(base, **body)
        t.check("/v1/compute = /math", new["status"] == "success" and new["content"]["value"]["value"] == old, f"{body}: {new}")
    new = compute(base, op="eval", expr="1/0")
    t.check("/v1/compute recusa com código", new["status"] == "invalid_input" and new["refusal"]["code"] == "division_by_zero", new)
    new = compute(base, op="eval", expr="sqrt(2)", mode="exact")
    t.check("/v1/compute recusa com código", new["status"] == "unsupported", new)


def compute(base, **body):
    request = urllib.request.Request(base.rstrip("/") + "/v1/compute", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def exact_expression(rng, depth):
    """Texto e valor construídos juntos, sem passar por nenhum analisador."""
    if depth == 0 or rng.random() < 0.25:
        whole, places = rng.randint(0, 999), rng.randint(1, 9)
        digits = rng.randint(0, 10**places - 1)
        return f"{whole}.{digits:0{places}d}", Fraction(whole) + Fraction(digits, 10**places)
    op = rng.choice("+-*/^")
    left, a = exact_expression(rng, depth - 1)
    if op == "^":
        power = rng.randint(1, 5)
        return f"({left})^{power}", a**power
    right, b = exact_expression(rng, depth - 1)
    if op == "/" and b == 0:
        op = "*"
    value = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else a}[op]
    return f"({left}) {op} ({right})", value


def exact_mode(base, t, rng, count):
    for _ in range(count):
        text, value = exact_expression(rng, 3)
        if len(text) > 500:
            continue
        out = compute(base, op="eval", expr=text, mode="exact")
        shown = out.get("content", {}).get("value", {})
        got = shown.get("text") if shown.get("kind") == "integer" else f"{shown.get('num')}/{shown.get('den')}"
        want = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
        t.check("exact x fractions", out.get("nature") == "exact" and got == want, f"{text}: {out}")


HEAVY = "3^165000 / 7^93000 + 5^112000 / 11^75000 + 13^70000 / 17^64000 + 19^61000 / 23^57000"


def budget(base, t):
    """Pior caso real: prazo de 2 s e 4 vagas. Só as folgas vêm desta medição; o prazo está no servidor."""
    started = time.time()
    out = compute(base, op="eval", expr=HEAVY, mode="exact")
    elapsed = time.time() - started
    print(f"conta pesada: {elapsed:.2f} s, {out['status']}, {out.get('refusal', {}).get('code')}")
    t.check("prazo respeitado", out.get("refusal", {}).get("code") == "limit_time" and elapsed < 2.5, f"{elapsed:.2f} s: {str(out)[:200]}")

    results = []
    def worker():
        begin = time.time()
        results.append((compute(base, op="eval", expr=HEAVY, mode="exact"), time.time() - begin))
    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    codes = sorted(r[0].get("refusal", {}).get("code", "ok") for r in results)
    slowest = max(r[1] for r in results)
    print(f"12 simultâneas: {codes.count('limit_time')} por tempo, {codes.count('limit_concurrency')} sem vaga, a mais lenta {slowest:.2f} s")
    t.check("concorrência limitada", codes.count("limit_concurrency") >= 8 and codes.count("limit_time") <= 4 and slowest < 3.0, f"{codes} {slowest:.2f}")

    # Cliente desiste no meio: a vaga precisa voltar bem antes do prazo de 2 s.
    addr = urllib.parse.urlparse(base)
    payload = json.dumps(dict(op="eval", expr=HEAVY, mode="exact")).encode()
    connections = []
    for _ in range(4):
        connection = http.client.HTTPConnection(addr.hostname, addr.port)
        connection.request("POST", "/v1/compute", payload, {"Content-Type": "application/json"})
        connections.append(connection)
    time.sleep(0.2)
    blocked = compute(base, op="eval", expr="1 + 1").get("refusal", {}).get("code")
    for connection in connections:
        connection.close()
    begin, freed = time.time(), None
    while time.time() - begin < 3:
        if compute(base, op="eval", expr="1 + 1").get("status") == "success":
            freed = time.time() - begin
            break
        time.sleep(0.01)
    print(f"desistência: antes {blocked}, vaga de volta em {freed if freed is None else round(freed, 3)} s")
    t.check("desistir libera a vaga", blocked == "limit_concurrency" and freed is not None and freed < 1.0, f"{blocked} {freed}")


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
    newton(args.base, t, rng, args.count)
    contract(args.base, t)
    exact_mode(args.base, t, rng, args.count)
    budget(args.base, t)
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
