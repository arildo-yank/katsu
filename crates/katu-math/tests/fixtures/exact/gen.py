#!/usr/bin/env python3
"""Gera os gabaritos de aritmética exata com os inteiros e `fractions` do Python, motor independente.

Uso: python3 crates/katu-math/tests/fixtures/exact/gen.py
O teste só lê os .txt: Python não é dependência de compilação nem de execução.
"""
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.set_int_max_str_digits(0)
HERE = Path(__file__).parent
rng = random.Random(20261101)


def limbs(count):
    """Inteiro com dígitos de 32 bits escolhidos para forçar transporte e a estimativa da divisão longa."""
    value = 0
    for _ in range(count):
        value = (value << 32) | rng.choice([0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, rng.getrandbits(32), rng.getrandbits(32)])
    return value


def signed(count):
    value = limbs(rng.randint(0, count))
    return -value if rng.random() < 0.5 else value


def truncated(a, b):
    q = abs(a) // abs(b)
    q = -q if (a < 0) != (b < 0) else q
    return q, a - q * b


def write(name, lines):
    (HERE / name).write_text("\n".join(lines) + "\n")


ints = []
for _ in range(400):
    a, b = signed(20), signed(12)
    if b == 0:
        b = 1
    q, r = truncated(a, b)
    ints.append("\t".join(str(v) for v in (a, b, a + b, a - b, a * b, q, r)))
write("int_ops.txt", ints)

powers = []
for _ in range(60):
    base, exponent = signed(3), rng.randint(0, 40)
    powers.append(f"{base}\t{exponent}\t{base ** exponent}")
write("int_pow.txt", powers)


def fraction():
    den = signed(3)
    return Fraction(signed(3), den if den else 1)


rationals = []
for _ in range(300):
    a, b = fraction(), fraction()
    if b == 0:
        b = Fraction(1, 3)
    rationals.append("\t".join(f"{v.numerator}/{v.denominator}" for v in (a, b, a + b, a - b, a * b, a / b)))
write("rational_ops.txt", rationals)

floats = []
for _ in range(600):
    kind = rng.random()
    if kind < 0.5:
        value = fraction()
    elif kind < 0.8:
        # Faixa inteira do f64, inclusive subnormais.
        shift = rng.randint(-1100, 1000)
        value = Fraction(rng.getrandbits(70) + 1) * Fraction(2) ** shift
    else:
        # Exatamente no meio de dois f64 vizinhos, ou quase: é onde o arredondamento se denuncia.
        m = rng.getrandbits(53) | (1 << 52)
        value = Fraction(2 * m + 1) * Fraction(2) ** rng.randint(-1100, 900)
        value += rng.choice([0, 0, Fraction(1, 10 ** 700), -Fraction(1, 10 ** 700)])
    try:
        out = float(value)
    except OverflowError:
        continue
    floats.append(f"{value.numerator}/{value.denominator}\t{out.hex()}")
write("rational_float.txt", floats)


def expression(depth):
    """Texto e valor construídos juntos, sem passar por nenhum analisador."""
    if depth == 0 or rng.random() < 0.25:
        if rng.random() < 0.5:
            value = rng.randint(0, 10 ** rng.randint(1, 12))
            return str(value), Fraction(value)
        whole, places = rng.randint(0, 999), rng.randint(1, 9)
        digits = rng.randint(0, 10 ** places - 1)
        return f"{whole}.{digits:0{places}d}", Fraction(whole) + Fraction(digits, 10 ** places)
    op = rng.choice("+-*/^n")
    left, a = expression(depth - 1)
    if op == "n":
        return f"-({left})", -a
    if op == "^":
        power = rng.randint(-4, 6)
        if a == 0 and power <= 0:
            power = 2
        return f"({left})^({power})", a ** power
    right, b = expression(depth - 1)
    if op == "/" and b == 0:
        op = "*"
    value = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else None}[op]
    return f"({left}) {op} ({right})", value


evals = []
while len(evals) < 300:
    text, value = expression(4)
    if len(text) <= 500:
        evals.append(f"{text}\t{value.numerator}/{value.denominator}")
write("eval.txt", evals)
