"""Katu: arrays com tipo, cálculo científico e gráficos, com a conta feita em Rust.

    import katu
    a = katu.array([1, 2, 3])
    katu.math.derivative("x^2 - 5*x")   # '2 * x - 5'
    katu.plot.plot("1/x", x=(-1, 1)).save("inversa.svg")

Manual: docs/MANUAL_KATU_PYTHON.md.
"""
from ._native import (
    Array,
    __version__,
    arange,
    array,
    concatenate,
    load,
    matmul,
    stack,
    zeros,
)
from . import math, plot

__all__ = ["Array", "__version__", "arange", "array", "concatenate", "load", "math", "matmul", "plot", "stack", "zeros"]
