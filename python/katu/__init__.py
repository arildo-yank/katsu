"""Katu: arrays com tipo e cálculo científico, com a conta feita em Rust.

    import katu
    a = katu.array([1, 2, 3])
    katu.math.derivative("x^2 - 5*x")   # '2 * x - 5'

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
from . import math

__all__ = ["Array", "__version__", "arange", "array", "concatenate", "load", "math", "matmul", "stack", "zeros"]
