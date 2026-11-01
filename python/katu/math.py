"""Cálculo do katu-math: derivada simbólica, primitivas conferidas, integral, raízes e EDO.

Responde exato ou recusa com ValueError; não executa código. Ângulos em radianos.
"""
from ._native import antiderivative, derivative, evaluate, integrate, latex, ode, root

__all__ = ["antiderivative", "derivative", "evaluate", "integrate", "latex", "ode", "root"]
