"""Gráficos 2D de funções, do katu-plot: quem amostra e classifica as quebras é o motor, em Rust.

    fig = katu.plot.figure(x=(-1, 1))
    fig.function("1/x")
    grafico = fig.build()
    grafico.save("inversa.svg")

Recusa com `PlotRefused`, sem gráfico parcial. Contrato em docs/KATU_PLOT_CONTRATO.md.
"""
import json
from pathlib import Path

from ._native import _plot

__all__ = ["Figure", "Plot", "PlotRefused", "figure", "plot"]


class PlotRefused(ValueError):
    """O motor recusou o gráfico. `status` e `code` são os do contrato de resultados."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


class Plot:
    """O artefato `katu.plot/1`, com as exportações geradas dele."""

    def __init__(self, answer):
        self.data = answer["artifact"]
        self.svg = answer["svg"]
        self.csv = answer["csv"]

    series = property(lambda self: self.data["series"])
    axes = property(lambda self: self.data["axes"])
    warnings = property(lambda self: self.data["warnings"])
    alt_text = property(lambda self: self.data["alt_text"])

    def to_json(self):
        return json.dumps(self.data, ensure_ascii=False)

    def save(self, path):
        """Grava conforme a extensão: .svg, .csv ou .json. Devolve o caminho."""
        target = Path(path)
        contents = {".svg": self.svg, ".csv": self.csv, ".json": self.to_json()}.get(target.suffix.lower())
        if contents is None:
            raise ValueError(f"extensão desconhecida: {target.suffix!r}. Há .svg, .csv e .json")
        target.write_text(contents, encoding="utf-8")
        return target

    def _repr_svg_(self):
        return self.svg

    def __repr__(self):
        names = ", ".join(one["label"] for one in self.series)
        return f"Plot({names}; x de {self.axes['x']['range'][0]} a {self.axes['x']['range'][1]})"


class Figure:
    """Um gráfico sendo montado. Sem estado global: cada figura é só o seu pedido."""

    def __init__(self, x=None, y="auto", x_label=None, y_label=None, width=800, height=500, theme="light", params=None):
        self._request = {
            "schema": "katu.plot.request/1",
            "axes": {"x": {}, "y": {"range": list(y) if y != "auto" else "auto"}},
            "series": [],
            "viewport": {"width_px": width, "height_px": height},
            "theme": theme,
            "formats": ["svg", "csv"],
        }
        if x is not None:
            self._request["axes"]["x"]["range"] = list(x)
        if x_label is not None:
            self._request["axes"]["x"]["label"] = x_label
        if y_label is not None:
            self._request["axes"]["y"]["label"] = y_label
        if params:
            self._request["params"] = dict(params)

    def function(self, expr, var="x", label=None):
        """Acrescenta uma curva. Devolve a própria figura, para encadear."""
        series = {"id": f"s{len(self._request['series']) + 1}", "kind": "function", "expr": expr, "var": var}
        if label is not None:
            series["label"] = label
        self._request["series"].append(series)
        return self

    def build(self):
        answer = json.loads(_plot(json.dumps(self._request)))
        refusal = answer.get("refusal")
        if refusal:
            raise PlotRefused(refusal["status"], refusal["code"], refusal["message"])
        return Plot(answer)


def figure(x=None, **options):
    return Figure(x, **options)


def plot(*expressions, x=None, **options):
    """Atalho para uma figura com estas funções: `katu.plot.plot("sin(x)", "cos(x)", x=(0, 6.28))`."""
    fig = Figure(x, **options)
    for expr in expressions:
        fig.function(expr)
    return fig.build()
