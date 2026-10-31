"""Gera os gabaritos do NumPy para tests/array.rs. Roda uma vez, fora do teste:

    python3 crates/katsu-core/tests/fixtures/array/gen.py

Cada linha de cases.txt: nome ; operação ; entradas ; argumentos ; esperado ; tolerância relativa.
Fatias: "i:k" é índice inteiro; "r:início:fim:passo", com campo vazio para o padrão do Python.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(20261101)
cases = []


def save(name, array):
    np.save(HERE / f"{name}.npy", array)
    return f"{name}.npy"


def case(name, op, inputs, args, expected, tol=0.0):
    cases.append(f"{name} ; {op} ; {','.join(inputs)} ; {args} ; {save(name + '_out', expected)} ; {tol}")


def parse(spec):
    out = []
    for part in spec.split("|"):
        kind, *rest = part.split(":")
        if kind == "i":
            out.append(int(rest[0]))
        else:
            out.append(slice(*(int(v) if v else None for v in rest)))
    return tuple(out)


cube = save("cube_f64", np.arange(60, dtype=np.float64).reshape(3, 4, 5))
cube_v = np.load(HERE / cube)
for i, spec in enumerate(["r:1::|r:::-1|r:0:5:2", "i:-1|r::|r:::2", "r:::-2|i:1|r:-3::",
                          "r:10:20:|r:2:0:-1", "r:5:1:", "i:0|i:0|i:0", "r:-100:100:3|r:-1:-5:-2"]):
    case(f"slice{i}", "slice", [cube], spec, cube_v[parse(spec)])

case("transpose", "transpose", [cube], "-", cube_v.T)
case("permute", "permute", [cube], "2,0,1", cube_v.transpose(2, 0, 1))
case("reshape_view", "slice_reshape", [cube], "r:::-1|r:1:3:|r:::2;6,3", cube_v[::-1, 1:3, ::2].reshape(6, 3))
case("broadcast_to", "broadcast_to", [save("row_i64", np.arange(4, dtype=np.int64))], "3,4",
     np.broadcast_to(np.arange(4, dtype=np.int64), (3, 4)))

col_f32 = save("col_f32", rng.standard_normal((3, 1)).astype(np.float32))
row_f32 = save("row_f32", rng.standard_normal(4).astype(np.float32))
mat_i64 = save("mat_i64", rng.integers(-50, 50, (3, 4)).astype(np.int64))
vec_i64 = save("vec_i64", rng.integers(1, 9, 4).astype(np.int64))
mask_a = save("mask_a", rng.random((3, 4)) > 0.5)
mask_b = save("mask_b", rng.random(4) > 0.5)
mat_f64 = save("mat_f64", rng.standard_normal((4, 3)))
big_i64 = save("big_i64", np.array([2**62, 2**62, -3], dtype=np.int64))
L = {k: np.load(HERE / v) for k, v in dict(col_f32=col_f32, row_f32=row_f32, mat_i64=mat_i64, vec_i64=vec_i64,
                                            mask_a=mask_a, mask_b=mask_b, mat_f64=mat_f64, big_i64=big_i64).items()}

case("add_f32", "add", [col_f32, row_f32], "-", L["col_f32"] + L["row_f32"])
case("mul_f32", "mul", [col_f32, row_f32], "-", L["col_f32"] * L["row_f32"])
case("add_i64_f32", "add", [mat_i64, row_f32], "-", L["mat_i64"] + L["row_f32"])
case("sub_i64", "sub", [mat_i64, vec_i64], "-", L["mat_i64"] - L["vec_i64"])
case("div_i64", "div", [mat_i64, vec_i64], "-", L["mat_i64"] / L["vec_i64"])
case("div_f32", "div", [col_f32, row_f32], "-", L["col_f32"] / L["row_f32"])
case("add_bool", "add", [mask_a, mask_b], "-", L["mask_a"] + L["mask_b"])
case("mul_bool", "mul", [mask_a, mask_b], "-", L["mask_a"] * L["mask_b"])
case("add_bool_i64", "add", [mask_a, vec_i64], "-", L["mask_a"] + L["vec_i64"])
case("wrap_i64", "add", [big_i64, big_i64], "-", L["big_i64"] + L["big_i64"])
case("sub_view_f64", "sub_transposed", [mat_f64, cube], "-", L["mat_f64"].T - cube_v[0, :3, :4])
case("lt_i64_f32", "lt", [mat_i64, row_f32], "-", L["mat_i64"] < L["row_f32"])
case("eq_i64", "eq", [mat_i64, vec_i64], "-", L["mat_i64"] == L["vec_i64"])

nan_f32 = rng.standard_normal((3, 4)).astype(np.float32)
nan_f32[1, 2] = np.nan
nan_f32 = save("nan_f32", nan_f32)
L["nan_f32"] = np.load(HERE / nan_f32)
big_f64 = save("big_f64", rng.standard_normal((50, 40)))
L["big_f64"] = np.load(HERE / big_f64)
for axis in ("none", "0", "1"):
    ax = None if axis == "none" else int(axis)
    case(f"sum_f64_{axis}", "sum", [big_f64], axis, np.asarray(L["big_f64"].sum(axis=ax)), 1e-12)
    case(f"sum_i64_{axis}", "sum", [mat_i64], axis, np.asarray(L["mat_i64"].sum(axis=ax)))
    case(f"sum_bool_{axis}", "sum", [mask_a], axis, np.asarray(L["mask_a"].sum(axis=ax)))
    case(f"mean_i64_{axis}", "mean", [mat_i64], axis, np.asarray(L["mat_i64"].mean(axis=ax)), 1e-15)
    case(f"max_f32_{axis}", "max", [nan_f32], axis, np.asarray(L["nan_f32"].max(axis=ax)))
    case(f"min_i64_{axis}", "min", [mat_i64], axis, np.asarray(L["mat_i64"].min(axis=ax)))
case("sum_f32_1", "sum", [col_f32], "1", L["col_f32"].sum(axis=1), 1e-6)
case("mean_f32_none", "mean", [row_f32], "none", np.asarray(L["row_f32"].mean()), 1e-6)

case("astype_f64_i64", "astype", [save("frac_f64", np.array([-2.7, -0.5, 0.0, 0.5, 2.7, 1e15]))], "i64",
     np.array([-2.7, -0.5, 0.0, 0.5, 2.7, 1e15]).astype(np.int64))
case("astype_i64_bool", "astype", [mat_i64], "bool", L["mat_i64"].astype(bool))
case("astype_bool_f32", "astype", [mask_a], "f32", L["mask_a"].astype(np.float32))
case("astype_f64_f32", "astype", [big_f64], "f32", L["big_f64"].astype(np.float32))

fortran = HERE / "fortran_f64.npy"
np.save(fortran, np.asfortranarray(cube_v[0]))
case("fortran_read", "identity", ["fortran_f64.npy"], "-", np.ascontiguousarray(cube_v[0]))
scalar = save("scalar_f64", np.float64(3.25) * np.ones(()))
case("scalar", "identity", [scalar], "-", np.ones(()) * 3.25)

(HERE / "cases.txt").write_text(f"# Gerado por gen.py com NumPy {np.__version__}. Não editar à mão.\n" + "\n".join(cases) + "\n")
print(len(cases), "casos, NumPy", np.__version__)
