//! Ponte Python do Katu: `katu.Array` (o `NdArray` do `katu-core`) e as funções do `katu-math`.
//! Só a casca: a conta acontece no Rust. Fase 7 de `docs/KATU_ARRAY_PLAN.md`; uso em
//! `docs/MANUAL_KATU_PYTHON.md`.

use katu_core::{promote, sgemm, ArrayError, Compare, DType, Hub, Index, NdArray};
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PySlice, PyTuple};
use pyo3::IntoPyObjectExt;

fn error(e: ArrayError) -> PyErr {
    match e {
        ArrayError::Index(_) | ArrayError::Axis { .. } => PyIndexError::new_err(e.to_string()),
        ArrayError::DType(_) => PyTypeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn math_error(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn dtype_from(name: &str) -> PyResult<DType> {
    Ok(match name {
        "bool" => DType::Bool,
        "u8" | "uint8" => DType::U8,
        "i32" | "int32" => DType::I32,
        "i64" | "int64" | "int" => DType::I64,
        "f16" | "float16" => DType::F16,
        "bf16" | "bfloat16" => DType::Bf16,
        "f32" | "float32" => DType::F32,
        "f64" | "float64" | "float" => DType::F64,
        other => return Err(PyTypeError::new_err(format!("tipo desconhecido: {other}"))),
    })
}

fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::Bool => "bool",
        DType::U8 => "u8",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
    }
}

/// Valores lidos de listas aninhadas do Python, antes de virar array.
enum Leaf {
    Bool(bool),
    Int(i64),
    Float(f64),
}

fn flatten(
    obj: &Bound<'_, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    out: &mut Vec<Leaf>,
) -> PyResult<()> {
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'_, PyAny>> = obj.try_iter()?.collect::<PyResult<_>>()?;
        if shape.len() == depth {
            shape.push(items.len());
        } else if shape.len() < depth || shape[depth] != items.len() {
            return Err(PyValueError::new_err(
                "listas de tamanhos diferentes no mesmo nível",
            ));
        }
        for item in &items {
            flatten(item, depth + 1, shape, out)?;
        }
        return Ok(());
    }
    if depth != shape.len() {
        return Err(PyValueError::new_err("número no lugar de uma lista"));
    }
    out.push(if obj.is_instance_of::<PyBool>() {
        Leaf::Bool(obj.extract()?)
    } else if obj.is_instance_of::<PyInt>() {
        Leaf::Int(obj.extract()?)
    } else if obj.is_instance_of::<PyFloat>() {
        Leaf::Float(obj.extract()?)
    } else {
        return Err(PyTypeError::new_err(format!(
            "valor não numérico: {}",
            obj.get_type().name()?
        )));
    });
    Ok(())
}

fn from_python(data: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<NdArray> {
    if let Ok(array) = data.extract::<PyRef<'_, Array>>() {
        return match dtype {
            Some(d) => array.inner.astype(dtype_from(d)?).map_err(error),
            None => Ok(array.inner.clone()),
        };
    }
    let (mut shape, mut leaves) = (Vec::new(), Vec::new());
    flatten(data, 0, &mut shape, &mut leaves)?;
    let any_float = leaves.iter().any(|l| matches!(l, Leaf::Float(_)));
    let all_bool = !leaves.is_empty() && leaves.iter().all(|l| matches!(l, Leaf::Bool(_)));
    let base = if all_bool {
        NdArray::from_vec(
            leaves
                .iter()
                .map(|l| matches!(l, Leaf::Bool(true)))
                .collect(),
            &shape,
        )
    } else if any_float || leaves.is_empty() {
        let values = leaves.iter().map(|l| match l {
            Leaf::Bool(b) => f64::from(u8::from(*b)),
            Leaf::Int(i) => *i as f64,
            Leaf::Float(f) => *f,
        });
        NdArray::from_vec(values.collect(), &shape)
    } else {
        let values = leaves.iter().map(|l| match l {
            Leaf::Bool(b) => i64::from(*b),
            Leaf::Int(i) => *i,
            Leaf::Float(f) => *f as i64,
        });
        NdArray::from_vec(values.collect(), &shape)
    }
    .map_err(error)?;
    match dtype {
        Some(d) => base.astype(dtype_from(d)?).map_err(error),
        None => Ok(base),
    }
}

/// Número do Python numa conta com um array: assume o tipo do array quando cabe nele, como os escalares
/// "fracos" do NumPy 2 (NEP 50). Um float com um array de inteiros dá f64.
fn scalar_like(obj: &Bound<'_, PyAny>, like: DType) -> PyResult<Option<NdArray>> {
    let value = if obj.is_instance_of::<PyBool>() {
        NdArray::from_vec(vec![obj.extract::<bool>()?], &[])
    } else if obj.is_instance_of::<PyInt>() {
        NdArray::from_vec(vec![obj.extract::<i64>()?], &[])
    } else if obj.is_instance_of::<PyFloat>() {
        NdArray::from_vec(vec![obj.extract::<f64>()?], &[])
    } else {
        return Ok(None);
    }
    .map_err(error)?;
    let target = match value.dtype() {
        DType::Bool => like,
        DType::I64 if like != DType::Bool => like,
        DType::F64 if like.is_float() => like,
        other => promote(other, like),
    };
    value.astype(target).map(Some).map_err(error)
}

fn operand(other: &Bound<'_, PyAny>, like: DType) -> PyResult<NdArray> {
    if let Ok(array) = other.extract::<PyRef<'_, Array>>() {
        return Ok(array.inner.clone());
    }
    scalar_like(other, like)?.ok_or_else(|| {
        PyTypeError::new_err("operação entre Array e um valor que não é número nem Array")
    })
}

fn hub_to_py(py: Python<'_>, value: Hub) -> PyResult<Py<PyAny>> {
    match value {
        Hub::Bool(v) => v.into_py_any(py),
        Hub::Int(v) => v.into_py_any(py),
        Hub::Float(v) => v.into_py_any(py),
    }
}

fn hubs(array: &NdArray) -> Vec<Hub> {
    match array.dtype() {
        DType::Bool => array
            .values_as::<bool>()
            .into_iter()
            .map(Hub::Bool)
            .collect(),
        d if d.is_float() => array
            .values_as::<f64>()
            .into_iter()
            .map(Hub::Float)
            .collect(),
        _ => array.values_as::<i64>().into_iter().map(Hub::Int).collect(),
    }
}

fn nested(py: Python<'_>, values: &[Hub], shape: &[usize]) -> PyResult<Py<PyAny>> {
    let Some((&first, rest)) = shape.split_first() else {
        return hub_to_py(py, values[0]);
    };
    let block = rest.iter().product::<usize>();
    let items = (0..first)
        .map(|i| nested(py, &values[i * block..(i + 1) * block], rest))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)?.into_py_any(py)
}

/// Array 0-d vira número do Python; os outros ficam Array.
fn output(py: Python<'_>, array: NdArray) -> PyResult<Py<PyAny>> {
    if array.ndim() == 0 {
        return hub_to_py(py, hubs(&array)[0]);
    }
    Array { inner: array }.into_py_any(py)
}

fn index(key: &Bound<'_, PyAny>) -> PyResult<Index> {
    if key.is_instance_of::<PySlice>() {
        let part = |name: &str| -> PyResult<Option<isize>> { key.getattr(name)?.extract() };
        return Ok(Index::Range {
            start: part("start")?,
            stop: part("stop")?,
            step: part("step")?.unwrap_or(1),
        });
    }
    if key.is_instance_of::<PyInt>() && !key.is_instance_of::<PyBool>() {
        return Ok(Index::At(key.extract()?));
    }
    Err(PyIndexError::new_err(
        "índice aceito: inteiro, fatia, tupla deles, Array de bool (máscara) ou lista de inteiros",
    ))
}

/// Array N-dimensional com tipo, no modelo do NumPy. Fatias e transposição não copiam os dados.
#[pyclass(name = "Array", module = "katu", frozen)]
pub struct Array {
    inner: NdArray,
}

#[pymethods]
impl Array {
    #[new]
    #[pyo3(signature = (data, dtype = None))]
    fn new(data: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Self> {
        Ok(Self {
            inner: from_python(data, dtype)?,
        })
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape())
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.dtype())
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.len()
    }

    /// Transposta, sem cópia.
    #[getter(T)]
    fn transposed(&self) -> Self {
        Self {
            inner: self.inner.transpose(),
        }
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .shape()
            .first()
            .copied()
            .ok_or_else(|| PyTypeError::new_err("array 0-d não tem tamanho"))
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        nested(py, &hubs(&self.inner), self.inner.shape())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let values = self.tolist(py)?;
        Ok(format!(
            "Array({}, dtype='{}')",
            values.bind(py).repr()?,
            self.dtype()
        ))
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(mask) = key.extract::<PyRef<'_, Array>>() {
            let picked = match mask.inner.dtype() {
                DType::Bool => self.inner.select(&mask.inner),
                d if !d.is_float() => self.inner.take(&mask.inner.values_as::<i64>(), 0),
                _ => return Err(PyIndexError::new_err("índice com Array de float")),
            };
            return output(py, picked.map_err(error)?);
        }
        if key.is_instance_of::<PyList>() {
            let rows: Vec<i64> = key.extract()?;
            return output(py, self.inner.take(&rows, 0).map_err(error)?);
        }
        let indices = if key.is_instance_of::<PyTuple>() {
            key.try_iter()?
                .map(|k| index(&k?))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            vec![index(key)?]
        };
        output(py, self.inner.slice(&indices).map_err(error)?)
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.reshape(&shape).map_err(error)?,
        })
    }

    #[pyo3(signature = (*axes))]
    fn transpose(&self, axes: Vec<usize>) -> PyResult<Self> {
        if axes.is_empty() {
            return Ok(self.transposed());
        }
        Ok(Self {
            inner: self.inner.permute(&axes).map_err(error)?,
        })
    }

    fn astype(&self, dtype: &str) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.astype(dtype_from(dtype)?).map_err(error)?,
        })
    }

    fn broadcast_to(&self, shape: Vec<usize>) -> PyResult<Self> {
        Ok(Self {
            inner: self.inner.broadcast_to(&shape).map_err(error)?,
        })
    }

    #[pyo3(signature = (axis = None))]
    fn sum(&self, py: Python<'_>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        output(py, self.inner.sum(axis).map_err(error)?)
    }

    #[pyo3(signature = (axis = None))]
    fn mean(&self, py: Python<'_>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        output(py, self.inner.mean(axis).map_err(error)?)
    }

    #[pyo3(signature = (axis = None, ddof = 0))]
    fn var(&self, py: Python<'_>, axis: Option<usize>, ddof: usize) -> PyResult<Py<PyAny>> {
        output(py, self.inner.var(axis, ddof).map_err(error)?)
    }

    #[pyo3(signature = (axis = None, ddof = 0))]
    fn std(&self, py: Python<'_>, axis: Option<usize>, ddof: usize) -> PyResult<Py<PyAny>> {
        output(py, self.inner.std(axis, ddof).map_err(error)?)
    }

    #[pyo3(signature = (axis = None))]
    fn max(&self, py: Python<'_>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        output(py, self.inner.max(axis).map_err(error)?)
    }

    #[pyo3(signature = (axis = None))]
    fn min(&self, py: Python<'_>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        output(py, self.inner.min(axis).map_err(error)?)
    }

    #[pyo3(signature = (axis = None))]
    fn argmax(&self, py: Python<'_>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        output(py, self.inner.argmax(axis).map_err(error)?)
    }

    #[pyo3(signature = (axis = None))]
    fn argmin(&self, py: Python<'_>, axis: Option<usize>) -> PyResult<Py<PyAny>> {
        output(py, self.inner.argmin(axis).map_err(error)?)
    }

    /// Grava em `.npy`, legível pelo `numpy.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.write_npy(path).map_err(error)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, false, NdArray::add)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, true, NdArray::add)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, false, NdArray::sub)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, true, NdArray::sub)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, false, NdArray::mul)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, true, NdArray::mul)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, false, NdArray::div)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.binary(other, true, NdArray::div)
    }

    fn __neg__(&self) -> PyResult<Self> {
        let zero = NdArray::zeros(self.inner.dtype(), &[]).map_err(error)?;
        Ok(Self {
            inner: zero.sub(&self.inner).map_err(error)?,
        })
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Self> {
        let other = operand(other, self.inner.dtype())?;
        let op = match op {
            CompareOp::Lt => Compare::Lt,
            CompareOp::Le => Compare::Le,
            CompareOp::Eq => Compare::Eq,
            CompareOp::Ne => Compare::Ne,
            CompareOp::Gt => Compare::Gt,
            CompareOp::Ge => Compare::Ge,
        };
        Ok(Self {
            inner: self.inner.compare(&other, op).map_err(error)?,
        })
    }

    /// Produto de matrizes 2-D em f32, pelo GEMM em blocos do Katu. Converta com `astype("f32")`.
    fn __matmul__(&self, other: PyRef<'_, Array>) -> PyResult<Self> {
        matmul(self, &other)
    }
}

impl Array {
    fn binary(
        &self,
        other: &Bound<'_, PyAny>,
        reflected: bool,
        op: fn(&NdArray, &NdArray) -> Result<NdArray, ArrayError>,
    ) -> PyResult<Self> {
        let other = operand(other, self.inner.dtype())?;
        let result = if reflected {
            op(&other, &self.inner)
        } else {
            op(&self.inner, &other)
        };
        Ok(Self {
            inner: result.map_err(error)?,
        })
    }
}

/// Produto de matrizes 2-D em f32, pelo GEMM em blocos do Katu.
#[pyfunction]
fn matmul(a: &Array, b: &Array) -> PyResult<Array> {
    let (a, b) = (&a.inner, &b.inner);
    if a.dtype() != DType::F32 || b.dtype() != DType::F32 {
        return Err(PyTypeError::new_err(
            "o produto de matrizes ainda só existe em f32; converta com astype('f32')",
        ));
    }
    let (&[m, k], &[k2, n]) = (a.shape(), b.shape()) else {
        return Err(PyValueError::new_err(
            "o produto de matrizes pede dois arrays 2-D",
        ));
    };
    if k != k2 {
        return Err(PyValueError::new_err(format!(
            "formas incompatíveis: ({m}, {k}) @ ({k2}, {n})"
        )));
    }
    let (x, y) = (
        a.to_vec::<f32>().map_err(error)?,
        b.to_vec::<f32>().map_err(error)?,
    );
    let mut c = vec![0.0f32; m * n];
    sgemm(&x, &y, &mut c, m, k, n);
    Ok(Array {
        inner: NdArray::from_vec(c, &[m, n]).map_err(error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (data, dtype = None))]
fn array(data: &Bound<'_, PyAny>, dtype: Option<&str>) -> PyResult<Array> {
    Ok(Array {
        inner: from_python(data, dtype)?,
    })
}

#[pyfunction]
#[pyo3(signature = (shape, dtype = "f64"))]
fn zeros(shape: Vec<usize>, dtype: &str) -> PyResult<Array> {
    Ok(Array {
        inner: NdArray::zeros(dtype_from(dtype)?, &shape).map_err(error)?,
    })
}

#[pyfunction]
fn arange(n: usize) -> PyResult<Array> {
    Ok(Array {
        inner: NdArray::arange(n).map_err(error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
fn concatenate(arrays: Vec<PyRef<'_, Array>>, axis: usize) -> PyResult<Array> {
    let refs: Vec<&NdArray> = arrays.iter().map(|a| &a.inner).collect();
    Ok(Array {
        inner: NdArray::concatenate(&refs, axis).map_err(error)?,
    })
}

#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
fn stack(arrays: Vec<PyRef<'_, Array>>, axis: usize) -> PyResult<Array> {
    let refs: Vec<&NdArray> = arrays.iter().map(|a| &a.inner).collect();
    Ok(Array {
        inner: NdArray::stack(&refs, axis).map_err(error)?,
    })
}

/// Lê um `.npy` gravado pelo NumPy ou pelo Katu.
#[pyfunction]
fn load(path: &str) -> PyResult<Array> {
    Ok(Array {
        inner: NdArray::read_npy(path).map_err(error)?,
    })
}

fn parse(expr: &str) -> PyResult<katu_math::Expr> {
    katu_math::parse(expr).map_err(math_error)
}

/// Derivada simbólica de `expr` em `var`, como texto.
#[pyfunction]
#[pyo3(signature = (expr, var = "x"))]
fn derivative(expr: &str, var: &str) -> PyResult<String> {
    Ok(katu_math::diff(&parse(expr)?, var)
        .map_err(math_error)?
        .to_string())
}

/// Primitiva em forma fechada, conferida derivando; recusa o que não sabe escrever. Sem a constante C.
#[pyfunction]
#[pyo3(signature = (expr, var = "x"))]
fn antiderivative(expr: &str, var: &str) -> PyResult<String> {
    Ok(katu_math::antiderivative(&parse(expr)?, var)
        .map_err(math_error)?
        .to_string())
}

/// Valor da expressão, com as variáveis por nome: `evaluate("a*x + 1", x=2, a=3)`.
#[pyfunction]
#[pyo3(signature = (expr, **variables))]
fn evaluate(
    expr: &str,
    variables: Option<std::collections::HashMap<String, f64>>,
) -> PyResult<f64> {
    katu_math::eval(&parse(expr)?, &variables.unwrap_or_default()).map_err(math_error)
}

/// Integral definida, por Simpson adaptativo (aproximação numérica com a tolerância pedida).
#[pyfunction]
#[pyo3(signature = (expr, a, b, var = "x", tol = 1e-9))]
fn integrate(expr: &str, a: f64, b: f64, var: &str, tol: f64) -> PyResult<f64> {
    katu_math::integrate(&parse(expr)?, var, a, b, tol, &Default::default()).map_err(math_error)
}

/// Raiz por Newton-Raphson a partir de `start`; recusa quando não converge.
#[pyfunction]
#[pyo3(signature = (expr, start, var = "x", tol = 1e-12))]
fn root(expr: &str, start: f64, var: &str, tol: f64) -> PyResult<f64> {
    katu_math::newton(&parse(expr)?, var, start, tol, &Default::default()).map_err(math_error)
}

/// y(t1) da EDO dy/dt = f(t, y), com y(t0) = y0, por Runge-Kutta de 4ª ordem em `steps` passos.
#[pyfunction]
#[pyo3(signature = (f, t0, y0, t1, steps = 1000, t = "t", y = "y"))]
fn ode(f: &str, t0: f64, y0: f64, t1: f64, steps: usize, t: &str, y: &str) -> PyResult<f64> {
    katu_math::rk4(&parse(f)?, t, y, t0, y0, t1, steps, &Default::default()).map_err(math_error)
}

/// A expressão em LaTeX.
#[pyfunction]
fn latex(expr: &str) -> PyResult<String> {
    Ok(katu_math::latex(&parse(expr)?))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<Array>()?;
    for function in [
        wrap_pyfunction!(array, module)?,
        wrap_pyfunction!(zeros, module)?,
        wrap_pyfunction!(arange, module)?,
        wrap_pyfunction!(concatenate, module)?,
        wrap_pyfunction!(stack, module)?,
        wrap_pyfunction!(load, module)?,
        wrap_pyfunction!(matmul, module)?,
        wrap_pyfunction!(derivative, module)?,
        wrap_pyfunction!(antiderivative, module)?,
        wrap_pyfunction!(evaluate, module)?,
        wrap_pyfunction!(integrate, module)?,
        wrap_pyfunction!(root, module)?,
        wrap_pyfunction!(ode, module)?,
        wrap_pyfunction!(latex, module)?,
    ] {
        module.add_function(function)?;
    }
    Ok(())
}
