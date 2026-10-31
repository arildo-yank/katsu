//! Arrays N-dimensionais com tipo (bool, i64, f32, f64) e views sem cópia, no modelo do NumPy.
//! Uma view é forma, strides e deslocamento sobre o mesmo buffer compartilhado; as views são só de
//! leitura e a cópia é explícita. Promoção de tipos e reduções seguem o NumPy 1.26, conferidos contra
//! os gabaritos de `tests/fixtures/array`. Plano: `docs/KATSU_ARRAY_PLAN.md` (fase 0).

use std::fmt;
use std::path::Path;
use std::sync::Arc;

/// 2^63: o primeiro f64 que não cabe em i64.
const I64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    Bool,
    I64,
    F32,
    F64,
}

impl DType {
    fn npy_descr(self) -> &'static str {
        match self {
            DType::Bool => "|b1",
            DType::I64 => "<i8",
            DType::F32 => "<f4",
            DType::F64 => "<f8",
        }
    }

    fn size(self) -> usize {
        match self {
            DType::Bool => 1,
            DType::I64 | DType::F64 => 8,
            DType::F32 => 4,
        }
    }
}

/// Buffer compartilhado entre um array e as suas views.
#[derive(Clone, Debug)]
pub enum ArrayData {
    Bool(Arc<[bool]>),
    I64(Arc<[i64]>),
    F32(Arc<[f32]>),
    F64(Arc<[f64]>),
}

impl ArrayData {
    fn dtype(&self) -> DType {
        match self {
            ArrayData::Bool(_) => DType::Bool,
            ArrayData::I64(_) => DType::I64,
            ArrayData::F32(_) => DType::F32,
            ArrayData::F64(_) => DType::F64,
        }
    }

    fn same_buffer(&self, other: &ArrayData) -> bool {
        match (self, other) {
            (ArrayData::Bool(a), ArrayData::Bool(b)) => Arc::ptr_eq(a, b),
            (ArrayData::I64(a), ArrayData::I64(b)) => Arc::ptr_eq(a, b),
            (ArrayData::F32(a), ArrayData::F32(b)) => Arc::ptr_eq(a, b),
            (ArrayData::F64(a), ArrayData::F64(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// Os quatro tipos de elemento, com a conversão de cada um no sentido do `astype` do NumPy.
pub trait Element: Copy + PartialEq + fmt::Debug + 'static {
    const DTYPE: DType;
    fn wrap(values: Vec<Self>) -> ArrayData;
    fn slice(data: &ArrayData) -> Option<&[Self]>;
    fn from_bool(v: bool) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_f32(v: f32) -> Self;
    fn from_f64(v: f64) -> Self;
}

macro_rules! element {
    ($t:ty, $variant:ident, $b:expr, $i:expr, $f:expr, $d:expr) => {
        impl Element for $t {
            const DTYPE: DType = DType::$variant;
            fn wrap(values: Vec<Self>) -> ArrayData {
                ArrayData::$variant(values.into())
            }
            fn slice(data: &ArrayData) -> Option<&[Self]> {
                match data {
                    ArrayData::$variant(v) => Some(v),
                    _ => None,
                }
            }
            fn from_bool(v: bool) -> Self {
                $b(v)
            }
            fn from_i64(v: i64) -> Self {
                $i(v)
            }
            fn from_f32(v: f32) -> Self {
                $f(v)
            }
            fn from_f64(v: f64) -> Self {
                $d(v)
            }
        }
    };
}

element!(bool, Bool, |v| v, |v| v != 0, |v| v != 0.0, |v| v != 0.0);
element!(
    i64,
    I64,
    |v| v as i64,
    |v| v,
    |v: f32| v as i64,
    |v: f64| v as i64
);
element!(f32, F32, |v| u8::from(v) as f32, |v| v as f32, |v| v, |v| v
    as f32);
element!(
    f64,
    F64,
    |v| u8::from(v) as f64,
    |v| v as f64,
    |v| v as f64,
    |v| v
);

#[derive(Clone, Debug, PartialEq)]
pub enum ArrayError {
    DataLength { expected: usize, actual: usize },
    Overflow,
    Axis { axis: usize, ndim: usize },
    Index(String),
    Broadcast { left: Vec<usize>, right: Vec<usize> },
    Reshape { from: Vec<usize>, to: Vec<usize> },
    DType(String),
    Empty,
    Npy(String),
    Io(String),
}

impl fmt::Display for ArrayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayError::DataLength { expected, actual } => {
                write!(f, "a forma pede {expected} elementos e vieram {actual}")
            }
            ArrayError::Overflow => write!(f, "a forma estoura o tamanho máximo"),
            ArrayError::Axis { axis, ndim } => {
                write!(f, "eixo {axis} fora de um array de {ndim} dimensões")
            }
            ArrayError::Index(why) => write!(f, "índice inválido: {why}"),
            ArrayError::Broadcast { left, right } => {
                write!(
                    f,
                    "as formas {left:?} e {right:?} não são compatíveis para broadcasting"
                )
            }
            ArrayError::Reshape { from, to } => {
                write!(f, "não dá para mudar a forma {from:?} para {to:?}")
            }
            ArrayError::DType(why) => write!(f, "tipo: {why}"),
            ArrayError::Empty => write!(f, "redução sem elementos"),
            ArrayError::Npy(why) => write!(f, "arquivo .npy: {why}"),
            ArrayError::Io(why) => write!(f, "arquivo: {why}"),
        }
    }
}

impl std::error::Error for ArrayError {}

/// Um eixo de uma fatia: índice inteiro (o eixo some) ou intervalo com passo, como `a[1:5:2]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Index {
    At(isize),
    Range {
        start: Option<isize>,
        stop: Option<isize>,
        step: isize,
    },
}

impl Index {
    pub fn all() -> Self {
        Index::Range {
            start: None,
            stop: None,
            step: 1,
        }
    }
}

#[derive(Clone, Copy)]
enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Compare {
    fn test<T: PartialOrd>(self, a: T, b: T) -> bool {
        match self {
            Compare::Eq => a == b,
            Compare::Ne => a != b,
            Compare::Lt => a < b,
            Compare::Le => a <= b,
            Compare::Gt => a > b,
            Compare::Ge => a >= b,
        }
    }
}

#[derive(Clone, Copy)]
enum Reduce {
    Sum,
    Mean,
    Max,
    Min,
}

#[derive(Clone, Debug)]
pub struct NdArray {
    data: ArrayData,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

fn element_count(shape: &[usize]) -> Result<usize, ArrayError> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or(ArrayError::Overflow)
}

fn c_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![1isize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1].max(1) as isize;
    }
    strides
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let ndim = left.len().max(right.len());
    let mut shape = vec![0; ndim];
    for i in 0..ndim {
        let l = if i < ndim - left.len() {
            1
        } else {
            left[i - (ndim - left.len())]
        };
        let r = if i < ndim - right.len() {
            1
        } else {
            right[i - (ndim - right.len())]
        };
        shape[i] = match (l, r) {
            (l, r) if l == r => l,
            (1, r) => r,
            (l, 1) => l,
            _ => {
                return Err(ArrayError::Broadcast {
                    left: left.to_vec(),
                    right: right.to_vec(),
                })
            }
        };
    }
    Ok(shape)
}

/// Tipo do resultado de uma conta entre dois arrays, pela tabela do NumPy para estes quatro tipos.
fn promote(a: DType, b: DType) -> DType {
    match (a.min(b), a.max(b)) {
        (DType::I64, DType::F32) => DType::F64,
        (_, high) => high,
    }
}

impl NdArray {
    pub fn from_vec<T: Element>(values: Vec<T>, shape: &[usize]) -> Result<Self, ArrayError> {
        let expected = element_count(shape)?;
        if values.len() != expected {
            return Err(ArrayError::DataLength {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self::contiguous(T::wrap(values), shape.to_vec()))
    }

    fn contiguous(data: ArrayData, shape: Vec<usize>) -> Self {
        Self {
            strides: c_strides(&shape),
            data,
            shape,
            offset: 0,
        }
    }

    pub fn zeros(dtype: DType, shape: &[usize]) -> Result<Self, ArrayError> {
        let n = element_count(shape)?;
        let data = match dtype {
            DType::Bool => ArrayData::Bool(vec![false; n].into()),
            DType::I64 => ArrayData::I64(vec![0; n].into()),
            DType::F32 => ArrayData::F32(vec![0.0; n].into()),
            DType::F64 => ArrayData::F64(vec![0.0; n].into()),
        };
        Ok(Self::contiguous(data, shape.to_vec()))
    }

    /// 0, 1, …, n − 1, em i64.
    pub fn arange(n: usize) -> Result<Self, ArrayError> {
        Self::from_vec((0..n as i64).collect::<Vec<_>>(), &[n])
    }

    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Passo de cada eixo, em elementos (o NumPy dá em bytes).
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data(&self) -> &ArrayData {
        &self.data
    }

    /// Verdadeiro quando os dois arrays leem o mesmo buffer: uma view do outro, ou views do mesmo array.
    pub fn shares_memory(&self, other: &NdArray) -> bool {
        self.data.same_buffer(&other.data)
    }

    /// Elementos em ordem C, contíguos na memória.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1isize;
        for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dim > 1 && stride != expected {
                return false;
            }
            expected *= dim as isize;
        }
        true
    }

    /// Posição no buffer de cada elemento, em ordem C.
    fn offsets(&self) -> Vec<usize> {
        let n = self.len();
        if self.is_contiguous() {
            return (self.offset..self.offset + n).collect();
        }
        let mut out = Vec::with_capacity(n);
        let mut index = vec![0usize; self.ndim()];
        let mut position = self.offset as isize;
        for _ in 0..n {
            out.push(position as usize);
            for axis in (0..self.ndim()).rev() {
                index[axis] += 1;
                position += self.strides[axis];
                if index[axis] < self.shape[axis] {
                    break;
                }
                position -= self.strides[axis] * self.shape[axis] as isize;
                index[axis] = 0;
            }
        }
        out
    }

    fn gather<T: Copy>(&self, buffer: &[T]) -> Vec<T> {
        if self.is_contiguous() {
            return buffer[self.offset..self.offset + self.len()].to_vec();
        }
        self.offsets().into_iter().map(|i| buffer[i]).collect()
    }

    /// Valores em ordem C, convertidos para `T` como no `astype` do NumPy.
    pub fn values_as<T: Element>(&self) -> Vec<T> {
        match &self.data {
            ArrayData::Bool(b) => self.gather(b).into_iter().map(T::from_bool).collect(),
            ArrayData::I64(b) => self.gather(b).into_iter().map(T::from_i64).collect(),
            ArrayData::F32(b) => self.gather(b).into_iter().map(T::from_f32).collect(),
            ArrayData::F64(b) => self.gather(b).into_iter().map(T::from_f64).collect(),
        }
    }

    /// Valores em ordem C, sem conversão: o tipo pedido precisa ser o do array.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>, ArrayError> {
        T::slice(&self.data)
            .map(|buffer| self.gather(buffer))
            .ok_or_else(|| {
                ArrayError::DType(format!("o array é {:?}, não {:?}", self.dtype(), T::DTYPE))
            })
    }

    /// Fatia sem cópia, com a semântica de `a[...]` do NumPy. Eixos não citados ficam inteiros.
    pub fn slice(&self, indices: &[Index]) -> Result<NdArray, ArrayError> {
        if indices.len() > self.ndim() {
            return Err(ArrayError::Index(format!(
                "{} índices para {} dimensões",
                indices.len(),
                self.ndim()
            )));
        }
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let mut offset = self.offset as isize;
        for axis in 0..self.ndim() {
            let len = self.shape[axis] as isize;
            let stride = self.strides[axis];
            match indices.get(axis).copied().unwrap_or_else(Index::all) {
                Index::At(k) => {
                    let k = if k < 0 { k + len } else { k };
                    if !(0..len).contains(&k) {
                        return Err(ArrayError::Index(format!(
                            "{k} fora do eixo {axis} de tamanho {len}"
                        )));
                    }
                    offset += k * stride;
                }
                Index::Range { start, stop, step } => {
                    let (first, count) = range(start, stop, step, len)?;
                    if count > 0 {
                        offset += first * stride;
                    }
                    shape.push(count as usize);
                    strides.push(stride * step);
                }
            }
        }
        Ok(NdArray {
            data: self.data.clone(),
            shape,
            strides,
            offset: offset as usize,
        })
    }

    /// Eixos em ordem invertida, sem cópia, como `a.T`.
    pub fn transpose(&self) -> NdArray {
        let axes: Vec<usize> = (0..self.ndim()).rev().collect();
        self.permute(&axes)
            .expect("a inversão dos eixos é uma permutação")
    }

    /// Eixos reordenados, sem cópia, como `a.transpose(axes)`.
    pub fn permute(&self, axes: &[usize]) -> Result<NdArray, ArrayError> {
        let mut seen = vec![false; self.ndim()];
        if axes.len() != self.ndim()
            || axes
                .iter()
                .any(|&a| a >= self.ndim() || std::mem::replace(&mut seen[a], true))
        {
            return Err(ArrayError::Index(format!(
                "{axes:?} não é uma permutação de {} eixos",
                self.ndim()
            )));
        }
        Ok(NdArray {
            data: self.data.clone(),
            shape: axes.iter().map(|&a| self.shape[a]).collect(),
            strides: axes.iter().map(|&a| self.strides[a]).collect(),
            offset: self.offset,
        })
    }

    /// A mesma forma lida em outra. Sem cópia quando o array é contíguo; senão copia antes, como o NumPy.
    pub fn reshape(&self, shape: &[usize]) -> Result<NdArray, ArrayError> {
        if element_count(shape)? != self.len() {
            return Err(ArrayError::Reshape {
                from: self.shape.clone(),
                to: shape.to_vec(),
            });
        }
        let base = self.to_contiguous();
        Ok(NdArray {
            strides: c_strides(shape),
            shape: shape.to_vec(),
            ..base
        })
    }

    /// Repete os eixos de tamanho 1 (e os que faltam à esquerda) com passo 0, sem cópia.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<NdArray, ArrayError> {
        let error = || ArrayError::Broadcast {
            left: self.shape.clone(),
            right: shape.to_vec(),
        };
        if shape.len() < self.ndim() {
            return Err(error());
        }
        let extra = shape.len() - self.ndim();
        let mut strides = vec![0isize; shape.len()];
        for (axis, &dim) in shape.iter().enumerate().skip(extra) {
            let own = self.shape[axis - extra];
            strides[axis] = match own {
                d if d == dim => self.strides[axis - extra],
                1 => 0,
                _ => return Err(error()),
            };
        }
        Ok(NdArray {
            data: self.data.clone(),
            shape: shape.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    /// O próprio array se já for contíguo; senão, uma cópia em ordem C.
    pub fn to_contiguous(&self) -> NdArray {
        if self.is_contiguous() {
            return NdArray {
                strides: c_strides(&self.shape),
                ..self.clone()
            };
        }
        let data = match &self.data {
            ArrayData::Bool(b) => ArrayData::Bool(self.gather(b).into()),
            ArrayData::I64(b) => ArrayData::I64(self.gather(b).into()),
            ArrayData::F32(b) => ArrayData::F32(self.gather(b).into()),
            ArrayData::F64(b) => ArrayData::F64(self.gather(b).into()),
        };
        Self::contiguous(data, self.shape.clone())
    }

    /// Cópia convertida para outro tipo. Diferente do NumPy, NaN ou infinito para i64 é erro, não lixo.
    pub fn astype(&self, dtype: DType) -> Result<NdArray, ArrayError> {
        let shape = &self.shape;
        if dtype == DType::I64 && matches!(self.dtype(), DType::F32 | DType::F64) {
            let values: Vec<f64> = self.values_as();
            if values
                .iter()
                .any(|v| !v.is_finite() || v.abs() >= I64_LIMIT)
            {
                return Err(ArrayError::DType(
                    "valor não finito ou grande demais para i64".into(),
                ));
            }
        }
        match dtype {
            DType::Bool => Self::from_vec(self.values_as::<bool>(), shape),
            DType::I64 => Self::from_vec(self.values_as::<i64>(), shape),
            DType::F32 => Self::from_vec(self.values_as::<f32>(), shape),
            DType::F64 => Self::from_vec(self.values_as::<f64>(), shape),
        }
    }

    fn arith(&self, other: &NdArray, op: Arith) -> Result<NdArray, ArrayError> {
        let shape = broadcast_shape(&self.shape, &other.shape)?;
        let (a, b) = (self.broadcast_to(&shape)?, other.broadcast_to(&shape)?);
        let kind = match op {
            Arith::Div if (a.dtype(), b.dtype()) == (DType::F32, DType::F32) => DType::F32,
            Arith::Div => DType::F64,
            _ => promote(a.dtype(), b.dtype()),
        };
        match kind {
            DType::Bool => {
                let pair = a.values_as::<bool>().into_iter().zip(b.values_as::<bool>());
                let values = match op {
                    Arith::Add => pair.map(|(x, y)| x | y).collect(),
                    Arith::Mul => pair.map(|(x, y)| x & y).collect(),
                    _ => {
                        return Err(ArrayError::DType(
                            "subtração de bool; converta com astype antes".into(),
                        ))
                    }
                };
                Self::from_vec::<bool>(values, &shape)
            }
            DType::I64 => {
                let pair = a.values_as::<i64>().into_iter().zip(b.values_as::<i64>());
                let values = match op {
                    Arith::Add => pair.map(|(x, y)| x.wrapping_add(y)).collect(),
                    Arith::Sub => pair.map(|(x, y)| x.wrapping_sub(y)).collect(),
                    Arith::Mul => pair.map(|(x, y)| x.wrapping_mul(y)).collect(),
                    Arith::Div => unreachable!("a divisão nunca é i64"),
                };
                Self::from_vec::<i64>(values, &shape)
            }
            DType::F32 => Self::from_vec(
                float_op(a.values_as::<f32>(), b.values_as::<f32>(), op),
                &shape,
            ),
            DType::F64 => Self::from_vec(
                float_op(a.values_as::<f64>(), b.values_as::<f64>(), op),
                &shape,
            ),
        }
    }

    pub fn add(&self, other: &NdArray) -> Result<NdArray, ArrayError> {
        self.arith(other, Arith::Add)
    }

    pub fn sub(&self, other: &NdArray) -> Result<NdArray, ArrayError> {
        self.arith(other, Arith::Sub)
    }

    pub fn mul(&self, other: &NdArray) -> Result<NdArray, ArrayError> {
        self.arith(other, Arith::Mul)
    }

    /// Divisão verdadeira, como `/` no NumPy: o resultado é sempre de ponto flutuante.
    pub fn div(&self, other: &NdArray) -> Result<NdArray, ArrayError> {
        self.arith(other, Arith::Div)
    }

    /// Comparação elemento a elemento, com broadcasting; o resultado é bool.
    pub fn compare(&self, other: &NdArray, op: Compare) -> Result<NdArray, ArrayError> {
        let shape = broadcast_shape(&self.shape, &other.shape)?;
        let (a, b) = (self.broadcast_to(&shape)?, other.broadcast_to(&shape)?);
        let values: Vec<bool> = match promote(a.dtype(), b.dtype()) {
            DType::Bool => zip_test(a.values_as::<bool>(), b.values_as::<bool>(), op),
            DType::I64 => zip_test(a.values_as::<i64>(), b.values_as::<i64>(), op),
            DType::F32 => zip_test(a.values_as::<f32>(), b.values_as::<f32>(), op),
            DType::F64 => zip_test(a.values_as::<f64>(), b.values_as::<f64>(), op),
        };
        Self::from_vec(values, &shape)
    }

    /// Eixo movido para o fim; devolve a forma do resultado e o tamanho de cada fileira reduzida.
    fn lanes(&self, axis: Option<usize>) -> Result<(NdArray, Vec<usize>, usize), ArrayError> {
        let Some(axis) = axis else {
            return Ok((self.clone(), Vec::new(), self.len()));
        };
        if axis >= self.ndim() {
            return Err(ArrayError::Axis {
                axis,
                ndim: self.ndim(),
            });
        }
        let mut order: Vec<usize> = (0..self.ndim()).filter(|&a| a != axis).collect();
        let out_shape = order.iter().map(|&a| self.shape[a]).collect();
        order.push(axis);
        Ok((self.permute(&order)?, out_shape, self.shape[axis]))
    }

    fn reduce(&self, axis: Option<usize>, how: Reduce) -> Result<NdArray, ArrayError> {
        let (moved, shape, lane) = self.lanes(axis)?;
        let out = element_count(&shape)?;
        if lane == 0 && matches!(how, Reduce::Max | Reduce::Min) {
            return Err(ArrayError::Empty);
        }
        match (how, self.dtype()) {
            (Reduce::Sum, DType::Bool | DType::I64) => {
                let sums = fold(moved.values_as::<i64>(), lane, out, |v| {
                    v.iter().fold(0i64, |s, &x| s.wrapping_add(x))
                });
                Self::from_vec(sums, &shape)
            }
            (Reduce::Sum, DType::F32) => {
                let sums = fold(moved.values_as::<f64>(), lane, out, |v| {
                    v.iter().sum::<f64>() as f32
                });
                Self::from_vec(sums, &shape)
            }
            (Reduce::Sum, DType::F64) => {
                Self::from_vec(fold(moved.values_as::<f64>(), lane, out, pairwise), &shape)
            }
            (Reduce::Mean, DType::F32) => {
                let means = fold(moved.values_as::<f64>(), lane, out, |v| {
                    (v.iter().sum::<f64>() / lane as f64) as f32
                });
                Self::from_vec(means, &shape)
            }
            (Reduce::Mean, _) => {
                let means = fold(moved.values_as::<f64>(), lane, out, |v| {
                    pairwise(v) / lane as f64
                });
                Self::from_vec(means, &shape)
            }
            (Reduce::Max | Reduce::Min, dtype) => {
                let max = matches!(how, Reduce::Max);
                match dtype {
                    DType::Bool => Self::from_vec(
                        fold(moved.values_as::<bool>(), lane, out, |v| extreme(v, max)),
                        &shape,
                    ),
                    DType::I64 => Self::from_vec(
                        fold(moved.values_as::<i64>(), lane, out, |v| extreme(v, max)),
                        &shape,
                    ),
                    DType::F32 => Self::from_vec(
                        fold(moved.values_as::<f32>(), lane, out, |v| extreme(v, max)),
                        &shape,
                    ),
                    DType::F64 => Self::from_vec(
                        fold(moved.values_as::<f64>(), lane, out, |v| extreme(v, max)),
                        &shape,
                    ),
                }
            }
        }
    }

    /// Soma em um eixo, ou em todos com `None`. bool e i64 somam em i64.
    pub fn sum(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.reduce(axis, Reduce::Sum)
    }

    /// Média; bool e i64 dão f64. Sem elementos, NaN, como o NumPy.
    pub fn mean(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.reduce(axis, Reduce::Mean)
    }

    /// Máximo; NaN se a fileira tiver NaN, como o NumPy.
    pub fn max(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.reduce(axis, Reduce::Max)
    }

    pub fn min(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.reduce(axis, Reduce::Min)
    }

    /// Bytes de um `.npy` versão 1.0, em ordem C, com o mesmo cabeçalho que o NumPy escreve.
    pub fn to_npy_bytes(&self) -> Vec<u8> {
        let shape = match self.shape.as_slice() {
            [] => "()".to_string(),
            [n] => format!("({n},)"),
            dims => format!(
                "({})",
                dims.iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let mut header = format!(
            "{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}",
            self.dtype().npy_descr()
        );
        let total = 10 + header.len() + 1;
        header.push_str(&" ".repeat((64 - total % 64) % 64));
        header.push('\n');
        let mut out = Vec::with_capacity(10 + header.len() + self.len() * self.dtype().size());
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        match &self.data {
            ArrayData::Bool(b) => out.extend(self.gather(b).into_iter().map(u8::from)),
            ArrayData::I64(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            ArrayData::F32(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
            ArrayData::F64(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        }
        out
    }

    /// Lê um `.npy` (versões 1, 2 e 3) de `<f8`, `<f4`, `<i8` ou `|b1`. Ordem Fortran vira strides, sem cópia.
    pub fn from_npy_bytes(bytes: &[u8]) -> Result<NdArray, ArrayError> {
        let bad = |why: &str| ArrayError::Npy(why.to_string());
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err(bad("assinatura ausente"));
        }
        let (header_len, start) = match bytes[6] {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 | 3 if bytes.len() >= 12 => (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            ),
            v => return Err(ArrayError::Npy(format!("versão {v} não suportada"))),
        };
        let header = bytes
            .get(start..start + header_len)
            .and_then(|h| std::str::from_utf8(h).ok())
            .ok_or_else(|| bad("cabeçalho cortado"))?;
        let descr = header_field(header, "descr").ok_or_else(|| bad("sem descr"))?;
        let dtype = match descr.trim_matches(|c| c == '\'' || c == '"') {
            "|b1" | "<b1" => DType::Bool,
            "<i8" => DType::I64,
            "<f4" => DType::F32,
            "<f8" => DType::F64,
            other => return Err(ArrayError::Npy(format!("tipo {other} não suportado"))),
        };
        let fortran = match header_field(header, "fortran_order") {
            Some("True") => true,
            Some("False") => false,
            _ => return Err(bad("sem fortran_order")),
        };
        let shape_text = header_field(header, "shape").ok_or_else(|| bad("sem shape"))?;
        let shape = shape_text
            .trim_matches(|c| c == '(' || c == ')')
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().map_err(|_| bad("dimensão inválida")))
            .collect::<Result<Vec<_>, _>>()?;
        let n = element_count(&shape)?;
        let body = &bytes[start + header_len..];
        if body.len() != n * dtype.size() {
            return Err(ArrayError::DataLength {
                expected: n * dtype.size(),
                actual: body.len(),
            });
        }
        let data = match dtype {
            DType::Bool => ArrayData::Bool(body.iter().map(|&b| b != 0).collect::<Vec<_>>().into()),
            DType::I64 => ArrayData::I64(
                body.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            DType::F32 => ArrayData::F32(
                body.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            DType::F64 => ArrayData::F64(
                body.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| f64::from_le_bytes(*c))
                    .collect::<Vec<_>>()
                    .into(),
            ),
        };
        if !fortran {
            return Ok(Self::contiguous(data, shape));
        }
        let reversed: Vec<usize> = shape.iter().rev().copied().collect();
        let strides = c_strides(&reversed).into_iter().rev().collect();
        Ok(NdArray {
            data,
            shape,
            strides,
            offset: 0,
        })
    }

    pub fn read_npy(path: impl AsRef<Path>) -> Result<NdArray, ArrayError> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| ArrayError::Io(e.to_string()))?;
        Self::from_npy_bytes(&bytes)
    }

    pub fn write_npy(&self, path: impl AsRef<Path>) -> Result<(), ArrayError> {
        std::fs::write(path.as_ref(), self.to_npy_bytes())
            .map_err(|e| ArrayError::Io(e.to_string()))
    }
}

/// Primeiro índice e quantidade de elementos de `start:stop:step`, com as regras do Python.
fn range(
    start: Option<isize>,
    stop: Option<isize>,
    step: isize,
    len: isize,
) -> Result<(isize, isize), ArrayError> {
    if step == 0 {
        return Err(ArrayError::Index("passo zero".into()));
    }
    let fix = |v: isize, low: isize, high: isize| {
        let v = if v < 0 { v + len } else { v };
        v.clamp(low, high)
    };
    if step > 0 {
        let first = start.map_or(0, |v| fix(v, 0, len));
        let last = stop.map_or(len, |v| fix(v, 0, len));
        let count = if last > first {
            (last - first + step - 1) / step
        } else {
            0
        };
        Ok((first, count))
    } else {
        let first = start.map_or(len - 1, |v| fix(v, -1, len - 1));
        let last = stop.map_or(-1, |v| fix(v, -1, len - 1));
        let count = if first > last {
            (first - last - 1) / (-step) + 1
        } else {
            0
        };
        Ok((first, count))
    }
}

fn header_field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let rest = &header[header.find(&format!("'{key}'"))? + key.len() + 2..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else {
        rest.find(',')?
    };
    Some(rest[..end].trim())
}

fn float_op<T>(a: Vec<T>, b: Vec<T>, op: Arith) -> Vec<T>
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>,
{
    let pair = a.into_iter().zip(b);
    match op {
        Arith::Add => pair.map(|(x, y)| x + y).collect(),
        Arith::Sub => pair.map(|(x, y)| x - y).collect(),
        Arith::Mul => pair.map(|(x, y)| x * y).collect(),
        Arith::Div => pair.map(|(x, y)| x / y).collect(),
    }
}

fn zip_test<T: PartialOrd>(a: Vec<T>, b: Vec<T>, op: Compare) -> Vec<bool> {
    a.into_iter().zip(b).map(|(x, y)| op.test(x, y)).collect()
}

/// Aplica `f` a cada fileira de `lane` elementos; com fileiras vazias, a `f` recebe uma fatia vazia.
fn fold<T, R>(values: Vec<T>, lane: usize, out: usize, f: impl Fn(&[T]) -> R) -> Vec<R> {
    if lane == 0 {
        return (0..out).map(|_| f(&[])).collect();
    }
    values.chunks(lane).map(f).collect()
}

/// Soma em pares, como o NumPy: o erro cresce com log n, não com n.
fn pairwise(v: &[f64]) -> f64 {
    if v.len() <= 8 {
        return v.iter().sum();
    }
    let (left, right) = v.split_at(v.len() / 2);
    pairwise(left) + pairwise(right)
}

/// Máximo ou mínimo; um NaN contamina o resultado, como no NumPy.
fn extreme<T: Copy + PartialOrd>(v: &[T], max: bool) -> T {
    let nan = |x: T| x.partial_cmp(&x).is_none();
    v.iter().copied().fold(v[0], |best, x| {
        if nan(best) {
            best
        } else if nan(x) || (max && x > best) || (!max && x < best) {
            x
        } else {
            best
        }
    })
}
