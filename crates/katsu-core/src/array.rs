//! Arrays N-dimensionais com tipo (bool, u8, i32, i64, f16, bf16, f32, f64) e views sem cópia, no
//! modelo do NumPy. Uma view é forma, strides e deslocamento sobre o mesmo buffer compartilhado; as views
//! são só de leitura e a cópia é explícita. Promoção de tipos e reduções seguem o NumPy 1.26, conferidos
//! contra os gabaritos de `tests/fixtures/array`. Plano: `docs/KATSU_ARRAY_PLAN.md` (fase 0).

use std::fmt;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I32,
    I64,
    F16,
    Bf16,
    F32,
    F64,
}

impl DType {
    fn npy_descr(self) -> Option<&'static str> {
        Some(match self {
            DType::Bool => "|b1",
            DType::U8 => "|u1",
            DType::I32 => "<i4",
            DType::I64 => "<i8",
            DType::F16 => "<f2",
            DType::Bf16 => return None,
            DType::F32 => "<f4",
            DType::F64 => "<f8",
        })
    }

    pub fn size(self) -> usize {
        match self {
            DType::Bool | DType::U8 => 1,
            DType::F16 | DType::Bf16 => 2,
            DType::I32 | DType::F32 => 4,
            DType::I64 | DType::F64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::Bf16 | DType::F32 | DType::F64)
    }

    fn is_int(self) -> bool {
        matches!(self, DType::U8 | DType::I32 | DType::I64)
    }

    /// Faixa de inteiros representáveis, para recusar conversões de float fora dela.
    fn int_range(self) -> (f64, f64) {
        match self {
            DType::U8 => (0.0, 256.0),
            DType::I32 => (-2_147_483_648.0, 2_147_483_648.0),
            _ => (-I64_LIMIT, I64_LIMIT),
        }
    }

    /// O menor float que guarda o inteiro exatamente: é o que o NumPy usa na promoção com floats.
    fn as_float(self) -> DType {
        match self {
            DType::Bool | DType::U8 => DType::F16,
            DType::I32 | DType::I64 => DType::F64,
            float => float,
        }
    }
}

/// 2^63: o primeiro f64 que não cabe em i64.
const I64_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// Meia precisão IEEE 754 (binary16), guardada nos bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct F16(pub u16);

/// bfloat16: os 16 bits altos de um f32, sem o NumPy como referência (vem do `ml_dtypes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Bf16(pub u16);

impl F16 {
    pub fn from_f64(x: f64) -> Self {
        F16(encode(x, 5, 10) as u16)
    }

    pub fn to_f64(self) -> f64 {
        decode(self.0 as u32, 5, 10)
    }
}

impl Bf16 {
    pub fn from_f64(x: f64) -> Self {
        Bf16(encode(x, 8, 7) as u16)
    }

    pub fn to_f64(self) -> f64 {
        decode(self.0 as u32, 8, 7)
    }
}

/// f64 para um float menor de `ebits` de expoente e `mbits` de mantissa, arredondando ao par mais
/// próximo numa só etapa (passar por f32 antes poderia arredondar duas vezes).
fn encode(x: f64, ebits: u32, mbits: u32) -> u32 {
    let bits = x.to_bits();
    let sign = ((bits >> 63) as u32) << (ebits + mbits);
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let man = bits & ((1u64 << 52) - 1);
    let emax = (1u32 << ebits) - 1;
    let bias = (1i64 << (ebits - 1)) - 1;
    if exp == 0x7ff {
        let nan = if man != 0 { 1u32 << (mbits - 1) } else { 0 };
        return sign | (emax << mbits) | nan;
    }
    if exp == 0 {
        return sign;
    }
    let e = exp - 1023 + bias;
    let drop = 52 - mbits as i64;
    let (kept, shift) = if e >= 1 {
        ((e as u64) << mbits | (man >> drop), drop)
    } else {
        let shift = drop + 1 - e;
        if shift > 53 {
            return sign;
        }
        ((man | 1u64 << 52) >> shift, shift)
    };
    let full = if e >= 1 { man } else { man | 1u64 << 52 };
    let rest = full & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let rounded = kept + u64::from(rest > half || (rest == half && kept & 1 == 1));
    let limit = (emax as u64) << mbits;
    sign | rounded.min(limit) as u32
}

fn decode(h: u32, ebits: u32, mbits: u32) -> f64 {
    let sign = if (h >> (ebits + mbits)) & 1 == 1 {
        -1.0
    } else {
        1.0
    };
    let exp = ((h >> mbits) & ((1 << ebits) - 1)) as i32;
    let man = (h & ((1 << mbits) - 1)) as f64;
    let bias = (1i32 << (ebits - 1)) - 1;
    let scale = (1u64 << mbits) as f64;
    if exp == (1 << ebits) - 1 {
        return if man != 0.0 {
            f64::NAN
        } else {
            sign * f64::INFINITY
        };
    }
    if exp == 0 {
        return sign * man / scale * 2f64.powi(1 - bias);
    }
    sign * (1.0 + man / scale) * 2f64.powi(exp - bias)
}

/// Buffer compartilhado entre um array e as suas views.
#[derive(Clone, Debug)]
pub enum ArrayData {
    Bool(Arc<[bool]>),
    U8(Arc<[u8]>),
    I32(Arc<[i32]>),
    I64(Arc<[i64]>),
    F16(Arc<[F16]>),
    Bf16(Arc<[Bf16]>),
    F32(Arc<[f32]>),
    F64(Arc<[f64]>),
}

/// Executa `$body` com `$v` ligado ao buffer, qualquer que seja o tipo.
macro_rules! each {
    ($data:expr, $v:ident => $body:expr) => {
        match $data {
            ArrayData::Bool($v) => $body,
            ArrayData::U8($v) => $body,
            ArrayData::I32($v) => $body,
            ArrayData::I64($v) => $body,
            ArrayData::F16($v) => $body,
            ArrayData::Bf16($v) => $body,
            ArrayData::F32($v) => $body,
            ArrayData::F64($v) => $body,
        }
    };
}

/// Executa `$body` com `$t` sendo o tipo Rust de `$dtype`.
macro_rules! with_type {
    ($dtype:expr, $t:ident => $body:expr) => {
        match $dtype {
            DType::Bool => {
                type $t = bool;
                $body
            }
            DType::U8 => {
                type $t = u8;
                $body
            }
            DType::I32 => {
                type $t = i32;
                $body
            }
            DType::I64 => {
                type $t = i64;
                $body
            }
            DType::F16 => {
                type $t = F16;
                $body
            }
            DType::Bf16 => {
                type $t = Bf16;
                $body
            }
            DType::F32 => {
                type $t = f32;
                $body
            }
            DType::F64 => {
                type $t = f64;
                $body
            }
        }
    };
}

impl ArrayData {
    fn dtype(&self) -> DType {
        match self {
            ArrayData::Bool(_) => DType::Bool,
            ArrayData::U8(_) => DType::U8,
            ArrayData::I32(_) => DType::I32,
            ArrayData::I64(_) => DType::I64,
            ArrayData::F16(_) => DType::F16,
            ArrayData::Bf16(_) => DType::Bf16,
            ArrayData::F32(_) => DType::F32,
            ArrayData::F64(_) => DType::F64,
        }
    }

    fn address(&self) -> *const u8 {
        each!(self, v => v.as_ptr().cast::<u8>())
    }
}

/// Os tipos de elemento. A conversão segue o `astype` do NumPy: inteiros passam por i64 (dando a volta
/// no tipo menor), floats por f64 (exato para todos eles).
pub trait Element: Copy + PartialEq + fmt::Debug + Send + Sync + 'static {
    const DTYPE: DType;
    fn wrap(values: Vec<Self>) -> ArrayData;
    fn slice(data: &ArrayData) -> Option<&[Self]>;
    fn from_bool(v: bool) -> Self;
    fn from_i64(v: i64) -> Self;
    fn from_f64(v: f64) -> Self;
    /// O valor no tipo de cálculo da sua família: bool, i64 ou f64.
    fn hub(self) -> Hub;
}

/// Valor de qualquer elemento no tipo amplo da sua família.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hub {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl Hub {
    fn to<T: Element>(self) -> T {
        match self {
            Hub::Bool(v) => T::from_bool(v),
            Hub::Int(v) => T::from_i64(v),
            Hub::Float(v) => T::from_f64(v),
        }
    }
}

macro_rules! element {
    ($t:ty, $variant:ident, $hub:expr, $b:expr, $i:expr, $f:expr) => {
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
            fn from_f64(v: f64) -> Self {
                $f(v)
            }
            fn hub(self) -> Hub {
                $hub(self)
            }
        }
    };
}

element!(bool, Bool, Hub::Bool, |v| v, |v| v != 0, |v| v != 0.0);
element!(
    u8,
    U8,
    |v: u8| Hub::Int(v as i64),
    u8::from,
    |v| v as u8,
    |v: f64| v as i64 as u8
);
element!(
    i32,
    I32,
    |v: i32| Hub::Int(v as i64),
    i32::from,
    |v| v as i32,
    |v: f64| v as i64 as i32
);
element!(i64, I64, Hub::Int, i64::from, |v| v, |v: f64| v as i64);
element!(
    F16,
    F16,
    |v: F16| Hub::Float(v.to_f64()),
    |v| F16::from_f64(f64::from(u8::from(v))),
    |v| F16::from_f64(v as f64),
    F16::from_f64
);
element!(
    Bf16,
    Bf16,
    |v: Bf16| Hub::Float(v.to_f64()),
    |v| Bf16::from_f64(f64::from(u8::from(v))),
    |v| Bf16::from_f64(v as f64),
    Bf16::from_f64
);
element!(
    f32,
    F32,
    |v: f32| Hub::Float(v as f64),
    |v| f32::from(u8::from(v)),
    |v| v as f32,
    |v| v as f32
);
element!(
    f64,
    F64,
    Hub::Float,
    |v| f64::from(u8::from(v)),
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
    let dim = |s: &[usize], i: usize| {
        if i < ndim - s.len() {
            1
        } else {
            s[i - (ndim - s.len())]
        }
    };
    (0..ndim)
        .map(|i| match (dim(left, i), dim(right, i)) {
            (l, r) if l == r => Ok(l),
            (1, r) => Ok(r),
            (l, 1) => Ok(l),
            _ => Err(ArrayError::Broadcast {
                left: left.to_vec(),
                right: right.to_vec(),
            }),
        })
        .collect()
}

/// Tipo do resultado de uma conta entre dois arrays, pela tabela do NumPy 1.26 (conferida em
/// `tests/fixtures/array/promotion.txt`). bf16 com f16 dá f32, como no `ml_dtypes`.
pub fn promote(a: DType, b: DType) -> DType {
    use DType::*;
    if a == b {
        return a;
    }
    let rank = |d: DType| match d {
        Bool => 0,
        U8 => 1,
        I32 => 2,
        I64 => 3,
        F16 | Bf16 => 4,
        F32 => 5,
        F64 => 6,
    };
    let (low, high) = if rank(a) <= rank(b) { (a, b) } else { (b, a) };
    match (low, high) {
        (Bool, other) => other,
        (_, high) if !high.is_float() => high,
        (F16, Bf16) | (Bf16, F16) => F32,
        (low, high) => match (low.as_float(), high) {
            (F64, _) => F64,
            (F16, Bf16) => F32,
            (_, high) => high,
        },
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
        with_type!(dtype, T => Self::from_vec(vec![T::from_i64(0); n], shape))
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
        self.dtype() == other.dtype() && self.data.address() == other.data.address()
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
        if let Some(buffer) = T::slice(&self.data) {
            return self.gather(buffer);
        }
        each!(&self.data, b => self.gather(b).into_iter().map(|v| v.hub().to::<T>()).collect())
    }

    /// Valores em ordem C, sem conversão: o tipo pedido precisa ser o do array.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>, ArrayError> {
        T::slice(&self.data)
            .map(|buffer| self.gather(buffer))
            .ok_or_else(|| {
                ArrayError::DType(format!("o array é {:?}, não {:?}", self.dtype(), T::DTYPE))
            })
    }

    fn with_data<T: Element>(&self, values: Vec<T>) -> Result<NdArray, ArrayError> {
        Self::from_vec(values, &self.shape)
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
                    offset += normalize(k, len, axis)? * stride;
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

    /// Eixo `axis` levado para a frente, sem cópia; devolve também a ordem que desfaz a troca.
    fn axis_first(&self, axis: usize) -> Result<(NdArray, Vec<usize>), ArrayError> {
        if axis >= self.ndim() {
            return Err(ArrayError::Axis {
                axis,
                ndim: self.ndim(),
            });
        }
        let mut order = vec![axis];
        order.extend((0..self.ndim()).filter(|&a| a != axis));
        let mut back = vec![0; self.ndim()];
        for (position, &a) in order.iter().enumerate() {
            back[a] = position;
        }
        Ok((self.permute(&order)?, back))
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
            strides[axis] = match self.shape[axis - extra] {
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
        let data = each!(&self.data, b => Element::wrap(self.gather(b)));
        Self::contiguous(data, self.shape.clone())
    }

    /// Cópia convertida para outro tipo. Diferente do NumPy, float fora da faixa do inteiro (NaN e
    /// infinito inclusive) é erro, não um valor indefinido.
    pub fn astype(&self, dtype: DType) -> Result<NdArray, ArrayError> {
        if dtype.is_int() && self.dtype().is_float() {
            let (low, high) = dtype.int_range();
            if self
                .values_as::<f64>()
                .iter()
                .any(|v| !v.is_finite() || v.trunc() < low || v.trunc() >= high)
            {
                return Err(ArrayError::DType(format!(
                    "valor não finito ou fora da faixa de {dtype:?}"
                )));
            }
        }
        with_type!(dtype, T => self.with_data(self.values_as::<T>()))
    }

    fn arith(&self, other: &NdArray, op: Arith) -> Result<NdArray, ArrayError> {
        let shape = broadcast_shape(&self.shape, &other.shape)?;
        let (a, b) = (self.broadcast_to(&shape)?, other.broadcast_to(&shape)?);
        let common = promote(a.dtype(), b.dtype());
        let kind = match op {
            Arith::Div if !common.is_float() => DType::F64,
            _ => common,
        };
        match kind {
            DType::Bool => {
                let pair = a.values_as::<bool>().into_iter().zip(b.values_as::<bool>());
                let values: Vec<bool> = match op {
                    Arith::Add => pair.map(|(x, y)| x | y).collect(),
                    Arith::Mul => pair.map(|(x, y)| x & y).collect(),
                    _ => {
                        return Err(ArrayError::DType(
                            "subtração de bool; converta com astype antes".into(),
                        ))
                    }
                };
                Self::from_vec(values, &shape)
            }
            DType::U8 | DType::I32 | DType::I64 => {
                let pair = a.values_as::<i64>().into_iter().zip(b.values_as::<i64>());
                let wide: Vec<i64> = match op {
                    Arith::Add => pair.map(|(x, y)| x.wrapping_add(y)).collect(),
                    Arith::Sub => pair.map(|(x, y)| x.wrapping_sub(y)).collect(),
                    Arith::Mul => pair.map(|(x, y)| x.wrapping_mul(y)).collect(),
                    Arith::Div => unreachable!("a divisão de inteiros dá f64"),
                };
                // A conta em i64 seguida do corte dá o mesmo que a conta com volta no tipo menor.
                let wide = Self::from_vec(wide, &shape)?;
                with_type!(kind, T => wide.with_data(wide.values_as::<T>()))
            }
            DType::F64 => Self::from_vec(
                float_op(a.values_as::<f64>(), b.values_as::<f64>(), op),
                &shape,
            ),
            // f16 e bf16 fazem a conta em f32 e arredondam, como o NumPy; f32 tem bits de sobra para
            // que o arredondamento duplo não mude o resultado de + − × ÷.
            float => {
                let values = float_op(a.values_as::<f32>(), b.values_as::<f32>(), op);
                let wide = Self::from_vec(values, &shape)?;
                with_type!(float, T => wide.with_data(wide.values_as::<T>()))
            }
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

    /// Divisão verdadeira, como `/` no NumPy: inteiros e bool dão f64.
    pub fn div(&self, other: &NdArray) -> Result<NdArray, ArrayError> {
        self.arith(other, Arith::Div)
    }

    /// Comparação elemento a elemento, com broadcasting; o resultado é bool.
    pub fn compare(&self, other: &NdArray, op: Compare) -> Result<NdArray, ArrayError> {
        let shape = broadcast_shape(&self.shape, &other.shape)?;
        let (a, b) = (self.broadcast_to(&shape)?, other.broadcast_to(&shape)?);
        let common = promote(a.dtype(), b.dtype());
        let values: Vec<bool> = if common == DType::Bool {
            zip_test(a.values_as::<bool>(), b.values_as::<bool>(), op)
        } else if common.is_float() {
            zip_test(a.values_as::<f64>(), b.values_as::<f64>(), op)
        } else {
            zip_test(a.values_as::<i64>(), b.values_as::<i64>(), op)
        };
        Self::from_vec(values, &shape)
    }

    /// Fileiras do eixo reduzido: cada uma contígua no vetor devolvido, junto da forma do resultado.
    fn lanes<T: Element>(
        &self,
        axis: Option<usize>,
    ) -> Result<(Vec<T>, Vec<usize>, usize), ArrayError> {
        let Some(axis) = axis else {
            return Ok((self.values_as(), Vec::new(), self.len()));
        };
        let (moved, _) = self.axis_first(axis)?;
        let order: Vec<usize> = (1..self.ndim()).chain([0]).collect();
        let moved = moved.permute(&order)?;
        let shape = moved.shape[..moved.ndim() - 1].to_vec();
        Ok((moved.values_as(), shape, self.shape[axis]))
    }

    /// Soma em um eixo, ou em todos com `None`. bool e inteiros somam em i64; o NumPy dá u64 para u8,
    /// com os mesmos valores.
    pub fn sum(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        let dtype = self.dtype();
        if !dtype.is_float() {
            let (values, shape, lane) = self.lanes::<i64>(axis)?;
            let sums = fold(&values, lane, &shape, |v| {
                v.iter().fold(0i64, |s, &x| s.wrapping_add(x))
            });
            return Self::from_vec(sums, &shape);
        }
        let (values, shape, lane) = self.lanes::<f64>(axis)?;
        let wide = Self::from_vec(fold(&values, lane, &shape, pairwise), &shape)?;
        wide.astype(dtype)
    }

    /// Média; bool e inteiros dão f64, floats ficam no próprio tipo. Sem elementos, NaN, como o NumPy.
    pub fn mean(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        let (values, shape, lane) = self.lanes::<f64>(axis)?;
        let means = fold(&values, lane, &shape, |v| pairwise(v) / lane as f64);
        self.float_result(means, &shape)
    }

    /// Variância com `ddof` graus de liberdade descontados, em duas passadas (média, depois desvios).
    pub fn var(&self, axis: Option<usize>, ddof: usize) -> Result<NdArray, ArrayError> {
        let (values, shape, lane) = self.lanes::<f64>(axis)?;
        let vars = fold(&values, lane, &shape, |v| variance(v, ddof));
        self.float_result(vars, &shape)
    }

    /// Desvio padrão: a raiz da variância.
    pub fn std(&self, axis: Option<usize>, ddof: usize) -> Result<NdArray, ArrayError> {
        let (values, shape, lane) = self.lanes::<f64>(axis)?;
        let stds = fold(&values, lane, &shape, |v| variance(v, ddof).sqrt());
        self.float_result(stds, &shape)
    }

    /// Resultado em f64, convertido para o tipo do array quando ele é float (regra do NumPy em
    /// média, variância e desvio padrão).
    fn float_result(&self, values: Vec<f64>, shape: &[usize]) -> Result<NdArray, ArrayError> {
        let wide = Self::from_vec(values, shape)?;
        if self.dtype().is_float() {
            wide.astype(self.dtype())
        } else {
            Ok(wide)
        }
    }

    fn extreme(&self, axis: Option<usize>, max: bool) -> Result<NdArray, ArrayError> {
        if self.dtype().is_float() {
            let (values, shape, lane) = self.lanes::<f64>(axis)?;
            if lane == 0 {
                return Err(ArrayError::Empty);
            }
            let picked = fold(&values, lane, &shape, |v| v[best_index(v, max)]);
            return Self::from_vec(picked, &shape)?.astype(self.dtype());
        }
        // Inteiros comparados em i64: um i64 grande não cabe em f64 sem perda.
        let (values, shape, lane) = self.lanes::<i64>(axis)?;
        if lane == 0 {
            return Err(ArrayError::Empty);
        }
        let wide = Self::from_vec(
            fold(&values, lane, &shape, |v| v[best_index(v, max)]),
            &shape,
        )?;
        with_type!(self.dtype(), T => wide.with_data(wide.values_as::<T>()))
    }

    /// Máximo; NaN se a fileira tiver NaN, como o NumPy.
    pub fn max(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.extreme(axis, true)
    }

    pub fn min(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.extreme(axis, false)
    }

    /// Índice do máximo, em i64; o primeiro em caso de empate, e o primeiro NaN se houver, como o NumPy.
    /// Com `None`, o índice é no array achatado.
    pub fn argmax(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.arg(axis, true)
    }

    pub fn argmin(&self, axis: Option<usize>) -> Result<NdArray, ArrayError> {
        self.arg(axis, false)
    }

    fn arg(&self, axis: Option<usize>, max: bool) -> Result<NdArray, ArrayError> {
        let (positions, shape) = if self.dtype().is_float() {
            let (values, shape, lane) = self.lanes::<f64>(axis)?;
            if lane == 0 {
                return Err(ArrayError::Empty);
            }
            (
                fold(&values, lane, &shape, |v| best_index(v, max) as i64),
                shape,
            )
        } else {
            let (values, shape, lane) = self.lanes::<i64>(axis)?;
            if lane == 0 {
                return Err(ArrayError::Empty);
            }
            (
                fold(&values, lane, &shape, |v| best_index(v, max) as i64),
                shape,
            )
        };
        Self::from_vec(positions, &shape)
    }

    /// Elementos onde a máscara é verdadeira, como `a[mask]`. A máscara cobre os primeiros eixos; o
    /// resultado tem um eixo com a contagem e os eixos restantes inteiros. Sempre copia.
    pub fn select(&self, mask: &NdArray) -> Result<NdArray, ArrayError> {
        if mask.dtype() != DType::Bool {
            return Err(ArrayError::DType("a máscara precisa ser bool".into()));
        }
        if mask.ndim() > self.ndim() || mask.shape[..] != self.shape[..mask.ndim()] {
            return Err(ArrayError::Index(format!(
                "máscara de forma {:?} para um array de forma {:?}",
                mask.shape, self.shape
            )));
        }
        let keep = mask.to_vec::<bool>()?;
        let rest: Vec<usize> = self.shape[mask.ndim()..].to_vec();
        let block = rest.iter().product::<usize>();
        let mut shape = vec![keep.iter().filter(|&&k| k).count()];
        shape.extend(&rest);
        each!(&self.data, b => {
            let values = self.gather(b);
            let picked: Vec<_> = if block == 0 {
                Vec::new()
            } else {
                values
                    .chunks(block)
                    .zip(&keep)
                    .filter(|(_, k)| **k)
                    .flat_map(|(c, _)| c.iter().copied())
                    .collect()
            };
            Self::from_vec(picked, &shape)
        })
    }

    /// Linhas do eixo `axis` nas posições dadas, como `np.take`; aceita índice negativo. Sempre copia.
    pub fn take(&self, indices: &[i64], axis: usize) -> Result<NdArray, ArrayError> {
        let (front, back) = self.axis_first(axis)?;
        let len = front.shape[0] as isize;
        let rows = indices
            .iter()
            .map(|&i| normalize(i as isize, len, axis).map(|k| k as usize))
            .collect::<Result<Vec<_>, _>>()?;
        let block: usize = front.shape[1..].iter().product();
        let mut shape = front.shape.clone();
        shape[0] = rows.len();
        let taken = each!(&front.data, b => {
            let values = front.gather(b);
            let picked: Vec<_> = rows
                .iter()
                .flat_map(|&r| values[r * block..(r + 1) * block].iter().copied())
                .collect();
            Self::from_vec(picked, &shape)?
        });
        Ok(taken.permute(&back)?.to_contiguous())
    }

    /// Junta arrays ao longo de um eixo existente, como `np.concatenate`. O tipo é a promoção de todos.
    pub fn concatenate(arrays: &[&NdArray], axis: usize) -> Result<NdArray, ArrayError> {
        let first = arrays
            .first()
            .ok_or_else(|| ArrayError::Index("nada para concatenar".into()))?;
        let dtype = arrays
            .iter()
            .skip(1)
            .fold(first.dtype(), |d, a| promote(d, a.dtype()));
        let mut fronts = Vec::with_capacity(arrays.len());
        let mut back = Vec::new();
        for array in arrays {
            let same = array.ndim() == first.ndim()
                && (0..first.ndim()).all(|a| a == axis || array.shape[a] == first.shape[a]);
            if !same {
                return Err(ArrayError::Broadcast {
                    left: first.shape.clone(),
                    right: array.shape.clone(),
                });
            }
            let (front, order) = array.axis_first(axis)?;
            back = order;
            fronts.push(front);
        }
        let mut shape = fronts[0].shape.clone();
        shape[0] = fronts.iter().map(|f| f.shape[0]).sum();
        let joined = with_type!(dtype, T => {
            let values: Vec<T> = fronts.iter().flat_map(|f| f.values_as::<T>()).collect();
            Self::from_vec(values, &shape)?
        });
        Ok(joined.permute(&back)?.to_contiguous())
    }

    /// Empilha arrays de mesma forma num eixo novo, como `np.stack`.
    pub fn stack(arrays: &[&NdArray], axis: usize) -> Result<NdArray, ArrayError> {
        let first = arrays
            .first()
            .ok_or_else(|| ArrayError::Index("nada para empilhar".into()))?;
        if axis > first.ndim() {
            return Err(ArrayError::Axis {
                axis,
                ndim: first.ndim() + 1,
            });
        }
        let mut shape = first.shape.clone();
        shape.insert(axis, 1);
        let expanded = arrays
            .iter()
            .map(|a| {
                if a.shape != first.shape {
                    return Err(ArrayError::Broadcast {
                        left: first.shape.clone(),
                        right: a.shape.clone(),
                    });
                }
                a.reshape(&shape)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::concatenate(&expanded.iter().collect::<Vec<_>>(), axis)
    }

    /// Bytes de um `.npy` versão 1.0, em ordem C, com o mesmo cabeçalho que o NumPy escreve. bf16 não
    /// existe no formato e é recusado.
    pub fn to_npy_bytes(&self) -> Result<Vec<u8>, ArrayError> {
        let descr = self
            .dtype()
            .npy_descr()
            .ok_or_else(|| ArrayError::Npy("bf16 não existe no formato .npy".into()))?;
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
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        let total = 10 + header.len() + 1;
        header.push_str(&" ".repeat((64 - total % 64) % 64));
        header.push('\n');
        let mut out = Vec::with_capacity(10 + header.len() + self.len() * self.dtype().size());
        out.extend_from_slice(b"\x93NUMPY\x01\x00");
        out.extend_from_slice(&(header.len() as u16).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        match &self.data {
            ArrayData::Bool(b) => out.extend(self.gather(b).into_iter().map(u8::from)),
            ArrayData::U8(b) => out.extend(self.gather(b)),
            ArrayData::I32(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend(v.to_le_bytes())),
            ArrayData::I64(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend(v.to_le_bytes())),
            ArrayData::F16(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend(v.0.to_le_bytes())),
            ArrayData::Bf16(_) => unreachable!("recusado acima"),
            ArrayData::F32(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend(v.to_le_bytes())),
            ArrayData::F64(b) => self
                .gather(b)
                .iter()
                .for_each(|v| out.extend(v.to_le_bytes())),
        }
        Ok(out)
    }

    /// Lê um `.npy` (versões 1, 2 e 3) dos tipos suportados, little-endian. Ordem Fortran vira strides,
    /// sem cópia.
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
            "|u1" | "<u1" => DType::U8,
            "<i4" => DType::I32,
            "<i8" => DType::I64,
            "<f2" => DType::F16,
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
            DType::U8 => ArrayData::U8(body.into()),
            DType::I32 => ArrayData::I32(
                body.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes(*c))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            DType::I64 => ArrayData::I64(
                body.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            DType::F16 => ArrayData::F16(
                body.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| F16(u16::from_le_bytes(*c)))
                    .collect::<Vec<_>>()
                    .into(),
            ),
            DType::Bf16 => unreachable!("não vem de .npy"),
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
        std::fs::write(path.as_ref(), self.to_npy_bytes()?)
            .map_err(|e| ArrayError::Io(e.to_string()))
    }
}

fn normalize(k: isize, len: isize, axis: usize) -> Result<isize, ArrayError> {
    let k = if k < 0 { k + len } else { k };
    if (0..len).contains(&k) {
        Ok(k)
    } else {
        Err(ArrayError::Index(format!(
            "{k} fora do eixo {axis} de tamanho {len}"
        )))
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
    let fix =
        |v: isize, low: isize, high: isize| (if v < 0 { v + len } else { v }).clamp(low, high);
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
fn fold<T, R>(values: &[T], lane: usize, shape: &[usize], f: impl Fn(&[T]) -> R) -> Vec<R> {
    if lane == 0 {
        return (0..shape.iter().product::<usize>())
            .map(|_| f(&[]))
            .collect();
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

fn variance(v: &[f64], ddof: usize) -> f64 {
    let mean = pairwise(v) / v.len() as f64;
    let squares: Vec<f64> = v.iter().map(|x| (x - mean) * (x - mean)).collect();
    pairwise(&squares) / v.len().saturating_sub(ddof) as f64
}

/// Posição do máximo (ou mínimo): o primeiro em empate, e o primeiro NaN se houver, como o NumPy.
fn best_index<T: Copy + PartialOrd>(v: &[T], max: bool) -> usize {
    let nan = |x: T| x.partial_cmp(&x).is_none();
    if let Some(i) = v.iter().position(|&x| nan(x)) {
        return i;
    }
    let mut best = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if (max && x > v[best]) || (!max && x < v[best]) {
            best = i;
        }
    }
    best
}
