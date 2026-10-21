use std::{
    cell::{Cell, Ref, RefCell},
    collections::{HashMap, HashSet, TryReserveError},
    error::Error,
    fmt,
    rc::Rc,
};

#[derive(Debug)]
pub enum TensorError {
    ShapeOverflow,
    InvalidConv(&'static str),
    MissingForward,
    NonScalarBackward,
    GradientBorrowed,
    DataLength {
        expected: usize,
        actual: usize,
    },
    ShapeMismatch,
    RankMismatch {
        left: usize,
        right: usize,
    },
    MatmulMismatch {
        left_columns: usize,
        right_rows: usize,
    },
    Allocation(TryReserveError),
    InvalidLearningRate,
    MissingGradient {
        index: usize,
    },
    NonFiniteLoss {
        epoch: usize,
    },
    InvalidShape(&'static str),
    IndexOutOfRange {
        index: usize,
        classes: usize,
    },
    OptimizerState,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConv(message) => write!(f, "invalid Conv2D: {message}"),
            Self::MissingForward => write!(f, "Conv2D backward requires a successful forward"),
            Self::NonScalarBackward => write!(
                f,
                "backward requires one element; use sum() or backward_with(seed)"
            ),
            Self::GradientBorrowed => {
                write!(f, "release gradient borrows before backward or zero_grad")
            }
            Self::ShapeOverflow => write!(f, "shape exceeds addressable tensor storage"),
            Self::DataLength { expected, actual } => {
                write!(f, "expected {expected} elements, got {actual}")
            }
            Self::ShapeMismatch => {
                write!(f, "elementwise shapes are not broadcast-compatible")
            }
            Self::RankMismatch { left, right } => {
                write!(f, "matmul requires rank 2, got {left} and {right}")
            }
            Self::MatmulMismatch {
                left_columns,
                right_rows,
            } => write!(
                f,
                "matmul inner dimensions differ: {left_columns} and {right_rows}"
            ),
            Self::Allocation(error) => write!(f, "tensor allocation failed: {error}"),
            Self::InvalidLearningRate => {
                write!(f, "learning rate must be finite and positive")
            }
            Self::MissingGradient { index } => {
                write!(f, "parameter {index} has no gradient; run backward first")
            }
            Self::NonFiniteLoss { epoch } => {
                write!(f, "loss is NaN or infinite at epoch {epoch}")
            }
            Self::InvalidShape(message) => write!(f, "invalid shape: {message}"),
            Self::IndexOutOfRange { index, classes } => {
                write!(f, "class index {index} out of range for {classes} classes")
            }
            Self::OptimizerState => {
                write!(f, "parameters do not match the optimizer state")
            }
        }
    }
}

impl Error for TensorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TryReserveError> for TensorError {
    fn from(error: TryReserveError) -> Self {
        Self::Allocation(error)
    }
}

/// Operação registrada no grafo. As derivadas locais são produtos vetor-Jacobiano.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Leaf,
    Add,
    Mul,
    Matmul,
    Conv2D { stride: usize, padding: usize },
    Sum,
    Relu,
    Transpose,
    Reshape,
    Gelu,
    Softmax,
    LogSoftmax,
    LayerNorm,
}

/// Handle compartilhado: clone preserva a identidade do nó, sem copiar os dados.
/// Valores e arestas são imutáveis; apenas o gradiente usa mutabilidade interior.
/// Rc torna este motor deliberadamente single-threaded.
#[derive(Clone)]
pub struct Tensor(Rc<TensorNode>);

struct TensorNode {
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: RefCell<Option<Vec<f32>>>,
    retain_grad: Cell<bool>,
    parents: Vec<Tensor>,
    op: Op,
}

/// Desmonta o grafo iterativamente: o drop recursivo padrão estoura a pilha
/// em cadeias profundas (ex.: centenas de milhares de operações).
impl Drop for TensorNode {
    fn drop(&mut self) {
        let mut stack = std::mem::take(&mut self.parents);
        while let Some(parent) = stack.pop() {
            // Só desmonta nós sem outros donos; os demais continuam vivos.
            if let Ok(mut node) = Rc::try_unwrap(parent.0) {
                stack.append(&mut node.parents);
            }
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Não percorre recursivamente o grafo (pode ser grande ou compartilhado).
        f.debug_struct("Tensor")
            .field("data", &self.data())
            .field("shape", &self.shape())
            .field("op", &self.op())
            .finish()
    }
}

pub(crate) fn zeros_buffer(count: usize) -> Result<Vec<f32>, TensorError> {
    let mut result = Vec::new();
    result.try_reserve_exact(count)?;
    result.resize(count, 0.0);
    Ok(result)
}

pub(crate) fn element_count(shape: &[usize]) -> Result<usize, TensorError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    let count = shape.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or(TensorError::ShapeOverflow)
    })?;
    if count > (isize::MAX as usize) / std::mem::size_of::<f32>() {
        return Err(TensorError::ShapeOverflow);
    }
    Ok(count)
}

fn copy_shape(shape: &[usize]) -> Result<Vec<usize>, TensorError> {
    let mut output = Vec::new();
    output.try_reserve_exact(shape.len())?;
    output.extend_from_slice(shape);
    Ok(output)
}

impl Tensor {
    /// Assume ownership dos dados existentes sem copiá-los.
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Result<Self, TensorError> {
        let expected = element_count(shape)?;
        if data.len() != expected {
            return Err(TensorError::DataLength {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self::node(data, copy_shape(shape)?, Op::Leaf, Vec::new()))
    }

    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        let count = element_count(shape)?;
        let shape = copy_shape(shape)?;
        let mut data = Vec::new();
        data.try_reserve_exact(count)?;
        data.resize(count, 0.0);
        Ok(Self::node(data, shape, Op::Leaf, Vec::new()))
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.0.data
    }
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }
    pub fn numel(&self) -> usize {
        self.0.data.len()
    }

    pub(crate) fn node(data: Vec<f32>, shape: Vec<usize>, op: Op, parents: Vec<Self>) -> Self {
        Self(Rc::new(TensorNode {
            data,
            shape,
            grad: RefCell::new(None),
            retain_grad: Cell::new(false),
            parents,
            op,
        }))
    }

    pub fn op(&self) -> Op {
        self.0.op
    }
    pub fn parents(&self) -> &[Self] {
        &self.0.parents
    }

    /// O guard deve ser liberado antes de backward/zero_grad.
    pub fn grad(&self) -> Result<Ref<'_, Option<Vec<f32>>>, TensorError> {
        self.0
            .grad
            .try_borrow()
            .map_err(|_| TensorError::GradientBorrowed)
    }

    /// Pede que backward publique o gradiente deste nó intermediário.
    /// Folhas sempre recebem gradiente; intermediários só com retain_grad.
    pub fn retain_grad(&self) {
        self.0.retain_grad.set(true);
    }

    /// Indica se backward publica o gradiente deste nó.
    pub fn retains_grad(&self) -> bool {
        self.0.op == Op::Leaf || self.0.retain_grad.get()
    }

    /// Limpa apenas este nó. Chamadas backward sucessivas acumulam gradientes.
    pub fn zero_grad(&self) -> Result<(), TensorError> {
        *self
            .0
            .grad
            .try_borrow_mut()
            .map_err(|_| TensorError::GradientBorrowed)? = None;
        Ok(())
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.elementwise(rhs, Op::Add)
    }

    /// Multiplicação elemento a elemento (não é matmul).
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.elementwise(rhs, Op::Mul)
    }

    /// Shapes compatíveis por broadcasting (regras do NumPy).
    fn elementwise(&self, rhs: &Self, op: Op) -> Result<Self, TensorError> {
        let shape = crate::ops::broadcast_shape(self.shape(), rhs.shape())?;
        let left = crate::ops::broadcast_map(&shape, self.shape())?;
        let right = crate::ops::broadcast_map(&shape, rhs.shape())?;
        let mut data = zeros_buffer(element_count(&shape)?)?;
        for (i, out) in data.iter_mut().enumerate() {
            let a = self.data()[crate::ops::at(&left, i)];
            let b = rhs.data()[crate::ops::at(&right, i)];
            *out = if op == Op::Add { a + b } else { a * b };
        }
        Ok(Self::node(data, shape, op, vec![self.clone(), rhs.clone()]))
    }

    /// max(0, x) elemento a elemento. A derivada em x = 0 é definida como 0.
    pub fn relu(&self) -> Result<Self, TensorError> {
        let mut data = zeros_buffer(self.numel())?;
        for (out, x) in data.iter_mut().zip(self.data()) {
            *out = x.max(0.0);
        }
        Ok(Self::node(
            data,
            copy_shape(self.shape())?,
            Op::Relu,
            vec![self.clone()],
        ))
    }

    /// Redução para um escalar, incluindo tensores vazios (soma zero).
    pub fn sum(&self) -> Result<Self, TensorError> {
        let mut data = zeros_buffer(1)?;
        data[0] = self.data().iter().sum();
        Ok(Self::node(data, Vec::new(), Op::Sum, vec![self.clone()]))
    }

    /// `[m,k] @ [k,n] -> [m,n]`, com arestas para ambos os operandos.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.rank() != 2 || rhs.rank() != 2 {
            return Err(TensorError::RankMismatch {
                left: self.rank(),
                right: rhs.rank(),
            });
        }
        let (m, k, n) = (self.shape()[0], self.shape()[1], rhs.shape()[1]);
        if k != rhs.shape()[0] {
            return Err(TensorError::MatmulMismatch {
                left_columns: k,
                right_rows: rhs.shape()[0],
            });
        }
        let mut data = zeros_buffer(element_count(&[m, n])?)?;
        if !data.is_empty() && k != 0 {
            for i in 0..m {
                for p in 0..k {
                    for j in 0..n {
                        data[i * n + j] += self.data()[i * k + p] * rhs.data()[p * n + j];
                    }
                }
            }
        }
        Ok(Self::node(
            data,
            copy_shape(&[m, n])?,
            Op::Matmul,
            vec![self.clone(), rhs.clone()],
        ))
    }

    fn id(&self) -> *const TensorNode {
        Rc::as_ptr(&self.0)
    }

    /// DFS iterativa: pais antes dos filhos, deduplicando por identidade.
    /// A API só cria arestas para nós existentes e nunca permite ciclos.
    fn topological_order(&self) -> Vec<Self> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![(self.clone(), false)];
        while let Some((node, expanded)) = stack.pop() {
            if expanded {
                order.push(node);
                continue;
            }
            if !visited.insert(node.id()) {
                continue;
            }
            stack.push((node.clone(), true));
            for parent in node.parents().iter().rev() {
                stack.push((parent.clone(), false));
            }
        }
        order
    }

    /// Inicializa dL/dL = 1. Para saídas vetoriais, use sum ou backward_with.
    pub fn backward(&self) -> Result<(), TensorError> {
        if self.numel() != 1 {
            return Err(TensorError::NonScalarBackward);
        }
        self.backward_with(&[1.0])
    }

    /// Calcula o VJP com seed no mesmo layout achatado da saída.
    /// Cada chamada usa adjuntos temporários novos: gradientes antigos nunca
    /// são propagados novamente. Só o resultado final é acumulado em grad,
    /// e apenas em folhas e nós marcados com retain_grad.
    /// Gradientes são buffers numéricos; derivadas de ordem superior não são suportadas.
    pub fn backward_with(&self, seed: &[f32]) -> Result<(), TensorError> {
        if seed.len() != self.numel() {
            return Err(TensorError::DataLength {
                expected: self.numel(),
                actual: seed.len(),
            });
        }
        let order = self.topological_order();
        // Adjuntos alocados sob demanda: só existem entre o primeiro filho
        // processado e o consumo do próprio nó, reduzindo o pico de memória.
        let mut pending = HashMap::new();
        adjoint(&mut pending, self)?.copy_from_slice(seed);
        let mut retained = Vec::new();
        for node in order.iter().rev() {
            let upstream = match pending.remove(&node.id()) {
                Some(upstream) => upstream,
                None => zeros_buffer(node.numel())?,
            };
            node.op().backward(node, &upstream, &mut pending)?;
            // Adjuntos de intermediários não retidos são liberados assim que consumidos.
            if node.retains_grad() {
                retained.push((node, upstream));
            }
        }
        // Pré-valida todos os borrows antes de modificar qualquer grad.
        let mut guards = Vec::new();
        for (node, _) in &retained {
            guards.push(
                node.0
                    .grad
                    .try_borrow_mut()
                    .map_err(|_| TensorError::GradientBorrowed)?,
            );
        }
        for ((_, gradient), mut stored) in retained.into_iter().zip(guards) {
            if let Some(previous) = stored.as_mut() {
                for (old, new) in previous.iter_mut().zip(gradient) {
                    *old += new;
                }
            } else {
                *stored = Some(gradient);
            }
        }
        Ok(())
    }
}

/// Buffer de adjunto do nó, criado com zeros na primeira contribuição.
fn adjoint<'a>(
    pending: &'a mut HashMap<*const TensorNode, Vec<f32>>,
    node: &Tensor,
) -> Result<&'a mut Vec<f32>, TensorError> {
    use std::collections::hash_map::Entry;
    Ok(match pending.entry(node.id()) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => entry.insert(zeros_buffer(node.numel())?),
    })
}

impl Op {
    fn backward(
        self,
        node: &Tensor,
        g: &[f32],
        pending: &mut HashMap<*const TensorNode, Vec<f32>>,
    ) -> Result<(), TensorError> {
        match self {
            Self::Leaf => {}
            Self::Conv2D { stride, padding } => {
                let (dx, dw) = crate::conv2d::backward_buffers(
                    &node.parents()[0],
                    &node.parents()[1],
                    g,
                    stride,
                    padding,
                )?;
                // Empréstimos sequenciais também suportam pais com identidade igual.
                for (parent, gradient) in node.parents().iter().zip([dx, dw]) {
                    let target = adjoint(pending, parent)?;
                    for (out, value) in target.iter_mut().zip(gradient) {
                        *out += value;
                    }
                }
            }
            Self::Add | Self::Mul => {
                // Visita as duas arestas mesmo quando apontam para o mesmo nó.
                for side in 0..2 {
                    let parent = &node.parents()[side];
                    let other = &node.parents()[1 - side];
                    // Dimensões expandidas somam as contribuições no mesmo índice.
                    let to_parent = crate::ops::broadcast_map(node.shape(), parent.shape())?;
                    let to_other = crate::ops::broadcast_map(node.shape(), other.shape())?;
                    let target = adjoint(pending, parent)?;
                    for (index, grad) in g.iter().enumerate() {
                        // Add: d(a+b)/da = 1. Mul: d(a*b)/da = b.
                        target[crate::ops::at(&to_parent, index)] += if self == Self::Add {
                            *grad
                        } else {
                            grad * other.data()[crate::ops::at(&to_other, index)]
                        };
                    }
                }
            }
            Self::Transpose
            | Self::Reshape
            | Self::Gelu
            | Self::Softmax
            | Self::LogSoftmax
            | Self::LayerNorm => {
                let parent = &node.parents()[0];
                let target = adjoint(pending, parent)?;
                crate::ops::unary_backward(self, parent, node, g, target);
            }
            Self::Relu => {
                let parent = &node.parents()[0];
                let target = adjoint(pending, parent)?;
                for ((out, grad), x) in target.iter_mut().zip(g).zip(parent.data()) {
                    if *x > 0.0 {
                        *out += grad;
                    }
                }
            }
            Self::Sum => {
                let target = adjoint(pending, &node.parents()[0])?;
                for out in target {
                    *out += g[0];
                }
            }
            Self::Matmul => {
                let (a, b) = (&node.parents()[0], &node.parents()[1]);
                let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
                if g.is_empty() || k == 0 {
                    return Ok(());
                }
                // dA = G @ B^T; dB = A^T @ G, sem materializar transpostas.
                {
                    let da = adjoint(pending, a)?;
                    for i in 0..m {
                        for p in 0..k {
                            for j in 0..n {
                                da[i * k + p] += g[i * n + j] * b.data()[p * n + j];
                            }
                        }
                    }
                }
                {
                    let db = adjoint(pending, b)?;
                    for i in 0..m {
                        for p in 0..k {
                            for j in 0..n {
                                db[p * n + j] += a.data()[i * k + p] * g[i * n + j];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
