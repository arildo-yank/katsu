use std::{
    cell::{Ref, RefCell},
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
                write!(
                    f,
                    "elementwise operations require identical shapes (no broadcasting)"
                )
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
    parents: Vec<Tensor>,
    op: Op,
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

    fn elementwise(&self, rhs: &Self, op: Op) -> Result<Self, TensorError> {
        if self.shape() != rhs.shape() {
            return Err(TensorError::ShapeMismatch);
        }
        let mut data = zeros_buffer(self.numel())?;
        for ((out, a), b) in data.iter_mut().zip(self.data()).zip(rhs.data()) {
            *out = if op == Op::Add { a + b } else { a * b };
        }
        Ok(Self::node(
            data,
            copy_shape(self.shape())?,
            op,
            vec![self.clone(), rhs.clone()],
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
    /// são propagados novamente. Só o resultado final é acumulado em grad.
    /// Gradientes são buffers numéricos; derivadas de ordem superior não são suportadas.
    pub fn backward_with(&self, seed: &[f32]) -> Result<(), TensorError> {
        if seed.len() != self.numel() {
            return Err(TensorError::DataLength {
                expected: self.numel(),
                actual: seed.len(),
            });
        }
        let order = self.topological_order();
        let mut pending = HashMap::new();
        for node in &order {
            pending.insert(node.id(), zeros_buffer(node.numel())?);
        }
        pending
            .get_mut(&self.id())
            .expect("root is in topology")
            .copy_from_slice(seed);
        for node in order.iter().rev() {
            let upstream = pending.remove(&node.id()).expect("node is in topology");
            node.op().backward(node, &upstream, &mut pending)?;
            pending.insert(node.id(), upstream);
        }
        // Pré-valida todos os borrows e aloca antes de modificar qualquer grad.
        let mut guards = Vec::new();
        for node in &order {
            guards.push(
                node.0
                    .grad
                    .try_borrow_mut()
                    .map_err(|_| TensorError::GradientBorrowed)?,
            );
        }
        for (node, mut stored) in order.iter().zip(guards) {
            let gradient = pending.remove(&node.id()).expect("gradient was computed");
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
                    let target = pending
                        .get_mut(&parent.id())
                        .expect("parent is in topology");
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
                    let target = pending
                        .get_mut(&parent.id())
                        .expect("parent is in topology");
                    for (index, (out, grad)) in target.iter_mut().zip(g).enumerate() {
                        // Add: d(a+b)/da = 1. Mul: d(a*b)/da = b.
                        *out += if self == Self::Add {
                            *grad
                        } else {
                            grad * other.data()[index]
                        };
                    }
                }
            }
            Self::Sum => {
                let target = pending
                    .get_mut(&node.parents()[0].id())
                    .expect("parent is in topology");
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
                    let da = pending.get_mut(&a.id()).expect("parent is in topology");
                    for i in 0..m {
                        for p in 0..k {
                            for j in 0..n {
                                da[i * k + p] += g[i * n + j] * b.data()[p * n + j];
                            }
                        }
                    }
                }
                {
                    let db = pending.get_mut(&b.id()).expect("parent is in topology");
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
