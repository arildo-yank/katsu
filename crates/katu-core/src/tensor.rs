use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet, TryReserveError},
    error::Error,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, RwLock, RwLockReadGuard,
    },
};

#[derive(Debug)]
pub enum TensorError {
    BackendUnavailable(&'static str),
    LockPoisoned,
    BackendFailure(String),
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
            Self::BackendUnavailable(message) => write!(f, "backend unavailable: {message}"),
            Self::BackendFailure(message) => write!(f, "backend error: {message}"),
            Self::LockPoisoned => write!(f, "gradient lock poisoned"),
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
    BatchedMatmul,
    MatmulMany,
    Columns,
    GatherRows,
    CrossEntropy,
    Conv2D { stride: usize, padding: usize },
    Concat { axis: usize },
    CausalAttention,
    Linear,
    LayerNormAffine,
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
/// Arc + RwLock permitem compartilhar o grafo entre threads. Publicação é serializada.
#[derive(Clone)]
pub struct Tensor(Arc<TensorNode>);

struct TensorNode {
    // Arc: reshape compartilha os dados do pai em vez de copiá-los. Nunca muda depois de criado.
    data: Arc<Vec<f32>>,
    shape: Vec<usize>,
    grad: RwLock<Option<Vec<f32>>>,
    retain_grad: AtomicBool,
    parents: Vec<Tensor>,
    op: Op,
    backend: Option<Arc<dyn crate::Backend>>,
    indices: Vec<usize>,
    // Folha cujos números vivem num ParamStore (data fica vazio): só serve de lado direito de
    // matmul/linear, e o gradiente é acumulado no próprio store.
    resident: Option<Residency>,
}

#[derive(Clone)]
struct Residency {
    store: Arc<dyn crate::ParamStore>,
    view: crate::View,
}

/// Desmonta o grafo iterativamente: o drop recursivo padrão estoura a pilha
/// em cadeias profundas (ex.: centenas de milhares de operações).
impl Drop for TensorNode {
    fn drop(&mut self) {
        let mut stack = std::mem::take(&mut self.parents);
        while let Some(parent) = stack.pop() {
            // Só desmonta nós sem outros donos; os demais continuam vivos.
            if let Ok(mut node) = Arc::try_unwrap(parent.0) {
                stack.append(&mut node.parents);
            }
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Não percorre recursivamente o grafo (pode ser grande ou compartilhado).
        if let Some(resident) = &self.0.resident {
            return f
                .debug_struct("Tensor")
                .field("resident", &resident.store.name())
                .field("view", &resident.view)
                .field("shape", &self.shape())
                .finish();
        }
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
    /// Dados no host. Um tensor residente não tem cópia no host: use `materialize`.
    pub fn data(&self) -> &[f32] {
        assert!(
            self.0.resident.is_none(),
            "tensor residente: os números estão no ParamStore; use katu_core::materialize"
        );
        &self.0.data
    }
    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }
    pub fn numel(&self) -> usize {
        match &self.0.resident {
            Some(resident) => resident.view.len,
            None => self.0.data.len(),
        }
    }

    /// Folha residente: `view` do store com o `shape` dado. Sem cópia no host.
    pub fn resident(
        store: Arc<dyn crate::ParamStore>,
        view: crate::View,
        shape: &[usize],
    ) -> Result<Self, TensorError> {
        crate::resident::check_view(store.capacity(), view)?;
        let expected = element_count(shape)?;
        if view.len != expected {
            return Err(TensorError::DataLength {
                expected,
                actual: view.len,
            });
        }
        let mut tensor = Self::node(Vec::new(), copy_shape(shape)?, Op::Leaf, Vec::new());
        Arc::get_mut(&mut tensor.0)
            .expect("new resident node")
            .resident = Some(Residency { store, view });
        Ok(tensor)
    }

    /// Store e faixa de um tensor residente.
    pub fn residency(&self) -> Option<(&Arc<dyn crate::ParamStore>, crate::View)> {
        self.0.resident.as_ref().map(|r| (&r.store, r.view))
    }

    pub fn is_resident(&self) -> bool {
        self.0.resident.is_some()
    }

    pub(crate) fn node(data: Vec<f32>, shape: Vec<usize>, op: Op, parents: Vec<Self>) -> Self {
        Self::shared_node(Arc::new(data), shape, op, parents)
    }

    /// Nó que reaproveita um buffer existente (ex.: reshape), sem copiar os dados.
    pub(crate) fn shared_node(
        data: Arc<Vec<f32>>,
        shape: Vec<usize>,
        op: Op,
        parents: Vec<Self>,
    ) -> Self {
        Self(Arc::new(TensorNode {
            data,
            shape,
            grad: RwLock::new(None),
            retain_grad: AtomicBool::new(false),
            parents,
            op,
            backend: None,
            indices: Vec::new(),
            resident: None,
        }))
    }

    pub(crate) fn with_indices(mut self, indices: &[usize]) -> Self {
        Arc::get_mut(&mut self.0).expect("new indexed node").indices = indices.to_vec();
        self
    }

    pub(crate) fn with_backend(mut self, backend: Arc<dyn crate::Backend>) -> Self {
        Arc::get_mut(&mut self.0).expect("new backend node").backend = Some(backend);
        self
    }

    pub(crate) fn storage(&self) -> &Arc<Vec<f32>> {
        &self.0.data
    }

    /// Este handle é o único dono do nó e do buffer de dados.
    pub(crate) fn is_unique(&self) -> bool {
        Arc::strong_count(&self.0) == 1 && Arc::strong_count(&self.0.data) == 1
    }

    /// Dados, shape e gradiente por valor, quando este handle é o único dono do nó e do
    /// buffer (ninguém mais pode observá-los). Senão devolve o próprio tensor intacto.
    pub(crate) fn try_into_parts(self) -> Result<OwnedParts, Self> {
        if !self.is_unique() {
            return Err(self);
        }
        let mut node = Arc::try_unwrap(self.0).map_err(Self)?;
        let data = std::mem::take(&mut node.data);
        let data = Arc::try_unwrap(data).unwrap_or_else(|shared| shared.as_ref().clone());
        let grad = node.grad.get_mut().map(Option::take).unwrap_or_default();
        Ok(OwnedParts {
            data,
            shape: std::mem::take(&mut node.shape),
            grad,
        })
    }

    pub fn op(&self) -> Op {
        self.0.op
    }
    pub fn parents(&self) -> &[Self] {
        &self.0.parents
    }

    /// O guard deve ser liberado antes de backward/zero_grad.
    pub fn grad(&self) -> Result<RwLockReadGuard<'_, Option<Vec<f32>>>, TensorError> {
        self.0
            .grad
            .try_read()
            .map_err(|_| TensorError::GradientBorrowed)
    }

    /// Pede que backward publique o gradiente deste nó intermediário.
    /// Folhas sempre recebem gradiente; intermediários só com retain_grad.
    pub fn retain_grad(&self) {
        self.0.retain_grad.store(true, Ordering::Relaxed);
    }

    /// Indica se backward publica o gradiente deste nó.
    pub fn retains_grad(&self) -> bool {
        self.0.op == Op::Leaf || self.0.retain_grad.load(Ordering::Relaxed)
    }

    /// Limpa apenas este nó. Chamadas backward sucessivas acumulam gradientes.
    pub fn zero_grad(&self) -> Result<(), TensorError> {
        *self
            .0
            .grad
            .try_write()
            .map_err(|_| TensorError::GradientBorrowed)? = None;
        Ok(())
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.add_with_backend(rhs, Arc::new(crate::CpuBackend::default()))
    }

    pub fn add_with_backend(
        &self,
        rhs: &Self,
        backend: Arc<dyn crate::Backend>,
    ) -> Result<Self, TensorError> {
        let shape = crate::ops::broadcast_shape(self.shape(), rhs.shape())?;
        let left = crate::ops::broadcast_map(&shape, self.shape())?;
        let right = crate::ops::broadcast_map(&shape, rhs.shape())?;
        let count = element_count(&shape)?;
        let a: Vec<f32> = (0..count)
            .map(|i| self.data()[crate::ops::at(&left, i)])
            .collect();
        let b: Vec<f32> = (0..count)
            .map(|i| rhs.data()[crate::ops::at(&right, i)])
            .collect();
        let data = backend.add(&a, &b)?;
        if data.len() != count {
            return Err(TensorError::InvalidShape("backend returned wrong add size"));
        }
        Ok(Self::node(data, shape, Op::Add, vec![self.clone(), rhs.clone()]).with_backend(backend))
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

    /// `[m,k] @ [k,n] -> [m,n]`, no backend CPU padrão.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.matmul_with_backend(rhs, Arc::new(crate::CpuBackend::default()))
    }

    pub fn matmul_with_backend(
        &self,
        rhs: &Self,
        backend: Arc<dyn crate::Backend>,
    ) -> Result<Self, TensorError> {
        self.matrix_product(rhs, backend, false)
    }

    /// `[batch,m,k] @ [batch,k,n]`; sem broadcasting entre batches.
    pub fn bmm(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.bmm_with_backend(rhs, Arc::new(crate::CpuBackend::default()))
    }

    pub fn bmm_with_backend(
        &self,
        rhs: &Self,
        backend: Arc<dyn crate::Backend>,
    ) -> Result<Self, TensorError> {
        self.matrix_product(rhs, backend, true)
    }

    fn matrix_product(
        &self,
        rhs: &Self,
        backend: Arc<dyn crate::Backend>,
        batched: bool,
    ) -> Result<Self, TensorError> {
        let rank = if batched { 3 } else { 2 };
        if self.is_resident() || (batched && rhs.is_resident()) {
            return Err(TensorError::InvalidShape(
                "resident tensors only as the rhs of a 2D matmul",
            ));
        }
        if self.rank() != rank || rhs.rank() != rank {
            return Err(if batched {
                TensorError::InvalidShape("bmm needs rank 3")
            } else {
                TensorError::RankMismatch {
                    left: self.rank(),
                    right: rhs.rank(),
                }
            });
        }
        let offset = rank - 2;
        let (m, k, n) = (
            self.shape()[offset],
            self.shape()[offset + 1],
            rhs.shape()[offset + 1],
        );
        if k != rhs.shape()[offset] {
            return Err(TensorError::MatmulMismatch {
                left_columns: k,
                right_rows: rhs.shape()[offset],
            });
        }
        let batch = if batched { self.shape()[0] } else { 1 };
        if batched && batch != rhs.shape()[0] {
            return Err(TensorError::InvalidShape("bmm batch sizes differ"));
        }
        let shape = if batched {
            vec![batch, m, n]
        } else {
            vec![m, n]
        };
        let mut data = zeros_buffer(element_count(&shape)?)?;
        if let Some((store, view)) = rhs.residency() {
            if m != 0 && n != 0 && k != 0 {
                data = store.matmul(self.data(), view, m, k, n)?;
                if data.len() != m * n {
                    return Err(TensorError::InvalidShape(
                        "store returned wrong output size",
                    ));
                }
            }
        } else if m != 0 && n != 0 && k != 0 {
            for i in 0..batch {
                let result = backend.matmul(
                    &self.data()[i * m * k..(i + 1) * m * k],
                    &rhs.data()[i * k * n..(i + 1) * k * n],
                    m,
                    k,
                    n,
                )?;
                if result.len() != m * n {
                    return Err(TensorError::InvalidShape(
                        "backend returned wrong output size",
                    ));
                }
                data[i * m * n..(i + 1) * m * n].copy_from_slice(&result);
            }
        } else if batch != 0 && !batched {
            // Mesmo um produto vazio deve verificar disponibilidade do backend.
            backend.matmul(self.data(), rhs.data(), m, k, n)?;
        }
        let mut out = Self::node(
            data,
            shape,
            if batched {
                Op::BatchedMatmul
            } else {
                Op::Matmul
            },
            vec![self.clone(), rhs.clone()],
        );
        Arc::get_mut(&mut out.0).expect("new node").backend = Some(backend);
        Ok(out)
    }

    /// Lookup de linhas com scatter-add no backward; evita one_hot × tabela.
    pub fn gather_rows(&self, indices: &[usize]) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::InvalidShape("gather_rows needs a matrix"));
        }
        let (rows, width) = (self.shape()[0], self.shape()[1]);
        let mut data = zeros_buffer(element_count(&[indices.len(), width])?)?;
        for (i, &index) in indices.iter().enumerate() {
            if index >= rows {
                return Err(TensorError::IndexOutOfRange {
                    index,
                    classes: rows,
                });
            }
            data[i * width..(i + 1) * width]
                .copy_from_slice(&self.data()[index * width..(index + 1) * width]);
        }
        let mut out = Self::node(
            data,
            vec![indices.len(), width],
            Op::GatherRows,
            vec![self.clone()],
        );
        Arc::get_mut(&mut out.0).expect("new node").indices = indices.to_vec();
        Ok(out)
    }

    fn id(&self) -> *const TensorNode {
        Arc::as_ptr(&self.0)
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
        Self::propagate(order, self.id(), seed)
    }

    /// Como `backward`, mas consome o grafo: cada nó intermediário sem outro dono é
    /// liberado assim que seu gradiente chega aos pais. Os dados de um nó só servem ao
    /// backward dele mesmo e dos filhos, que vêm antes na ordem reversa. Use quando a
    /// perda não for mais necessária (leia o valor antes); gradientes ficam nas folhas.
    pub fn backward_release(self) -> Result<(), TensorError> {
        if self.numel() != 1 {
            return Err(TensorError::NonScalarBackward);
        }
        let order = self.topological_order();
        let root = self.id();
        drop(self);
        Self::propagate(order, root, &[1.0])
    }

    fn propagate(
        mut order: Vec<Self>,
        root: *const TensorNode,
        seed: &[f32],
    ) -> Result<(), TensorError> {
        // Adjuntos alocados sob demanda: só existem entre o primeiro filho
        // processado e o consumo do próprio nó, reduzindo o pico de memória.
        let mut pending = HashMap::new();
        pending.insert(root, seed.to_vec());
        let (mut retained, mut gradients) = (Vec::new(), Vec::new());
        // Ordem reversa, soltando cada handle depois do uso: sem outros donos, o nó
        // (e seus dados) é liberado na hora.
        while let Some(node) = order.pop() {
            if node.is_resident() {
                // O gradiente já foi somado no store pelo backward do matmul.
                pending.remove(&node.id());
                continue;
            }
            let upstream = match pending.remove(&node.id()) {
                Some(upstream) => upstream,
                None => zeros_buffer(node.numel())?,
            };
            node.op().backward(&node, &upstream, &mut pending)?;
            // Adjuntos de intermediários não retidos são liberados assim que consumidos.
            if node.retains_grad() {
                retained.push(node);
                gradients.push(upstream);
            }
        }
        // Serializa apenas a publicação; VJPs e forwards podem executar em paralelo.
        static PUBLISH: Mutex<()> = Mutex::new(());
        let _publication = PUBLISH.lock().map_err(|_| TensorError::LockPoisoned)?;
        // Pré-valida todos os locks antes de modificar qualquer grad.
        let mut guards = Vec::new();
        for node in &retained {
            guards.push(
                node.0
                    .grad
                    .try_write()
                    .map_err(|_| TensorError::GradientBorrowed)?,
            );
        }
        for (gradient, mut stored) in gradients.into_iter().zip(guards) {
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

/// Soma `source` em `target`, elemento a elemento; em paralelo quando vale a pena.
/// Cada elemento recebe exatamente a mesma soma do laço sequencial.
fn accumulate(target: &mut [f32], source: &[f32]) {
    const PARALLEL: usize = 1 << 16;
    if target.len() >= PARALLEL {
        target
            .par_chunks_mut(PARALLEL / 4)
            .zip(source.par_chunks(PARALLEL / 4))
            .for_each(|(out, value)| {
                for (out, value) in out.iter_mut().zip(value) {
                    *out += value;
                }
            });
    } else {
        for (out, value) in target.iter_mut().zip(source) {
            *out += value;
        }
    }
}

/// Entrega um gradiente completo ao pai. Na primeira contribuição o buffer vira o adjunto
/// (0 + x = x), sem zerar memória nem somar; nas seguintes, soma.
fn give(pending: &mut HashMap<*const TensorNode, Vec<f32>>, parent: &Tensor, gradient: Vec<f32>) {
    use std::collections::hash_map::Entry;
    match pending.entry(parent.id()) {
        Entry::Occupied(mut entry) => accumulate(entry.get_mut(), &gradient),
        Entry::Vacant(entry) => {
            entry.insert(gradient);
        }
    }
}

/// Conteúdo de um tensor que tinha um único dono (ver `Tensor::try_into_parts`).
pub(crate) struct OwnedParts {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
    pub(crate) grad: Option<Vec<f32>>,
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
                    give(pending, parent, gradient);
                }
            }
            Self::Add | Self::Mul => {
                if self == Self::Add {
                    let a = &node.parents()[0];
                    let b = &node.parents()[1];
                    let map_a = crate::ops::broadcast_map(node.shape(), a.shape())?;
                    let map_b = crate::ops::broadcast_map(node.shape(), b.shape())?;
                    let indices_a: Vec<usize> =
                        (0..g.len()).map(|i| crate::ops::at(&map_a, i)).collect();
                    let indices_b: Vec<usize> =
                        (0..g.len()).map(|i| crate::ops::at(&map_b, i)).collect();
                    let (da, db) = node.0.backend.as_ref().expect("add backend").add_backward(
                        g,
                        &indices_a,
                        &indices_b,
                        a.numel(),
                        b.numel(),
                    )?;
                    for (parent, gradient) in [(a, da), (b, db)] {
                        if gradient.len() != parent.numel() {
                            return Err(TensorError::InvalidShape(
                                "backend returned wrong add gradient size",
                            ));
                        }
                        give(pending, parent, gradient);
                    }
                    return Ok(());
                }
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
            Self::CausalAttention => {
                // Refaz kᵀ e reproduz os backward de softmax, + máscara, × escala e bmm na
                // mesma ordem do grafo sem fusão; só as probabilidades ficaram guardadas.
                let (q, k) = (&node.parents()[0], &node.parents()[1]);
                let (batch, t, dh) = (q.shape()[0], q.shape()[1], q.shape()[2]);
                if t == 0 || dh == 0 {
                    return Ok(());
                }
                let scale = 1.0 / (dh as f32).sqrt();
                let backend = node.0.backend.as_ref().expect("attention backend");
                for b in 0..batch {
                    let probs = &node.data()[b * t * t..(b + 1) * t * t];
                    let mut scores = zeros_buffer(t * t)?;
                    scores
                        .par_chunks_mut(t)
                        .zip(probs.par_chunks(t))
                        .zip(g[b * t * t..(b + 1) * t * t].par_chunks(t))
                        .for_each(|((out, y), gr)| {
                            let dot: f32 = gr.iter().zip(y).map(|(a, b)| a * b).sum();
                            for ((o, &gi), &yi) in out.iter_mut().zip(gr).zip(y) {
                                *o = yi * (gi - dot) * scale;
                            }
                        });
                    let span = b * t * dh..(b + 1) * t * dh;
                    let kt = crate::ops::transposed(&k.data()[span.clone()], t, dh)?;
                    let (dq, dkt) =
                        backend.matmul_backward(&q.data()[span.clone()], &kt, &scores, t, dh, t)?;
                    if dq.len() != t * dh || dkt.len() != dh * t {
                        return Err(TensorError::InvalidShape(
                            "backend returned wrong gradient size",
                        ));
                    }
                    accumulate(&mut adjoint(pending, q)?[span.clone()], &dq);
                    let target = &mut adjoint(pending, k)?[span];
                    for i in 0..t {
                        for j in 0..dh {
                            target[i * dh + j] += dkt[j * t + i];
                        }
                    }
                }
            }
            Self::Linear => {
                let (x, w, bias) = (&node.parents()[0], &node.parents()[1], &node.parents()[2]);
                let (m, k, n) = (x.shape()[0], x.shape()[1], w.shape()[1]);
                if m == 0 || k == 0 || n == 0 {
                    return Ok(());
                }
                // Bias: soma das linhas em buffer próprio, na ordem do add_backward.
                let mut db = zeros_buffer(n)?;
                for row in g.chunks(n) {
                    for (out, value) in db.iter_mut().zip(row) {
                        *out += value;
                    }
                }
                for (out, value) in adjoint(pending, bias)?.iter_mut().zip(db) {
                    *out += value;
                }
                if let Some((store, view)) = w.residency() {
                    let dx = store.matmul_backward(x.data(), view, g, m, k, n)?;
                    if dx.len() != m * k {
                        return Err(TensorError::InvalidShape(
                            "store returned wrong gradient size",
                        ));
                    }
                    give(pending, x, dx);
                    return Ok(());
                }
                let backend = node.0.backend.as_ref().expect("linear backend");
                let (dx, dw) = backend.matmul_backward(x.data(), w.data(), g, m, k, n)?;
                if dx.len() != m * k || dw.len() != k * n {
                    return Err(TensorError::InvalidShape(
                        "backend returned wrong gradient size",
                    ));
                }
                for (parent, gradient) in [(x, dx), (w, dw)] {
                    give(pending, parent, gradient);
                }
            }
            Self::LayerNormAffine => {
                let (x, gamma, beta) = (&node.parents()[0], &node.parents()[1], &node.parents()[2]);
                let width = *x.shape().last().expect("layer norm has rank >= 1");
                if width == 0 {
                    return Ok(());
                }
                let rows = x.numel() / width;
                // β: soma das linhas em buffer próprio, na ordem do add_backward.
                let mut db = zeros_buffer(width)?;
                for row in g.chunks(width) {
                    for (out, value) in db.iter_mut().zip(row) {
                        *out += value;
                    }
                }
                for (out, value) in adjoint(pending, beta)?.iter_mut().zip(db) {
                    *out += value;
                }
                // x̂ refeito de x; dγ += g·x̂ e dx̂ = g·γ, como no backward do mul.
                let mut normalized = zeros_buffer(x.numel())?;
                let mut stats = Vec::with_capacity(rows);
                for (r, out) in normalized.chunks_mut(width).enumerate() {
                    let row = &x.data()[r * width..(r + 1) * width];
                    let (mean, inv_std) = crate::ops::row_stats(row);
                    for (o, &value) in out.iter_mut().zip(row) {
                        *o = (value - mean) * inv_std;
                    }
                    stats.push(inv_std);
                }
                let mut dy = zeros_buffer(x.numel())?;
                for (i, (out, &grad)) in dy.iter_mut().zip(g).enumerate() {
                    *out += grad * gamma.data()[i % width];
                }
                let target = adjoint(pending, gamma)?;
                for (i, (&grad, &y)) in g.iter().zip(&normalized).enumerate() {
                    target[i % width] += grad * y;
                }
                // dx = inv_std · (dx̂ - média(dx̂) - x̂ · média(dx̂·x̂)), como no layer_norm.
                let target = adjoint(pending, x)?;
                let n = width as f32;
                for (r, inv_std) in stats.into_iter().enumerate() {
                    let span = r * width..(r + 1) * width;
                    let (y, gr, out) = (
                        &normalized[span.clone()],
                        &dy[span.clone()],
                        &mut target[span],
                    );
                    let mean_g = gr.iter().sum::<f32>() / n;
                    let mean_gy = gr.iter().zip(y).map(|(a, b)| a * b).sum::<f32>() / n;
                    for ((o, &gi), &yi) in out.iter_mut().zip(gr).zip(y) {
                        *o += inv_std * (gi - mean_g - yi * mean_gy);
                    }
                }
            }
            Self::MatmulMany => {
                let (x, weights) = (&node.parents()[0], &node.parents()[1..]);
                let (m, k, n) = (x.shape()[0], x.shape()[1], weights[0].shape()[1]);
                if m == 0 || k == 0 || n == 0 {
                    return Ok(());
                }
                if let Some((store, _)) = weights[0].residency() {
                    // O forward garante: todos residentes no mesmo store.
                    let views: Vec<crate::View> = weights
                        .iter()
                        .map(|w| w.residency().expect("resident weights").1)
                        .collect();
                    let dx = store.matmul_many_backward(x.data(), &views, g, m, k, n)?;
                    if dx.len() != m * k {
                        return Err(TensorError::InvalidShape(
                            "store returned wrong gradient size",
                        ));
                    }
                    give(pending, x, dx);
                    return Ok(());
                }
                let backend = node.0.backend.as_ref().expect("matmul_many backend");
                // dx = Σᵢ gᵢ·Wᵢᵀ, somado na ordem dos pesos (a mesma do store residente).
                let mut dx = zeros_buffer(m * k)?;
                for (i, w) in weights.iter().enumerate() {
                    let gi = crate::resident::columns_of(g, m, n, weights.len(), i);
                    let (dxi, dwi) = backend.matmul_backward(x.data(), w.data(), &gi, m, k, n)?;
                    if dxi.len() != m * k || dwi.len() != k * n {
                        return Err(TensorError::InvalidShape(
                            "backend returned wrong gradient size",
                        ));
                    }
                    give(pending, w, dwi);
                    for (out, value) in dx.iter_mut().zip(dxi) {
                        *out += value;
                    }
                }
                give(pending, x, dx);
            }
            Self::Columns => {
                let parent = &node.parents()[0];
                let (start, width) = (node.0.indices[0], node.0.indices[1]);
                let (rows, total) = (parent.shape()[0], parent.shape()[1]);
                let target = adjoint(pending, parent)?;
                for row in 0..rows {
                    let span = row * total + start..row * total + start + width;
                    for (out, value) in target[span]
                        .iter_mut()
                        .zip(&g[row * width..(row + 1) * width])
                    {
                        *out += value;
                    }
                }
            }
            Self::Concat { axis } => {
                // Cada pai recebe o seu pedaço do gradiente, na mesma posição do forward.
                let total = node.shape()[1];
                let mut offset = 0;
                for parent in node.parents() {
                    let (rows, width) = (parent.shape()[0], parent.shape()[1]);
                    let target = adjoint(pending, parent)?;
                    if axis == 0 {
                        for (out, value) in target.iter_mut().zip(&g[offset..offset + rows * width])
                        {
                            *out += value;
                        }
                        offset += rows * width;
                    } else {
                        for row in 0..rows {
                            let source = &g[row * total + offset..row * total + offset + width];
                            for (out, value) in target[row * width..(row + 1) * width]
                                .iter_mut()
                                .zip(source)
                            {
                                *out += value;
                            }
                        }
                        offset += width;
                    }
                }
            }
            Self::Sum => {
                let target = adjoint(pending, &node.parents()[0])?;
                for out in target {
                    *out += g[0];
                }
            }
            Self::CrossEntropy => {
                let logits = &node.parents()[0];
                let values = node
                    .0
                    .backend
                    .as_ref()
                    .expect("CE backend")
                    .cross_entropy_backward(
                        logits.data(),
                        &node.0.indices,
                        logits.shape()[1],
                        g[0],
                    )?;
                if values.len() != logits.numel() {
                    return Err(TensorError::InvalidShape(
                        "backend returned wrong CE gradient size",
                    ));
                }
                give(pending, logits, values);
            }
            Self::GatherRows => {
                let parent = &node.parents()[0];
                let width = parent.shape()[1];
                let target = adjoint(pending, parent)?;
                for (i, &index) in node.0.indices.iter().enumerate() {
                    for j in 0..width {
                        target[index * width + j] += g[i * width + j];
                    }
                }
            }
            Self::Matmul | Self::BatchedMatmul => {
                let (a, b) = (&node.parents()[0], &node.parents()[1]);
                let offset = a.rank() - 2;
                let (m, k, n) = (
                    a.shape()[offset],
                    a.shape()[offset + 1],
                    b.shape()[offset + 1],
                );
                let batch = if self == Self::BatchedMatmul {
                    a.shape()[0]
                } else {
                    1
                };
                if m == 0 || k == 0 || n == 0 {
                    return Ok(());
                }
                if let Some((store, view)) = b.residency() {
                    // O forward só aceita residente em matmul 2D: batch == 1.
                    let da = store.matmul_backward(a.data(), view, g, m, k, n)?;
                    if da.len() != m * k {
                        return Err(TensorError::InvalidShape(
                            "store returned wrong gradient size",
                        ));
                    }
                    give(pending, a, da);
                    return Ok(());
                }
                let backend = node.0.backend.as_ref().expect("matmul backend");
                for i in 0..batch {
                    let (da, db) = backend.matmul_backward(
                        &a.data()[i * m * k..(i + 1) * m * k],
                        &b.data()[i * k * n..(i + 1) * k * n],
                        &g[i * m * n..(i + 1) * m * n],
                        m,
                        k,
                        n,
                    )?;
                    if da.len() != m * k || db.len() != k * n {
                        return Err(TensorError::InvalidShape(
                            "backend returned wrong gradient size",
                        ));
                    }
                    for (parent, gradient, start) in [(a, da, i * m * k), (b, db, i * k * n)] {
                        if batch == 1 {
                            give(pending, parent, gradient);
                        } else {
                            let target = adjoint(pending, parent)?;
                            accumulate(&mut target[start..start + gradient.len()], &gradient);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
