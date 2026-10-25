//! Backend CUDA experimental híbrido: cada chamada transfere buffers host/device.
//! `CudaParamStore` mantém pesos, gradientes e Adam na VRAM (ver `resident`).
//! Sem driver NVIDIA/NVRTC retorna erro; nunca substitui por CPU.
use crate::{
    backend::{check, validate_ce, validate_maps},
    tensor::{element_count, zeros_buffer},
    Backend, TensorError,
};
use crate::{
    resident::{check_matmul, check_view, columns_of, put_columns},
    AdamStep, Buffer, ParamStore, Transfers, View,
};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};
type Result<T> = std::result::Result<T, TensorError>;
fn error(e: impl std::fmt::Debug) -> TensorError {
    TensorError::BackendFailure(format!("CUDA: {e:?}"))
}
fn int(n: usize) -> Result<i32> {
    i32::try_from(n)
        .map_err(|_| TensorError::InvalidShape("CUDA reference kernel exceeds i32 indexing"))
}

#[derive(Debug)]
pub struct CudaBackend {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    launches: AtomicU64,
}
impl CudaBackend {
    pub fn new(device: usize) -> Result<Self> {
        if !cfg!(target_os = "linux") {
            return Err(TensorError::BackendUnavailable(
                "CUDA requires Linux/NVIDIA; not available on this OS",
            ));
        }
        // cudarc's dynamic loader can panic when shared libraries are absent.
        std::panic::catch_unwind(|| Self::initialize(device)).map_err(|_| {
            TensorError::BackendUnavailable(
                "CUDA driver or NVRTC shared library could not be loaded",
            )
        })?
    }
    fn initialize(device: usize) -> Result<Self> {
        let ctx = CudaContext::new(device).map_err(error)?;
        let (major, minor) = ctx.compute_capability().map_err(error)?;
        let ptx = compile_ptx_with_opts(
            include_str!("cuda/kernels.cu"),
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                fmad: Some(false),
                ftz: Some(false),
                prec_div: Some(true),
                prec_sqrt: Some(true),
                ..Default::default()
            },
        )
        .map_err(error)?;
        let module = ctx.load_module(ptx).map_err(error)?;
        Ok(Self {
            stream: ctx.default_stream(),
            module,
            launches: AtomicU64::new(0),
        })
    }
    pub fn kernel_launches(&self) -> u64 {
        self.launches.load(Ordering::Relaxed)
    }
    fn launched(&self) {
        self.launches.fetch_add(1, Ordering::Relaxed);
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        ta: bool,
        tb: bool,
    ) -> Result<Vec<f32>> {
        check(a, &[m, k])?;
        check(b, &[k, n])?;
        let count = element_count(&[m, n])?;
        int(a.len())?;
        int(b.len())?;
        int(count)?;
        let (mi, ki, ni) = (int(m)?, int(k)?, int(n)?);
        if m == 0 || n == 0 || k == 0 {
            return zeros_buffer(count);
        }
        if m.div_ceil(16) > 65535 {
            return Err(TensorError::InvalidShape("CUDA grid y exceeds 65535"));
        }
        let ad = self.stream.clone_htod(a).map_err(error)?;
        let bd = self.stream.clone_htod(b).map_err(error)?;
        let mut out = self.stream.alloc_zeros::<f32>(count).map_err(error)?;
        let f = self.module.load_function("gemm").map_err(error)?;
        let (ta, tb) = (i32::from(ta), i32::from(tb));
        let mut launch = self.stream.launch_builder(&f);
        launch
            .arg(&ad)
            .arg(&bd)
            .arg(&mut out)
            .arg(&mi)
            .arg(&ki)
            .arg(&ni)
            .arg(&ta)
            .arg(&tb);
        // SAFETY: lengths, indices, grid and argument order match kernels.cu; allocations live through sync copy.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(error)?;
        self.launched();
        self.stream.clone_dtoh(&out).map_err(error)
    }
    fn ce(
        &self,
        logits: &[f32],
        targets: &[usize],
        classes: usize,
        seed: Option<f32>,
    ) -> Result<Vec<f32>> {
        validate_ce(logits, targets, classes)?;
        int(logits.len())?;
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        let rows = int(targets.len())?;
        let classes = int(classes)?;
        let x = self.stream.clone_htod(logits).map_err(error)?;
        let labels: Vec<u64> = targets.iter().map(|&v| v as u64).collect();
        let labels = self.stream.clone_htod(&labels).map_err(error)?;
        let size = if seed.is_some() {
            logits.len()
        } else {
            targets.len()
        };
        let mut out = self.stream.alloc_zeros::<f32>(size).map_err(error)?;
        let f = self
            .module
            .load_function(if seed.is_some() {
                "ce_backward"
            } else {
                "cross_entropy"
            })
            .map_err(error)?;
        let mut launch = self.stream.launch_builder(&f);
        launch
            .arg(&x)
            .arg(&labels)
            .arg(&mut out)
            .arg(&rows)
            .arg(&classes);
        if let Some(ref seed) = seed {
            launch.arg(seed);
        }
        // SAFETY: validated row/class counts and labels bound all device accesses.
        unsafe { launch.launch(LaunchConfig::for_num_elems(rows as u32)) }.map_err(error)?;
        self.launched();
        self.stream.clone_dtoh(&out).map_err(error)
    }
}
impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda-hybrid"
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        self.gemm(a, b, m, k, n, false, false)
    }
    fn matmul_backward(
        &self,
        a: &[f32],
        b: &[f32],
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        check(a, &[m, k])?;
        check(b, &[k, n])?;
        check(g, &[m, n])?;
        Ok((
            self.gemm(g, b, m, n, k, false, true)?,
            self.gemm(a, g, k, m, n, true, false)?,
        ))
    }
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        check(b, &[a.len()])?;
        let count = int(a.len())?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let a = self.stream.clone_htod(a).map_err(error)?;
        let b = self.stream.clone_htod(b).map_err(error)?;
        let mut out = self
            .stream
            .alloc_zeros::<f32>(count as usize)
            .map_err(error)?;
        let f = self.module.load_function("add").map_err(error)?;
        let mut launch = self.stream.launch_builder(&f);
        launch.arg(&a).arg(&b).arg(&mut out).arg(&count);
        // SAFETY: all three buffers have count elements, matching the CUDA signature.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.map_err(error)?;
        self.launched();
        self.stream.clone_dtoh(&out).map_err(error)
    }
    fn add_backward(
        &self,
        g: &[f32],
        left: &[usize],
        right: &[usize],
        left_len: usize,
        right_len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        validate_maps(g, left, right, left_len, right_len)?;
        let count = int(g.len())?;
        int(left_len)?;
        int(right_len)?;
        if count == 0 {
            return Ok((zeros_buffer(left_len)?, zeros_buffer(right_len)?));
        }
        let g = self.stream.clone_htod(g).map_err(error)?;
        let left: Vec<u64> = left.iter().map(|&v| v as u64).collect();
        let right: Vec<u64> = right.iter().map(|&v| v as u64).collect();
        let left = self.stream.clone_htod(&left).map_err(error)?;
        let right = self.stream.clone_htod(&right).map_err(error)?;
        let mut da = self.stream.alloc_zeros::<f32>(left_len).map_err(error)?;
        let mut db = self.stream.alloc_zeros::<f32>(right_len).map_err(error)?;
        let f = self.module.load_function("add_backward").map_err(error)?;
        let mut launch = self.stream.launch_builder(&f);
        launch
            .arg(&g)
            .arg(&left)
            .arg(&right)
            .arg(&mut da)
            .arg(&mut db)
            .arg(&count);
        // SAFETY: maps validated against da/db lengths; atomicAdd resolves repeated indices.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.map_err(error)?;
        self.launched();
        Ok((
            self.stream.clone_dtoh(&da).map_err(error)?,
            self.stream.clone_dtoh(&db).map_err(error)?,
        ))
    }
    fn cross_entropy(&self, logits: &[f32], targets: &[usize], classes: usize) -> Result<f32> {
        Ok(self.ce(logits, targets, classes, None)?.iter().sum::<f32>()
            / targets.len().max(1) as f32)
    }
    fn cross_entropy_backward(
        &self,
        logits: &[f32],
        targets: &[usize],
        classes: usize,
        seed: f32,
    ) -> Result<Vec<f32>> {
        self.ce(logits, targets, classes, Some(seed))
    }
}

fn gemm_config(rows: usize, cols: usize) -> Result<LaunchConfig> {
    if rows.div_ceil(16) > 65535 {
        return Err(TensorError::InvalidShape("CUDA grid y exceeds 65535"));
    }
    Ok(LaunchConfig {
        grid_dim: (cols.div_ceil(16) as u32, rows.div_ceil(16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    })
}

struct DeviceArena {
    weights: CudaSlice<f32>,
    grad: CudaSlice<f32>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
}

impl DeviceArena {
    fn buffer(&mut self, buffer: Buffer) -> &mut CudaSlice<f32> {
        match buffer {
            Buffer::Weights => &mut self.weights,
            Buffer::Grad => &mut self.grad,
            Buffer::M => &mut self.m,
            Buffer::V => &mut self.v,
        }
    }
}

/// Pesos, gradientes e momentos do Adam residentes na VRAM, numa arena por buffer. Por passo
/// só as ativações cruzam o PCIe; o Adam e a norma do gradiente rodam na GPU.
pub struct CudaParamStore {
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    arena: Mutex<DeviceArena>,
    capacity: usize,
    to_device: AtomicU64,
    to_host: AtomicU64,
}

impl std::fmt::Debug for CudaParamStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaParamStore")
            .field("capacity", &self.capacity)
            .field("transfers", &self.transfers())
            .finish()
    }
}

impl CudaBackend {
    /// Store residente no mesmo contexto e stream deste backend: `capacity` números por
    /// buffer (pesos, gradiente, m, v), ou seja, 16 bytes por parâmetro residente na VRAM.
    pub fn param_store(&self, capacity: usize) -> Result<CudaParamStore> {
        int(capacity)?;
        let zeros = || {
            self.stream
                .alloc_zeros::<f32>(capacity.max(1))
                .map_err(error)
        };
        Ok(CudaParamStore {
            stream: self.stream.clone(),
            module: self.module.clone(),
            arena: Mutex::new(DeviceArena {
                weights: zeros()?,
                grad: zeros()?,
                m: zeros()?,
                v: zeros()?,
            }),
            capacity,
            to_device: AtomicU64::new(0),
            to_host: AtomicU64::new(0),
        })
    }
}

impl CudaParamStore {
    fn arena(&self) -> Result<std::sync::MutexGuard<'_, DeviceArena>> {
        self.arena.lock().map_err(|_| TensorError::LockPoisoned)
    }
    fn upload(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        self.to_device
            .fetch_add(4 * data.len() as u64, Ordering::Relaxed);
        self.stream.clone_htod(data).map_err(error)
    }
    fn received(&self, bytes: usize) {
        self.to_host.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

impl ParamStore for CudaParamStore {
    fn name(&self) -> &'static str {
        "cuda-resident"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn matmul(&self, a: &[f32], rhs: View, m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        check_matmul(a, rhs, m, k, n, self.capacity)?;
        let count = element_count(&[m, n])?;
        if m == 0 || k == 0 || n == 0 {
            return zeros_buffer(count);
        }
        let (mi, ki, ni) = (int(m)?, int(k)?, int(n)?);
        int(count)?;
        let config = gemm_config(m, n)?;
        let arena = self.arena()?;
        let ad = self.upload(a)?;
        let mut out = self.stream.alloc_zeros::<f32>(count).map_err(error)?;
        let w = arena.weights.slice(rhs.range());
        let f = self.module.load_function("gemm").map_err(error)?;
        let (ta, tb) = (0i32, 0i32);
        let mut launch = self.stream.launch_builder(&f);
        launch
            .arg(&ad)
            .arg(&w)
            .arg(&mut out)
            .arg(&mi)
            .arg(&ki)
            .arg(&ni)
            .arg(&ta)
            .arg(&tb);
        // SAFETY: a [m,k], w [k,n] (view checked against the arena), out [m,n]; grid covers out.
        unsafe { launch.launch(config) }.map_err(error)?;
        let host = self.stream.clone_dtoh(&out).map_err(error)?;
        self.received(4 * host.len());
        Ok(host)
    }

    fn matmul_backward(
        &self,
        a: &[f32],
        rhs: View,
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        check_matmul(a, rhs, m, k, n, self.capacity)?;
        check(g, &[m, n])?;
        if m == 0 || k == 0 || n == 0 {
            return zeros_buffer(m * k);
        }
        let (mi, ki, ni) = (int(m)?, int(k)?, int(n)?);
        let (to_da, to_dw) = (gemm_config(m, k)?, gemm_config(k, n)?);
        let mut arena = self.arena()?;
        let DeviceArena { weights, grad, .. } = &mut *arena;
        let (ad, gd) = (self.upload(a)?, self.upload(g)?);
        let mut da = self.stream.alloc_zeros::<f32>(m * k).map_err(error)?;
        let w = weights.slice(rhs.range());
        let mut dw = grad.slice_mut(rhs.range());
        let gemm = self.module.load_function("gemm").map_err(error)?;
        let gemm_acc = self.module.load_function("gemm_acc").map_err(error)?;
        let (zero, one) = (0i32, 1i32);
        // dA [m,k] = g [m,n] · Wᵀ.
        let mut launch = self.stream.launch_builder(&gemm);
        launch
            .arg(&gd)
            .arg(&w)
            .arg(&mut da)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki)
            .arg(&zero)
            .arg(&one);
        // SAFETY: g [m,n], w [k,n] read transposed, da [m,k]; grid covers da.
        unsafe { launch.launch(to_da) }.map_err(error)?;
        // grad [k,n] += Aᵀ [k,m] · g [m,n].
        let mut launch = self.stream.launch_builder(&gemm_acc);
        launch
            .arg(&ad)
            .arg(&gd)
            .arg(&mut dw)
            .arg(&ki)
            .arg(&mi)
            .arg(&ni)
            .arg(&one)
            .arg(&zero);
        // SAFETY: a [m,k] read transposed, g [m,n], dw = grad view [k,n]; grid covers dw.
        unsafe { launch.launch(to_dw) }.map_err(error)?;
        let host = self.stream.clone_dtoh(&da).map_err(error)?;
        self.received(4 * host.len());
        Ok(host)
    }

    fn matmul_many(
        &self,
        a: &[f32],
        rhs: &[View],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        for &view in rhs {
            check_matmul(a, view, m, k, n, self.capacity)?;
        }
        let part = element_count(&[m, n])?;
        let total = part
            .checked_mul(rhs.len())
            .ok_or(TensorError::ShapeOverflow)?;
        if total == 0 || k == 0 {
            return zeros_buffer(total);
        }
        let (mi, ki, ni) = (int(m)?, int(k)?, int(n)?);
        int(total)?;
        let config = gemm_config(m, n)?;
        let arena = self.arena()?;
        let ad = self.upload(a)?;
        // Saídas [R, m, n] contíguas na GPU; o host as põe lado a lado em [m, R·n].
        let mut out = self.stream.alloc_zeros::<f32>(total).map_err(error)?;
        let f = self.module.load_function("gemm").map_err(error)?;
        let (ta, tb) = (0i32, 0i32);
        for (i, view) in rhs.iter().enumerate() {
            let w = arena.weights.slice(view.range());
            let mut target = out.slice_mut(i * part..(i + 1) * part);
            let mut launch = self.stream.launch_builder(&f);
            launch
                .arg(&ad)
                .arg(&w)
                .arg(&mut target)
                .arg(&mi)
                .arg(&ki)
                .arg(&ni)
                .arg(&ta)
                .arg(&tb);
            // SAFETY: a [m,k], w [k,n], target is the i-th [m,n] block of out.
            unsafe { launch.launch(config) }.map_err(error)?;
        }
        let blocks = self.stream.clone_dtoh(&out).map_err(error)?;
        self.received(4 * blocks.len());
        let mut host = zeros_buffer(total)?;
        for (i, block) in blocks.chunks(part).enumerate() {
            put_columns(&mut host, block, m, n, rhs.len(), i);
        }
        Ok(host)
    }

    fn matmul_many_backward(
        &self,
        a: &[f32],
        rhs: &[View],
        g: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        for &view in rhs {
            check_matmul(a, view, m, k, n, self.capacity)?;
        }
        check(g, &[m, n * rhs.len()])?;
        if m == 0 || k == 0 || n == 0 || rhs.is_empty() {
            return zeros_buffer(m * k);
        }
        let (mi, ki, ni) = (int(m)?, int(k)?, int(n)?);
        let (to_da, to_dw) = (gemm_config(m, k)?, gemm_config(k, n)?);
        let part = m * n;
        // g [m, R·n] vira blocos [R, m, n] contíguos antes de subir (uma cópia só).
        let mut blocks = Vec::with_capacity(g.len());
        for i in 0..rhs.len() {
            blocks.extend(columns_of(g, m, n, rhs.len(), i));
        }
        let mut arena = self.arena()?;
        let DeviceArena { weights, grad, .. } = &mut *arena;
        let (ad, gd) = (self.upload(a)?, self.upload(&blocks)?);
        let mut da = self.stream.alloc_zeros::<f32>(m * k).map_err(error)?;
        let gemm_acc = self.module.load_function("gemm_acc").map_err(error)?;
        let (zero, one) = (0i32, 1i32);
        for (i, view) in rhs.iter().enumerate() {
            let gi = gd.slice(i * part..(i + 1) * part);
            let w = weights.slice(view.range());
            // dA += gᵢ · Wᵢᵀ, na ordem dos pesos (como o caminho CPU).
            let mut launch = self.stream.launch_builder(&gemm_acc);
            launch
                .arg(&gi)
                .arg(&w)
                .arg(&mut da)
                .arg(&mi)
                .arg(&ni)
                .arg(&ki)
                .arg(&zero)
                .arg(&one);
            // SAFETY: gi [m,n], w [k,n] read transposed, da [m,k].
            unsafe { launch.launch(to_da) }.map_err(error)?;
            let mut dw = grad.slice_mut(view.range());
            let mut launch = self.stream.launch_builder(&gemm_acc);
            launch
                .arg(&ad)
                .arg(&gi)
                .arg(&mut dw)
                .arg(&ki)
                .arg(&mi)
                .arg(&ni)
                .arg(&one)
                .arg(&zero);
            // SAFETY: a [m,k] read transposed, gi [m,n], dw = grad view [k,n].
            unsafe { launch.launch(to_dw) }.map_err(error)?;
        }
        let host = self.stream.clone_dtoh(&da).map_err(error)?;
        self.received(4 * host.len());
        Ok(host)
    }

    fn grad_stats(&self) -> Result<(f64, usize)> {
        if self.capacity == 0 {
            return Ok((0.0, 0));
        }
        const BLOCKS: usize = 256;
        let count = int(self.capacity)?;
        let arena = self.arena()?;
        let mut partial = self.stream.alloc_zeros::<f64>(BLOCKS).map_err(error)?;
        let mut nonfinite = self.stream.alloc_zeros::<u32>(1).map_err(error)?;
        let f = self.module.load_function("sumsq").map_err(error)?;
        let mut launch = self.stream.launch_builder(&f);
        launch
            .arg(&arena.grad)
            .arg(&count)
            .arg(&mut partial)
            .arg(&mut nonfinite);
        // SAFETY: grid-stride loop bounded by count; partial has one slot per block.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (BLOCKS as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(error)?;
        let sums = self.stream.clone_dtoh(&partial).map_err(error)?;
        let bad = self.stream.clone_dtoh(&nonfinite).map_err(error)?;
        self.received(8 * BLOCKS + 4);
        Ok((sums.iter().sum(), bad[0] as usize))
    }

    fn adam_step(&self, step: AdamStep) -> Result<usize> {
        if self.capacity == 0 {
            return Ok(0);
        }
        let count = int(self.capacity)?;
        let mut arena = self.arena()?;
        let DeviceArena {
            weights,
            grad,
            m,
            v,
        } = &mut *arena;
        let mut nonfinite = self.stream.alloc_zeros::<u32>(1).map_err(error)?;
        let f = self.module.load_function("adam").map_err(error)?;
        let mut launch = self.stream.launch_builder(&f);
        launch
            .arg(weights)
            .arg(grad)
            .arg(m)
            .arg(v)
            .arg(&count)
            .arg(&step.beta1)
            .arg(&step.beta2)
            .arg(&step.lr)
            .arg(&step.eps)
            .arg(&step.correction1)
            .arg(&step.correction2)
            .arg(&mut nonfinite);
        // SAFETY: the four buffers hold `capacity` elements; one thread per element.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.map_err(error)?;
        let bad = self.stream.clone_dtoh(&nonfinite).map_err(error)?;
        self.received(4);
        Ok(bad[0] as usize)
    }

    fn read(&self, buffer: Buffer, view: View) -> Result<Vec<f32>> {
        check_view(self.capacity, view)?;
        if view.len == 0 {
            return Ok(Vec::new());
        }
        let mut arena = self.arena()?;
        let host = self
            .stream
            .clone_dtoh(&arena.buffer(buffer).slice(view.range()))
            .map_err(error)?;
        self.received(4 * host.len());
        Ok(host)
    }

    fn write(&self, buffer: Buffer, view: View, data: &[f32]) -> Result<()> {
        check_view(self.capacity, view)?;
        if data.len() != view.len {
            return Err(TensorError::DataLength {
                expected: view.len,
                actual: data.len(),
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let mut arena = self.arena()?;
        let mut target = arena.buffer(buffer).slice_mut(view.range());
        self.stream.memcpy_htod(data, &mut target).map_err(error)?;
        self.to_device
            .fetch_add(4 * data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    fn transfers(&self) -> Transfers {
        Transfers {
            to_device: self.to_device.load(Ordering::Relaxed),
            to_host: self.to_host.load(Ordering::Relaxed),
        }
    }
}
