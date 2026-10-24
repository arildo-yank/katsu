//! Backend CUDA experimental híbrido: cada chamada transfere buffers host/device.
//! Sem driver NVIDIA/NVRTC retorna erro; nunca substitui por CPU.
use crate::{
    backend::{check, validate_ce, validate_maps},
    tensor::{element_count, zeros_buffer},
    Backend, TensorError,
};
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
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
