//! Treino do Katsu LM inteiro na GPU: pesos, gradientes, momentos do Adam e ativações ficam
//! na VRAM, e um passo não copia nada entre host e dispositivo além dos tokens de entrada e
//! das perdas por linha (alguns KB). Forward e backward são explícitos para a arquitetura do
//! `KatsuLmConfig` (sem grafo dinâmico), com as mesmas fórmulas do backend CPU; a paridade é
//! conferida em `tests/gpu_train.rs`. O checkpoint continua no formato do host
//! (`GpuLayout` faz a permutação de ida e volta).
use crate::{
    gpu_layout::{GpuLayout, SftBatch},
    Adam, KatsuLmConfig, Tensor, TensorError,
};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
    },
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use std::{collections::HashMap, sync::Arc};

type Result<T> = std::result::Result<T, TensorError>;

fn error(e: impl std::fmt::Debug) -> TensorError {
    TensorError::BackendFailure(format!("CUDA: {e:?}"))
}
fn int(n: usize) -> Result<i32> {
    i32::try_from(n).map_err(|_| TensorError::InvalidShape("GPU trainer dimension exceeds i32"))
}
fn long(n: usize) -> i64 {
    n as i64
}
/// Endereço de `floats` posições depois de `base`.
fn at(base: u64, floats: usize) -> u64 {
    base + 4 * floats as u64
}

/// Qual GEMM usar (as etapas seguintes acrescentam variantes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmKernel {
    /// Tiles 16×16 em memória compartilhada, um resultado por thread.
    Reference16,
    /// Etapa 3: tiles 64×64, 4×4 resultados por thread em registradores (FP32).
    Tiled64,
    /// Etapa 3b: Tensor Cores (WMMA) com entradas em TF32 e acumulação em FP32. Pesos,
    /// gradientes e Adam continuam em FP32; só os produtos perdem precisão (~1e-3 relativo).
    Tf32,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuTrainOptions {
    pub gemm: GemmKernel,
    /// Etapa 2: bias, GELU e resíduo no epílogo do GEMM, e gelu' no GEMM do backward, em vez
    /// de passadas separadas pela memória.
    pub fused: bool,
}

impl Default for GpuTrainOptions {
    fn default() -> Self {
        Self {
            gemm: GemmKernel::Reference16,
            fused: true,
        }
    }
}

/// O que o epílogo do GEMM faz além de escrever A·B (ver `epilogue` em train.cu).
#[derive(Debug, Clone, Copy)]
struct Epi {
    bias: u64,
    res: u64,
    pre: u64,
    /// 0: nada; 1: GELU guardando a entrada em `pre`; 2: multiplica por gelu'(pre)
    mode: i32,
    acc: bool,
}
const PLAIN: Epi = Epi {
    bias: 0,
    res: 0,
    pre: 0,
    mode: 0,
    acc: false,
};
const ACC: Epi = Epi { acc: true, ..PLAIN };

/// Argumento de kernel: o tipo precisa bater com a assinatura em train.cu.
enum Arg {
    /// ponteiro (float*, int*, ...)
    P(u64),
    /// int
    I(i32),
    /// long long
    L(i64),
    /// float
    F(f32),
}
use Arg::{F, I, L, P};

const KERNELS: &[&str] = &[
    "fill0",
    "add2",
    "copy",
    "embed_fwd",
    "embed_bwd",
    "layernorm_fwd",
    "layernorm_bwd",
    "gemm16",
    "gemm64",
    "bias_add",
    "colsum_acc",
    "gelu_fwd",
    "gelu_bwd",
    "attn_fwd",
    "attn_bwd_rows",
    "attn_bwd_cols",
    "ce_fwd_bwd",
    "adam",
    "sumsq",
];
const SUMSQ_BLOCKS: usize = 1024;

/// Começo de cada região de ativações; as por camada têm passo `*_stride`.
#[derive(Debug, Clone, Copy, Default)]
struct Acts {
    x: u64,
    xmid: u64,
    ln1: u64,
    ln2: u64,
    mean1: u64,
    rstd1: u64,
    mean2: u64,
    rstd2: u64,
    qkv: u64,
    att: u64,
    atty: u64,
    f1: u64,
    gl: u64,
    lnf: u64,
    meanf: u64,
    rstdf: u64,
    logits: u64,
    dres: u64,
    dh: u64,
    dqkv: u64,
    datty: u64,
    dbig: u64,
    ds: u64,
    tmp: u64,
    // passos por camada, em floats
    sd: usize,
    sf: usize,
    sq: usize,
    sa: usize,
    sr: usize,
}

pub struct GpuTrainer {
    config: KatsuLmConfig,
    layout: GpuLayout,
    max_batch: usize,
    max_t: usize,
    options: GpuTrainOptions,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
    weights: CudaSlice<f32>,
    grads: CudaSlice<f32>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    _act: CudaSlice<f32>,
    tok: CudaSlice<i32>,
    tgt: CudaSlice<i32>,
    wt: CudaSlice<f32>,
    losses: CudaSlice<f32>,
    partial: CudaSlice<f64>,
    nonfinite: CudaSlice<u32>,
    w: u64,
    g: u64,
    pm: u64,
    pv: u64,
    a: Acts,
    hyper: (f32, f32, f32, f32, i32),
    launches: u64,
}

impl std::fmt::Debug for GpuTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTrainer")
            .field("config", &self.config)
            .field("max_batch", &self.max_batch)
            .field("max_t", &self.max_t)
            .field("options", &self.options)
            .finish()
    }
}

impl GpuTrainer {
    /// Sobe os pesos (e o estado do Adam, se houver) e reserva ativações para lotes de até
    /// `max_batch` sequências de até `max_t` tokens. Falha antes de alocar se a VRAM livre
    /// não comportar.
    pub fn new(
        config: KatsuLmConfig,
        params: &[Tensor],
        adam: &Adam,
        max_batch: usize,
        max_t: usize,
        device: usize,
        options: GpuTrainOptions,
    ) -> Result<Self> {
        if !cfg!(target_os = "linux") {
            return Err(TensorError::BackendUnavailable(
                "GPU trainer requires Linux/NVIDIA",
            ));
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::initialize(config, params, adam, max_batch, max_t, device, options)
        }))
        .map_err(|_| {
            TensorError::BackendUnavailable(
                "CUDA driver or NVRTC shared library could not be loaded",
            )
        })?
    }

    /// Floats de ativação para o lote máximo (sem contar pesos, gradientes e Adam).
    pub fn activation_floats(config: &KatsuLmConfig, max_batch: usize, max_t: usize) -> usize {
        let (l, d, h, v) = (
            config.layers,
            config.embed_dim,
            config.heads,
            config.vocab_size,
        );
        let rows = max_batch * max_t;
        let (sd, sf, sq, sa, sr) = (
            rows * d,
            rows * 4 * d,
            rows * 3 * d,
            max_batch * h * max_t * max_t,
            rows,
        );
        (l + 1) * sd                 // x
            + l * (sd * 4)           // xmid, ln1, ln2, atty
            + l * (sr * 4)           // médias e rstd
            + l * sq                 // qkv
            + l * sa                 // att
            + l * (sf * 2)           // f1, gl
            + sd + 2 * sr            // lnf, meanf, rstdf
            + rows * v               // logits
            + sd * 4 + sq + sf + sa // dres, dh, datty, tmp, dqkv, dbig, ds
    }

    #[allow(clippy::too_many_arguments)]
    fn initialize(
        config: KatsuLmConfig,
        params: &[Tensor],
        adam: &Adam,
        max_batch: usize,
        max_t: usize,
        device: usize,
        options: GpuTrainOptions,
    ) -> Result<Self> {
        let layout = GpuLayout::new(config)?;
        if max_batch == 0 || max_t == 0 || max_t > config.block_size {
            return Err(TensorError::InvalidShape(
                "GPU trainer needs 0 < max_t <= block_size",
            ));
        }
        let ctx = CudaContext::new(device).map_err(error)?;
        // Um stream só: a ordem do stream já sincroniza tudo, e os ponteiros crus da arena
        // não passam pelo rastreamento de eventos do cudarc.
        // SAFETY: todo trabalho deste contexto vai para `stream`; nada é usado em outro stream.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.default_stream();
        let (major, minor) = ctx.compute_capability().map_err(error)?;
        // O GEMM de Tensor Cores precisa de mma.h (dos headers do CUDA) e só entra quando pedido.
        let tensor_cores = options.gemm == GemmKernel::Tf32;
        let mut source = include_str!("cuda/train.cu").to_owned();
        if tensor_cores {
            if major < 8 {
                return Err(TensorError::BackendUnavailable(
                    "TF32 Tensor Cores need sm_80 or newer",
                ));
            }
            source.push_str(include_str!("cuda/train_tc.cu"));
        }
        let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions {
                options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
                prec_div: Some(true),
                prec_sqrt: Some(true),
                include_paths: if tensor_cores {
                    vec![format!("{cuda_home}/include")]
                } else {
                    Vec::new()
                },
                ..Default::default()
            },
        )
        .map_err(error)?;
        let module = ctx.load_module(ptx).map_err(error)?;
        let mut kernels = HashMap::new();
        for &name in KERNELS {
            kernels.insert(name, module.load_function(name).map_err(error)?);
        }
        if tensor_cores {
            kernels.insert("gemm_tc", module.load_function("gemm_tc").map_err(error)?);
        }

        let total = layout.total;
        let act_floats = Self::activation_floats(&config, max_batch, max_t);
        let rows = max_batch * max_t;
        let need = 4 * (4 * total + act_floats + 2 * rows) + 8 * SUMSQ_BLOCKS;
        let (free, _) = ctx.mem_get_info().map_err(error)?;
        if need > free {
            return Err(TensorError::BackendFailure(format!(
                "GPU trainer needs {:.2} GiB, {:.2} GiB free (lower the batch)",
                need as f64 / (1u64 << 30) as f64,
                free as f64 / (1u64 << 30) as f64
            )));
        }

        let arena = layout.pack_params(params)?;
        let weights = stream.clone_htod(&arena).map_err(error)?;
        drop(arena);
        let grads = stream.alloc_zeros::<f32>(total).map_err(error)?;
        let m = stream.alloc_zeros::<f32>(total).map_err(error)?;
        let v = stream.alloc_zeros::<f32>(total).map_err(error)?;
        let act = stream.alloc_zeros::<f32>(act_floats).map_err(error)?;
        let tok = stream.alloc_zeros::<i32>(rows).map_err(error)?;
        let tgt = stream.alloc_zeros::<i32>(rows).map_err(error)?;
        let wt = stream.alloc_zeros::<f32>(rows).map_err(error)?;
        let losses = stream.alloc_zeros::<f32>(rows).map_err(error)?;
        let partial = stream.alloc_zeros::<f64>(SUMSQ_BLOCKS).map_err(error)?;
        let nonfinite = stream.alloc_zeros::<u32>(1).map_err(error)?;

        let ptr = |s: &CudaSlice<f32>| s.device_ptr(&stream).0;
        let (w, g, pm, pv, base) = (ptr(&weights), ptr(&grads), ptr(&m), ptr(&v), ptr(&act));

        let (l, d, h, vocab) = (
            config.layers,
            config.embed_dim,
            config.heads,
            config.vocab_size,
        );
        let (sd, sf, sq, sa, sr) = (
            rows * d,
            rows * 4 * d,
            rows * 3 * d,
            max_batch * h * max_t * max_t,
            rows,
        );
        let mut next = 0;
        let mut take = |n: usize| {
            let p = at(base, next);
            next += n;
            p
        };
        let a = Acts {
            x: take((l + 1) * sd),
            xmid: take(l * sd),
            ln1: take(l * sd),
            ln2: take(l * sd),
            atty: take(l * sd),
            mean1: take(l * sr),
            rstd1: take(l * sr),
            mean2: take(l * sr),
            rstd2: take(l * sr),
            qkv: take(l * sq),
            att: take(l * sa),
            f1: take(l * sf),
            gl: take(l * sf),
            lnf: take(sd),
            meanf: take(sr),
            rstdf: take(sr),
            logits: take(rows * vocab),
            dres: take(sd),
            dh: take(sd),
            datty: take(sd),
            tmp: take(sd),
            dqkv: take(sq),
            dbig: take(sf),
            ds: take(sa),
            sd,
            sf,
            sq,
            sa,
            sr,
        };
        debug_assert_eq!(next, act_floats);
        let mut trainer = Self {
            config,
            layout,
            max_batch,
            max_t,
            options,
            ctx,
            stream,
            kernels,
            weights,
            grads,
            m,
            v,
            _act: act,
            tok,
            tgt,
            wt,
            losses,
            partial,
            nonfinite,
            w,
            g,
            pm,
            pv,
            a,
            hyper: adam.hyperparameters(),
            launches: 0,
        };
        trainer.load_adam(adam)?;
        Ok(trainer)
    }

    /// Troca o estado do Adam (hiperparâmetros, passo e momentos), ex.: ao passar da fase de
    /// adaptação ao corpus para o SFT, que começa com um Adam novo.
    pub fn load_adam(&mut self, adam: &Adam) -> Result<()> {
        let moments = adam.moments();
        if moments.is_empty() {
            let (m, v) = (self.pm, self.pv);
            let total = self.layout.total;
            self.elementwise("fill0", total, &[P(m), L(long(total))])?;
            self.elementwise("fill0", total, &[P(v), L(long(total))])?;
        } else {
            let ms: Vec<&[f32]> = moments.iter().map(|(m, _)| m.as_slice()).collect();
            let vs: Vec<&[f32]> = moments.iter().map(|(_, v)| v.as_slice()).collect();
            self.stream
                .memcpy_htod(&self.layout.pack(&ms)?, &mut self.m)
                .map_err(error)?;
            self.stream
                .memcpy_htod(&self.layout.pack(&vs)?, &mut self.v)
                .map_err(error)?;
        }
        self.hyper = adam.hyperparameters();
        Ok(())
    }

    pub fn config(&self) -> &KatsuLmConfig {
        &self.config
    }
    pub fn kernel_launches(&self) -> u64 {
        self.launches
    }
    /// (livre, total) da VRAM, em bytes.
    pub fn memory(&self) -> Result<(usize, usize)> {
        self.ctx.mem_get_info().map_err(error)
    }
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<()> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(TensorError::InvalidLearningRate);
        }
        self.hyper.0 = lr;
        Ok(())
    }

    fn run(
        &mut self,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        smem: usize,
        args: &[Arg],
    ) -> Result<()> {
        let f = &self.kernels[name];
        let mut launch = self.stream.launch_builder(f);
        for arg in args {
            match arg {
                P(x) => launch.arg(x),
                I(x) => launch.arg(x),
                L(x) => launch.arg(x),
                F(x) => launch.arg(x),
            };
        }
        // SAFETY: a ordem e os tipos dos argumentos seguem a assinatura em train.cu; todo
        // ponteiro aponta para dentro de uma alocação deste trainer, com o tamanho conferido
        // pelo chamador.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: smem as u32,
            })
        }
        .map_err(error)?;
        self.launches += 1;
        Ok(())
    }
    /// Kernel elemento a elemento sobre `n` posições, blocos de 256.
    fn elementwise(&mut self, name: &str, n: usize, args: &[Arg]) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.run(name, (n.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, args)
    }

    /// C[m,n] = op(A)·op(B) com o epílogo `e`, row-major com leading dimensions.
    fn gemm(
        &mut self,
        (a, ta, lda): (u64, bool, usize),
        (b, tb, ldb): (u64, bool, usize),
        (c, ldc): (u64, usize),
        (m, n, k): (usize, usize, usize),
        e: Epi,
    ) -> Result<()> {
        if m.div_ceil(16) > 65535 {
            return Err(TensorError::InvalidShape("GEMM grid y exceeds 65535"));
        }
        let acc = e.acc;
        let args = [
            P(a),
            P(b),
            P(c),
            P(e.bias),
            P(e.res),
            P(e.pre),
            I(e.mode),
            I(int(m)?),
            I(int(n)?),
            I(int(k)?),
            I(int(lda)?),
            I(int(ldb)?),
            I(int(ldc)?),
            I(ta as i32),
            I(tb as i32),
            I(acc as i32),
        ];
        let tiles = |size: usize| (n.div_ceil(size) as u32, m.div_ceil(size) as u32, 1);
        match self.options.gemm {
            GemmKernel::Reference16 => self.run("gemm16", tiles(16), (16, 16, 1), 0, &args),
            GemmKernel::Tiled64 => self.run("gemm64", tiles(64), (256, 1, 1), 0, &args),
            GemmKernel::Tf32 => self.run("gemm_tc", tiles(64), (128, 1, 1), 0, &args),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn layernorm(
        &mut self,
        x: u64,
        gamma: usize,
        beta: usize,
        out: u64,
        mean: u64,
        rstd: u64,
        rows: usize,
    ) -> Result<()> {
        let d = self.config.embed_dim;
        let (g, b) = (at(self.w, gamma), at(self.w, beta));
        self.run(
            "layernorm_fwd",
            ((rows * 32).div_ceil(256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                P(x),
                P(g),
                P(b),
                P(out),
                P(mean),
                P(rstd),
                I(int(rows)?),
                I(int(d)?),
            ],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn layernorm_backward(
        &mut self,
        dout: u64,
        x: u64,
        mean: u64,
        rstd: u64,
        gamma: usize,
        beta: usize,
        rows: usize,
    ) -> Result<()> {
        let d = self.config.embed_dim;
        let blocks = rows.div_ceil(8).clamp(1, 512);
        let args = [
            P(dout),
            P(x),
            P(mean),
            P(rstd),
            P(at(self.w, gamma)),
            P(self.a.dres),
            P(at(self.g, gamma)),
            P(at(self.g, beta)),
            I(int(rows)?),
            I(int(d)?),
        ];
        self.run(
            "layernorm_bwd",
            (blocks as u32, 1, 1),
            (256, 1, 1),
            2 * d * 4,
            &args,
        )
    }
    fn colsum(&mut self, x: u64, out: u64, rows: usize, cols: usize) -> Result<()> {
        let slices = rows.div_ceil(256).clamp(1, 64);
        self.run(
            "colsum_acc",
            (cols.div_ceil(32) as u32, slices as u32, 1),
            (32, 8, 1),
            0,
            &[P(x), P(out), I(int(rows)?), I(int(cols)?)],
        )
    }

    fn upload(&mut self, batch: &SftBatch) -> Result<(usize, usize)> {
        let (b, t) = (batch.batch, batch.width);
        let rows = b * t;
        if b == 0 || t == 0 || b > self.max_batch || t > self.max_t {
            return Err(TensorError::InvalidShape(
                "batch exceeds GPU trainer capacity",
            ));
        }
        if batch.tokens.len() != rows || batch.targets.len() != rows || batch.weights.len() != rows
        {
            return Err(TensorError::InvalidShape(
                "SftBatch arrays do not match batch × width",
            ));
        }
        let vocab = self.config.vocab_size;
        let ids = |v: &[usize]| -> Result<Vec<i32>> {
            v.iter()
                .map(|&x| {
                    if x >= vocab {
                        Err(TensorError::IndexOutOfRange {
                            index: x,
                            classes: vocab,
                        })
                    } else {
                        Ok(x as i32)
                    }
                })
                .collect()
        };
        let (tok, tgt) = (ids(&batch.tokens)?, ids(&batch.targets)?);
        if batch.weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(TensorError::InvalidShape(
                "SFT weights must be finite and >= 0",
            ));
        }
        self.stream
            .memcpy_htod(&tok, &mut self.tok.slice_mut(0..rows))
            .map_err(error)?;
        self.stream
            .memcpy_htod(&tgt, &mut self.tgt.slice_mut(0..rows))
            .map_err(error)?;
        self.stream
            .memcpy_htod(&batch.weights, &mut self.wt.slice_mut(0..rows))
            .map_err(error)?;
        Ok((b, t))
    }

    /// Perda do lote, sem tocar nos gradientes.
    pub fn loss(&mut self, batch: &SftBatch) -> Result<f64> {
        let (b, t) = self.upload(batch)?;
        self.forward(b, t, false)
    }

    /// Perda do lote; soma os gradientes dele aos acumulados (o passo do Adam os zera).
    pub fn loss_and_grad(&mut self, batch: &SftBatch) -> Result<f64> {
        let (b, t) = self.upload(batch)?;
        let loss = self.forward(b, t, true)?;
        if loss.is_finite() {
            self.backward(b, t)?;
        }
        Ok(loss)
    }

    fn forward(&mut self, b: usize, t: usize, grad: bool) -> Result<f64> {
        let c = self.config;
        let (d, h, vocab) = (c.embed_dim, c.heads, c.vocab_size);
        let (dh, f, rows) = (d / h, 4 * d, b * t);
        let scale = 1.0 / (dh as f32).sqrt();
        let a = self.a;
        let tok = self.tok.device_ptr(&self.stream).0;
        let (wte, wpe) = (at(self.w, self.layout.wte), at(self.w, self.layout.wpe));
        self.elementwise(
            "embed_fwd",
            rows * d,
            &[
                P(tok),
                P(wte),
                P(wpe),
                P(a.x),
                I(int(rows)?),
                I(int(t)?),
                I(int(d)?),
            ],
        )?;
        for l in 0..c.layers {
            let s = self.layout.layers[l];
            let x = at(a.x, l * a.sd);
            let (ln1, mean1, rstd1) = (
                at(a.ln1, l * a.sd),
                at(a.mean1, l * a.sr),
                at(a.rstd1, l * a.sr),
            );
            let (qkv, att, atty) = (
                at(a.qkv, l * a.sq),
                at(a.att, l * a.sa),
                at(a.atty, l * a.sd),
            );
            let xmid = at(a.xmid, l * a.sd);
            let (ln2, mean2, rstd2) = (
                at(a.ln2, l * a.sd),
                at(a.mean2, l * a.sr),
                at(a.rstd2, l * a.sr),
            );
            let (f1, gl) = (at(a.f1, l * a.sf), at(a.gl, l * a.sf));
            let next = at(a.x, (l + 1) * a.sd);

            self.layernorm(x, s.ln1_g, s.ln1_b, ln1, mean1, rstd1, rows)?;
            self.gemm(
                (ln1, false, d),
                (at(self.w, s.wqkv), false, 3 * d),
                (qkv, 3 * d),
                (rows, 3 * d, d),
                PLAIN,
            )?;
            self.run(
                "attn_fwd",
                (t as u32, h as u32, b as u32),
                (128, 1, 1),
                (32 + dh + t) * 4,
                &[
                    P(qkv),
                    P(att),
                    P(atty),
                    I(int(t)?),
                    I(int(h)?),
                    I(int(dh)?),
                    F(scale),
                ],
            )?;
            let (wo, w1, b1, w2, b2) = (
                at(self.w, s.wo),
                at(self.w, s.w1),
                at(self.w, s.b1),
                at(self.w, s.w2),
                at(self.w, s.b2),
            );
            if self.options.fused {
                // xmid = x + atty·Wo; gl = gelu(ln2·W1 + b1), guardando f1; próximo = xmid + gl·W2 + b2
                let residual = Epi { res: x, ..PLAIN };
                self.gemm(
                    (atty, false, d),
                    (wo, false, d),
                    (xmid, d),
                    (rows, d, d),
                    residual,
                )?;
                self.layernorm(xmid, s.ln2_g, s.ln2_b, ln2, mean2, rstd2, rows)?;
                let fc1 = Epi {
                    bias: b1,
                    pre: f1,
                    mode: 1,
                    ..PLAIN
                };
                self.gemm((ln2, false, d), (w1, false, f), (gl, f), (rows, f, d), fc1)?;
                let fc2 = Epi {
                    bias: b2,
                    res: xmid,
                    ..PLAIN
                };
                self.gemm((gl, false, f), (w2, false, d), (next, d), (rows, d, f), fc2)?;
                continue;
            }
            self.gemm(
                (atty, false, d),
                (wo, false, d),
                (a.tmp, d),
                (rows, d, d),
                PLAIN,
            )?;
            self.elementwise(
                "add2",
                rows * d,
                &[P(x), P(a.tmp), P(xmid), L(long(rows * d))],
            )?;
            self.layernorm(xmid, s.ln2_g, s.ln2_b, ln2, mean2, rstd2, rows)?;
            self.gemm(
                (ln2, false, d),
                (w1, false, f),
                (f1, f),
                (rows, f, d),
                PLAIN,
            )?;
            self.elementwise(
                "bias_add",
                rows * f,
                &[P(f1), P(b1), L(long(rows)), I(int(f)?)],
            )?;
            self.elementwise("gelu_fwd", rows * f, &[P(f1), P(gl), L(long(rows * f))])?;
            self.gemm(
                (gl, false, f),
                (w2, false, d),
                (a.tmp, d),
                (rows, d, f),
                PLAIN,
            )?;
            self.elementwise(
                "bias_add",
                rows * d,
                &[P(a.tmp), P(b2), L(long(rows)), I(int(d)?)],
            )?;
            self.elementwise(
                "add2",
                rows * d,
                &[P(xmid), P(a.tmp), P(next), L(long(rows * d))],
            )?;
        }
        let last = at(a.x, c.layers * a.sd);
        self.layernorm(
            last,
            self.layout.lnf_g,
            self.layout.lnf_b,
            a.lnf,
            a.meanf,
            a.rstdf,
            rows,
        )?;
        self.gemm(
            (a.lnf, false, d),
            (at(self.w, self.layout.head), false, vocab),
            (a.logits, vocab),
            (rows, vocab, d),
            PLAIN,
        )?;
        let (tgt, wt, losses) = (
            self.tgt.device_ptr(&self.stream).0,
            self.wt.device_ptr(&self.stream).0,
            self.losses.device_ptr(&self.stream).0,
        );
        self.run(
            "ce_fwd_bwd",
            (rows as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                P(a.logits),
                P(tgt),
                P(wt),
                P(losses),
                I(int(vocab)?),
                I(grad as i32),
            ],
        )?;
        let mut per_row = vec![0.0f32; rows];
        self.stream
            .memcpy_dtoh(&self.losses.slice(0..rows), &mut per_row)
            .map_err(error)?;
        Ok(per_row.iter().map(|&x| f64::from(x)).sum())
    }

    fn backward(&mut self, b: usize, t: usize) -> Result<()> {
        let c = self.config;
        let (d, h, vocab) = (c.embed_dim, c.heads, c.vocab_size);
        let (dh, f, rows) = (d / h, 4 * d, b * t);
        let scale = 1.0 / (dh as f32).sqrt();
        let a = self.a;
        let (w, g) = (self.w, self.g);
        let head = self.layout.head;
        // logits já viraram dlogits no ce_fwd_bwd
        self.gemm(
            (a.logits, false, vocab),
            (at(w, head), true, vocab),
            (a.dh, d),
            (rows, d, vocab),
            PLAIN,
        )?;
        self.gemm(
            (a.lnf, true, d),
            (a.logits, false, vocab),
            (at(g, head), vocab),
            (d, vocab, rows),
            ACC,
        )?;
        self.elementwise("fill0", rows * d, &[P(a.dres), L(long(rows * d))])?;
        let last = at(a.x, c.layers * a.sd);
        self.layernorm_backward(
            a.dh,
            last,
            a.meanf,
            a.rstdf,
            self.layout.lnf_g,
            self.layout.lnf_b,
            rows,
        )?;
        for l in (0..c.layers).rev() {
            let s = self.layout.layers[l];
            let x = at(a.x, l * a.sd);
            let (ln1, mean1, rstd1) = (
                at(a.ln1, l * a.sd),
                at(a.mean1, l * a.sr),
                at(a.rstd1, l * a.sr),
            );
            let (qkv, att, atty) = (
                at(a.qkv, l * a.sq),
                at(a.att, l * a.sa),
                at(a.atty, l * a.sd),
            );
            let xmid = at(a.xmid, l * a.sd);
            let (ln2, mean2, rstd2) = (
                at(a.ln2, l * a.sd),
                at(a.mean2, l * a.sr),
                at(a.rstd2, l * a.sr),
            );
            let (f1, gl) = (at(a.f1, l * a.sf), at(a.gl, l * a.sf));

            // MLP: dres é dL/dx da saída da camada. Fundido, dF1 = (dres·W2ᵀ)·gelu'(f1) sai
            // direto do epílogo; sem fusão, o gelu' é uma passada a mais.
            let dgelu = if self.options.fused {
                Epi {
                    pre: f1,
                    mode: 2,
                    ..PLAIN
                }
            } else {
                PLAIN
            };
            self.gemm(
                (a.dres, false, d),
                (at(w, s.w2), true, d),
                (a.dbig, f),
                (rows, f, d),
                dgelu,
            )?;
            self.gemm(
                (gl, true, f),
                (a.dres, false, d),
                (at(g, s.w2), d),
                (f, d, rows),
                ACC,
            )?;
            self.colsum(a.dres, at(g, s.b2), rows, d)?;
            if !self.options.fused {
                self.elementwise("gelu_bwd", rows * f, &[P(a.dbig), P(f1), L(long(rows * f))])?;
            }
            self.gemm(
                (ln2, true, d),
                (a.dbig, false, f),
                (at(g, s.w1), f),
                (d, f, rows),
                ACC,
            )?;
            self.colsum(a.dbig, at(g, s.b1), rows, f)?;
            self.gemm(
                (a.dbig, false, f),
                (at(w, s.w1), true, f),
                (a.dh, d),
                (rows, d, f),
                PLAIN,
            )?;
            self.layernorm_backward(a.dh, xmid, mean2, rstd2, s.ln2_g, s.ln2_b, rows)?;
            // atenção: dres agora é dL/dxmid
            self.gemm(
                (atty, true, d),
                (a.dres, false, d),
                (at(g, s.wo), d),
                (d, d, rows),
                ACC,
            )?;
            self.gemm(
                (a.dres, false, d),
                (at(w, s.wo), true, d),
                (a.datty, d),
                (rows, d, d),
                PLAIN,
            )?;
            self.run(
                "attn_bwd_rows",
                (t as u32, h as u32, b as u32),
                (128, 1, 1),
                (32 + dh + 2 * t) * 4,
                &[
                    P(qkv),
                    P(att),
                    P(a.datty),
                    P(a.ds),
                    P(a.dqkv),
                    I(int(t)?),
                    I(int(h)?),
                    I(int(dh)?),
                    F(scale),
                ],
            )?;
            self.run(
                "attn_bwd_cols",
                (t as u32, h as u32, b as u32),
                (64, 1, 1),
                0,
                &[
                    P(qkv),
                    P(att),
                    P(a.ds),
                    P(a.datty),
                    P(a.dqkv),
                    I(int(t)?),
                    I(int(h)?),
                    I(int(dh)?),
                    F(scale),
                ],
            )?;
            self.gemm(
                (ln1, true, d),
                (a.dqkv, false, 3 * d),
                (at(g, s.wqkv), 3 * d),
                (d, 3 * d, rows),
                ACC,
            )?;
            self.gemm(
                (a.dqkv, false, 3 * d),
                (at(w, s.wqkv), true, 3 * d),
                (a.dh, d),
                (rows, d, 3 * d),
                PLAIN,
            )?;
            self.layernorm_backward(a.dh, x, mean1, rstd1, s.ln1_g, s.ln1_b, rows)?;
        }
        let tok = self.tok.device_ptr(&self.stream).0;
        let (dwte, dwpe) = (at(g, self.layout.wte), at(g, self.layout.wpe));
        self.elementwise(
            "embed_bwd",
            rows * d,
            &[
                P(a.dres),
                P(tok),
                P(dwte),
                P(dwpe),
                I(int(rows)?),
                I(int(t)?),
                I(int(d)?),
            ],
        )
    }

    /// Norma L2 dos gradientes acumulados (em f64) e quantos não são finitos.
    pub fn grad_stats(&mut self) -> Result<(f64, usize)> {
        let total = self.layout.total;
        self.stream
            .memcpy_htod(&[0u32], &mut self.nonfinite)
            .map_err(error)?;
        let (partial, nonfinite) = (
            self.partial.device_ptr(&self.stream).0,
            self.nonfinite.device_ptr(&self.stream).0,
        );
        self.run(
            "sumsq",
            (SUMSQ_BLOCKS as u32, 1, 1),
            (256, 1, 1),
            0,
            &[P(self.g), L(long(total)), P(partial), P(nonfinite)],
        )?;
        let mut sums = vec![0.0f64; SUMSQ_BLOCKS];
        self.stream
            .memcpy_dtoh(&self.partial, &mut sums)
            .map_err(error)?;
        let mut count = [0u32];
        self.stream
            .memcpy_dtoh(&self.nonfinite, &mut count)
            .map_err(error)?;
        Ok((sums.iter().sum::<f64>().sqrt(), count[0] as usize))
    }

    /// Um passo do Adam sobre a arena inteira, com as constantes do `Adam` do host; zera os
    /// gradientes. Devolve quantos pesos ficaram não finitos.
    pub fn adam_step(&mut self) -> Result<usize> {
        let (lr, beta1, beta2, eps, step) = self.hyper;
        let step = step.saturating_add(1);
        self.hyper.4 = step;
        let (c1, c2) = (1.0 - beta1.powi(step), 1.0 - beta2.powi(step));
        self.stream
            .memcpy_htod(&[0u32], &mut self.nonfinite)
            .map_err(error)?;
        let nonfinite = self.nonfinite.device_ptr(&self.stream).0;
        let total = self.layout.total;
        let args = [
            P(self.w),
            P(self.g),
            P(self.pm),
            P(self.pv),
            L(long(total)),
            F(beta1),
            F(beta2),
            F(lr),
            F(eps),
            F(c1),
            F(c2),
            P(nonfinite),
        ];
        self.elementwise("adam", total, &args)?;
        let mut count = [0u32];
        self.stream
            .memcpy_dtoh(&self.nonfinite, &mut count)
            .map_err(error)?;
        Ok(count[0] as usize)
    }

    /// Pesos atuais nos shapes do checkpoint.
    pub fn params(&self) -> Result<Vec<Tensor>> {
        let arena = self.stream.clone_dtoh(&self.weights).map_err(error)?;
        self.layout.unpack_params(&arena)
    }

    /// Gradientes acumulados, um buffer por tensor do checkpoint (para testes e diagnóstico).
    pub fn grads(&self) -> Result<Vec<Vec<f32>>> {
        let arena = self.stream.clone_dtoh(&self.grads).map_err(error)?;
        self.layout.unpack(&arena)
    }

    /// Estado do Adam no formato do host (para checkpoint).
    pub fn adam(&self) -> Result<Adam> {
        if self.hyper.4 == 0 {
            return Adam::from_state(self.hyper, Vec::new());
        }
        let m = self
            .layout
            .unpack(&self.stream.clone_dtoh(&self.m).map_err(error)?)?;
        let v = self
            .layout
            .unpack(&self.stream.clone_dtoh(&self.v).map_err(error)?)?;
        Adam::from_state(self.hyper, m.into_iter().zip(v).collect())
    }
}
