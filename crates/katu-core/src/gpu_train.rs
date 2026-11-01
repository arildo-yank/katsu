//! Treino do Katu LM inteiro na GPU: pesos, gradientes, momentos do Adam e ativações ficam
//! na VRAM, e um passo não copia nada entre host e dispositivo além dos tokens de entrada e
//! das perdas por linha (alguns KB). Forward e backward são explícitos para a arquitetura do
//! `KatuLmConfig` (sem grafo dinâmico), com as mesmas fórmulas do backend CPU; a paridade é
//! conferida em `tests/gpu_train.rs`. O checkpoint continua no formato do host
//! (`GpuLayout` faz a permutação de ida e volta).
//!
//! V2 (opções, todas desligadas por padrão): GEMM em BF16 nos Tensor Cores, Adam com momentos
//! de 8 bits (o formato de `adam8.rs`, com blocos sobre a arena) e recomputação de ativações
//! (guarda só a entrada de cada camada e refaz o forward dela no backward).
use crate::{
    adam8::State as Adam8State,
    gpu_layout::{GpuLayout, SftBatch},
    Adam, KatuLmConfig, Tensor, TensorError, ADAM8_BLOCK,
};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
        PushKernelArg,
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
    /// V2: Tensor Cores (WMMA) em BF16, acumulação em FP32, com a mistura do `Bf16Backend`:
    /// ativações e gradientes de ativação voltam a BF16, gradientes de peso ficam FP32. Os
    /// buffers continuam FP32 (a conversão é na carga): isto muda a conta, não a memória.
    Bf16,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuTrainOptions {
    pub gemm: GemmKernel,
    /// Etapa 2: bias, GELU e resíduo no epílogo do GEMM, e gelu' no GEMM do backward, em vez
    /// de passadas separadas pela memória.
    pub fused: bool,
    /// V2: momentos do Adam em 8 bits por blocos de `ADAM8_BLOCK` (~2,03 bytes/parâmetro em vez
    /// de 8), a mesma conta do `Adam8bit` da CPU.
    pub adam8: bool,
    /// V2: ativações de uma camada só; o backward refaz o forward de cada camada (um forward
    /// a mais por passo).
    pub recompute: bool,
}

impl Default for GpuTrainOptions {
    fn default() -> Self {
        Self {
            gemm: GemmKernel::Reference16,
            fused: true,
            adam8: false,
            recompute: false,
        }
    }
}

/// Estado do Adam na VRAM.
enum Moments {
    Fp32 {
        m: CudaSlice<f32>,
        v: CudaSlice<f32>,
    },
    /// Formato do `Adam8bit`: um byte por momento e uma escala FP32 por bloco.
    Int8 {
        m: CudaSlice<i8>,
        v: CudaSlice<u8>,
        m_scale: CudaSlice<f32>,
        v_scale: CudaSlice<f32>,
    },
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
    "prefix_fwd",
    "embed_bwd_prefix",
    "adam_masked",
    "adam8",
];
const SUMSQ_BLOCKS: usize = 1024;
// o kernel adam8 usa um bloco de 256 threads por bloco de quantização
const _: () = assert!(ADAM8_BLOCK == 256);

/// Compila train.cu (e o GEMM de Tensor Cores pedido) para a GPU do contexto.
fn build_module(ctx: &Arc<CudaContext>, gemm: GemmKernel) -> Result<Arc<CudaModule>> {
    let (major, minor) = ctx.compute_capability().map_err(error)?;
    // Os GEMMs de Tensor Cores precisam de mma.h (dos headers do CUDA) e só entram quando pedidos.
    let tensor_cores = matches!(gemm, GemmKernel::Tf32 | GemmKernel::Bf16);
    if tensor_cores && major < 8 {
        return Err(TensorError::BackendUnavailable(
            "TF32/BF16 Tensor Cores need sm_80 or newer",
        ));
    }
    let mut source = include_str!("cuda/train.cu").to_owned();
    match gemm {
        GemmKernel::Tf32 => source.push_str(include_str!("cuda/train_tc.cu")),
        GemmKernel::Bf16 => source.push_str(include_str!("cuda/train_bf16.cu")),
        _ => {}
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
    ctx.load_module(ptx).map_err(error)
}

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

/// Endereços das ativações de uma camada (ver `GpuTrainer::layer_acts`).
#[derive(Debug, Clone, Copy)]
struct LayerActs {
    x: u64,
    ln1: u64,
    mean1: u64,
    rstd1: u64,
    qkv: u64,
    att: u64,
    atty: u64,
    xmid: u64,
    ln2: u64,
    mean2: u64,
    rstd2: u64,
    f1: u64,
    gl: u64,
    next: u64,
}

pub struct GpuTrainer {
    config: KatuLmConfig,
    layout: GpuLayout,
    max_batch: usize,
    max_t: usize,
    options: GpuTrainOptions,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernels: HashMap<&'static str, CudaFunction>,
    weights: CudaSlice<f32>,
    grads: CudaSlice<f32>,
    moments: Moments,
    _act: CudaSlice<f32>,
    tok: CudaSlice<i32>,
    tgt: CudaSlice<i32>,
    wt: CudaSlice<f32>,
    losses: CudaSlice<f32>,
    partial: CudaSlice<f64>,
    nonfinite: CudaSlice<u32>,
    w: u64,
    g: u64,
    a: Acts,
    hyper: (f32, f32, f32, f32, i32),
    launches: u64,
    /// Arandu-Ve: vetores do prefixo e o gradiente deles, alocados no primeiro lote com prefixo.
    pre: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    /// (prefix_len, batch) do último lote enviado.
    prefix: (usize, usize),
    /// Máscara do Adam (1 = treinável), por peso da arena; `None` = nada congelado.
    trainable: Option<CudaSlice<u8>>,
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
        config: KatuLmConfig,
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

    /// Floats de ativação para o lote máximo (sem contar pesos, gradientes e Adam). Com
    /// `recompute`, só a entrada de cada camada fica guardada; o resto existe para uma camada.
    pub fn activation_floats(
        config: &KatuLmConfig,
        max_batch: usize,
        max_t: usize,
        recompute: bool,
    ) -> usize {
        let (layers, d, h, v) = (
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
        let l = if recompute { 1 } else { layers };
        (layers + 1) * sd            // x
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
        config: KatuLmConfig,
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
        let module = build_module(&ctx, options.gemm)?;
        let mut kernels = HashMap::new();
        for &name in KERNELS {
            kernels.insert(name, module.load_function(name).map_err(error)?);
        }
        let gemm = match options.gemm {
            GemmKernel::Tf32 => Some("gemm_tc"),
            GemmKernel::Bf16 => Some("gemm_bf16"),
            _ => None,
        };
        if let Some(name) = gemm {
            kernels.insert(name, module.load_function(name).map_err(error)?);
        }

        let total = layout.total;
        let act_floats = Self::activation_floats(&config, max_batch, max_t, options.recompute);
        let rows = max_batch * max_t;
        let blocks = total.div_ceil(ADAM8_BLOCK);
        // pesos e gradientes FP32, mais o Adam (8 bytes/parâmetro, ou 2 + 8 por bloco)
        let adam_bytes = if options.adam8 {
            2 * total + 8 * blocks
        } else {
            8 * total
        };
        let need = 4 * (2 * total + act_floats + 2 * rows) + adam_bytes + 8 * SUMSQ_BLOCKS;
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
        let moments = if options.adam8 {
            Moments::Int8 {
                m: stream.alloc_zeros::<i8>(total).map_err(error)?,
                v: stream.alloc_zeros::<u8>(total).map_err(error)?,
                m_scale: stream.alloc_zeros::<f32>(blocks).map_err(error)?,
                v_scale: stream.alloc_zeros::<f32>(blocks).map_err(error)?,
            }
        } else {
            Moments::Fp32 {
                m: stream.alloc_zeros::<f32>(total).map_err(error)?,
                v: stream.alloc_zeros::<f32>(total).map_err(error)?,
            }
        };
        let act = stream.alloc_zeros::<f32>(act_floats).map_err(error)?;
        let tok = stream.alloc_zeros::<i32>(rows).map_err(error)?;
        let tgt = stream.alloc_zeros::<i32>(rows).map_err(error)?;
        let wt = stream.alloc_zeros::<f32>(rows).map_err(error)?;
        let losses = stream.alloc_zeros::<f32>(rows).map_err(error)?;
        let partial = stream.alloc_zeros::<f64>(SUMSQ_BLOCKS).map_err(error)?;
        let nonfinite = stream.alloc_zeros::<u32>(1).map_err(error)?;

        let ptr = |s: &CudaSlice<f32>| s.device_ptr(&stream).0;
        let (w, g, base) = (ptr(&weights), ptr(&grads), ptr(&act));

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
        // por camada, ou uma camada só com recomputação
        let k = if options.recompute { 1 } else { l };
        let mut next = 0;
        let mut take = |n: usize| {
            let p = at(base, next);
            next += n;
            p
        };
        let a = Acts {
            x: take((l + 1) * sd),
            xmid: take(k * sd),
            ln1: take(k * sd),
            ln2: take(k * sd),
            atty: take(k * sd),
            mean1: take(k * sr),
            rstd1: take(k * sr),
            mean2: take(k * sr),
            rstd2: take(k * sr),
            qkv: take(k * sq),
            att: take(k * sa),
            f1: take(k * sf),
            gl: take(k * sf),
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
            moments,
            _act: act,
            tok,
            tgt,
            wt,
            losses,
            partial,
            nonfinite,
            w,
            g,
            a,
            hyper: adam.hyperparameters(),
            launches: 0,
            pre: None,
            prefix: (0, 0),
            trainable: None,
        };
        trainer.load_adam(adam)?;
        Ok(trainer)
    }

    /// Troca o estado do Adam (hiperparâmetros, passo e momentos), ex.: ao passar da fase de
    /// adaptação ao corpus para o SFT, que começa com um Adam novo.
    /// Com `adam8`, momentos FP32 são quantizados no host (escala no máximo de cada bloco).
    pub fn load_adam(&mut self, adam: &Adam) -> Result<()> {
        let moments = adam.moments();
        let stream = self.stream.clone();
        self.hyper = adam.hyperparameters();
        if moments.is_empty() {
            match &mut self.moments {
                Moments::Fp32 { m, v } => {
                    stream.memset_zeros(m).map_err(error)?;
                    stream.memset_zeros(v).map_err(error)?;
                }
                Moments::Int8 {
                    m,
                    v,
                    m_scale,
                    v_scale,
                } => {
                    stream.memset_zeros(m).map_err(error)?;
                    stream.memset_zeros(v).map_err(error)?;
                    stream.memset_zeros(m_scale).map_err(error)?;
                    stream.memset_zeros(v_scale).map_err(error)?;
                }
            }
            return Ok(());
        }
        let ms: Vec<&[f32]> = moments.iter().map(|(m, _)| m.as_slice()).collect();
        let vs: Vec<&[f32]> = moments.iter().map(|(_, v)| v.as_slice()).collect();
        let (m, v) = (self.layout.pack(&ms)?, self.layout.pack(&vs)?);
        match &mut self.moments {
            Moments::Fp32 { m: dm, v: dv } => {
                stream.memcpy_htod(&m, dm).map_err(error)?;
                stream.memcpy_htod(&v, dv).map_err(error)?;
            }
            Moments::Int8 {
                m: dm,
                v: dv,
                m_scale,
                v_scale,
            } => {
                let q = Adam8State::quantize(&m, &v)?;
                stream.memcpy_htod(&q.m, dm).map_err(error)?;
                stream.memcpy_htod(&q.v, dv).map_err(error)?;
                stream.memcpy_htod(&q.m_scale, m_scale).map_err(error)?;
                stream.memcpy_htod(&q.v_scale, v_scale).map_err(error)?;
            }
        }
        Ok(())
    }

    pub fn config(&self) -> &KatuLmConfig {
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
            GemmKernel::Bf16 => self.run("gemm_bf16", tiles(64), (128, 1, 1), 0, &args),
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
        let n = batch.prefix_len;
        if n > 0 {
            let d = self.config.embed_dim;
            if n >= t || batch.prefix.len() != b * n * d {
                return Err(TensorError::InvalidShape(
                    "prefix must be batch × prefix_len × d, shorter than the width",
                ));
            }
            if batch.prefix.iter().any(|x| !x.is_finite()) {
                return Err(TensorError::InvalidShape("prefix vectors must be finite"));
            }
            if self.pre.is_none() {
                let cap = self.max_batch * self.max_t * d;
                let pre = self.stream.alloc_zeros::<f32>(cap).map_err(error)?;
                let dpre = self.stream.alloc_zeros::<f32>(cap).map_err(error)?;
                self.pre = Some((pre, dpre));
            }
            let (pre, _) = self.pre.as_mut().expect("alocado acima");
            self.stream
                .memcpy_htod(&batch.prefix, &mut pre.slice_mut(0..b * n * d))
                .map_err(error)?;
        }
        self.prefix = (n, b);
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
        let (d, vocab, rows) = (c.embed_dim, c.vocab_size, b * t);
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
        if let (n @ 1.., Some((pre, _))) = (self.prefix.0, &self.pre) {
            let pre = pre.device_ptr(&self.stream).0;
            self.elementwise(
                "prefix_fwd",
                rows * d,
                &[
                    P(pre),
                    P(wpe),
                    P(a.x),
                    I(int(rows)?),
                    I(int(t)?),
                    I(int(n)?),
                    I(int(d)?),
                ],
            )?;
        }
        for l in 0..c.layers {
            self.layer_forward(l, b, t)?;
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
        let (d, vocab, rows) = (c.embed_dim, c.vocab_size, b * t);
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
            if self.options.recompute {
                self.layer_forward(l, b, t)?;
            }
            self.layer_backward(l, b, t)?;
        }
        let tok = self.tok.device_ptr(&self.stream).0;
        let (dwte, dwpe) = (at(g, self.layout.wte), at(g, self.layout.wpe));
        if let (n @ 1.., Some((_, dpre))) = (self.prefix.0, &self.pre) {
            let dpre = dpre.device_ptr(&self.stream).0;
            return self.elementwise(
                "embed_bwd_prefix",
                rows * d,
                &[
                    P(a.dres),
                    P(tok),
                    P(dwte),
                    P(dwpe),
                    P(dpre),
                    I(int(rows)?),
                    I(int(t)?),
                    I(int(n)?),
                    I(int(d)?),
                ],
            );
        }
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

    /// Ponteiros das ativações da camada `l`. Com recomputação só `x` e `next` são por camada;
    /// o resto é o buffer da camada única, refeito antes do backward de cada camada.
    fn layer_acts(&self, l: usize) -> LayerActs {
        let a = self.a;
        let k = if self.options.recompute { 0 } else { l };
        LayerActs {
            x: at(a.x, l * a.sd),
            ln1: at(a.ln1, k * a.sd),
            mean1: at(a.mean1, k * a.sr),
            rstd1: at(a.rstd1, k * a.sr),
            qkv: at(a.qkv, k * a.sq),
            att: at(a.att, k * a.sa),
            atty: at(a.atty, k * a.sd),
            xmid: at(a.xmid, k * a.sd),
            ln2: at(a.ln2, k * a.sd),
            mean2: at(a.mean2, k * a.sr),
            rstd2: at(a.rstd2, k * a.sr),
            f1: at(a.f1, k * a.sf),
            gl: at(a.gl, k * a.sf),
            next: at(a.x, (l + 1) * a.sd),
        }
    }

    /// Forward de uma camada: lê `x` da camada `l` e escreve o da `l + 1`.
    fn layer_forward(&mut self, l: usize, b: usize, t: usize) -> Result<()> {
        let c = self.config;
        let (d, h) = (c.embed_dim, c.heads);
        let (dh, f, rows) = (d / h, 4 * d, b * t);
        let scale = 1.0 / (dh as f32).sqrt();
        let s = self.layout.layers[l];
        let LayerActs {
            x,
            ln1,
            mean1,
            rstd1,
            qkv,
            att,
            atty,
            xmid,
            ln2,
            mean2,
            rstd2,
            f1,
            gl,
            next,
        } = self.layer_acts(l);
        let a = self.a;

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
            return Ok(());
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
        Ok(())
    }

    /// Backward de uma camada: `a.dres` entra como dL/d(saída) e sai como dL/d(entrada).
    fn layer_backward(&mut self, l: usize, b: usize, t: usize) -> Result<()> {
        let c = self.config;
        let (d, h) = (c.embed_dim, c.heads);
        let (dh, f, rows) = (d / h, 4 * d, b * t);
        let scale = 1.0 / (dh as f32).sqrt();
        let s = self.layout.layers[l];
        let LayerActs {
            x,
            ln1,
            mean1,
            rstd1,
            qkv,
            att,
            atty,
            xmid,
            ln2,
            mean2,
            rstd2,
            f1,
            gl,
            next: _,
        } = self.layer_acts(l);
        let a = self.a;
        let (w, g) = (self.w, self.g);

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
        Ok(())
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
        let stream = &self.stream;
        let nonfinite = self.nonfinite.device_ptr(stream).0;
        let total = self.layout.total;
        let mask = self.trainable.as_ref().map(|m| m.device_ptr(stream).0);
        let (w, g) = (self.w, self.g);
        let hyper = [
            F(beta1),
            F(beta2),
            F(lr),
            F(eps),
            F(c1),
            F(c2),
            P(nonfinite),
        ];
        match &self.moments {
            Moments::Int8 {
                m,
                v,
                m_scale,
                v_scale,
            } => {
                let mut args = vec![
                    P(w),
                    P(g),
                    P(m.device_ptr(stream).0),
                    P(v.device_ptr(stream).0),
                    P(m_scale.device_ptr(stream).0),
                    P(v_scale.device_ptr(stream).0),
                    P(mask.unwrap_or(0)),
                    L(long(total)),
                ];
                args.extend(hyper);
                let blocks = total.div_ceil(ADAM8_BLOCK) as u32;
                self.run("adam8", (blocks, 1, 1), (256, 1, 1), 0, &args)?;
            }
            Moments::Fp32 { m, v } => {
                let mut args = vec![
                    P(w),
                    P(g),
                    P(m.device_ptr(stream).0),
                    P(v.device_ptr(stream).0),
                ];
                if let Some(mask) = mask {
                    args.push(P(mask));
                }
                args.push(L(long(total)));
                args.extend(hyper);
                let name = if mask.is_some() {
                    "adam_masked"
                } else {
                    "adam"
                };
                self.elementwise(name, total, &args)?;
            }
        }
        let mut count = [0u32];
        self.stream
            .memcpy_dtoh(&self.nonfinite, &mut count)
            .map_err(error)?;
        Ok(count[0] as usize)
    }

    /// Gradiente da perda em relação aos vetores do prefixo do último `loss_and_grad`,
    /// `[batch·prefix_len·d]`, na ordem de `SftBatch::with_prefix`. O projetor (no host)
    /// continua o backward a partir daqui.
    pub fn prefix_grad(&self) -> Result<Vec<f32>> {
        let (n, b) = self.prefix;
        let Some((_, dpre)) = (n > 0).then_some(()).and(self.pre.as_ref()) else {
            return Err(TensorError::InvalidShape("last batch had no prefix"));
        };
        let mut all = self.stream.clone_dtoh(dpre).map_err(error)?;
        all.truncate(b * n * self.config.embed_dim);
        Ok(all)
    }

    /// Congela tensores do checkpoint (`frozen[i]`, na ordem de `params()`): o Adam deixa os
    /// pesos deles intactos e não acumula momento, e o gradiente deles é zerado a cada passo.
    /// Q, K e V são fatias intercaladas de uma matriz só; a máscara segue cada slot.
    pub fn set_frozen(&mut self, frozen: &[bool]) -> Result<()> {
        if frozen.len() != self.layout.slots.len() {
            return Err(TensorError::InvalidShape(
                "frozen mask must have one entry per checkpoint tensor",
            ));
        }
        if !frozen.iter().any(|&f| f) {
            self.trainable = None;
            return Ok(());
        }
        let mut mask = vec![1u8; self.layout.total];
        for (slot, _) in self.layout.slots.iter().zip(frozen).filter(|(_, &f)| f) {
            for r in 0..slot.rows {
                let at = slot.offset + r * slot.ld;
                mask[at..at + slot.cols].fill(0);
            }
        }
        self.trainable = Some(self.stream.clone_htod(&mask).map_err(error)?);
        Ok(())
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

    /// Estado do Adam no formato do host (para checkpoint). Com `adam8`, os momentos saem
    /// dequantizados para FP32.
    pub fn adam(&self) -> Result<Adam> {
        if self.hyper.4 == 0 {
            return Adam::from_state(self.hyper, Vec::new());
        }
        let host = |x: &CudaSlice<f32>| self.stream.clone_dtoh(x).map_err(error);
        let (m, v) = match &self.moments {
            Moments::Fp32 { m, v } => (host(m)?, host(v)?),
            Moments::Int8 {
                m,
                v,
                m_scale,
                v_scale,
            } => Adam8State {
                m: self.stream.clone_dtoh(m).map_err(error)?,
                v: self.stream.clone_dtoh(v).map_err(error)?,
                m_scale: host(m_scale)?,
                v_scale: host(v_scale)?,
            }
            .dequantize(),
        };
        let (m, v) = (self.layout.unpack(&m)?, self.layout.unpack(&v)?);
        Adam::from_state(self.hyper, m.into_iter().zip(v).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adam8, AdamStep};

    /// O kernel `adam8` sobre um vetor de 3 blocos e 1/3, 6 passos com gradientes variados,
    /// contra `adam8::apply` da CPU: pesos, momentos e escalas iguais bit a bit.
    #[test]
    fn adam8_kernel_is_bit_identical_to_cpu() {
        if !cfg!(target_os = "linux") {
            return;
        }
        let ctx = match std::panic::catch_unwind(|| CudaContext::new(0)) {
            Ok(Ok(ctx)) => ctx,
            _ => {
                eprintln!("sem GPU, teste pulado");
                return;
            }
        };
        let stream = ctx.default_stream();
        let module = build_module(&ctx, GemmKernel::Reference16).unwrap();
        let kernel = module.load_function("adam8").unwrap();
        let n = 3 * ADAM8_BLOCK + 85;
        let blocks = n.div_ceil(ADAM8_BLOCK);
        let wave = |seed: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7919 + seed * 104_729) % 2001) as f32 / 1000.0 - 1.0)
                .map(|x| x * scale)
                .collect()
        };
        let mut w_cpu = wave(0, 0.1);
        let mut state = adam8::State::zeros(n).unwrap();
        let w = stream.clone_htod(&w_cpu).unwrap();
        let mut g = stream.alloc_zeros::<f32>(n).unwrap();
        let qm = stream.alloc_zeros::<i8>(n).unwrap();
        let qv = stream.alloc_zeros::<u8>(n).unwrap();
        let sm = stream.alloc_zeros::<f32>(blocks).unwrap();
        let sv = stream.alloc_zeros::<f32>(blocks).unwrap();
        let bad = stream.alloc_zeros::<u32>(1).unwrap();
        let (lr, beta1, beta2, eps) = (1e-3f32, 0.9f32, 0.999f32, 1e-8f32);
        for step in 1..=6 {
            // um bloco com gradientes mil vezes menores que os outros
            let mut grad = wave(step, 1e-2);
            grad[ADAM8_BLOCK..2 * ADAM8_BLOCK]
                .iter_mut()
                .for_each(|x| *x *= 1e-3);
            let (c1, c2) = (1.0 - beta1.powi(step as i32), 1.0 - beta2.powi(step as i32));
            let c = AdamStep {
                beta1,
                beta2,
                lr,
                eps,
                correction1: c1,
                correction2: c2,
            };
            adam8::apply(c, &mut w_cpu, &grad, &mut state);
            stream.memcpy_htod(&grad, &mut g).unwrap();
            let (pw, pg) = (w.device_ptr(&stream).0, g.device_ptr(&stream).0);
            let (pm, pv) = (qm.device_ptr(&stream).0, qv.device_ptr(&stream).0);
            let (ps, pu) = (sm.device_ptr(&stream).0, sv.device_ptr(&stream).0);
            let pbad = bad.device_ptr(&stream).0;
            let (none, count) = (0u64, n as i64);
            let mut launch = stream.launch_builder(&kernel);
            launch
                .arg(&pw)
                .arg(&pg)
                .arg(&pm)
                .arg(&pv)
                .arg(&ps)
                .arg(&pu)
                .arg(&none)
                .arg(&count)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&lr)
                .arg(&eps)
                .arg(&c1)
                .arg(&c2)
                .arg(&pbad);
            // SAFETY: argumentos na ordem e nos tipos de `adam8` em train.cu; buffers de n
            // posições e `blocks` escalas.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (blocks as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        }
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&stream.clone_dtoh(&w).unwrap()), bits(&w_cpu), "pesos");
        assert_eq!(stream.clone_dtoh(&qm).unwrap(), state.m, "m quantizado");
        assert_eq!(stream.clone_dtoh(&qv).unwrap(), state.v, "v quantizado");
        assert_eq!(bits(&stream.clone_dtoh(&sm).unwrap()), bits(&state.m_scale));
        assert_eq!(bits(&stream.clone_dtoh(&sv).unwrap()), bits(&state.v_scale));
        assert!(stream.clone_dtoh(&g).unwrap().iter().all(|&x| x == 0.0));
    }
}
