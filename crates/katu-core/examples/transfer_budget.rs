//! Bytes que um passo de treino cruzaria entre host e GPU, no caminho antigo (cada chamada do
//! backend sobe as entradas e desce as saídas, como o `CudaBackend` híbrido) e no residente
//! (pesos, gradientes e Adam no store). Roda na CPU: mede o tráfego, não o tempo da GPU.
//!
//!   cargo run --release -p katu-core --example transfer_budget -- [layers] [batch] [tokens]
//!
//! Padrão: o 134M (32k de vocabulário, 768 de largura, 12 camadas, 12 cabeças), batch 4 × 256.
use katu_core::{
    arena_len, upload, Adam, Backend, HostParamStore, KatuLmConfig, ParamStore, SplitMix64, Tensor,
    TensorError,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

type Result<T> = std::result::Result<T, TensorError>;

/// Repassa ao CPU e anota, por categoria, os bytes que o `CudaBackend` copiaria.
#[derive(Debug)]
struct Counting {
    inner: Arc<dyn Backend>,
    dims: (usize, usize, usize), // (d, dh, t)
    bytes: Mutex<BTreeMap<&'static str, (u64, u64)>>,
}

impl Counting {
    fn kind(&self, m: usize, k: usize, n: usize) -> &'static str {
        let (d, dh, t) = self.dims;
        match (k, n) {
            (k, n) if k == d && n == dh => "Q/K/V por cabeça",
            (k, n) if (k == dh && n == t) || (k == t && n == dh) => "atenção (scores, valores)",
            (k, n) if k == d && n == d => "projeção Wo",
            (k, n) if (k == d && n == 4 * d) || (k == 4 * d && n == d) => "MLP",
            _ if m == 0 => "vazio",
            _ => "cabeça de saída",
        }
    }
    fn note(&self, kind: &'static str, up: usize, down: usize) {
        let mut bytes = self.bytes.lock().unwrap();
        let entry = bytes.entry(kind).or_default();
        entry.0 += 4 * up as u64;
        entry.1 += 4 * down as u64;
    }
}

impl Backend for Counting {
    fn name(&self) -> &'static str {
        "counting"
    }
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
        self.note(self.kind(m, k, n), a.len() + b.len(), m * n);
        self.inner.matmul(a, b, m, k, n)
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
        self.note(
            self.kind(m, k, n),
            a.len() + b.len() + g.len(),
            a.len() + b.len(),
        );
        self.inner.matmul_backward(a, b, g, m, k, n)
    }
    fn add(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        self.note("soma residual", a.len() + b.len(), a.len());
        self.inner.add(a, b)
    }
    fn add_backward(
        &self,
        g: &[f32],
        l: &[usize],
        r: &[usize],
        ll: usize,
        rl: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Os mapas de índices sobem como u64 (2 floats cada).
        self.note("soma residual", g.len() + 2 * (l.len() + r.len()), ll + rl);
        self.inner.add_backward(g, l, r, ll, rl)
    }
    fn cross_entropy(&self, x: &[f32], t: &[usize], c: usize) -> Result<f32> {
        self.note("entropia cruzada", x.len() + 2 * t.len(), t.len());
        self.inner.cross_entropy(x, t, c)
    }
    fn cross_entropy_backward(&self, x: &[f32], t: &[usize], c: usize, s: f32) -> Result<Vec<f32>> {
        self.note("entropia cruzada", x.len() + 2 * t.len(), x.len());
        self.inner.cross_entropy_backward(x, t, c, s)
    }
}

fn fast_backend() -> Arc<dyn Backend> {
    #[cfg(target_os = "macos")]
    {
        Arc::new(katu_core::AccelerateBackend::default())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Arc::new(katu_core::CpuBackend::default())
    }
}

fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

fn step(
    c: &KatuLmConfig,
    params: Vec<Tensor>,
    windows: &[Vec<usize>],
    backend: Arc<dyn Backend>,
    store: Option<&HostParamStore>,
) -> Result<()> {
    let refs: Vec<&[usize]> = windows.iter().map(Vec::as_slice).collect();
    let loss = c.loss_batch_with_backend(&params, &refs, backend)?;
    eprintln!("  loss {:.4}", loss.data()[0]);
    loss.backward_release()?;
    let mut adam = Adam::new(1e-4)?;
    match store {
        Some(store) => drop(adam.step_resident(params, store)?),
        None => drop(katu_core::Optimizer::step_owned(&mut adam, params)?),
    }
    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse())
        .collect::<std::result::Result<_, _>>()?;
    let (layers, batch, tokens) = (
        *args.first().unwrap_or(&12),
        *args.get(1).unwrap_or(&4),
        *args.get(2).unwrap_or(&256),
    );
    let c = KatuLmConfig {
        vocab_size: 32_000,
        block_size: 256,
        embed_dim: 768,
        layers,
        heads: 12,
    };
    let mut rng = SplitMix64(7);
    let windows: Vec<Vec<usize>> = (0..batch)
        .map(|_| (0..=tokens).map(|_| rng.below(c.vocab_size)).collect())
        .collect();
    let dims = (c.embed_dim, c.embed_dim / c.heads, tokens);
    eprintln!(
        "{} parâmetros, batch {batch} × {tokens} tokens; pesos: {}",
        c.param_count(),
        gb(4 * c.param_count() as u64)
    );

    eprintln!("caminho antigo (cada chamada copia entradas e saídas)...");
    let old = Arc::new(Counting {
        inner: fast_backend(),
        dims,
        bytes: Mutex::default(),
    });
    step(&c, c.init(1)?, &windows, old.clone(), None)?;
    let old = old.bytes.lock().unwrap().clone();

    eprintln!("caminho residente (pesos, gradientes e Adam no store)...");
    let params = c.init(1)?;
    let layout = c.resident_layout();
    let store = Arc::new(HostParamStore::new(
        arena_len(&params, &layout)?,
        fast_backend(),
    )?);
    let params = upload(params, &layout, store.clone())?;
    let upload_bytes = store.transfers().to_device;
    let rest = Arc::new(Counting {
        inner: fast_backend(),
        dims,
        bytes: Mutex::default(),
    });
    step(&c, params, &windows, rest.clone(), Some(store.as_ref()))?;
    let resident = store.transfers();
    let rest = rest.bytes.lock().unwrap().clone();

    let total = |m: &BTreeMap<&str, (u64, u64)>| m.values().map(|(u, d)| u + d).sum::<u64>();
    println!("\n## Caminho antigo, por categoria (sobe + desce)\n");
    println!("| Categoria | Sobe | Desce |\n|---|---|---|");
    for (kind, (up, down)) in &old {
        println!("| {kind} | {} | {} |", gb(*up), gb(*down));
    }
    let old_total = total(&old);
    let new_total = resident.to_device - upload_bytes + resident.to_host + total(&rest);
    println!("\n| Caminho | Bytes por passo |\n|---|---|");
    println!("| Antigo (híbrido) | {} |", gb(old_total));
    println!(
        "| Residente | {} (matmuls com peso no store: {}; somas e entropia: {}) |",
        gb(new_total),
        gb(resident.to_device - upload_bytes + resident.to_host),
        gb(total(&rest))
    );
    println!(
        "\nRedução: {:.0}%. Subida única dos pesos residentes: {}.",
        100.0 * (1.0 - new_total as f64 / old_total as f64),
        gb(upload_bytes)
    );
    Ok(())
}
