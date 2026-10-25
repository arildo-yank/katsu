//! Caminho residente (pesos, gradientes e Adam no store) contra o caminho CPU de sempre.
//! O `HostParamStore` faz as mesmas contas que o CPU: os resultados precisam bater bit a bit.
use katsu_core::{
    arena_len, materialize, save_params, upload, Adam, Backend, CpuBackend, HostParamStore,
    KatsuLmConfig, Optimizer, ParamStore, Tensor, TensorError,
};
use std::sync::Arc;

fn config() -> KatsuLmConfig {
    KatsuLmConfig {
        vocab_size: 48,
        block_size: 12,
        embed_dim: 32,
        layers: 2,
        heads: 4,
    }
}

fn windows(step: usize) -> Vec<Vec<usize>> {
    (0..2)
        .map(|b| (0..9).map(|i| (7 * step + 5 * b + 3 * i) % 48).collect())
        .collect()
}

fn loss(c: &KatsuLmConfig, params: &[Tensor], step: usize, backend: Arc<dyn Backend>) -> Tensor {
    let w = windows(step);
    let refs: Vec<&[usize]> = w.iter().map(Vec::as_slice).collect();
    c.loss_batch_with_backend(params, &refs, backend).unwrap()
}

/// Cópia com nós novos: `Vec<Tensor>::clone` compartilha os nós, e duas execuções somariam
/// gradientes nas mesmas folhas.
fn fresh(params: &[Tensor]) -> Vec<Tensor> {
    params
        .iter()
        .map(|p| Tensor::new(p.data().to_vec(), p.shape()).unwrap())
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn host_grad_sumsq(params: &[Tensor]) -> f64 {
    params
        .iter()
        .filter(|p| !p.is_resident())
        .flat_map(|p| p.grad().unwrap().clone().unwrap())
        .map(|g| f64::from(g).powi(2))
        .sum()
}

struct Run {
    losses: Vec<u32>,
    norms: Vec<f64>,
    params: Vec<Tensor>,
    adam: Adam,
}

fn cpu_run(c: &KatsuLmConfig, steps: std::ops::Range<usize>, params: &[Tensor], adam: Adam) -> Run {
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::default());
    let (mut params, mut adam) = (fresh(params), adam);
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    for step in steps {
        let l = loss(c, &params, step, cpu.clone());
        losses.push(l.data()[0].to_bits());
        l.backward_release().unwrap();
        norms.push(host_grad_sumsq(&params));
        params = adam.step_owned(params).unwrap();
    }
    Run {
        losses,
        norms,
        params,
        adam,
    }
}

fn resident_run(
    c: &KatsuLmConfig,
    steps: std::ops::Range<usize>,
    params: &[Tensor],
    adam: Adam,
    store: &Arc<HostParamStore>,
) -> Run {
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::default());
    let mut adam = adam;
    let mut params = params.to_vec();
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    for step in steps {
        let l = loss(c, &params, step, cpu.clone());
        losses.push(l.data()[0].to_bits());
        l.backward_release().unwrap();
        let (resident, nonfinite) = store.grad_stats().unwrap();
        assert_eq!(nonfinite, 0);
        norms.push(host_grad_sumsq(&params) + resident);
        let (next, bad) = adam.step_resident(params, store.as_ref()).unwrap();
        assert_eq!(bad, 0);
        params = next;
    }
    Run {
        losses,
        norms,
        params,
        adam,
    }
}

fn new_store(c: &KatsuLmConfig, params: &[Tensor]) -> Arc<HostParamStore> {
    let len = arena_len(params, &c.resident_layout()).unwrap();
    Arc::new(HostParamStore::new(len, Arc::new(CpuBackend::default())).unwrap())
}

fn assert_same_weights(resident: &[Tensor], cpu: &[Tensor]) {
    let resident = materialize(resident).unwrap();
    assert_eq!(resident.len(), cpu.len());
    for (i, (r, c)) in resident.iter().zip(cpu).enumerate() {
        assert_eq!(r.shape(), c.shape(), "param {i}");
        assert_eq!(bits(r.data()), bits(c.data()), "param {i}");
    }
}

#[test]
fn resident_training_matches_cpu_bit_for_bit() {
    let c = config();
    let init = c.init(11).unwrap();
    let expected = cpu_run(&c, 0..6, &init, Adam::new(3e-3).unwrap());
    let store = new_store(&c, &init);
    let params = upload(fresh(&init), &c.resident_layout(), store.clone()).unwrap();
    let got = resident_run(&c, 0..6, &params, Adam::new(3e-3).unwrap(), &store);
    assert_eq!(got.losses, expected.losses, "losses");
    assert_same_weights(&got.params, &expected.params);
    // A norma soma na ordem da arena: igual até o arredondamento do f64.
    for (g, e) in got.norms.iter().zip(&expected.norms) {
        assert!((g - e).abs() <= 1e-12 * e.max(1.0), "{g} vs {e}");
    }
}

#[test]
fn checkpoints_are_identical_to_the_cpu_path() {
    let c = config();
    let init = c.init(5).unwrap();
    let expected = cpu_run(&c, 0..3, &init, Adam::new(1e-3).unwrap());
    let store = new_store(&c, &init);
    let params = upload(fresh(&init), &c.resident_layout(), store.clone()).unwrap();
    let got = resident_run(&c, 0..3, &params, Adam::new(1e-3).unwrap(), &store);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    expected.adam.save_state(&mut a).unwrap();
    got.adam.save_state_resident(&mut b, &got.params).unwrap();
    assert_eq!(a, b, "Adam state");
    let (mut a, mut b) = (Vec::new(), Vec::new());
    save_params(&expected.params, &mut a).unwrap();
    save_params(&materialize(&got.params).unwrap(), &mut b).unwrap();
    assert_eq!(a, b, "weights");
}

#[test]
fn resuming_from_a_host_checkpoint_continues_identically() {
    let c = config();
    let first = cpu_run(&c, 0..2, &c.init(3).unwrap(), Adam::new(2e-3).unwrap());
    let mut saved = Vec::new();
    first.adam.save_state(&mut saved).unwrap();
    let expected = cpu_run(&c, 2..5, &first.params, first.adam);

    let mut adam = Adam::load_state(&mut saved.as_slice(), &first.params).unwrap();
    let store = new_store(&c, &first.params);
    let params = upload(fresh(&first.params), &c.resident_layout(), store.clone()).unwrap();
    adam.offload_moments(&params).unwrap();
    let got = resident_run(&c, 2..5, &params, adam, &store);
    assert_eq!(got.losses, expected.losses);
    assert_same_weights(&got.params, &expected.params);
}

#[test]
fn a_step_moves_activations_not_weights() {
    // Pesos ∝ d², ativações ∝ linhas·d: com 4 linhas e d = 128 os pesos dominam, como num
    // passo de verdade em que o caminho antigo subia os pesos a cada chamada.
    let c = KatsuLmConfig {
        vocab_size: 512,
        block_size: 8,
        embed_dim: 128,
        layers: 2,
        heads: 4,
    };
    let init = c.init(1).unwrap();
    let store = new_store(&c, &init);
    let weight_bytes = 4 * store.capacity() as u64;
    let params = upload(init, &c.resident_layout(), store.clone()).unwrap();
    assert_eq!(
        store.transfers().to_device,
        weight_bytes,
        "one upload of the weights"
    );
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend::default());
    let mut adam = Adam::new(1e-3).unwrap();
    let mut params = params;
    let mut per_step = Vec::new();
    // O primeiro passo de um Adam novo zera os momentos no store (uma vez); mede os seguintes.
    for _ in 0..3 {
        let before = store.transfers();
        let l = c
            .loss_batch_with_backend(&params, &[&[1, 2, 3, 4, 5][..]], cpu.clone())
            .unwrap();
        l.backward_release().unwrap();
        params = adam.step_resident(params, store.as_ref()).unwrap().0;
        let after = store.transfers();
        per_step.push((
            after.to_device - before.to_device,
            after.to_host - before.to_host,
        ));
    }
    assert_eq!(per_step[1], per_step[2], "same traffic every step");
    let (up, down) = per_step[1];
    // O caminho antigo subia todos os pesos no forward e de novo no backward e descia todos
    // os gradientes: ≥ 3× os pesos por passo. Aqui só ativações.
    assert!(
        up + down < weight_bytes / 2,
        "{up} + {down} vs weights {weight_bytes}"
    );
}

#[test]
fn layout_holds_exactly_the_matmul_weights() {
    let c = config();
    let params = c.init(0).unwrap();
    let layout = c.resident_layout();
    let mut sorted = layout.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), layout.len(), "no duplicates");
    assert!(layout.iter().all(|&i| params[i].rank() == 2));
    assert!(
        !layout.contains(&0) && !layout.contains(&1),
        "embeddings stay on the host"
    );
    let (v, d, l) = (c.vocab_size, c.embed_dim, c.layers);
    // Por camada: 4·d² da atenção + 8·d² do MLP; mais a cabeça d·v.
    assert_eq!(arena_len(&params, &layout).unwrap(), l * 12 * d * d + d * v);
    // O que sobra no host: embeddings, LayerNorm e bias.
    let host: usize = (0..params.len())
        .filter(|i| !layout.contains(i))
        .map(|i| params[i].numel())
        .sum();
    assert_eq!(host + l * 12 * d * d + d * v, c.param_count());
}

#[test]
fn misuse_fails_loudly() {
    let c = config();
    let init = c.init(2).unwrap();
    let store = new_store(&c, &init);
    let layout = c.resident_layout();
    let params = upload(init, &layout, store.clone()).unwrap();
    let wo = &params[layout[3 * c.heads]];
    let x = Tensor::zeros(&[3, c.embed_dim / c.heads]).unwrap();
    // Residente só do lado direito, e reshape não tem dados no host para compartilhar.
    assert!(wo.matmul(&Tensor::zeros(&[16, 2]).unwrap()).is_err());
    assert!(wo.reshape(&[wo.numel()]).is_err());
    assert!(x.matmul(wo).is_ok());
    // Concat só de faixas vizinhas: os Wo das cabeças 0 e 2 não são; 0 e 1 são.
    let (wo0, wo1, wo2) = (
        wo,
        &params[layout[3 * c.heads + 1]],
        &params[layout[3 * c.heads + 2]],
    );
    assert!(matches!(
        Tensor::concat(&[wo0.clone(), wo2.clone()], 0),
        Err(TensorError::InvalidShape(_))
    ));
    let joined = Tensor::concat(&[wo0.clone(), wo1.clone()], 0).unwrap();
    assert_eq!(joined.residency().unwrap().1.len, 2 * wo0.numel());
    // step_owned não sabe atualizar residentes: falta gradiente no host.
    let mut adam = Adam::new(1e-3).unwrap();
    let l = loss(&c, &params, 0, Arc::new(CpuBackend::default()));
    l.backward_release().unwrap();
    assert!(matches!(
        adam.step_owned(params.clone()),
        Err(TensorError::MissingGradient { .. })
    ));
    // E o store tem de ser o dos parâmetros.
    let other = new_store(&c, &c.init(2).unwrap());
    assert!(adam.step_resident(params, other.as_ref()).is_err());
}

#[test]
#[should_panic(expected = "tensor residente")]
fn reading_resident_data_on_the_host_panics() {
    let c = config();
    let init = c.init(4).unwrap();
    let store = new_store(&c, &init);
    let params = upload(init, &c.resident_layout(), store).unwrap();
    let _ = params[c.resident_layout()[0]].data();
}

#[test]
fn a_fresh_adam_starts_from_zero_moments_in_the_store() {
    // Como o train_sft: adaptação com um Adam, SFT com outro, no mesmo store.
    let c = config();
    let init = c.init(9).unwrap();
    let first = cpu_run(&c, 0..2, &init, Adam::new(2e-3).unwrap());
    let expected = cpu_run(&c, 2..4, &first.params, Adam::new(1e-3).unwrap());
    let store = new_store(&c, &init);
    let params = upload(fresh(&init), &c.resident_layout(), store.clone()).unwrap();
    let warm = resident_run(&c, 0..2, &params, Adam::new(2e-3).unwrap(), &store);
    let got = resident_run(&c, 2..4, &warm.params, Adam::new(1e-3).unwrap(), &store);
    assert_eq!(got.losses, expected.losses);
    assert_same_weights(&got.params, &expected.params);
}
