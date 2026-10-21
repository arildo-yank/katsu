use katsu_core::{fit, Adam, Op, Optimizer, Sgd, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

fn grad(t: &Tensor) -> Vec<f32> {
    t.grad().unwrap().as_ref().unwrap().clone()
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} != {y}");
    }
}

/// Dados determinísticos, sem empates, para diferenças finitas.
fn sample(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.23 + 0.05)
        .collect()
}

/// Reduz a saída a um escalar com pesos distintos, para que cada elemento importe.
fn weighted_sum(y: &Tensor) -> Result<Tensor> {
    let w = Tensor::new(
        (0..y.numel())
            .map(|i| ((i * 5 % 7) as f32 - 3.0) * 0.4 + 0.1)
            .collect(),
        y.shape(),
    )?;
    y.mul(&w)?.sum()
}

/// Compara o gradiente analítico de `f` em `x` com diferenças centrais.
fn check_gradient(shape: &[usize], f: impl Fn(&Tensor) -> Result<Tensor>) -> Result<()> {
    let values = sample(shape.iter().product());
    let x = Tensor::new(values.clone(), shape)?;
    weighted_sum(&f(&x)?)?.backward()?;
    let analytical = grad(&x);
    let eps = 1e-2;
    for i in 0..values.len() {
        let eval = |delta: f32| -> Result<f32> {
            let mut v = values.clone();
            v[i] += delta;
            Ok(weighted_sum(&f(&Tensor::new(v, shape)?)?)?.data()[0])
        };
        let numerical = (eval(eps)? - eval(-eps)?) / (2.0 * eps);
        let tol = 2e-2 * numerical.abs().max(1.0);
        assert!(
            (analytical[i] - numerical).abs() <= tol,
            "index {i}: analytical {} vs numerical {numerical}",
            analytical[i]
        );
    }
    Ok(())
}

#[test]
fn broadcasting_forward_follows_numpy_rules() -> Result<()> {
    let m = Tensor::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3])?;
    let row = Tensor::new(vec![10., 20., 30.], &[3])?;
    let col = Tensor::new(vec![2., 3.], &[2, 1])?;
    assert_eq!(m.add(&row)?.data(), &[11., 22., 33., 14., 25., 36.]);
    assert_eq!(m.mul(&col)?.data(), &[2., 4., 6., 12., 15., 18.]);
    let outer = col.mul(&row)?; // [2,1] * [3] -> [2,3]
    assert_eq!(outer.shape(), &[2, 3]);
    assert_eq!(outer.data(), &[20., 40., 60., 30., 60., 90.]);
    assert_eq!(
        m.mul(&Tensor::scalar(0.5)?)?.data(),
        &[0.5, 1., 1.5, 2., 2.5, 3.]
    );
    assert!(matches!(
        m.add(&Tensor::zeros(&[2])?),
        Err(TensorError::ShapeMismatch)
    ));
    Ok(())
}

#[test]
fn broadcasting_gradients_sum_over_expanded_dims() -> Result<()> {
    let m = Tensor::new(sample(6), &[2, 3])?;
    let bias = Tensor::new(vec![0.1, 0.2, 0.3], &[3])?;
    let scale = Tensor::new(vec![2., 3.], &[2, 1])?;
    weighted_sum(&m.add(&bias)?.mul(&scale)?)?.backward()?;
    // Pesos de weighted_sum para 6 elementos: [-1.1, 0.9, 0.1, -0.7, 1.3, 0.5].
    close(
        &grad(&bias),
        &[
            -1.1 * 2. - 0.7 * 3.,
            0.9 * 2. + 1.3 * 3.,
            0.1 * 2. + 0.5 * 3.,
        ],
        1e-5,
    );
    check_gradient(&[3], |b| m.add(b)?.mul(&scale))?;
    check_gradient(&[2, 1], |s| m.mul(s))?;
    check_gradient(&[], |s| m.mul(s))
}

#[test]
fn transpose_and_reshape() -> Result<()> {
    let x = Tensor::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3])?;
    let t = x.transpose()?;
    assert_eq!(
        (t.shape(), t.data()),
        (&[3, 2][..], &[1., 4., 2., 5., 3., 6.][..])
    );
    assert_eq!(t.op(), Op::Transpose);
    assert_eq!(x.reshape(&[3, 2])?.data(), x.data());
    assert!(matches!(
        x.reshape(&[4]),
        Err(TensorError::DataLength { .. })
    ));
    assert!(matches!(
        Tensor::zeros(&[2])?.transpose(),
        Err(TensorError::InvalidShape(_))
    ));
    check_gradient(&[2, 3], |x| x.transpose())?;
    check_gradient(&[2, 3], |x| x.reshape(&[6, 1]))
}

#[test]
fn gelu_values_and_gradient() -> Result<()> {
    let y = Tensor::new(vec![-3., 0., 1., 3.], &[4])?.gelu()?;
    close(y.data(), &[-0.003_637, 0., 0.841_192, 2.996_363], 1e-4);
    check_gradient(&[2, 5], |x| x.gelu())
}

#[test]
fn softmax_rows_sum_to_one_and_differentiate() -> Result<()> {
    let s = Tensor::new(vec![1., 2., 3., 1000., 1000., 1000.], &[2, 3])?.softmax()?;
    close(&s.data()[..3], &[0.090_031, 0.244_728, 0.665_241], 1e-5);
    close(&s.data()[3..], &[1. / 3.; 3], 1e-6); // estável com valores grandes
    check_gradient(&[3, 4], |x| x.softmax())?;
    check_gradient(&[3, 4], |x| x.log_softmax())
}

#[test]
fn causal_mask_hides_the_future() -> Result<()> {
    let scores = Tensor::new(sample(9), &[3, 3])?;
    let p = scores.add(&Tensor::causal_mask(3)?)?.softmax()?;
    assert_eq!(p.data()[1..3], [0., 0.]);
    assert_eq!(p.data()[5], 0.);
    assert_eq!(p.data()[0], 1.);
    check_gradient(&[3, 3], |x| x.add(&Tensor::causal_mask(3)?)?.softmax())
}

#[test]
fn layer_norm_normalizes_rows() -> Result<()> {
    let y = Tensor::new(vec![1., 2., 3., 4., 10., 10., 10., 14.], &[2, 4])?.layer_norm()?;
    for row in y.data().chunks(4) {
        let mean: f32 = row.iter().sum::<f32>() / 4.;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.;
        assert!(mean.abs() < 1e-5 && (var - 1.).abs() < 1e-3, "{row:?}");
    }
    check_gradient(&[3, 4], |x| x.layer_norm())
}

#[test]
fn cross_entropy_matches_hand_value() -> Result<()> {
    let uniform = Tensor::zeros(&[2, 3])?.cross_entropy(&[0, 2])?;
    close(uniform.data(), &[3f32.ln()], 1e-6);
    let logits = Tensor::new(sample(8), &[2, 4])?;
    let loss = logits.cross_entropy(&[1, 3])?;
    loss.backward()?;
    // dL/dlogits = (softmax - one_hot) / n
    let p = logits.softmax()?;
    let mut expected: Vec<f32> = p.data().iter().map(|v| v / 2.).collect();
    expected[1] -= 0.5;
    expected[7] -= 0.5;
    close(&grad(&logits), &expected, 1e-6);
    assert!(matches!(
        logits.cross_entropy(&[0, 4]),
        Err(TensorError::IndexOutOfRange {
            index: 4,
            classes: 4
        })
    ));
    assert!(matches!(
        logits.cross_entropy(&[0]),
        Err(TensorError::DataLength { .. })
    ));
    Ok(())
}

/// Atenção causal de uma cabeça: softmax(QKᵀ/√d + máscara) V, com LayerNorm e GELU.
fn attention_block(x: &Tensor, wq: &Tensor, wk: &Tensor, wv: &Tensor) -> Result<Tensor> {
    let (q, k, v) = (x.matmul(wq)?, x.matmul(wk)?, x.matmul(wv)?);
    let d = wq.shape()[1] as f32;
    let scores = q
        .matmul(&k.transpose()?)?
        .mul(&Tensor::scalar(1. / d.sqrt())?)?
        .add(&Tensor::causal_mask(x.shape()[0])?)?;
    scores.softmax()?.matmul(&v)?.add(x)?.layer_norm()?.gelu()
}

#[test]
fn attention_block_gradients_match_finite_differences() -> Result<()> {
    let x = Tensor::new(sample(12), &[3, 4])?;
    let w = |seed: usize| -> Result<Tensor> {
        Tensor::new(
            (0..16)
                .map(|i| (((i + seed) * 13 % 9) as f32 - 4.) * 0.15)
                .collect(),
            &[4, 4],
        )
    };
    let (wk, wv) = (w(1)?, w(2)?);
    check_gradient(&[4, 4], |wq| attention_block(&x, wq, &wk, &wv))?;
    let wq = w(3)?;
    check_gradient(&[3, 4], |x| attention_block(x, &wq, &wk, &wv))
}

#[test]
fn adam_first_step_moves_by_learning_rate() -> Result<()> {
    let w = Tensor::new(vec![1., -2., 0.5], &[3])?;
    w.mul(&w)?.sum()?.backward()?;
    let mut adam = Adam::new(0.1)?;
    let next = adam.step(std::slice::from_ref(&w))?;
    // No primeiro passo, m̂/√v̂ = sinal(g): cada peso anda exatamente lr.
    close(next[0].data(), &[0.9, -1.9, 0.4], 1e-6);
    assert_eq!(adam.learning_rate(), 0.1);
    let other = Tensor::new(vec![1., 2.], &[2])?;
    other.sum()?.backward()?;
    assert!(matches!(
        adam.step(&[other]),
        Err(TensorError::OptimizerState)
    ));
    assert!(matches!(
        Adam::with_betas(0.1, 1., 0.999, 1e-8),
        Err(TensorError::InvalidLearningRate)
    ));
    Ok(())
}

#[test]
fn adam_trains_a_softmax_classifier() -> Result<()> {
    // 4 pontos, 2 features + bias; a classe é o sinal da primeira feature.
    let x = Tensor::new(vec![2., 1., -1., -2., 1.5, -0.5, -1., 0.5], &[4, 2])?;
    let targets = [0, 1, 0, 1];
    let params = vec![Tensor::zeros(&[2, 2])?, Tensor::zeros(&[2])?];
    let mut losses = Vec::new();
    let trained = fit(
        params,
        &mut Adam::new(0.1)?,
        100,
        |p| x.matmul(&p[0])?.add(&p[1])?.cross_entropy(&targets),
        |m| losses.push(m.loss),
    )?;
    close(&losses[..1], &[2f32.ln()], 1e-6);
    assert!(losses[99] < 0.05, "final loss {}", losses[99]);
    let logits = x.matmul(&trained[0])?.add(&trained[1])?;
    for (row, &t) in logits.data().chunks(2).zip(&targets) {
        assert_eq!(usize::from(row[1] > row[0]), t);
    }
    // O mesmo problema com SGD continua funcionando pelo trait.
    let sgd_params = vec![Tensor::zeros(&[2, 2])?, Tensor::zeros(&[2])?];
    fit(
        sgd_params,
        &mut Sgd::new(0.5)?,
        10,
        |p| x.matmul(&p[0])?.add(&p[1])?.cross_entropy(&targets),
        |_| {},
    )?;
    Ok(())
}
