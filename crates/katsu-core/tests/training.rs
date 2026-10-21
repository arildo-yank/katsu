use katsu_core::{fit, EpochMetrics, Op, Sgd, Tensor, TensorError};

/// Regressão linear y = 2x + 1 com bias via coluna de uns (não há broadcasting).
fn linear_problem() -> Result<(Tensor, Tensor, Tensor), TensorError> {
    let xs: Vec<f32> = (0..8).map(|i| -1. + i as f32 * 2. / 7.).collect();
    let design = xs.iter().flat_map(|&x| [x, 1.]).collect();
    let neg_y = xs.iter().map(|&x| -(2. * x + 1.)).collect();
    Ok((
        Tensor::new(design, &[8, 2])?,
        Tensor::new(neg_y, &[8, 1])?,
        Tensor::new(vec![1. / 8.], &[])?,
    ))
}

fn mse(
    params: &[Tensor],
    x: &Tensor,
    neg_y: &Tensor,
    inv_n: &Tensor,
) -> Result<Tensor, TensorError> {
    let residual = x.matmul(&params[0])?.add(neg_y)?;
    residual.mul(&residual)?.sum()?.mul(inv_n)
}

#[test]
fn sgd_step_matches_hand_computation() -> Result<(), TensorError> {
    let w = Tensor::new(vec![1., -2.], &[2])?;
    w.mul(&w)?.sum()?.backward()?; // grad = 2w = [2, -4]
    let next = Sgd::new(0.25)?.step(std::slice::from_ref(&w))?;
    assert_eq!(next[0].data(), &[0.5, -1.]);
    assert_eq!(next[0].shape(), &[2]);
    assert_eq!(next[0].op(), Op::Leaf);
    assert!(next[0].grad()?.is_none());
    assert_eq!(w.data(), &[1., -2.]); // o parâmetro antigo não muda
    Ok(())
}

#[test]
fn sgd_rejects_bad_learning_rates_and_missing_gradients() -> Result<(), TensorError> {
    for lr in [0., -0.1, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            Sgd::new(lr),
            Err(TensorError::InvalidLearningRate)
        ));
    }
    let a = Tensor::new(vec![1.], &[])?;
    let b = Tensor::new(vec![2.], &[])?;
    a.mul(&a)?.backward()?;
    assert!(matches!(
        Sgd::new(0.1)?.step(&[a, b]),
        Err(TensorError::MissingGradient { index: 1 })
    ));
    Ok(())
}

#[test]
fn fit_learns_linear_regression_and_reports_metrics() -> Result<(), TensorError> {
    let (x, neg_y, inv_n) = linear_problem()?;
    let mut history = Vec::new();
    let params = fit(
        vec![Tensor::zeros(&[2, 1])?],
        &mut Sgd::new(0.1)?,
        300,
        |p| mse(p, &x, &neg_y, &inv_n),
        |m| history.push(*m),
    )?;
    assert_eq!(history.len(), 300);
    assert_eq!(history[0].epoch, 1);
    assert_eq!(history[299].epoch, 300);
    assert!(history.iter().all(|m| m.learning_rate == 0.1));
    // Perda inicial com W = 0 é mean(y²); a curva decresce monotonicamente.
    assert!(history[0].loss > 1.);
    assert!(history.windows(2).all(|w| w[1].loss <= w[0].loss));
    assert!(history[299].loss < 1e-6);
    let w = params[0].data();
    assert!(
        (w[0] - 2.).abs() < 1e-3 && (w[1] - 1.).abs() < 1e-3,
        "{w:?}"
    );
    Ok(())
}

#[test]
fn fit_stops_on_divergence() -> Result<(), TensorError> {
    let (x, neg_y, inv_n) = linear_problem()?;
    let mut emitted = 0;
    let result = fit(
        vec![Tensor::zeros(&[2, 1])?],
        &mut Sgd::new(1e6)?,
        100,
        |p| mse(p, &x, &neg_y, &inv_n),
        |_| emitted += 1,
    );
    let Err(TensorError::NonFiniteLoss { epoch }) = result else {
        panic!("expected divergence, got {result:?}");
    };
    assert_eq!(emitted, epoch - 1);
    Ok(())
}

#[test]
fn metrics_serialize_as_json_lines() {
    let line = EpochMetrics {
        epoch: 86,
        loss: 0.114578,
        learning_rate: 0.001,
    }
    .to_string();
    assert_eq!(
        line,
        r#"{"epoch":86,"loss":0.114578,"learning_rate":0.001}"#
    );
}

#[test]
fn relu_forward_and_gradient() -> Result<(), TensorError> {
    let x = Tensor::new(vec![-2., -0., 0.5, 3.], &[4])?;
    let y = x.relu()?;
    assert_eq!(y.data(), &[0., 0., 0.5, 3.]);
    assert_eq!(y.op(), Op::Relu);
    let scale = Tensor::new(vec![1., 2., 3., 4.], &[4])?;
    y.mul(&scale)?.sum()?.backward()?;
    assert_eq!(x.grad()?.as_deref(), Some(&[0., 0., 3., 4.][..]));
    Ok(())
}
