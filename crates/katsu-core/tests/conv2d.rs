use katsu_core::{Conv2D, Tensor, TensorError};

type Result<T> = std::result::Result<T, TensorError>;

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tolerance * (1.0 + x.abs().max(y.abs())),
            "element {i}: {x} != {y}"
        );
    }
}

#[test]
fn known_forward_and_overlap_gradients() -> Result<()> {
    let x = Tensor::new((1..=9).map(|i| i as f32).collect(), &[1, 1, 3, 3])?;
    let layer = Conv2D::from_weights(Tensor::new(vec![1., 2., 3., 4.], &[1, 1, 2, 2])?, 1, 0)?;
    assert!(layer.get_activation_map().is_none());
    let y = layer.forward(&x)?;
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    assert_eq!(y.data(), &[37., 47., 67., 77.]);
    let g = Tensor::new(vec![1.; 4], y.shape())?;
    let (dx, dw) = layer.backward(&g)?;
    assert_eq!(dx.data(), &[1., 3., 2., 4., 10., 6., 3., 7., 4.]);
    assert_eq!(dw.data(), &[12., 16., 24., 28.]);
    assert!(x.grad()?.is_none()); // VJP local não publica gradientes.
    y.sum()?.backward()?;
    close(x.grad()?.as_ref().unwrap(), dx.data(), 0.);
    close(layer.weights().grad()?.as_ref().unwrap(), dw.data(), 0.);
    let map = layer.get_activation_map().unwrap();
    assert_eq!(map.grad()?.as_ref().unwrap(), &[1.; 4]);
    layer.clear_cache();
    assert!(layer.get_activation_map().is_none());
    assert!(matches!(
        layer.backward(&g),
        Err(TensorError::MissingForward)
    ));
    // O handle original continua válido depois de liberar o cache.
    y.sum()?.backward()?;
    assert_eq!(x.grad()?.as_ref().unwrap()[0], 2.);
    Ok(())
}

#[test]
fn finite_differences_input_and_weights() -> Result<()> {
    // Batch >1, múltiplos canais, kernel retangular, padding e stride com resto.
    for (stride, padding) in [(1, 0), (1, 1), (2, 1)] {
        let xs = [2, 2, 3, 4];
        let ws = [2, 2, 2, 3];
        let xv: Vec<f32> = (0..48).map(|i| (i % 11) as f32 * 0.07 - 0.3).collect();
        let wv: Vec<f32> = (0..24).map(|i| (i % 7) as f32 * 0.09 - 0.2).collect();
        let x = Tensor::new(xv.clone(), &xs)?;
        let layer = Conv2D::from_weights(Tensor::new(wv.clone(), &ws)?, stride, padding)?;
        let y = layer.forward(&x)?;
        let seed: Vec<f32> = (0..y.numel())
            .map(|i| (i % 5) as f32 * 0.13 - 0.2)
            .collect();
        let (dx, dw) = layer.backward(&Tensor::new(seed.clone(), y.shape())?)?;
        y.backward_with(&seed)?;
        close(x.grad()?.as_ref().unwrap(), dx.data(), 1e-6);
        close(layer.weights().grad()?.as_ref().unwrap(), dw.data(), 1e-6);
        let eval = |data: Vec<f32>, weights: Vec<f32>| -> Result<f32> {
            let conv = Conv2D::from_weights(Tensor::new(weights, &ws)?, stride, padding)?;
            let out = conv.forward(&Tensor::new(data, &xs)?)?;
            Ok(out.data().iter().zip(&seed).map(|(v, g)| v * g).sum())
        };
        for side in 0..2 {
            let analytical = if side == 0 { dx.data() } else { dw.data() };
            for (i, &expected) in analytical.iter().enumerate() {
                let (mut xp, mut xm, mut wp, mut wm) =
                    (xv.clone(), xv.clone(), wv.clone(), wv.clone());
                let eps = 0.001;
                if side == 0 {
                    xp[i] += eps;
                    xm[i] -= eps;
                } else {
                    wp[i] += eps;
                    wm[i] -= eps;
                }
                let numerical = (eval(xp, wp)? - eval(xm, wm)?) / (2. * eps);
                close(&[expected], &[numerical], 3e-4);
            }
        }
    }
    Ok(())
}

#[test]
fn reused_layer_keeps_independent_graphs() -> Result<()> {
    let layer = Conv2D::from_weights(Tensor::new(vec![2.], &[1, 1, 1, 1])?, 1, 0)?;
    let a = Tensor::new(vec![3.], &[1, 1, 1, 1])?;
    let b = Tensor::new(vec![5.], &[1, 1, 1, 1])?;
    let ya = layer.forward(&a)?;
    let yb = layer.forward(&b)?;
    ya.add(&yb)?.sum()?.backward()?;
    assert_eq!(layer.weights().grad()?.as_ref().unwrap(), &[8.]);
    assert_eq!(a.grad()?.as_ref().unwrap(), &[2.]);
    assert_eq!(b.grad()?.as_ref().unwrap(), &[2.]);
    let (_, dw) = layer.backward(&Tensor::new(vec![1.], yb.shape())?)?;
    assert_eq!(dw.data(), &[5.]);
    // Forward inválido preserva o último cache bem-sucedido.
    assert!(layer.forward(&Tensor::zeros(&[1])?).is_err());
    assert_eq!(layer.get_activation_map().unwrap().data(), &[10.]);
    Ok(())
}

#[test]
fn shared_operand_and_chained_convolution() -> Result<()> {
    let x = Tensor::new(vec![2.], &[1, 1, 1, 1])?;
    let layer = Conv2D::from_weights(x.clone(), 1, 0)?;
    let y = layer.forward(&x)?; // x²
    layer.forward(&y)?.sum()?.backward()?; // x³
    assert_eq!(x.grad()?.as_ref().unwrap(), &[12.]);
    Ok(())
}

#[test]
fn invalid_shapes_and_overflow() -> Result<()> {
    assert!(Conv2D::new(0, 1, (3, 3), 1, 0).is_err());
    assert!(Conv2D::new(1, 1, (0, 3), 1, 0).is_err());
    assert!(Conv2D::new(1, 1, (1, 1), 0, 0).is_err());
    assert!(Conv2D::from_weights(Tensor::zeros(&[1, 1, 1])?, 1, 0).is_err());
    assert!(Conv2D::new(1, 1, (1, 1), 1, usize::MAX).is_err());
    let layer = Conv2D::new(1, 1, (3, 3), 1, 0)?;
    assert!(matches!(
        layer.backward(&Tensor::zeros(&[1])?),
        Err(TensorError::MissingForward)
    ));
    assert!(layer.forward(&Tensor::zeros(&[1, 2, 3, 3])?).is_err());
    assert!(layer.forward(&Tensor::zeros(&[1, 1, 2, 2])?).is_err());
    assert!(layer.forward(&Tensor::zeros(&[1, 1, 0, 2])?).is_err());
    layer.forward(&Tensor::zeros(&[1, 1, 3, 3])?)?;
    // Mesmo numel, rank incorreto.
    assert!(layer.backward(&Tensor::zeros(&[1])?).is_err());
    let huge_pad = Conv2D::new(1, 1, (1, 1), 1, usize::MAX / 2)?;
    assert!(matches!(
        huge_pad.forward(&Tensor::zeros(&[0, 1, 2, 2])?),
        Err(TensorError::ShapeOverflow)
    ));
    Ok(())
}

#[test]
fn empty_batch_and_seeded_initialization() -> Result<()> {
    let layer = Conv2D::new_seeded(3, 2, (3, 3), 1, 1, 7)?;
    let same = Conv2D::new_seeded(3, 2, (3, 3), 1, 1, 7)?;
    let other = Conv2D::new_seeded(3, 2, (3, 3), 1, 1, 8)?;
    assert_eq!(layer.weights().data(), same.weights().data());
    assert_ne!(layer.weights().data(), other.weights().data());
    assert!(layer
        .weights()
        .data()
        .iter()
        .all(|w| w.abs() <= 1. / 27_f32.sqrt()));
    let x = Tensor::zeros(&[0, 3, 4, 5])?;
    let y = layer.forward(&x)?;
    assert_eq!(y.shape(), &[0, 2, 4, 5]);
    y.sum()?.backward()?;
    assert!(x.grad()?.as_ref().unwrap().is_empty());
    assert!(layer
        .weights()
        .grad()?
        .as_ref()
        .unwrap()
        .iter()
        .all(|&w| w == 0.));
    Ok(())
}

#[test]
fn known_stride_and_padding() -> Result<()> {
    let x = Tensor::new((1..=9).map(|i| i as f32).collect(), &[1, 1, 3, 3])?;
    let layer = Conv2D::from_weights(Tensor::new(vec![1., 2., 3., 4.], &[1, 1, 2, 2])?, 2, 1)?;
    let y = layer.forward(&x)?;
    assert_eq!(y.shape(), &[1, 1, 2, 2]);
    assert_eq!(y.data(), &[4., 18., 36., 77.]);
    Ok(())
}
