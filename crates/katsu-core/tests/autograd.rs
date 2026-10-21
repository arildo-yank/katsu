use katsu_core::{Op, Tensor, TensorError};

fn grad(t: &Tensor) -> Vec<f32> {
    t.grad().unwrap().as_ref().unwrap().clone()
}

#[test]
fn squared_error_shared_nodes_and_repeated_backward() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1., 2.], &[2])?;
    let b = Tensor::new(vec![3., 4.], &[2])?;
    let s = a.add(&b)?;
    let loss = s.mul(&s)?.sum()?;
    assert_eq!(s.op(), Op::Add);
    assert_eq!(s.parents().len(), 2);
    assert!(a.grad()?.is_none());
    loss.backward()?;
    assert_eq!(loss.data(), &[52.]);
    assert_eq!(grad(&a), vec![8., 12.]);
    assert_eq!(grad(&b), vec![8., 12.]);
    assert_eq!(grad(&s), vec![8., 12.]);
    loss.backward()?;
    assert_eq!(grad(&a), vec![16., 24.]);
    a.zero_grad()?;
    loss.backward()?;
    assert_eq!(grad(&a), vec![8., 12.]);
    assert_eq!(grad(&b), vec![24., 36.]);
    Ok(())
}

#[test]
fn diamond_and_clone_identity() -> Result<(), TensorError> {
    let x = Tensor::new(vec![2.], &[])?;
    let alias = x.clone();
    let square = x.mul(&alias)?;
    square.add(&square.mul(&x)?)?.backward()?; // x² + x³
    assert_eq!(grad(&alias), vec![16.]);
    assert_eq!(grad(&square), vec![3.]);
    Ok(())
}

#[test]
fn explicit_seed_and_errors_are_atomic() -> Result<(), TensorError> {
    let x = Tensor::new(vec![2., 3.], &[2])?;
    let y = x.mul(&x)?;
    assert!(matches!(y.backward(), Err(TensorError::NonScalarBackward)));
    assert!(matches!(
        y.backward_with(&[1.]),
        Err(TensorError::DataLength { .. })
    ));
    let borrowed = x.grad()?;
    assert!(matches!(
        y.backward_with(&[2., 4.]),
        Err(TensorError::GradientBorrowed)
    ));
    assert!(y.grad()?.is_none());
    drop(borrowed);
    y.backward_with(&[2., 4.])?;
    assert_eq!(grad(&x), vec![8., 24.]);
    Ok(())
}

#[test]
fn matmul_finite_differences() -> Result<(), TensorError> {
    let av = vec![0.2, -0.4, 0.7, 0.3, 0.8, -0.1];
    let bv = vec![0.1, 0.5, -0.3, 0.6, 0.9, -0.2];
    let eval = |a: Vec<f32>, b: Vec<f32>| -> Result<f32, TensorError> {
        let y = Tensor::new(a, &[2, 3])?.matmul(&Tensor::new(b, &[3, 2])?)?;
        Ok(y.mul(&y)?.sum()?.data()[0])
    };
    let a = Tensor::new(av.clone(), &[2, 3])?;
    let b = Tensor::new(bv.clone(), &[3, 2])?;
    let y = a.matmul(&b)?;
    y.mul(&y)?.sum()?.backward()?;
    for side in 0..2 {
        let analytical = if side == 0 { grad(&a) } else { grad(&b) };
        for i in 0..6 {
            let (mut ap, mut am, mut bp, mut bm) = (av.clone(), av.clone(), bv.clone(), bv.clone());
            let eps = 0.001;
            if side == 0 {
                ap[i] += eps;
                am[i] -= eps;
            } else {
                bp[i] += eps;
                bm[i] -= eps;
            }
            let numerical = (eval(ap, bp)? - eval(am, bm)?) / (2. * eps);
            assert!(
                (analytical[i] - numerical).abs() < 0.001,
                "side={side} i={i}: {} != {numerical}",
                analytical[i]
            );
        }
    }
    Ok(())
}

#[test]
fn matmul_same_parent_accumulates_both_edges() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1., 2., 3., 4.], &[2, 2])?;
    a.matmul(&a)?.sum()?.backward()?;
    assert_eq!(grad(&a), vec![7., 11., 9., 13.]);
    Ok(())
}

#[test]
fn empty_graph_operations() -> Result<(), TensorError> {
    let a = Tensor::zeros(&[2, 0])?;
    let b = Tensor::zeros(&[0, 3])?;
    a.matmul(&b)?.sum()?.backward()?;
    assert!(grad(&a).is_empty());
    assert!(grad(&b).is_empty());
    let x = Tensor::zeros(&[0])?;
    x.mul(&x)?.add(&x)?.sum()?.backward()?;
    assert!(grad(&x).is_empty());
    Ok(())
}
