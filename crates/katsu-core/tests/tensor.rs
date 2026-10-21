use katsu_core::{Tensor, TensorError};

#[test]
fn validates_data_and_storage_size() {
    assert!(matches!(
        Tensor::new(vec![1.0], &[2]),
        Err(TensorError::DataLength { .. })
    ));
    assert!(matches!(
        Tensor::zeros(&[usize::MAX, 2]),
        Err(TensorError::ShapeOverflow)
    ));
    assert!(matches!(
        Tensor::zeros(&[isize::MAX as usize]),
        Err(TensorError::ShapeOverflow)
    ));
}

#[test]
fn scalar_and_empty_shapes() -> Result<(), TensorError> {
    assert_eq!(Tensor::new(vec![3.0], &[])?.rank(), 0);
    assert_eq!(Tensor::zeros(&[])?.data(), &[0.0]);
    assert_eq!(Tensor::zeros(&[usize::MAX, 0])?.numel(), 0);
    Ok(())
}

#[test]
fn addition_preserves_inputs() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1.0, 2.0], &[2])?;
    let b = Tensor::new(vec![3.0, 4.0], &[2])?;
    assert_eq!(a.add(&b)?.data(), &[4.0, 6.0]);
    assert_eq!(a.data(), &[1.0, 2.0]);
    assert!(matches!(
        a.add(&Tensor::zeros(&[3])?),
        Err(TensorError::ShapeMismatch)
    ));
    Ok(())
}

#[test]
fn rectangular_matmul() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3])?;
    let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], &[3, 2])?;
    let c = a.matmul(&b)?;
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[58., 64., 139., 154.]);
    Ok(())
}

#[test]
fn rejects_invalid_matmul() -> Result<(), TensorError> {
    let a = Tensor::zeros(&[2, 3])?;
    assert!(matches!(
        a.matmul(&Tensor::zeros(&[2, 2])?),
        Err(TensorError::MatmulMismatch { .. })
    ));
    assert!(matches!(
        a.matmul(&Tensor::zeros(&[3])?),
        Err(TensorError::RankMismatch { .. })
    ));
    Ok(())
}

#[test]
fn empty_matmul_and_output_overflow() -> Result<(), TensorError> {
    let c = Tensor::zeros(&[2, 0])?.matmul(&Tensor::zeros(&[0, 3])?)?;
    assert_eq!(c.shape(), &[2, 3]);
    assert_eq!(c.data(), &[0.; 6]);
    assert_eq!(
        Tensor::zeros(&[0, 2])?
            .matmul(&Tensor::zeros(&[2, 3])?)?
            .shape(),
        &[0, 3]
    );
    assert_eq!(
        Tensor::zeros(&[usize::MAX, 0])?
            .matmul(&Tensor::zeros(&[0, 0])?)?
            .numel(),
        0
    );
    assert!(matches!(
        Tensor::zeros(&[usize::MAX, 0])?.matmul(&Tensor::zeros(&[0, 2])?),
        Err(TensorError::ShapeOverflow)
    ));
    Ok(())
}
