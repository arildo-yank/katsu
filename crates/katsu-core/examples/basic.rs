use katsu_core::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], &[2, 3])?;
    let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], &[3, 2])?;
    println!("A + A = {:?}", a.add(&a)?);
    println!("A @ B = {:?}", a.matmul(&b)?);
    Ok(())
}
