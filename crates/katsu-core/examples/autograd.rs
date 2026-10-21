use katsu_core::{Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::new(vec![1., 2.], &[2])?;
    let b = Tensor::new(vec![3., 4.], &[2])?;
    let prediction = a.add(&b)?;
    // Erro quadrático contra alvo zero: L = sum((a+b)²).
    let loss = prediction.mul(&prediction)?.sum()?;
    loss.backward()?;
    println!("loss = {:?}", loss.data()); // [52]
    println!("dL/da = {:?}", a.grad()?); // Some([8, 12])
    println!("dL/db = {:?}", b.grad()?); // Some([8, 12])
    Ok(())
}
