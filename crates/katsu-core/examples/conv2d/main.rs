use katsu_core::{Conv2D, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // Uma imagem RGB sintética em NCHW, valores normalizados para [0,1].
    let pixels = (0..3 * 224 * 224)
        .map(|i| (i % 256) as f32 / 255.)
        .collect();
    let image = Tensor::new(pixels, &[1, 3, 224, 224])?;
    let conv = Conv2D::new_seeded(3, 8, (3, 3), 1, 1, 123)?;
    let output = conv.forward(&image)?;
    println!("Entrada: {:?}; saída: {:?}", image.shape(), output.shape());

    // Loss de demonstração: soma de quadrados das features, alvo zero.
    let loss = output.mul(&output)?.sum()?;
    loss.backward()?;
    println!("Loss: {}", loss.data()[0]);
    println!("dX: {:?}; dW: {:?}", image.shape(), conv.weights().shape());
    println!(
        "Primeiros gradientes dX: {:?}",
        image.grad()?.as_ref().map(|g| &g[..5])
    );

    // Mesmo nó do forward: guarda features e, após backward, seus gradientes.
    if let Some(activations) = conv.get_activation_map() {
        println!(
            "Ativações: {:?}; gradiente disponível: {}",
            activations.shape(),
            activations.grad()?.is_some()
        );
    }
    // Para L=sum(Y²), dL/dY=2Y. O backward local retorna os mesmos adjuntos,
    // mas não modifica os gradientes já preenchidos pelo autograd.
    let grad_output = Tensor::new(
        output.data().iter().map(|v| 2. * v).collect(),
        output.shape(),
    )?;
    let (dx, dw) = conv.backward(&grad_output)?;
    println!("Backward local: dX {:?}, dW {:?}", dx.shape(), dw.shape());
    Ok(())
}
