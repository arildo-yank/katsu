//! Treino mínimo: regressão linear y = 2x + 1 com SGD.
//!
//! Emite uma linha JSON por época (formato do monitor arandu):
//!   cargo run -p katsu-core --example train                       # stdout
//!   cargo run -p katsu-core --example train -- metrics.jsonl      # arquivo
use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use katsu_core::{fit, Sgd, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xs: Vec<f32> = (0..32).map(|i| -1. + i as f32 * 2. / 31.).collect();
    let x = Tensor::new(xs.iter().flat_map(|&x| [x, 1.]).collect(), &[32, 2])?;
    let neg_y = Tensor::new(xs.iter().map(|&x| -(2. * x + 1.)).collect(), &[32, 1])?;
    let inv_n = Tensor::new(vec![1. / 32.], &[])?;

    let mut out: Box<dyn Write> = match std::env::args().nth(1) {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(io::stdout().lock()),
    };
    let mut write_error = None;
    let params = fit(
        vec![Tensor::zeros(&[2, 1])?],
        &Sgd::new(0.1)?,
        200,
        |p| {
            let residual = x.matmul(&p[0])?.add(&neg_y)?;
            residual.mul(&residual)?.sum()?.mul(&inv_n)
        },
        |metrics| {
            // Flush por época: o monitor lê o arquivo enquanto o treino roda.
            if write_error.is_none() {
                write_error = writeln!(out, "{metrics}").and_then(|_| out.flush()).err();
            }
        },
    )?;
    if let Some(error) = write_error {
        return Err(error.into());
    }
    let w = params[0].data();
    eprintln!("aprendido: y = {:.4}x + {:.4} (alvo: 2x + 1)", w[0], w[1]);
    Ok(())
}
