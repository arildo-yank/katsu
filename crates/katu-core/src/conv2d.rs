//! Convolução direta CPU: NCHW na entrada e OIHW nos pesos.
use std::cell::RefCell;

use crate::{
    tensor::{element_count, zeros_buffer},
    Op, Tensor, TensorError,
};

type Result<T> = std::result::Result<T, TensorError>;

/// Correlação cruzada 2D, como nas camadas Conv2D de deep learning.
/// Sem bias, groups ou dilation. Stride e padding são iguais nos dois eixos.
/// O cache guarda apenas o último forward; cada saída mantém seu próprio grafo.
#[derive(Debug)]
pub struct Conv2D {
    weights: Tensor,
    stride: usize,
    padding: usize,
    last_forward: RefCell<Option<(Tensor, Tensor)>>,
}

impl Conv2D {
    /// Inicialização uniforme em [-1/sqrt(fan_in), 1/sqrt(fan_in)].
    /// Seed padrão fixa para reprodução; use new_seeded para variar entre camadas.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        Self::new_seeded(in_channels, out_channels, kernel_size, stride, padding, 42)
    }

    pub fn new_seeded(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
        seed: u64,
    ) -> Result<Self> {
        let shape = [out_channels, in_channels, kernel_size.0, kernel_size.1];
        validate_weights(&shape, stride)?;
        padding.checked_mul(2).ok_or(TensorError::ShapeOverflow)?;
        let mut data = zeros_buffer(element_count(&shape)?)?;
        let fan_in = element_count(&shape[1..])?;
        let bound = 1.0 / (fan_in as f32).sqrt();
        let mut state = seed;
        for value in &mut data {
            // SplitMix64: PRNG local e reproduzível; não serve para criptografia.
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            let uniform = (z >> 40) as f32 / 16_777_216.0;
            *value = (2.0 * uniform - 1.0) * bound;
        }
        Self::from_weights(Tensor::new(data, &shape)?, stride, padding)
    }

    /// Pesos explícitos no formato [out_channels, in_channels, kh, kw].
    /// Preserva identidade e grafo dos pesos recebidos.
    pub fn from_weights(weights: Tensor, stride: usize, padding: usize) -> Result<Self> {
        validate_weights(weights.shape(), stride)?;
        padding.checked_mul(2).ok_or(TensorError::ShapeOverflow)?;
        Ok(Self {
            weights,
            stride,
            padding,
            last_forward: RefCell::new(None),
        })
    }

    pub fn weights(&self) -> &Tensor {
        &self.weights
    }
    pub fn in_channels(&self) -> usize {
        self.weights.shape()[1]
    }
    pub fn out_channels(&self) -> usize {
        self.weights.shape()[0]
    }
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.weights.shape()[2], self.weights.shape()[3])
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let geometry = Geometry::new(input, &self.weights, self.stride, self.padding)?;
        let shape = geometry.output_shape();
        let mut data = zeros_buffer(element_count(&shape)?)?;
        // Padding virtual: não alocamos imagem expandida nem patches im2col.
        let (x, w) = (input.data(), self.weights.data());
        geometry.visit(|xi, wi, yi| {
            data[yi] += x[xi] * w[wi];
        });
        let output = Tensor::node(
            data,
            shape.to_vec(),
            Op::Conv2D {
                stride: self.stride,
                padding: self.padding,
            },
            vec![input.clone(), self.weights.clone()],
        );
        // A ativação é o alvo do Grad-CAM: publica seu gradiente no backward.
        output.retain_grad();
        *self.last_forward.borrow_mut() = Some((input.clone(), output.clone()));
        Ok(output)
    }

    /// VJP local do último forward bem-sucedido. Retorna folhas numéricas dX e dW;
    /// não altera os campos grad nem percorre o grafo. Para isso, use loss.backward().
    pub fn backward(&self, grad_output: &Tensor) -> Result<(Tensor, Tensor)> {
        let cached = self.last_forward.borrow();
        let (input, output) = cached.as_ref().ok_or(TensorError::MissingForward)?;
        if grad_output.shape() != output.shape() {
            return Err(TensorError::InvalidConv(
                "grad_output must match the output NCHW shape",
            ));
        }
        let (dx, dw) = backward_buffers(
            input,
            &self.weights,
            grad_output.data(),
            self.stride,
            self.padding,
        )?;
        Ok((
            Tensor::new(dx, input.shape())?,
            Tensor::new(dw, self.weights.shape())?,
        ))
    }

    /// Última saída pré-ativação, compartilhando o nó original (não faz detach).
    /// Após loss.backward(), map.grad() contém o gradiente dessa ativação.
    /// Ainda não calcula um heatmap Grad-CAM: retorna as features NCHW completas.
    pub fn get_activation_map(&self) -> Option<Tensor> {
        self.last_forward
            .borrow()
            .as_ref()
            .map(|(_, output)| output.clone())
    }

    /// Libera os handles retidos pela camada. Saídas guardadas pelo usuário continuam válidas.
    pub fn clear_cache(&self) {
        self.last_forward.borrow_mut().take();
    }
}

fn validate_weights(shape: &[usize], stride: usize) -> Result<()> {
    if shape.len() != 4 || shape.contains(&0) {
        return Err(TensorError::InvalidConv(
            "weights must have positive OIHW dimensions",
        ));
    }
    if stride == 0 {
        return Err(TensorError::InvalidConv("stride must be positive"));
    }
    Ok(())
}

struct Geometry {
    batch: usize,
    cin: usize,
    cout: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    stride: usize,
    padding: usize,
}

impl Geometry {
    fn new(input: &Tensor, weights: &Tensor, stride: usize, padding: usize) -> Result<Self> {
        validate_weights(weights.shape(), stride)?;
        if input.rank() != 4 {
            return Err(TensorError::InvalidConv("input must be rank 4 (NCHW)"));
        }
        let x = input.shape();
        let k = weights.shape();
        if x[1] != k[1] || x[2] == 0 || x[3] == 0 {
            return Err(TensorError::InvalidConv(
                "input channels must match weights; H and W must be positive",
            ));
        }
        // Mesmo batch vazio exige dimensões individuais representáveis.
        element_count(&x[1..])?;
        let pad2 = padding.checked_mul(2).ok_or(TensorError::ShapeOverflow)?;
        let output_dim = |size: usize, kernel: usize| -> Result<usize> {
            let padded = size.checked_add(pad2).ok_or(TensorError::ShapeOverflow)?;
            let span = padded
                .checked_sub(kernel)
                .ok_or(TensorError::InvalidConv("kernel exceeds padded input"))?;
            (span / stride)
                .checked_add(1)
                .ok_or(TensorError::ShapeOverflow)
        };
        let oh = output_dim(x[2], k[2])?;
        let ow = output_dim(x[3], k[3])?;
        element_count(&[k[0], oh, ow])?;
        element_count(&[x[0], k[0], oh, ow])?;
        Ok(Self {
            batch: x[0],
            cin: x[1],
            cout: k[0],
            h: x[2],
            w: x[3],
            kh: k[2],
            kw: k[3],
            oh,
            ow,
            stride,
            padding,
        })
    }

    fn output_shape(&self) -> [usize; 4] {
        [self.batch, self.cout, self.oh, self.ow]
    }

    /// Visita apenas as contribuições válidas (zero-padding não contribui).
    /// O loop kw lê posições contíguas da imagem e do filtro.
    fn visit(&self, mut contribution: impl FnMut(usize, usize, usize)) {
        for n in 0..self.batch {
            for oc in 0..self.cout {
                for oy in 0..self.oh {
                    for ox in 0..self.ow {
                        let yi = ((n * self.cout + oc) * self.oh + oy) * self.ow + ox;
                        for ic in 0..self.cin {
                            for ky in 0..self.kh {
                                // oy*stride+ky <= padded_height-1 pela fórmula do output.
                                let py = oy * self.stride + ky;
                                let Some(iy) = py.checked_sub(self.padding).filter(|&y| y < self.h)
                                else {
                                    continue;
                                };
                                let xbase = ((n * self.cin + ic) * self.h + iy) * self.w;
                                let wbase = ((oc * self.cin + ic) * self.kh + ky) * self.kw;
                                for kx in 0..self.kw {
                                    let px = ox * self.stride + kx;
                                    if let Some(ix) =
                                        px.checked_sub(self.padding).filter(|&x| x < self.w)
                                    {
                                        contribution(xbase + ix, wbase + kx, yi);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Mesma implementação para backward local e para o motor de autograd.
/// Y[y] += X[x]*W[w] implica dX[x] += G[y]*W[w] e dW[w] += G[y]*X[x].
/// += soma contribuições de janelas sobrepostas, canais e itens do batch.
/// Não se deve inverter novamente o filtro: os índices do forward já definem a derivada.
pub(crate) fn backward_buffers(
    input: &Tensor,
    weights: &Tensor,
    grad: &[f32],
    stride: usize,
    padding: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let geometry = Geometry::new(input, weights, stride, padding)?;
    let expected = element_count(&geometry.output_shape())?;
    if grad.len() != expected {
        return Err(TensorError::DataLength {
            expected,
            actual: grad.len(),
        });
    }
    let mut dx = zeros_buffer(input.numel())?;
    let mut dw = zeros_buffer(weights.numel())?;
    let (x, w) = (input.data(), weights.data());
    geometry.visit(|xi, wi, yi| {
        dx[xi] += grad[yi] * w[wi];
        dw[wi] += grad[yi] * x[xi];
    });
    Ok((dx, dw))
}
