# Conv2D e explicabilidade

[Índice](README.md) · [Implementação](../crates/katsu-core/src/conv2d.rs)

![Layouts e fluxo de derivadas da Conv2D](assets/conv2d.svg)

## Shapes e parâmetros

Entrada X: `[N, Cin, H, W]`. Pesos W: `[Cout, Cin, Kh, Kw]`.
Saída Y: `[N, Cout, Hout, Wout]`.

```text
Hout = floor((H + 2*padding - Kh) / stride) + 1
Wout = floor((W + 2*padding - Kw) / stride) + 1
```

Stride e padding são escalares aplicados igualmente aos dois eixos. Kernel pode
ser retangular. Canais e dimensões espaciais devem ser positivos; batch zero é
aceito. Stride deve ser positivo. Dimensões e aritmética de tamanhos são validadas.

O forward usa correlação cruzada: não inverte o kernel. Essa convenção é usada
em [camadas Conv2D de deep learning](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html).
Não há bias, groups, dilation ou ativação embutida.

## Forward e derivadas

Para uma saída `(n, oc, oy, ox)`, percorremos canais e posições do filtro:

```text
iy = oy*stride + ky - padding
ix = ox*stride + kx - padding
Y[n,oc,oy,ox] += X[n,ic,iy,ix] * W[oc,ic,ky,kx]
```

Posições fora da imagem equivalem a zero. A implementação usa subtração checada
antes de indexar slices, evitando underflow de `usize`. Não materializa padding.

Se G é o gradiente recebido em Y, a regra da cadeia para cada termo acima é:

```text
dX[n,ic,iy,ix] += G[n,oc,oy,ox] * W[oc,ic,ky,kx]
dW[oc,ic,ky,kx] += G[n,oc,oy,ox] * X[n,ic,iy,ix]
```

O `+=` soma janelas sobrepostas, canais de saída e itens do batch. A implementação
visita os mesmos índices do forward; não é necessário inverter explicitamente
o kernel nesse método. Formulações alternativas com convoluções transpostas
precisam tratar rotação, stride e bordas conforme sua convenção.

## Inicialização e acesso aos pesos

`new(...)` usa seed fixa 42. `new_seeded(..., seed)` permite escolher uma seed por
camada. O PRNG local é SplitMix64, usado para amostrar pesos uniformes em
`[-1/sqrt(fan_in), +1/sqrt(fan_in)]`, onde `fan_in=Cin*Kh*Kw`.

Duas camadas com mesma configuração e seed começam com pesos iguais. Para pesos
pré-definidos, use `from_weights(tensor, stride, padding)`; o nó do Tensor é preservado.
`weights()` expõe esse nó para leitura e inspeção dos gradientes, não para mutação.

## Dois caminhos de backward

| API | Contexto | Resultado |
| --- | --- | --- |
| `loss.backward()` | Grafo da loss e todos os ancestrais | Acumula campos grad |
| `conv.backward(&grad_output)` | Último forward válido da camada | Retorna folhas `(dX,dW)`, sem alterar grad |

Cada saída guarda seus próprios pais no grafo. Reutilizar a camada não muda
saídas anteriores. O cache local guarda apenas o último forward bem-sucedido;
um forward que falha preserva o cache anterior. `clear_cache()` o libera.

O grad_output local deve ter exatamente o shape NCHW da saída, não apenas o
mesmo número de elementos. Chamar backward local sem cache retorna `MissingForward`.

## Exemplo numérico completo

```rust
use katsu_core::{Conv2D, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let x = Tensor::new((1..=9).map(|n| n as f32).collect(), &[1,1,3,3])?;
    let w = Tensor::new(vec![1.,2.,3.,4.], &[1,1,2,2])?;
    let conv = Conv2D::from_weights(w, 1, 0)?;
    let y = conv.forward(&x)?;
    assert_eq!(y.data(), &[37.,47.,67.,77.]);
    let seed = Tensor::new(vec![1.; 4], y.shape())?;
    let (dx, dw) = conv.backward(&seed)?;
    assert_eq!(dx.data(), &[1.,3.,2.,4.,10.,6.,3.,7.,4.]);
    assert_eq!(dw.data(), &[12.,16.,24.,28.]);
    y.sum()?.backward()?;
    assert_eq!(x.grad()?.as_deref(), Some(dx.data()));
    Ok(())
}
```

Para RGB 224×224, execute o [exemplo maior](../crates/katsu-core/examples/conv2d/main.rs):

```sh
cargo run -p katsu-core --release --example conv2d
```

## O que falta para Grad-CAM

`get_activation_map()` retorna `Option<Tensor>` com as features pré-ativação do
último forward. O Tensor compartilha a identidade da saída; depois de backward,
`grad()` fornece as derivadas do objetivo em relação a essas features.

Isso é a matéria-prima, não o heatmap. Um futuro método deverá escolher um score
alvo, calcular os pesos por canal a partir dos gradientes espaciais, combinar as
features, aplicar o tratamento do mapa e redimensionar para a imagem original.
Não há atualmente classificador, score de classe, normalização ou overlay integrado.
Um mapa de atribuição não deve ser anunciado como prova causal ou explicação exata.

## Performance e memória

Custo aproximado do kernel: `N*Cout*Hout*Wout*Cin*Kh*Kw`. Os loops internos acessam
posições contíguas do filtro e da imagem. Não há alocação por janela, im2col ou
cópia da imagem com padding. Forward aloca a saída; backward local aloca dX e dW;
o autograd também mantém seus buffers de acumulação.

Ainda não há benchmark comparativo nem estimativa de latência garantida. Otimizações
com blocagem, SIMD, threads ou GPU dependerão de medições reproduzíveis.
