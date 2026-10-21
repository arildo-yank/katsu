# Treinamento e telemetria

[Índice](README.md)

## SGD sem mutação

Os dados de um `Tensor` são imutáveis, porque o backward lê os valores do forward.
Por isso `Sgd::step` não altera parâmetros no lugar: devolve **novas folhas**
`p - lr * grad`, na mesma ordem e com os mesmos shapes. Consequências:

- grafos e parâmetros anteriores continuam válidos;
- os novos parâmetros começam sem gradiente, então não há `zero_grad` por passo;
- `step` falha com `MissingGradient { index }` se algum parâmetro não tiver grad,
  sem devolver nenhum parâmetro parcial.

```rust
use katsu_core::{fit, Sgd, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    // y = 2x + 1; o bias entra como uma coluna de uns (não há broadcasting).
    let x = Tensor::new(vec![-1., 1., 0., 1., 1., 1.], &[3, 2])?;
    let neg_y = Tensor::new(vec![1., -1., -3.], &[3, 1])?;
    let params = fit(
        vec![Tensor::zeros(&[2, 1])?],
        &Sgd::new(0.1)?,
        200,
        |p| {
            let r = x.matmul(&p[0])?.add(&neg_y)?;
            r.mul(&r)?.sum()
        },
        |metrics| println!("{metrics}"),
    )?;
    println!("{:?}", params[0].data()); // ≈ [2, 1]
    Ok(())
}
```

## Loop `fit`

Em cada época, `fit` executa: `loss_fn(&params)` → backward → `on_epoch(métricas)`
→ `step`. A perda precisa ser escalar (`NonScalarBackward`) e finita. Uma perda
NaN/Inf interrompe o treino com `NonFiniteLoss { epoch }`, antes de emitir métricas
dessa época; as épocas anteriores já foram emitidas.

## Contrato com o monitor arandu

Cada `EpochMetrics` formatado com `Display` é **uma linha JSON** (JSON Lines):

```json
{"epoch":86,"loss":0.114578,"learning_rate":0.001}
```

| Campo | Tipo | Significado |
| --- | --- | --- |
| `epoch` | inteiro ≥ 1 | Época concluída |
| `loss` | número finito | Perda antes da atualização dessa época |
| `learning_rate` | número > 0 | Taxa usada no passo |

Os números nunca são NaN/Inf, então a linha é sempre JSON válido. O exemplo
executável grava uma linha por época e faz flush a cada uma, para o monitor
acompanhar o arquivo enquanto o treino roda:

```sh
cargo run -p katsu-core --example train -- metrics.jsonl
```

Sem argumento, as linhas vão para stdout e o resumo final vai para stderr.

## Limites atuais

Taxa de aprendizado fixa, sem momentum, Adam, mini-batches ou agendamento de lr.
O treino roda no processo do chamador, em uma thread; não há servidor HTTP.
