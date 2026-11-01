# Fase 1: relatório das correções

Data: 01/11/2026. Branch `fix/motor-computacional-fase1`, a partir de KATSU `2709b2b`, na worktree
`/Volumes/500G DEV/KATSU-motor-fase1`. Commits de `c17cef1` a `b76780c`. Depois integrados em `feat/katsu-lm-v1`, sem push, e compilados em `target/release/katu-server` em 01/11/2026 às 20:10; o app
só passa a usá-los ao abrir de novo.

## Os seis defeitos

Cada reprodução foi executada como teste antes da correção e falhou. Na mesma suíte há testes de preservação, como matriz
singular e broadcast de tensor vazio, que já passavam; e os de rótulos vieram depois, junto com a mudança.

| # | Reprodução | Antes | Agora | Local |
|---|---|---|---|---|
| 1 | Bisseção de `1/(x-0.1)` em `[0,1]` | Polo como raiz | Recusa: polo ou salto | `katu-math/src/lib.rs` |
| 2 | Primitiva de `0.0000000001*x` | `0 + C` | `0.00000000005 * x^2 + C` | `katu-math/src/integral.rs` |
| 3 | `det(diag(1e13, 1))` | `0` | `1e13` | `katu-math/src/linalg.rs` |
| 4 | Derivada de `ln(x)` em `x=-1` | `-1` | Recusa: fora do domínio | `katu-math/src/lib.rs`, `katu-server/src/math.rs` |
| 5 | NdArray: broadcast para `[usize::MAX, 2]` | Aceito | `ArrayError::Overflow` | `katu-core/src/array.rs` |
| 6 | Concat de `[usize::MAX, 0]` duas vezes | Soma estoura | `TensorError::ShapeOverflow` | `katu-core/src/ops.rs` |

Rótulos de precisão: `eval` agora só diz `approximate: false` quando a conta inteira ficou em inteiros abaixo de 2^53, e
`derive` informa o valor no ponto em `value_at_approximate`.

### Correções vizinhas, da mesma causa

- `inverse(diag(1e13, 1))` também recusava como singular.
- `det` devolvia zero por underflow em `diag(1e-200, 1e-200)` e estourava no meio do produto em
  `diag(1e200, 1e200, 1e-300)`. Agora recusa o que não cabe em f64 e calcula o que cabe.
- A conferência de primitivas tinha folga absoluta, que aprovava qualquer função pequena. Virou relativa.
- `0.3000000001` era impressa como `3/10`. A fração só absorve arredondamento.
- `NdArray::concatenate` tinha a mesma soma sem conferência do defeito 6.
- `Tensor::concat` lado a lado de vazios com `usize::MAX` linhas percorreria todas elas.
- `broadcast_map` multiplicava passos de saída vazia e podia estourar.
- A busca de raiz por texto parava em polo: em `1/(x-0.12) + 1 = 0`, agora passa dele e chega à raiz em -0,88.

## Mudanças de comportamento para consumidores

Todas em `/math`, sem campo removido:

1. Polos e saltos deixam de sair como raiz e voltam como erro 400.
2. Derivada fora do domínio volta como erro 400.
3. `det` só devolve `0` quando o pivô não se distingue do arredondamento. `[[1,1],[1,1+1e-10]]` devolvia 0 e agora devolve
   cerca de 1e-10, com a precisão que esse condicionamento permite.
4. `inv` julga a singularidade na matriz balanceada. Matrizes só mal escaladas passam a ter inversa; quase singulares
   seguem recusadas.
5. `eval` com decimais agora marca `approximate: true`, e ganhou nota de arredondamento. Na tela de matemática do Arandu
   (`arandu-desktop/src/Matematica.jsx`), o selo dessas contas passa de "Calculado" para "Aproximado", mesmo em
   `15/100*2300`, que dá 345 em f64 mas passa por 0,15. É conservador e intencional. O modelo recebe o campo no JSON da ferramenta.

## O que foi medido

Apple M3, macOS, rustc 1.99.1, modo debug salvo indicação.

| Conjunto | Referência da auditoria | Agora |
|---|---|---|
| katu-core | 112 | 116 aprovados, 0 falhas |
| katu-math | 22 | 33 aprovados, 0 falhas |
| katu-server, rota de matemática | 5 | 7 aprovados, 0 falhas |
| Total comparável | 139 | 156 aprovados, 0 falhas |
| katu-server, todos | não citado | 36 aprovados, 0 falhas |

Em modo release, com LTO: katu-math 33, testes `array` e `ops` do core 9 e 13, rota de matemática 7, todos aprovados. Isto
importa porque estouro de `usize` se comporta de outro jeito nos dois modos.

Passos do CI, executados localmente com `--offline`: `fmt`, testes, `clippy -D warnings` em core e math, exemplos `conv2d`
e `train`, `cargo check -p katu-python`, SBOM ("em dia: 69 componentes"). Lint do `katu-server` não entrou no CI, porque
`train_sft` já falhava com `-D warnings` antes destas mudanças.

Integração com o consumidor, no ARANDU `1d3bac0` sem alterações:

- `arandu-web/test_math.py`: 17 aprovados, com serviços simulados.
- `math_tool.run` por HTTP real, com `KATU_MATH_URL` apontando para um servidor temporário na porta 18792 e `KATU_URL`
  diferente dele: 7 chamadas, com recusas e resultados corretos. O servidor já foi desligado.

Comparação com referência independente, `scripts/math_reference/check.py`, semente 20261101, NumPy 2.4.6, SciPy 1.17.1:

| Grupo | Resultado |
|---|---|
| Reproduções pela rota HTTP (defeitos 1 a 4) | 4/4 |
| `det` contra NumPy, com escala por linha de 1e-13 a 1e13 | 500/500 |
| `inv`, resíduo | 500/500 |
| Matriz singular: `inv` recusa | 100/100 |
| Matriz singular: `det` dá zero | 100/100 |
| Raiz contra `brentq` | 500/500 |
| Polo recusado | 500/500 |
| Primitiva contra `quad` | 500/500 |
| Derivada contra diferença central | 385/385 |
| Derivada fora do domínio recusada | 115/115 |

É uma amostra escolhida para os defeitos corrigidos, não certificação do motor. A folga de `det` e `inv` cresce com o
condicionamento da matriz sorteada, e casos com condicionamento acima de 1e8 ficaram fora.

Custo da mudança na bisseção: exemplo `tempo` em modo debug mediu 26 µs. Não há medida equivalente em release nem de antes
da mudança, então nenhuma comparação é afirmada.

## O que não foi medido

- **CUDA não foi validado.** Os testes que exigem GPU NVIDIA ficam ignorados neste Mac.
- **Linux x86 e GitHub Actions não rodaram**: nada foi enviado, então a alteração em `ci.yml` só foi conferida executando
  os mesmos comandos no Mac.
- **Serviços instalados:** o app não foi reiniciado nem reinstalado. A 8792 não estava escutando durante os testes.
- **Fluxo com o DeepSeek** não rodou, por custar API; a volta da recusa para o modelo só foi testada com simulação.
- **Paridade de treino:** o único código do caminho de treino alterado é `Tensor::concat` e `broadcast_map`, em ramos de
  erro e de saída vazia. Os testes de core passam; não houve treino de comparação, nem era autorizado.

## Limites das correções

- A bisseção decide por amostragem: |f| precisa cair à metade em dez metades do intervalo, ou o resíduo cair abaixo de
  1e-6 do valor inicial. Função contínua com expoente de Hölder até 0,1 é recusada sem ser polo. Polo ou salto minúsculo,
  abaixo de 1e-6 da escala da função, pode passar. Nenhuma amostragem finita detecta tudo.
- Resíduo pequeno não é raiz precisa: no polinômio de grau 9 expandido de `(x-1)^9`, o teste aceita folga de 0,1 em `x`.
- `det` em ponto flutuante ainda devolve `0` para matrizes indistinguíveis de singulares. No contrato novo isso será
  `inconclusive`.
- Coeficientes que estouram para zero já no f64 (`1e-200 * 1e-200`) continuam zero. Só o tipo exato resolve.

## Lacunas encontradas e não corrigidas

Fora dos seis defeitos; viraram tarefas em [KATU_MARCO1_TAREFAS.md](KATU_MARCO1_TAREFAS.md).

| ID | Observado | Tarefa |
|---|---|---|
| L1 | **Newton devolvia assíntota como raiz:** `1/x` a partir de 1 dava `1099511627776`, `exp(-x)` dava `28`. Corrigido em `feat/motor-contrato-v1`, já integrada | B1, feita |
| L2 | `0^0` retorna `1` sem declarar a convenção | B2 |
| L3 | O LaTeX arredonda a entrada mostrada: `1.000000000000001` aparece como `1` | B3 |
| L4 | `inv` dizia "singular" também para matriz apenas quase singular. Corrigido na mesma branch | B4, feita |
| L5 | Autograd de concat vazio com dimensão enorme percorreria as linhas. Não testado com `backward`; deixado para depois do piloto | B5 |
| L6 | Integral num polo recusa com "não convergiu": correto, mas sem diagnóstico | B6 |
| L7 | O servidor precisa de checkpoint válido só para responder contas | A6 |

## Estado preservado

- KATSU: HEAD `2709b2b` no início; alterações locais e arquivos não rastreados intactos; `KATU_ARRAY_PLAN.md`,
  `KATU_MATH_PLAN.md`, formatos de checkpoint e tokenizador sem mudança.
- ARANDU: HEAD `1d3bac0`, sem nenhuma alteração desta tarefa.
- NdArray e GEMM novo continuam fora do caminho de treino. Armazenamento do Tensor e GEMM do `CpuBackend` não mudaram.
- Piloto 1B em andamento e não tocado: guarda `guarda_retoma.py` vivo; último registro lido às 19:15, passo 210.000,
  loss de validação 3,127, cerca de US$ 16,90.
