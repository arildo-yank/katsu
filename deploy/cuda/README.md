# Validação CUDA em Linux/NVIDIA

Backend **experimental e híbrido**. Matmul, add e cross-entropy e seus VJPs
usam kernels CUDA escritos no projeto, carregados por cudarc/NVRTC. LayerNorm,
GELU, softmax, gather, operações de forma e Adam continuam na CPU. Cada kernel
transfere buffers host/device. Isto ainda não é um trainer inteiramente GPU.

Não foi alugado servidor nem configurada conta Hostinger. Este ambiente pode ser
usado num host Linux com GPU NVIDIA, driver e NVIDIA Container Toolkit disponíveis.
Não há preço/capacidade de provedor presumidos.

```sh
docker build -t katsu-cuda -f deploy/cuda/Dockerfile .
docker run --rm --gpus all -v "$PWD:/workspace" katsu-cuda
```

O container não publica portas. Faz download das dependências de build, mas não
baixa pesos/dados, não usa APIs de modelos e não faz upload do corpus.
O comando padrão roda a suíte CPU e depois **obrigatoriamente os testes GPU**:
1. kernels versus CPU, incluindo matmul retangular, soma com broadcast e CE;
2. dez passos Adam do modelo de 2.061.152 parâmetros, com mesmos seeds e janelas,
   comparação de todos os gradientes e todos os pesos a cada passo.

Tolerâncias declaradas nos testes: kernels `atol=2e-5, rtol=2e-5`; treino
`grad atol=3e-4, rtol=3e-3`; pesos `atol=5e-4, rtol=5e-3`. O teste imprime o erro
absoluto máximo por passo. Broadcast backward usa atomicAdd, cuja ordem não é
bit a bit determinística. As tolerâncias são critérios propostos, **não resultados
validados**: não houve GPU disponível nesta entrega.

Somente depois de os testes GPU passarem, dimensione o host:

```sh
python3 scripts/scale_preflight.py --tokenizer runs/native-smoke-20261024/tokenizer.bpe --output preflight-linux.json
```

Os dois perfis são aproximados: com 32k tokens têm 141.649.152 e 343.089.152
parâmetros. Para um smoke inicial, use sequência 16, batch 1 e dez passos.
A tabela de posições continua em 512: isso não reduz a arquitetura dos pesos.
O orçamento deve caber na memória **disponível** do host, deixando margem para
processos, contexto CUDA e cópias. Não copie cegamente o exemplo de orçamento:

```sh
cargo run --release -p katsu-core --features cuda --example train_native -- \
  --corpus corpus --tokenizer runs/native-smoke-20261024/tokenizer.bpe \
  --profile 139m --batch-size 1 --sequence-length 16 --steps 10 \
  --backend cuda --memory-budget-mib 8192 --output runs/cuda-139m-001
# Para 349m, faça novo preflight e aumente o orçamento somente se o host comportar.
```

O orçamento padrão de 1024 MiB bloqueia treinos grandes acidentais. A estimativa
não garante ausência de OOM. O backend CUDA é um passo de correção funcional,
não uma promessa de performance para 139M/349M. Permanência de tensores no device,
kernels restantes, Adam GPU, reduções otimizadas e mixed precision são próximos trabalhos.
