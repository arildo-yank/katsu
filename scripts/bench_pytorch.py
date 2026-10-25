"""Treina o mesmo 134M do Katsu em PyTorch, sobre os lotes exatos que o Katsu treinou, para
comparar velocidade com os mesmos dados na mesma GPU.

  # 1) o Katsu grava os lotes de cada passo
  KATSU_ENGINE=gpu KATSU_DUMP_BATCHES=batches.jsonl ... train_sft ... runs/bench-katsu 100 0 0.0003
  # 2) o PyTorch treina os mesmos lotes
  python3 scripts/bench_pytorch.py batches.jsonl --mode fp32 --out runs/bench-torch-fp32.json

Modos:
  fp32  matmul em FP32 puro (sem TF32) e atenção explícita: as mesmas contas do Katsu (tiled64)
  tf32  matmul em TF32 e scaled_dot_product_attention (o PyTorch padrão numa GPU moderna)
  bf16  autocast BF16, FlashAttention e Adam fused (o PyTorch otimizado; --compile soma torch.compile)

Com --weights runs/<run>/migrated.ckpt (e o config.json ao lado), o PyTorch parte dos mesmos pesos
do Katsu; no modo fp32 a perda do passo 1 tem de bater com a do Katsu.

Mesma arquitetura (pre-LN, GELU tanh, sem bias em Q/K/V e Wo, cabeça sem bias), mesma perda
(entropia cruzada ponderada por linha, como o SftBatch) e mesmo Adam (lr 3e-4, betas 0.9/0.999,
eps 1e-8). Tokens/s contam só os tokens reais de entrada, como no Katsu.
"""
import argparse
import json
import math
import time

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from export_to_gguf import checkpoint_index

VOCAB, CONTEXT, DIM, LAYERS, HEADS = 32000, 256, 768, 12, 12


class Block(nn.Module):
    def __init__(self, d, h, flash):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.heads, self.flash = h, flash

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(d // self.heads)
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
            y = att.masked_fill(mask, float("-inf")).softmax(-1) @ v
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class LM(nn.Module):
    def __init__(self, flash):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, DIM)
        self.wpe = nn.Embedding(CONTEXT, DIM)
        self.blocks = nn.ModuleList(Block(DIM, HEADS, flash) for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(DIM, eps=1e-5)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, tokens):
        t = tokens.shape[1]
        x = self.wte(tokens) + self.wpe(torch.arange(t, device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.lnf(x))


def load_katsu(model, ckpt):
    """Copia um checkpoint KATSUCK1 para o modelo. O Katsu multiplica x·W com W [entrada, saída];
    o nn.Linear guarda [saída, entrada], então as matrizes entram transpostas. Q, K e V de cada
    cabeça viram colunas [q_0..q_H | k_0..k_H | v_0..v_H], na ordem do split do forward."""
    config = json.load(open(Path(ckpt).parent / "config.json"))
    raw = np.memmap(ckpt, dtype=np.uint8, mode="r")
    tensors = {name: np.frombuffer(raw[off:off + n], dtype="<f4").reshape(dims)
               for name, dims, off, n in checkpoint_index(ckpt, config)}
    t = lambda a: torch.from_numpy(np.array(a, dtype=np.float32))  # noqa: E731
    with torch.no_grad():
        model.wte.weight.copy_(t(tensors["token_embd.weight"]))
        model.wpe.weight.copy_(t(tensors["position_embd.weight"]))
        for i, block in enumerate(model.blocks):
            p = f"blk.{i}"
            block.ln1.weight.copy_(t(tensors[p + ".attn_norm.weight"]))
            block.ln1.bias.copy_(t(tensors[p + ".attn_norm.bias"]))
            cols = [tensors[f"{p}.head.{h}.attn_{n}.weight"] for n in "qkv" for h in range(HEADS)]
            block.qkv.weight.copy_(t(np.concatenate(cols, axis=1)).T)
            rows = [tensors[f"{p}.head.{h}.attn_output.weight"] for h in range(HEADS)]
            block.proj.weight.copy_(t(np.concatenate(rows, axis=0)).T)
            block.ln2.weight.copy_(t(tensors[p + ".ffn_norm.weight"]))
            block.ln2.bias.copy_(t(tensors[p + ".ffn_norm.bias"]))
            block.fc1.weight.copy_(t(tensors[p + ".ffn_up.weight"]).T)
            block.fc1.bias.copy_(t(tensors[p + ".ffn_up.bias"]))
            block.fc2.weight.copy_(t(tensors[p + ".ffn_down.weight"]).T)
            block.fc2.bias.copy_(t(tensors[p + ".ffn_down.bias"]))
        model.lnf.weight.copy_(t(tensors["output_norm.weight"]))
        model.lnf.bias.copy_(t(tensors["output_norm.bias"]))
        model.head.weight.copy_(t(tensors["output.weight"]).T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batches")
    parser.add_argument("--mode", choices=["fp32", "tf32", "bf16"], required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weights", help="checkpoint KATSUCK1 de partida (com config.json ao lado)")
    parser.add_argument("--out")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = args.mode != "fp32"
    torch.backends.cudnn.allow_tf32 = args.mode != "fp32"
    device = "cuda"
    model = LM(flash=args.mode != "fp32")
    if args.weights:
        load_katsu(model, args.weights)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    if params != 134_367_744:
        raise SystemExit(f"modelo diferente do Katsu: {params} parâmetros")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                           fused=args.mode == "bf16")
    step_fn = torch.compile(model) if args.compile else model

    batches = [json.loads(line) for line in open(args.batches)]
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for i, b in enumerate(batches):
        shape = (b["batch"], b["width"])
        tokens = torch.tensor(b["tokens"], device=device).view(shape)
        targets = torch.tensor(b["targets"], device=device).view(-1)
        weights = torch.tensor(b["weights"], device=device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.mode == "bf16"):
            logits = step_fn(tokens)
        per_row = F.cross_entropy(logits.view(-1, VOCAB).float(), targets, reduction="none")
        loss = (per_row * weights).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        rows.append({"step": b["step"], "seconds": time.perf_counter() - started,
                     "input_tokens": b["input_tokens"], "padded_tokens": shape[0] * shape[1],
                     "loss": loss.item()})
    timed = rows[args.warmup:]
    seconds = sum(r["seconds"] for r in timed)
    summary = {
        "framework": f"PyTorch {torch.__version__}", "mode": args.mode, "compile": args.compile,
        "gpu": torch.cuda.get_device_name(), "parameters": params, "steps": len(rows),
        "timed_steps": len(timed), "seconds_per_step": seconds / len(timed),
        "real_tokens_per_second": sum(r["input_tokens"] for r in timed) / seconds,
        "padded_tokens_per_second": sum(r["padded_tokens"] for r in timed) / seconds,
        "peak_memory_mib": torch.cuda.max_memory_allocated() // (1 << 20),
        "weights": args.weights, "first_loss": rows[0]["loss"], "last_loss": rows[-1]["loss"],
    }
    print(json.dumps(summary))
    if args.out:
        json.dump({"summary": summary, "steps": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
