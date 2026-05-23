#!/usr/bin/env python3
import os, sys
import torch
from types import SimpleNamespace

# make nanoGPT importable when running from workspace root
sys.path.insert(0, os.path.join(os.getcwd(), "nanoGPT"))
from model import GPT, GPTConfig
# MiniBlock is in workspace root
from mini_block import MiniBlock

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def human(n):
    for u in ['','K','M','B']:
        if abs(n) < 1000.0:
            return f"{n:,.0f}{u}"
        n /= 1000.0
    return f"{n:,.1f}T"

def report_model(name, model):
    params = count_params(model)
    flops = params * 2  # per spec: "2 × params for one forward"
    print(f"{name}:")
    print(f"  params: {params:,} ({human(params)})")
    print(f"  FLOPs (1 forward, approx): {flops:,} ({human(flops)})")
    print()

def make_tiny_gpt():
    # tiny config: 2 layers, 2 heads, 128 emb, small block size
    cfg = GPTConfig(block_size=32, vocab_size=50304, n_layer=2, n_head=2, n_embd=128, dropout=0.0, bias=True)
    return GPT(cfg), cfg

def make_miniblock():
    blk = MiniBlock(n_embd=128, n_head=2)
    return blk

def main():
    torch.set_grad_enabled(False)
    gpt, cfg = make_tiny_gpt()
    blk = make_miniblock()
    report_model("Tiny GPT (n_layer=2, n_head=2, n_embd=128)", gpt)
    report_model("MiniBlock (n_embd=128, n_head=2)", blk)
    # print q/k/v shape example
    B, T, C = 1, 16, cfg.n_embd
    x = torch.zeros(B, T, C, dtype=torch.long)  # dummy indices for GPT
    try:
        # for GPT: token embedding -> (B,T,n_embd); position embedding -> (T,n_embd)
        print("GPT token embedding -> shapes: token_emb (B,T,n_embd), pos_emb (T,n_embd)")
    except Exception:
        pass
    print("MiniBlock q/k/v shapes (given input x of shape (B,T,C)):")
    print("  qkv linear -> reshaped to (B, T, 3, n_head, head_dim)")
    print("  q, k, v -> each (B, T, n_head, head_dim)")
    print("  (standard/expected attention) re-arrange to (B, n_head, T, head_dim) then compute att (B, n_head, T, T)")
    print()
    print("Run this script to get exact numbers and copy them into README.md.")
    print()
    print("Command:")
    print("  python inspect_model.py")

if __name__ == '__main__':
    main()