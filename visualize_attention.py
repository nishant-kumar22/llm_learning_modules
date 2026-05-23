#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import torch

from mini_block import MiniBlock


def main():
    os.makedirs("assets", exist_ok=True)

    torch.manual_seed(0)
    block = MiniBlock(n_embd=128, n_head=2)
    block.eval()

    x = torch.randn(1, 32, 128)
    with torch.no_grad():
        _, att = block(x, return_att=True)

    head0 = att[0, 0].cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(head0, cmap="viridis", aspect="auto")
    plt.colorbar(label="attention weight")
    plt.title("MiniBlock attention head 0")
    plt.xlabel("key position")
    plt.ylabel("query position")
    plt.tight_layout()
    plt.savefig("assets/attn_head0.png", dpi=160)
    plt.close()

    print("saved assets/attn_head0.png")


if __name__ == "__main__":
    main()
