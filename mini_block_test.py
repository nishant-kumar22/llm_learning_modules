import torch
from mini_block import MiniBlock

blk = MiniBlock(n_embd=128, n_head=2)
x   = torch.randn(1, 10, 128)
y   = blk(x)
print("output shape:", y.shape)   # → torch.Size([1, 10, 128])