import torch, sys, platform, numpy, tqdm, tiktoken
from nanoGPT.model import GPT, GPTConfig

print("✔ Imports OK — Python", sys.version.split()[0], "|", platform.system())

cfg = GPTConfig(vocab_size=50304, block_size=64,
n_layer=2, n_head=2, n_embd=128)
model = GPT(cfg)
x = torch.randint(0, 1000, (1, 10))
logits, _ = model(x)
print("✔ Forward pass OK — logits shape:", logits.shape)