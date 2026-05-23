import torch
from nanoGPT.model import GPT, GPTConfig

print("✔ imports ok")

cfg = GPTConfig(
    vocab_size=50304,
    block_size=64,
    n_layer=2,         # tiny so it’s fast on CPU
    n_head=2,
    n_embd=128,
)
model = GPT(cfg)

x = torch.randint(0, 1000, (1, 10))  # fake 10-token prompt
logits, _ = model(x)

print("✔ forward pass ok — logits:", logits.shape)