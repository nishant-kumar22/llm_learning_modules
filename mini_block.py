import torch, torch.nn as nn, math

class MiniBlock(nn.Module):
    def __init__(self, n_embd=128, n_head=2):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.ff   = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, return_att=False):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.softmax(dim=-1)
        y   = att @ v
        y   = y.transpose(1, 2).contiguous().view(B, T, C)

        x = self.ln1(x + self.proj(y))  # residual 1
        x = self.ln2(x + self.ff(x))    # residual 2
        if return_att:
            return x, att
        return x