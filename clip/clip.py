# @Time     : 2023.4.11 10:04
# @Author   : Wang Yang


import torch
import torch.nn as nn

from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, func):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.func = func

    def forward(self, x, **kwargs):
        return self.func(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p=0.):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        assert dim % head == 0
        self.head_dim = dim // head
        self.head = head
        self.dim = dim
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x, mask=None):
        # x [b, s, d]
        # mask [b, s]

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # qkv [3, b, s, d]

        query, key, value = map(lambda t: rearrange(t, 'b s (h d) -> b h s d', h=self.head), qkv)
        # query key value [b, h, s, head_dim]

        attention_score = torch.einsum('...ik, ...jk -> ...ij', query, key)
        # 使用einsum计算attention_score

        if mask is not None:
            mask = rearrange(mask, 'b s -> b 1 1 s')
            # 需要扩展维度才能广播
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
            # mask矩阵中为0为需要mask的地方 mask值为负无穷则softmax为0
