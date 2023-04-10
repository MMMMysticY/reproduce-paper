# @Time     : 2023.4.10 09:30
# @Author   : Wang Yang


import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat


class PreNorm(nn.Module):
    # 实现pre-norm功能
    def __init__(self, norm_dim, func):
        """
        norm_dim: 进行layer norm的维度
        func: 需要加在norm后的函数方法
        """
        super().__init__()
        self.func = func
        self.norm = nn.LayerNorm(norm_dim)

    def forward(self, x, **kwargs):
        return self.func(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    # 实现feedforward 本质就是mlp
    def __init__(self, dim, hidden_dim, dropout_p=0.):
        """
        dim: 初始维度
        hidden_dim: 映射到的维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    # multi-head attention
    def __init__(self, head, dim, dropout=0.):
        """
        head: 头个数
        dim: 原始维度
        """
        super().__init__()
        assert dim % head == 0
        self.head = head
        self.dim = dim
        self.head_dim = dim // head
        self.to_qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        # 将qkv用一个linear一起完成
        self.scale = self.head_dim ** -0.5
        self.attend = nn.Softmax(-1)
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: [B, S, E]

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 经过to_qkv计算得到[B, S, 3*E]， 之后chunk分割得到qkv: [3, B S, E]

        query, key, value = map(lambda t: rearrange(t, 'b s (h d) -> b h s d', h=self.head), qkv)
        # 对于每个[b, s, e]的对象 分割为[b, s, h, d]再转置为[b h s d]

        attention_score = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        # [B, head, S, S]

        if mask is not None:
            attention_score = attention_score.mask_fill(mask == 0, -1e9)
            # mask矩阵中为0为需要mask的地方 mask值为负无穷则softmax为0
        attention = self.attend(attention_score)

        output = torch.matmul(attention, value)
        # [B, head, S, head_dim]
        output = rearrange(output, 'b h s d -> b s (h d)')

        output = self.output_proj(output)
        return output


class Transformer(nn.Module):
    def __init__(self, d_model, head, mlp_dim, num_layers, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, Attention(head, d_model, dropout)),
                PreNorm(d_model, FeedForward(d_model, mlp_dim, dropout)),
                ])
            )
        # 构建多层 每一层都是一个ModuleList 其中一个attention 一个ffw
        # 共有num_layers层

    def forward(self, x):
        for attn, ffw in self.layers:
            x = attn(x) + x
            x = ffw(x) + x
            # 实现add功能
        return x


class ViT(nn.Module):
    def __init__(self, image_sizes, path_sizes, d_model, head, mlp_dim, num_layers, num_classes, dropout=0., channel=3):
        super().__init__()
        image_height, image_width = image_sizes
        patch_height, patch_width = path_sizes

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        # 第一步分割图片部分
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # num_patches就是分块个数 也就是seq_len

        patch_dim = patch_height * patch_width * channel
        # patch_dim就是图像每个子块拉直后的维度

        self.cut_image = Rearrange('b (h p_h) (w p_w) c -> b (h w) (p_h p_w c)', p_h=patch_height, p_w=patch_width)
        # 使用Rearrange层 将原始b h w c 对高度切分为h patch_height 对宽度且分为 w patch width 然后组合起来 (h w)就是块的个数 (p_h p_w c)是每个块的像素数
        # p_h p_w c和patch_dim相同 h w和num_patches相同
        self.image_proj = nn.Linear(patch_dim, d_model)
        # 将每个子块映射至d_model维度

        self.to_image_embedding = nn.Sequential(
            self.cut_image,
            nn.LayerNorm(patch_dim),
            self.image_proj,
            nn.LayerNorm(d_model)
        )
        # 构建图像embedding模块

        self.pos_embedding = nn.Parameter(torch.randn(size=(1+num_patches, d_model)))
        # pos embedding 维度是1+patch个数

        self.cls_token = nn.Parameter(torch.randn(size=(1, d_model)))
        self.transformer = Transformer(d_model, head, mlp_dim, num_layers, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, img):
        # img [batch, height, width, channel]
        image_embedding = self.to_image_embedding(img)
        # [batch, num_patches, d_model]

        batch, seq_len, _ = image_embedding.shape
        all_cls = repeat(self.cls_token, '1 d -> b 1 d', b=batch)
        # 将cls token重复batch次  [batch, 1, d_model]

        x = torch.cat([all_cls, image_embedding], dim=1)
        x = x + self.pos_embedding

        x = self.transformer(x)
        # [batch, seq_len+1, d_model]

        feature = x[:, 0, :]
        output = self.mlp_head(feature)
        return output


def func():
    image_sizes = (224, 224)
    patch_sizes = (16, 16)
    d_model = 256
    head = 4
    mlp_dim = 1024
    num_layers = 4
    num_classes = 100
    vit = ViT(image_sizes, patch_sizes, d_model, head, mlp_dim, num_layers, num_classes)
    input_example = torch.randn(size=(32, 224, 224, 3))
    output = vit(input_example)
    print(output.shape)


if __name__ == '__main__':
    func()
