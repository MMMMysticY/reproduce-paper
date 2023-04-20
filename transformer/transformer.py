# @Time     : 2023.4.11 17:15
# @Author   : Wang Yang


import torch
import torch.nn as nn

from einops import rearrange, repeat
from utils import clones, TrainState, Batch

import math
import time


class ResidualPreNorm(nn.Module):
    """
    实现residual和prenorm
    顺序是输入x经过layernorn再经过func,得到的结果和x相加

    受限于存在两种attention 无法将参数中直接加入func，两种attention存在不同的参数值 所以需要在实际情况中采用lambda表达式将func作为参数传递
    """

    def __init__(self, norm_dim):
        super(ResidualPreNorm, self).__init__()
        self.norm = nn.LayerNorm(norm_dim)

    def forward(self, x, func):
        # x [batch, seq_len, d_model]
        # 可能的参数中往往存在各种mask
        output = func(self.norm(x))
        return output + x


class Attention(nn.Module):
    """
    实现multihead-attention计算 包括self-attention和cross-attention
    """

    def __init__(self, d_model, head, dropout=0.):
        super(Attention, self).__init__()
        assert d_model % head == 0
        self.head = head
        self.d_model = d_model
        self.head_dim = d_model // head
        self.dropout = dropout
        self.to_qkv = clones(nn.Linear(d_model, d_model), 3)
        # 若要并行计算，bias必须设置为False

        self.attend = nn.Softmax(-1)
        self.scale = d_model ** -0.5
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

        self.attn = None
        # 为了得到attention设定对象

    def forward(self, query, key, value, attn_mask=None, pad_mask=None):
        # query [batch, seq_len_out, d_model]
        # key value [batch, seq_len_in, d_model]
        # attn_mask [seq_len_out, seq_len_in]
        # pad_mask [batch, seq_len_in]

        # key与value的序列长度相同
        # attn_mask表示对所有样本而言，需要统一mask掉的位置
        # pad_mask表示对每个位置而言，各个样本各自不考虑那些pad token
        # 要求 mask矩阵中0代表需要mask

        qkv = [l(x) for l, x in zip(self.to_qkv, (query, key, value))]
        # 将原始值经过线性层映射 维度不变
        # [3, batch, seq_len_in/out, d_model]

        query, key, value = map(lambda t: rearrange(t, 'b s (h d) -> b h s d', h=self.head), qkv)
        # query [batch, head, seq_len_out, head_dim]
        # key value [batch, head, seq_len_in, head_dim]

        attention_score = torch.einsum('...ik,...jk->...ij', query, key)
        # attention_score [batch, head, seq_len_out, seq_len_in]

        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            attention_score.masked_fill(attn_mask == 0, -1e9)
        # 作用attn_mask

        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, 'b s -> b 1 1 s')
            attention_score.masked_fill(pad_mask == 0, -1e9)
        # 作用pad_mask

        attention = self.attend(attention_score)
        self.attn = attention

        output = torch.einsum('...ij,...jk->...ik', attention, value)
        # output [batch, head, seq_len_out, head_dim]
        output = rearrange(output, 'b h s d -> b s (h d)')
        # [batch, seq_len_out, d_model]
        output = self.output_proj(output)
        return output


class FeedForward(nn.Module):
    """
    实现feedforward模块 linear->gelu->dropout
    """

    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # [batch, seq_len, d_model]
        return self.layers(x)


class EncoderLayer(nn.Module):
    """
    构建单个Encoder Layer
    """

    def __init__(self, d_model, hidden_dim, head, dropout_attn=0., dropout_ffw=0.):
        super(EncoderLayer, self).__init__()
        self.attn_func = Attention(d_model, head, dropout_attn)
        self.ffw_func = FeedForward(d_model, hidden_dim, dropout_ffw)
        self.nets = clones(ResidualPreNorm(d_model), 2)
        # nets包括 attention和feedforward已经他们都带有的norm+add部分

    def forward(self, x, mask):
        # x [batch, seq_len, d_model]
        # mask [batch, seq_len]
        x = self.nets[0](x, lambda x: self.attn_func(x, x, x, pad_mask=mask))
        x = self.nets[1](x, self.ffw_func)
        return x


class Encoder(nn.Module):
    """
    encoder类构建N个encoder_layer并进行计算
    值得注意的是最后的norm是cross-attention前的key和value的norm
    """

    def __init__(self, d_model, encoder_layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(encoder_layer, N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x [batch, seq_len, d_model]
        # mask [batch, seq_len]

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, head, hidden_dim, dropout_attn=0., dropout_ffw=0.):
        super(DecoderLayer, self).__init__()
        self.self_attn = Attention(d_model, head, dropout_attn)
        self.cross_attn = Attention(d_model, head, dropout_attn)
        self.ffw = FeedForward(d_model, hidden_dim, dropout_ffw)

        self.nets = clones(ResidualPreNorm(d_model), 3)

    def forward(self, x, memory, src_pad_mask, tgt_pad_mask, tgt_mask):
        # x [batch, seq_len_de, d_model]
        # memory [batch, seq_len_en, d_model]
        # src_pad_mask [batch, seq_len_en]
        # tgt_pad_mask [batch, seq_len_de]
        # tgt_mask [seq_len_de, seq_len_de]

        x = self.nets[0](x, lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask, pad_mask=tgt_pad_mask))
        x = self.nets[1](x, lambda x: self.cross_attn(x, memory, memory, pad_mask=src_pad_mask))
        x = self.nets[2](x, self.ffw)
        return x


class Decoder(nn.Module):
    def __init__(self, decoder_layer, d_model, N):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_pad_mask, tgt_pad_mask, tgt_mask):
        # x [batch, seq_len_de, d_model]
        # memory [batch, seq_len_en, d_model]
        # src_pad_mask [batch, seq_len_en]
        # tgt_pad_mask [batch, seq_len_de]
        # tgt_mask [seq_len_de, seq_len_de]

        for layer in self.layers:
            x = layer(x, memory, src_pad_mask, tgt_pad_mask, tgt_mask)
        return self.norm(x)


class WordEmbedding(nn.Module):
    """
    实现word embedding 即对于每个token 选择对应的embedding值
    encoder与decoder采用同一个embedding矩阵
    """

    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__()
        self.dic = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # [batch, seq_len]
        return self.dic(x) * self.d_model ** 0.5


class PositionalEmbedding(nn.Module):
    """
    公式中的2i就是[0, 2, 4, ..., d_model] pos就是[0, 1, 2, ..., seq_len]
    二者相乘就是[[0, 0, 0, ... 0],
               [0, 2, 4, ... d_model],
               [0, 4, 8, ... 2d_model],
               ...
               [0, d_model, 2_d_model, ... d_model*seq_len]]
    此时用广播机制实现很方便 pos-> [seq_len, 1] 其他部分[d_model] 二者相乘[seq_len, d_model]
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEmbedding, self).__init__()
        assert d_model % 2 == 0
        # d_model为偶数时 才满足公式中2i的要求

        self.pe = torch.zeros(max_len, d_model)
        # pe的维度是[seq_len, d_model] 制作一个max_len, d_model的能够适应各种seq_len
        positions = torch.arange(0, max_len).unsqueeze(1)
        # 位置为[max_len, 1] 用以对每个位置加以不一样的pe，而同一个位置

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000) / d_model)
            # 前一项为[0, 2, 4.., d_model]表示公式中的2i的实现
            # 后一项为取了log后的剩余所有常数项 简单推导即可得出
        )

        the_matrix = positions * div_term
        # pos[max_len, 1] div [d_model//2] 采用广播机制计算得出结果[max_len, d_model//2]
        # 即成公式中的pos/10000^2i/d_model 表达式
        self.pe[:, 0::2] = torch.sin(the_matrix)
        self.pe[:, 1::2] = torch.cos(the_matrix)
        # 偶数位置为对the_matrix的sin值 奇数位置为cos值

        self.pe = self.pe.unsqueeze(0)
        # [1, seq_len, d_model]

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x [batch, seq_len, d_model]
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        # 加上seq_len长度的pe  注意设置不更新 torch中使用requires_grad_能够实现部分模块的冻结
        return self.dropout(x)


class Generator(nn.Module):
    """
    构建decoder末端的输出层
    注意！如果需要用cross entropy作为loss function时 不要加LogSoftmax
    cross entropy api: The input is expected to contain raw, unnormalized scores for each class
    """

    def __init__(self, vocab_size, d_model):
        super(Generator, self).__init__()
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.attend = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x [batch, seq_len, d_model]
        x = self.output_proj(x)
        # [batch, seq_len, vocab_size]
        return self.attend(x)
        # return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, hidden_dim, head, encoder_layer_num, decoder_layer_num, dropout=0.1):
        super(Transformer, self).__init__()
        self.word_embedding = WordEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model)

        encoder_layer = EncoderLayer(d_model, hidden_dim, head, dropout, dropout)
        decoder_layer = DecoderLayer(d_model, head, hidden_dim, dropout, dropout)

        self.encoder = Encoder(d_model, encoder_layer, encoder_layer_num)
        self.decoder = Decoder(decoder_layer, d_model, decoder_layer_num)

        self.generator = Generator(vocab_size, d_model)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        """
        src [batch, seq_len_en] 其中为int token
        src_pad_mask [batch, seq_len_en] 若用int或者float矩阵 0代表需要mask 若为Bool矩阵 则False代表需要mask
        tgt [batch, seq_len_de] 其中为int token
        tgt_pad_mask [batch, seq_len_de] 若用int或者float矩阵 0代表需要mask 若为Bool矩阵 则False代表需要mask
        tgt_mask [seq_len_de, seq_len_de] 若用int或者float矩阵 0代表需要mask 若为Bool矩阵 则False代表需要mask  一般为上三角矩阵
        """

        src = self.positional_embedding(self.word_embedding(src))
        # [batch, seq_len_en, d_model]
        memory = self.encoder(src, src_pad_mask)

        tgt = self.positional_embedding(self.word_embedding(tgt))
        output = self.decoder(tgt, memory, src_pad_mask, tgt_pad_mask, tgt_mask)
        output = self.generator(output)
        # [batch, seq_len_de, vocab_size]
        return output


def run_epoch(
        data_iter,  # 数据存储对象
        model,  # 模型
        loss_compute,  # 损失函数
        optimizer,  # 优化器
        scheduler,  # LambdaLR对象
        mode='train',
        accum_iter=1,
        # 梯度累计更新次数 ------------ 代表了经过accum_iter个mini-batch后再更新一次梯度
        train_state=TrainState()
):
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        output = model(batch.src, batch.tgt,batch.tgt_mask, batch.src_pad_mask, batch.tgt_pad_mask)
        # 模型推理
        loss, loss_node = loss_compute(output, batch.tgt_target, batch.all_tgt_tokens)
        # 计算损失函数
        # loss -> number 无梯度
        # loss_node -> 带梯度 反向传播使用

        if mode == 'train' or mode == 'train+log':
            # 反向传播
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.all_tgt_tokens
            # 每个mini-batch反向传播一次

            if i % accum_iter == 0 or (i + 1) == len(data_iter):
                # 每accum_iter个mini-batch更新一次模型参数 或者 在最后一次更新
                optimizer.step()
                optimizer.zero_grad()
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.all_tgt_tokens
        tokens += batch.all_tgt_tokens
        # 纪录信息
        if i % 10 == 1 and (mode == 'train' or mode == 'train+log'):
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time

            print(
                'Epoch Step: {} | Accumulation Step: {} | Loss: {} | Tokens/Sec : {} | lr : {}'.format(
                    i, n_accum, loss / batch.all_tgt_tokens, tokens / elapsed, lr
                )
            )
            start_time = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state
