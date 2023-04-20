# @Time     : 2023.4.19 14:44
# @Author   : Wang Yang


# 本样例仅用来测试模型的可用性 不包含任何训练过程与数据

import torch
import torch.nn as nn
from transformer import Transformer
from utils import generate_sequence_mask


def inference_test():
    """
    实现一个step-by-step的推理过程
    使用greedy search策略 即只选概率值最大的进行下一次输入
    """

    test_model = Transformer(11, 64, 256, 2, 2, 2)
    for p in test_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # 创建模型 初始化参数
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_pad_mask = torch.ones(size=(1, 10))
    # [1, 10]

    tgt = torch.zeros(size=(1, 1)).type_as(src)
    # 初始化的tgt batch=1 seq_len=1
    for i in range(9):
        prob = test_model(src, tgt, tgt_mask=generate_sequence_mask(tgt.shape[1]), src_pad_mask=src_pad_mask)[:, -1, :]
        # prob是最后一个位置的概率 [batch=1, vocab_size]
        _, next_word = torch.max(prob, dim=-1)
        # 找到概率最大的index [batch=1,]
        next_word = next_word[0]
        tgt = torch.cat([tgt, torch.empty(size=(1, 1)).type_as(src.data).fill_(next_word)], dim=-1)
        # 将生成的新token和之前的拼接 作为下一次的输入
        # 该过程就是递归生成模式
    print('result is : ', tgt)


if __name__ == '__main__':
    for _ in range(10):
        inference_test()
