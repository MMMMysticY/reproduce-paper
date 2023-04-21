# @Time     : 2023.4.19 14:50
# @Author   : Wang Yang


# 本样例极其简单 仅训练模型进行copy 重复输入样本

import torch
import torch.nn as nn
from utils import Batch, LabelSmoothing, rate, generate_sequence_mask, SimpleLossCompute
from transformer import Transformer, run_epoch
from torch.optim.lr_scheduler import LambdaLR


## 生成数据
def data_gen(max_vocab_num, batch_size, seq_len, nbatches):
    """
    所有数据中 0->pad token  1->[SOS]
    生成随机的src-tgt对 首个位置一定为[SOS]->1
    最终生成nbatches组样本 每组样本内有[batch_size, seq_len]个pair
    """
    for i in range(nbatches):
        data = torch.randint(1, max_vocab_num, size=(batch_size, seq_len))
        data[:, 0] = 1
        # 将第一个token设置为SOS
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


def main():
    vocab_size = 11
    # [0, 1, ... 10]为有效token 词表大小为11
    criterion = LabelSmoothing(vocab_size, 0, 0.0)
    # 由于任务是要重复 所以置信度没有必要降低 即smooth=0.0
    loss_compute = SimpleLossCompute(criterion)

    model = Transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, d_model=512, hidden_dim=2048, head=8,
                        encoder_layer_num=2,decoder_layer_num=2)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)

    lr_schedular = LambdaLR(optimizer,
                            lr_lambda=lambda step: rate(step, 512, 1.0, 400))

    batch_size = 80
    for epoch in range(15):
        # 训练部分
        model.train()
        run_epoch(
            data_iter=data_gen(vocab_size, batch_size, 10, 20),
            # 每次生成20组数据
            model=model,
            loss_compute=loss_compute,
            optimizer=optimizer,
            scheduler=lr_schedular,
            mode='train',
        )
        break
    # 测试部分
    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # 1为[SOS] 0为[PAD]
    batch, seq_len = src.shape
    src_pad_mask = src != 0

    tgt = torch.ones(size=(1, 1)).type_as(src)
    # 以[SOS]作为初始值进行推理
    result = torch.zeros(size=(1, 1)).type_as(src)
    # 纪录decoder最终的输出结果
    for i in range(seq_len):
        probs = model(src, tgt, tgt_mask=generate_sequence_mask(tgt.shape[1]), src_pad_mask=src_pad_mask)[:, -1, :]
        # 最后一个位置的概率  [batch, vocab_size]
        max_prob, next_words = torch.max(probs, dim=-1)
        # next_word [batch, ]
        print('probs: ', probs)
        print('max_prob: ', max_prob)
        tgt = torch.cat([tgt, next_words.unsqueeze(1)], dim=1)
        # tgt [batch, now_seq_len]  next_words.unsqueeze [batch, 1]
        # 生成 [batch, now_seq_len+1] 作为下一次的输入

        result = torch.cat([result, next_words.unsqueeze(1)], dim=1)
        # 纪录结果

    print(result[:, 1:])


if __name__ == '__main__':
    main()
