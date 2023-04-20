# @Time     : 2023.4.11 17:27
# @Author   : Wang Yang


import torch
import torch.nn as nn
from copy import deepcopy
from einops import repeat


def clones(module, N):
    """
    对module重复N次
    """
    modules = nn.ModuleList([])
    for _ in range(N):
        modules.append(deepcopy(module))
    return modules


def generate_sequence_mask(size):
    """
    生成decoder端的上三角矩阵mask
    后续代码要求0代表要mask
    类似于：
    tensor([
        [1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.]
        ])
    """
    mask = torch.triu(torch.full((size, size), -1.0), diagonal=1) + 1
    return mask


def rate(step, model_size, factor, warmup):
    """
    实现论文中的学习率先增高后下降
    step: 更新的步数 越往后值越大 注意为避免0值带来的影响 0值直接设为1
    model_size: d_model
    factor: 学习率
    warmup: warmup过程的步数(step)

    本方法作为LambdaLR的参数使用
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * (warmup**(-1.5)))
    )


class Batch:
    """
    Batch对象对输入的src和tgt生成对应的mask
    本实现的mask规则 若用int或者float矩阵 0代表需要mask 若为Bool矩阵 则False代表需要mask
    """
    def __init__(self, src, tgt=None, pad=0):
        # 0-> pad token
        # src tgt [batch, seq_len_in/out] int token

        self.src = src
        # 若实现用0进行pad，那么可以直接使用：
        # self.src_pad_mask = src -> 0代表需要mask 其他代表不mask
        self.src_pad_mask = self.src != pad
        # != pad 则在pad位置是False 表示需要mask掉

        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_target = tgt[:, 1:]
            # tgt为decoder输入 从第0个到倒数第二个token
            # tgt_target为输出 从第1个到最后一个token
            # 二者依次对应
        self.tgt_mask = generate_sequence_mask(self.tgt.shape[-1])
        self.tgt_pad_mask = self.tgt != pad
        self.all_tgt_tokens = self.tgt_pad_mask.data.sum()
        # 对True位置的加和 即表示非mask正常token的数量


class TrainState:
    """
    静态类 统计训练过程中的各个指标情况
    """
    step: int = 0           # 当前的步数
    accum_step: int = 0     # 梯度累计的次数 ------- 数个mini-batch更新一次网络参数，该参数记录了更新参数的次数
    samples: int = 0        # 已经用到的样本个数
    tokens: int = 0         # 用到的token个数


class LabelSmoothing(nn.Module):
    """
    label smoothing loss的实现
    其中包括了对句子中末尾的pad token不参与loss计算的处理
    注意：CrossEntropy作为loss function时 期望最后一层不加softmax或者logsoftmax
         而采用KLDiv作为loss function时 期望最后一层采用log-space number
    """
    def __init__(self, vocab_size, pad_idx, smooth=0.0):
        super(LabelSmoothing, self).__init__()
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.smooth = smooth
        self.confidence = 1.0 - smooth
        # confidence代表对true label的置信度 若无smooth即为1.0
        self.true_dist = None

    def forward(self, y, target):
        # y [*, vocab_size] 概率值
        # target [*] int token
        # 一般而言*为batch*seq_len 即y : [batch*seq_len, vocab_size] target [batch*seq_len]
        assert y.shape[-1] == self.vocab_size
        true_dist = y.clone().detach()
        # [batch*seq_len, vocab_size]
        true_dist.fill_(self.smooth / (self.vocab_size-2))
        # 除了true token和pad token 其他位置设置为缓冲值 而非原来的0值

        true_label_idx = target.detach().unsqueeze(1)
        # [batch*seq_len, 1]
        confidence_matrix = repeat(torch.FloatTensor([self.confidence]), '1 -> b 1', b=true_label_idx.shape[0])
        # [batch*seq_len, 1] 为了迎合scatter中index和src一一对应关系
        true_dist.scatter_(dim=1, index=true_label_idx, src=confidence_matrix)
        # scatter_分发 index为[batch*seq_len, 1] index表示的是每个样本对应的vocab_size的值 所以dim=1
        # 类似于one-hot的操作 对每个样本 在词表上对应的真实token位置赋值为confidence 而非之前的1值

        true_dist[:, self.pad_idx] = 0.
        # 将pad token位置设为0值 即概率为0 不参与计算ce

        mask = torch.nonzero(target.detach() == self.pad_idx)
        # target.detach()==self.pad_idx使用bool值代表target中为pad的部分
        # 使用torch.nonzero获得pad部分的index值
        # mask [pad_token_num, 1] 共有pad_token_num个pad位置 每个位置用单个数值表示
        # 即mask表示的是batch*seq_len个token里 哪些位置是pad token

        if mask.dim() > 0:
            true_dist.index_fill_(dim=0, index=mask.squeeze(), value=0.0)
            # 按照mask将pad位置所对应的样本全赋值为0 所以dim=0 mask的是某个样本对应的所有vocab_size的位置
            # [batch*seq_len, vocab_size] 即对某个样本的所有vocab_size全mask
        self.true_dist = true_dist
        return self.criterion(y, true_dist.clone().detach())


if __name__ == '__main__':
    # ------- Test of LabelSmoothing----------------
    crit = LabelSmoothing(5, 0, 0.1)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    true_label = torch.LongTensor([2, 1, 0, 3, 3])
    loss = crit(predict, true_label)
    print(crit.true_dist)
    print(loss)
