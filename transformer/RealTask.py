# @Time     : 2023.4.20 11:05
# @Author   : Wang Yang


import torch
import torch.nn as nn

import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torch.optim.lr_scheduler import LambdaLR

from transformer import Transformer, run_epoch
from utils import LabelSmoothing, rate, TrainState, Batch, SimpleLossCompute
from tensorboardX import SummaryWriter

import spacy
import os


def load_tokenizers():
    """
    加载spacy的tokenizer
    """
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    """
    使用tokenizer将text进行处理 对于英文来讲就是按照空格分割为list
    """
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    """
    将data_iter中的对象 经过tokenizer处理
    生成新的iterator对象
    """
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):
    """
    构建词表 包括tokenize和特殊字符的设定
    src->de tgt_en
    """

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print('Building German Vocabulary ...')
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(yield_tokens(train + val + test, tokenize_de, index=0),
                                          min_freq=2,
                                          specials=["<s>", "</s>", "<blank>", "<unk>"])
    # specials导致词表的0,1,2,3分别是[SOS] [EOS] [PAD] 和 [UNK]

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(yield_tokens(train + val + test, tokenize_en, index=1),
                                          min_freq=2,
                                          specials=["<s>", "</s>", "<blank>", "<unk>"])
    # 构建德文和英文词表 从Multi30k数据集构建

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])
    # 在出现Out of Vocabulary情况时 用unk代替
    # 返回值是torchtext.vocab对象 forward方法就是进行tokenize将token转化为int值
    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not os.path.exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), 'vocab.pt')
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished. \n Vocabulary Sizes: ")
    print('src: ', len(vocab_src))
    print('tgt: ', len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(batch, src_pipeline, tgt_pipeline, src_vocab, tgt_vocab, device, max_padding=128, pad_id=2):
    """
    为batch中的src和tgt 前后加上[SOS(0)]和[EOS(1)] 并pad到最大固定长度 并将其tokenize处理
    batch: collate_fn的必要参数 即__getitem__方法获得的一个batch内的对象
    src_pipeline/tgt_pipeline: 实际上是tokenize function即将一句话按照空格分割为list
    src_vocab/tgt_vocab: torchtext.vocab对象 forward方法将token转化为int值
    max_padding: 句子最大长度
    pad_id: pod token的int值
    """
    sos_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                sos_id,
                torch.tensor(src_vocab(src_pipeline(_src)),
                             dtype=torch.int64,
                             device=device),
                eos_id

            ],
            dim=0
        )
        processed_tgt = torch.cat(
            [
                sos_id,
                torch.tensor(tgt_vocab(tgt_pipeline(_tgt)),
                             dtype=torch.int64,
                             device=device),
                eos_id
            ],
            dim=0
        )
        # 进行tokenize和sos eos添加

        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),  # 左侧pad0个 右侧pad max_len-now_len个
                value=pad_id
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),  # 左侧pad0个 右侧pad max_len-now_len个
                value=pad_id
            )
        )
        # 进行pad至max_padding_len

        src = torch.stack(src_list, dim=0)
        tgt = torch.stack(tgt_list, dim=0)
        # [batch, max_padding_len]
        return src, tgt


def create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=128,
        max_padding=128,
        is_distributed=True,
):
    """
    按照collate_fn对训练集和验证集进行构建
    其中包括目前尚不理解的分布式策略
    """

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        """
        按照标准方式构建collate_fn
        返回值就是一个function 因为python万事万物皆对象
        """
        return collate_batch(
            batch,
            src_pipeline=tokenize_de,
            tgt_pipeline=tokenize_en,
            src_vocab=vocab_src,
            tgt_vocab=vocab_tgt,
            device=device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"]
        )

    # print(type(collate_fn)) --------- <class 'function'>
    train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=("de", "en"))

    train_iter_map = to_map_style_dataset(train_iter)
    # 事实上 train_iter对象无法构建Dataloader 变为map_style后才能有len等方法 进行训练使用
    train_sampler = (DistributedSampler(train_iter_map) if is_distributed else None)
    # 构建分布式sampler  ?
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = (DistributedSampler(valid_iter_map) if is_distributed else None)
    # 构建分布式sampler ?

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn
    )
    # 若无分布式 进行shuffle
    return train_dataloader, valid_dataloader


def data_process(dataloader, device, pad_idx):
    for b in dataloader:
        each_data = Batch(b[0], b[1], pad_idx)
        each_data.to_deivce(device)
        yield each_data


def train_worker(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        config,
        is_distribute=False,
):
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512

    # 构建模型
    model = Transformer(len(vocab_src), len(vocab_tgt), d_model, hidden_dim=2048, head=8, encoder_layer_num=6,
                        decoder_layer_num=6)
    model.cuda(gpu)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 损失函数
    criterion = LabelSmoothing(vocab_size=len(vocab_tgt), pad_idx=pad_idx, smooth=0.1)
    criterion.cuda(gpu)

    # 数据
    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
        is_distributed=is_distribute
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]))

    # 训练
    train_state = TrainState()
    writer = SummaryWriter(config["log_path"])
    min_valid_loss = 100.

    for epoch in range(config["num_epochs"]):
        print(f"Epoch {epoch} Training ====", flush=True)
        model.train()
        train_loss, train_state = run_epoch(
            data_iter=data_process(train_dataloader, gpu, pad_idx),
            model=model,
            loss_compute=SimpleLossCompute(criterion),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            mode='train',
            accum_iter=config["accum_iter"],
            train_state=train_state
        )
        torch.cuda.empty_cache()
        train_loss = train_loss.detach().item()
        print('Train Loss: ', train_loss)

        print(f"Epoch {epoch} Validation ====", flush=True)
        model.eval()
        valid_loss, _ = run_epoch(
            data_iter=data_process(valid_dataloader, gpu, pad_idx),
            model=model,
            loss_compute=SimpleLossCompute(criterion),
            optimizer=None,
            scheduler=None,
            mode='valid'
        )
        valid_loss = valid_loss.detach().item()
        print('Valid Loss: ', valid_loss)
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            save_path_name = os.path.join(config["model_path"], 'Epoch={}-Loss={}.ckpt'.format(epoch, valid_loss))
            torch.save(model.state_dict(), save_path_name)
        writer.add_scalars("train_valid_loss", {'train': train_loss, 'valid': valid_loss},
                           global_step=epoch)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    config = {
        "batch_size": 32,
        "max_padding": 72,
        "base_lr": 1.0,
        "warmup": 3000,
        "accum_iter": 2,
        "num_epochs": 8,
        "log_path": 'log',
        "model_path": 'model'
    }
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    train_worker(gpu=3, vocab_src=vocab_src, vocab_tgt=vocab_tgt, spacy_de=spacy_de, spacy_en=spacy_en, config=config)
