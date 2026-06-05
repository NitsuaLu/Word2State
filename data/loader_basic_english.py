"""
数据预处理模块（basic_english 版）：分词器 + Vocab 词表 + CBOW 滑动窗口 + DataLoader 工厂。

与 loader.py 的唯一差异在于分词器：
  loader.py          → 正则清洗 tokenize()，丢弃标点/特殊字符，仅保留字母数字
  loader_basic_english.py → 对齐 torchtext get_tokenizer("basic_english")，标点剥离为独立 token

其余组件（Vocab、CbowDataset、collate、工厂函数）代码结构完全相同，
仅各自调用本文件的 tokenize()。
"""

import os
import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

# ---- 常量（对齐 utils/constants.py）------------------------------------------------
CBOW_N_WORDS = 4
WINDOW_SIZE = CBOW_N_WORDS * 2 + 1   # = 9
MIN_WORD_FREQUENCY = 50
MAX_SEQUENCE_LENGTH = 256

# basic_english 分词正则：匹配连续字母、连续数字、或单个非空白非字母数字字符
# 等价于 torchtext get_tokenizer("basic_english") 的行为
_BE_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\s\w]")


# ---- 分词器（basic_english 对齐版）-----------------------------------------------
def tokenize(text: str) -> list[str]:
    """对齐 torchtext get_tokenizer("basic_english") 的分词行为。

    与 loader.py 的 tokenize() 的核心差异：
      - 本函数：标点/特殊字符剥离为独立 token（如 "hello," → ["hello", ","]）
      - loader.py：标点/特殊字符直接丢弃（如 "hello," → ["hello"]）

    额外处理：保护 <unk> 不被 basic_english 拆碎（"<" "unk" ">" → "<unk>"）。
    对齐原代码 build_vocab_from_iterator(specials=["<unk>"]) 的行为。
    """
    # 保护 <unk>：用纯字母占位符替换，分词完成后再还原。
    # 对齐原代码 build_vocab_from_iterator(specials=["<unk>"]) 的行为。
    text = text.replace("<unk>", "zzUNKzz")

    tokens = []
    for word in text.split():
        parts = _BE_PATTERN.findall(word)
        if parts:
            tokens.extend(parts)

    return ["<unk>" if t == "zzUNKzz" else t for t in tokens]


# ---- Vocab 词表 ------------------------------------------------------------------
class Vocab:
    """自建词表，接口对齐 torchtext.vocab.Vocab 的调用方式。

    原评估脚本（evaluate.py / process.py）依赖以下接口：
      - vocab[token]          → int idx（oov 返回 0）
      - len(vocab)            → 词表大小
      - vocab.get_stoi()      → {token: idx} 字典
      - vocab.lookup_token(i) → token 字符串
      - torch.save/load       → 序列化
    """

    def __init__(self, data_iter, min_freq: int = MIN_WORD_FREQUENCY):
        """遍历 data_iter（每行为一个文本），统计词频并构建词表。

        Args:
            data_iter: 可迭代对象，每次迭代返回一行文本字符串。
            min_freq: 最小词频阈值，低于此频率的词将被丢弃。
        """
        counter = Counter()
        for text in data_iter:
            for token in tokenize(text):
                counter[token] += 1

        # <unk> 固定为 0 号索引，兜底所有未登录词
        self.itos = {0: "<unk>"}
        self.stoi = {"<unk>": 0}

        idx = 1
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    def __len__(self) -> int:
        """返回词表大小（含 <unk>）。"""
        return len(self.stoi)

    def __getitem__(self, token: str) -> int:
        """返回 token 对应的索引，未登录词返回 <unk> 的索引 0。"""
        return self.stoi.get(token, 0)

    def get_stoi(self) -> dict[str, int]:
        """返回 {token: idx} 映射字典。原评估脚本通过此接口获取词表。"""
        return self.stoi

    def lookup_token(self, idx: int) -> str:
        """给定索引返回对应 token，越界或不存在返回 "<unk>"。"""
        return self.itos.get(idx, "<unk>")


# ---- CBOW 滑动窗口 ---------------------------------------------------------------
def build_cbow_windows(text: str) -> list[tuple[list[str], str]]:
    """对单行文本生成 CBOW 训练样本。

    流程（对齐原代码 dataloader.py:112-138）：
      1. 分词
      2. 跳过长度不足 WINDOW_SIZE 的文本
      3. 按 MAX_SEQUENCE_LENGTH 截断
      4. 滑动窗口（size=9, step=1），弹出中间词作为 target，剩余 8 词作为 context

    Returns:
        [(context_tokens, target_token), ...]  context_tokens 长度固定为 8。
    """
    tokens = tokenize(text)
    if len(tokens) < WINDOW_SIZE:
        return []
    tokens = tokens[:MAX_SEQUENCE_LENGTH]

    windows = []
    for i in range(len(tokens) - CBOW_N_WORDS * 2):
        window = tokens[i : i + WINDOW_SIZE]
        target = window.pop(CBOW_N_WORDS)
        context = window
        windows.append((context, target))
    return windows


# ---- Dataset -------------------------------------------------------------------
class CbowDataset(Dataset):
    """CBOW 数据集，读取 .tokens 文件并在构建时完成滑动窗口预处理。

    每个样本返回 (context_ids, target_id)：
      - context_ids: shape (8,)，dtype torch.long
      - target_id:   shape (),   dtype torch.long
    """

    def __init__(self, filepath: str, vocab: Vocab):
        """
        Args:
            filepath: .tokens 文件路径，每行为一段已分词文本。
            vocab: Vocab 实例，用于将 token 转为索引。
        """
        self.samples: list[tuple[list[int], int]] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                for ctx, tgt in build_cbow_windows(line):
                    ctx_ids = [vocab[t] for t in ctx]
                    tgt_id = vocab[tgt]
                    self.samples.append((ctx_ids, tgt_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ctx_ids, tgt_id = self.samples[idx]
        return (torch.tensor(ctx_ids, dtype=torch.long),
                torch.tensor(tgt_id, dtype=torch.long))


# ---- collate & DataLoader 工厂 ----------------------------------------------------
def collate_cbow(batch: list[tuple[torch.Tensor, torch.Tensor]]
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """对齐原代码 collate_cbow：堆叠 context 和 target 为 batch 张量。"""
    inputs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return inputs, labels


def get_cbow_dataloader(data_dir: str, ds_name: str, batch_size: int,
                         shuffle: bool, vocab: Vocab | None = None):
    """创建 CBOW 训练/验证 DataLoader + 词表。

    对齐原代码 get_dataloader_and_vocab 的调用约定：
      - 首次调用时不传 vocab，自动从训练集构建词表
      - 后续调用（如验证集）传入已构建的 vocab 以复用词表映射

    Args:
        data_dir:   数据集根目录（如 "dataset"）。
        ds_name:    数据集子目录名（如 "WikiText103"）。
        batch_size: 批大小。
        shuffle:    是否打乱训练集。
        vocab:      可选，复用已有 Vocab 实例。

    Returns:
        (train_dataloader, valid_dataloader, vocab)
    """
    train_path = os.path.join(data_dir, ds_name, "wiki.train.tokens")
    valid_path = os.path.join(data_dir, ds_name, "wiki.valid.tokens")

    if vocab is None:
        with open(train_path, "r", encoding="utf-8") as f:
            vocab = Vocab(f)

    train_ds = CbowDataset(train_path, vocab)
    valid_ds = CbowDataset(valid_path, vocab)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate_cbow)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                          collate_fn=collate_cbow)
    return train_dl, valid_dl, vocab
