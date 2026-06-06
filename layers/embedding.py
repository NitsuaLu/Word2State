# -*- coding: utf-8 -*-
"""
ComplexEmbedding：复值词嵌入层。

对齐原代码 layers/complexnn/embedding.py 的算法逻辑：
  幅度 embedding + 相位 embedding → 复向量 |amplitude| * e^(i*phase)

初始化：两个 nn.Embedding 使用 PyTorch 默认 N(0, 1)，对齐源码实际生效行为。
"""

import torch
import torch.nn as nn


class ComplexEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.amplitude_embed = nn.Embedding(vocab_size, embedding_dim)
        self.phase_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """将词索引映射为复值嵌入。

        Args:
            inputs: (B, L) 词索引张量。

        Returns:
            (B, L, D) 复值嵌入，dtype=torch.complex64。
        """
        amplitude = self.amplitude_embed(inputs)
        phase = self.phase_embed(inputs)
        real = phase.cos().mul(amplitude)
        imag = phase.sin().mul(amplitude)
        return torch.complex(real, imag)
