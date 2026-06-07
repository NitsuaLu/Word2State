# -*- coding: utf-8 -*-
"""
ComplexMeasurement：冯·诺依曼投影测量层。

对齐原代码 layers/complexnn/measurement.py 中 CBOW 模型使用的模式：
  - real_weight=True：权重存为 (n_units, D, 2) float，前向 view_as_complex
  - n_units=vocab_size，每个词一个测量基

公式：p(m) = ⟨k_m| ρ |k_m⟩ → 实值概率

初始化：torch.rand() → Uniform[0, 1)，对齐源码实际生效行为。
"""

import torch
import torch.nn as nn


class ComplexMeasurement(nn.Module):
    def __init__(self, embed_dim: int, n_units: int = -1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_units = embed_dim if n_units == -1 else n_units

        self.weight = nn.Parameter(
            torch.rand((self.n_units, self.embed_dim, 2), dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对密度矩阵做投影测量。

        Args:
            x: 密度矩阵，shape (B, D, D) 或 (B, L, D, D)。

        Returns:
            实值测量概率，(B, n_units) 或 (B, n_units, L)。
        """
        k = torch.view_as_complex(self.weight)

        if x.ndim == 4:
            einsum_expression = 'an,bcnm,ma->bca'
        elif x.ndim == 3:
            einsum_expression = 'an,bnm,ma->ba'

        output = torch.einsum(einsum_expression, k, x,
                              k.transpose(-2, -1).conj())
        return output.real
