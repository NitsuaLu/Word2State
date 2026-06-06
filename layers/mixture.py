# -*- coding: utf-8 -*-
"""
ComplexMixture：复值密度矩阵混合层。

对齐原代码 layers/complexnn/mixture.py 的算法逻辑：
  复向量序列 → 外积 |ψ⟩⟨ψ| → 加权求和 Σ w_i * ρ_i → 密度矩阵
"""

import torch
import torch.nn as nn


class ComplexMixture(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        """将复向量序列混合为密度矩阵。

        Args:
            inputs: (B, L, D) 复值向量序列。
            weights: (B, L, ...) 混合权重，None 时做平均混合。

        Returns:
            (B, D, D) 密度矩阵。
        """
        # 每个词的密度矩阵：|ψ⟩⟨ψ|
        inputs_conj = inputs.unsqueeze(dim=-2).conj()
        inputs = inputs.unsqueeze(dim=-1)
        outputs = torch.matmul(inputs, inputs_conj)

        if weights is None:
            outputs = torch.mean(outputs, dim=1)
        else:
            while weights.dim() < outputs.dim():
                weights = torch.unsqueeze(weights, dim=-1)
            outputs = outputs.mul(weights).sum(dim=1)

        return outputs
