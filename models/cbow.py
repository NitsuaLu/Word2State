"""
CBOW 模型：C_CBOW（复值） + R_CBOW（实值）。

对齐原代码 utils/model.py 中 PL 版的 forward 逻辑：
  - C_CBOW: ComplexEmbedding → normalize → ComplexMixture → ComplexMeasurement
  - R_CBOW: nn.Embedding → mean pool → nn.Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ComplexEmbedding, ComplexMixture, ComplexMeasurement


class C_CBOW(nn.Module):
    """复值 CBOW——词表示为密度矩阵态。

    论文公式（Section III）：
      w = A ⊙ e^(iθ)                       复值嵌入
      |w⟩ = w / ||w||                      L2 归一化
      ρ = Σ c_j · |w_j⟩⟨w_j|               加权混合
      p(m) = ⟨ψ_m| ρ |ψ_m⟩                 投影测量 → logits
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = ComplexEmbedding(vocab_size, embedding_dim)
        self.mixture = ComplexMixture()
        self.measurement = ComplexMeasurement(embedding_dim, n_units=vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, context_size) 上下文词索引。

        Returns:
            (B, vocab_size) logits。
        """
        x = self.embeddings(inputs)                              # (B, L, D)

        weights_ = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        weights = weights_.masked_fill(weights_ == 0, -10000)

        x = F.normalize(x, dim=-1, p=2)                          # |w⟩
        rho = self.mixture(x, torch.softmax(weights, dim=1))     # ρ = Σ c_j|w_j⟩⟨w_j|
        logits = self.measurement(rho)                            # p(m) = ⟨ψ_m|ρ|ψ_m⟩
        return logits


class R_CBOW(nn.Module):
    """实值 CBOW——基准对比模型。

    对齐 Word2Vec CBOW：embedding → mean → linear projection。
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, context_size) 上下文词索引。

        Returns:
            (B, vocab_size) logits。
        """
        x = self.embeddings(inputs)     # (B, L, D)
        x = x.mean(dim=1)               # (B, D)
        x = self.linear(x)              # (B, vocab_size)
        return x
