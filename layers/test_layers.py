"""
layers 模块单元测试。
用法：python layers/test_layers.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from layers import ComplexEmbedding, ComplexMixture, ComplexMeasurement


# ---- ComplexEmbedding ----

def test_embedding_shape():
    embed = ComplexEmbedding(vocab_size=100, embedding_dim=16)
    x = embed(torch.randint(0, 100, (2, 8)))
    assert x.shape == (2, 8, 16)
    assert x.is_complex()
    assert torch.isfinite(x).all()
    print(f"  [PASS] test_embedding_shape (shape={tuple(x.shape)}, dtype={x.dtype})")


def test_embedding_init():
    embed = ComplexEmbedding(1000, 50)
    amp = embed.amplitude_embed.weight
    ph = embed.phase_embed.weight
    assert abs(amp.mean().item()) < 0.1
    assert abs(ph.mean().item()) < 0.1
    assert 0.8 < amp.std().item() < 1.2
    assert 0.8 < ph.std().item() < 1.2
    print("  [PASS] test_embedding_init")


# ---- ComplexMixture ----

def test_mixture_average():
    mix = ComplexMixture()
    x = torch.randn(3, 4, 10, dtype=torch.cfloat)
    rho = mix(x)
    assert rho.shape == (3, 10, 10)
    print(f"  [PASS] test_mixture_average (shape={tuple(rho.shape)})")


def test_mixture_weighted():
    mix = ComplexMixture()
    x = torch.randn(3, 4, 10, dtype=torch.cfloat)
    w = torch.randn(3, 4)
    rho = mix(x, w)
    assert rho.shape == (3, 10, 10)
    print(f"  [PASS] test_mixture_weighted (shape={tuple(rho.shape)})")


def test_mixture_weight_dim():
    mix = ComplexMixture()
    x = torch.randn(3, 4, 10, dtype=torch.cfloat)
    w = torch.randn(3, 4, 1)
    rho = mix(x, w)
    assert rho.shape == (3, 10, 10)
    print(f"  [PASS] test_mixture_weight_dim (shape={tuple(rho.shape)})")


# ---- ComplexMeasurement ----

def test_measurement_3d():
    """3D 密度矩阵 → (B, n_units) 输出。"""
    mea = ComplexMeasurement(embed_dim=16, n_units=5)
    # 用 ComplexMixture 生成合法密度矩阵（正定、Hermitian）
    mix = ComplexMixture()
    psi = torch.randn(2, 4, 16, dtype=torch.cfloat)
    rho = mix(psi)
    probs = mea(rho)
    assert probs.shape == (2, 5)
    assert (probs >= 0).all()
    assert torch.isfinite(probs).all()
    print(f"  [PASS] test_measurement_3d (shape={tuple(probs.shape)})")


def test_measurement_4d():
    """4D 密度矩阵（带 seq_len）→ (B, n_units, L) 输出。"""
    mea = ComplexMeasurement(embed_dim=16, n_units=5)
    # 构造合法的 4D 密度矩阵序列
    psi = torch.randn(2, 4, 16, dtype=torch.cfloat)
    rho_per_pos = psi.unsqueeze(-1) @ psi.unsqueeze(-2).conj()  # (2,4,16,16)，每个位置一个纯态密度矩阵
    probs = mea(rho_per_pos)
    assert probs.shape == (2, 4, 5)
    assert (probs >= 0).all()
    print(f"  [PASS] test_measurement_4d (shape={tuple(probs.shape)})")


def test_measurement_cbow():
    """CBOW 实际调用方式：n_units 可大于 embed_dim。"""
    mea = ComplexMeasurement(embed_dim=16, n_units=10)
    psi = torch.randn(2, 4, 16, dtype=torch.cfloat)
    rho = ComplexMixture()(psi)
    probs = mea(rho)
    assert probs.shape == (2, 10)
    assert (probs >= 0).all()
    print(f"  [PASS] test_measurement_cbow (shape={tuple(probs.shape)})")


def test_measurement_n_units_default():
    """n_units=-1 时应等于 embed_dim。"""
    mea = ComplexMeasurement(embed_dim=16)
    assert mea.n_units == 16
    print(f"  [PASS] test_measurement_n_units_default (n_units={mea.n_units})")


if __name__ == "__main__":
    print("=== layers 单元测试 ===\n")
    print("ComplexEmbedding:")
    test_embedding_shape()
    test_embedding_init()
    print("\nComplexMixture:")
    test_mixture_average()
    test_mixture_weighted()
    test_mixture_weight_dim()
    print("\nComplexMeasurement:")
    test_measurement_3d()
    test_measurement_4d()
    test_measurement_cbow()
    test_measurement_n_units_default()
    print("\n=== 全部通过 ===")
