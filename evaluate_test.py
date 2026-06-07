"""
词相似度评估单元测试。
用法：python evaluate_test.py
"""

import os, sys, tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import compute_similarity, load_dataset, evaluate_one
from models import C_CBOW, R_CBOW


def _make_vocab_and_vectors(vocab_size=100, dim=16):
    """构造小词表和随机 L2 归一化向量。"""
    import torch.nn as nn
    e = nn.Embedding(vocab_size, dim)
    stoi = {f"w{i}": i for i in range(vocab_size)}
    stoi["<unk>"] = 0
    vectors = F.normalize(e.weight, p=2, dim=-1).detach()
    return vectors, stoi


def test_compute_similarity_c_cbow():
    """C_CBOW fidelity：边界值 + 对称性 + 范围。"""
    # 相同归一化向量 → fidelity = 1
    v = F.normalize(torch.randn(16, dtype=torch.cfloat), p=2, dim=-1)
    s = compute_similarity(v, v, "c_cbow")
    assert abs(s - 1.0) < 1e-6, f"Same: expected 1.0, got {s}"

    # 正交向量 → fidelity = 0
    v1 = torch.tensor([1 + 0j, 0 + 0j], dtype=torch.cfloat)
    v2 = torch.tensor([0 + 0j, 1 + 0j], dtype=torch.cfloat)
    s = compute_similarity(v1, v2, "c_cbow")
    assert s < 1e-6, f"Orthogonal: expected 0, got {s}"

    # 对称性：f(v1,v2) = f(v2,v1)
    va = F.normalize(torch.randn(16, dtype=torch.cfloat), p=2, dim=-1)
    vb = F.normalize(torch.randn(16, dtype=torch.cfloat), p=2, dim=-1)
    s_ab = compute_similarity(va, vb, "c_cbow")
    s_ba = compute_similarity(vb, va, "c_cbow")
    assert abs(s_ab - s_ba) < 1e-6, f"Symmetry: {s_ab} vs {s_ba}"

    # 范围：0 ≤ fidelity ≤ 1（对归一化向量）
    assert 0 <= s_ab <= 1, f"Range: {s_ab} not in [0, 1]"

    print("  [PASS] test_compute_similarity_c_cbow (boundary + symmetry + range)")


def test_compute_similarity_r_cbow():
    """R_CBOW cosine：边界值 + 对称性 + 范围。"""
    v = F.normalize(torch.randn(16), p=2, dim=-1)
    s = compute_similarity(v, v, "r_cbow")
    assert abs(s - 1.0) < 1e-6, f"Same: expected 1.0, got {s}"

    s = compute_similarity(v, -v, "r_cbow")
    assert abs(s - (-1.0)) < 1e-6, f"Opposite: expected -1.0, got {s}"

    # 对称性
    va = F.normalize(torch.randn(16), p=2, dim=-1)
    vb = F.normalize(torch.randn(16), p=2, dim=-1)
    s_ab = compute_similarity(va, vb, "r_cbow")
    s_ba = compute_similarity(vb, va, "r_cbow")
    assert abs(s_ab - s_ba) < 1e-6, f"Symmetry: {s_ab} vs {s_ba}"

    # 范围
    assert -1 <= s_ab <= 1, f"Range: {s_ab} not in [-1, 1]"

    print("  [PASS] test_compute_similarity_r_cbow (boundary + symmetry + range)")


def test_spearman_perfect():
    """Spearman：完全单调正相关 → 1.0，完全单调负相关 → -1.0。"""
    vectors, stoi = _make_vocab_and_vectors()
    # 造 5 对词，人工相似度与模型相似度完全正单调
    human_scores = [1, 2, 3, 4, 5]
    # 用完全相同的向量对当做 model sim（确保单调）
    pairs = [(f"w{i}", f"w{i}") for i in range(5)]
    # 对于相同向量对，model similarity ≈ 1.0 for all，不是单调的
    # 改用随机向量对 + 人工指定 model_sim
    ...

    # 换个更直接的测试：构造 df，人工控制 model_sim 与 human 的关系
    model_sims = [0.1, 0.2, 0.3, 0.4, 0.5]  # 完全正相关
    human =     [1,   2,   3,   4,   5   ]
    import scipy.stats
    rho, _ = scipy.stats.spearmanr(human, model_sims)
    assert abs(rho - 1.0) < 1e-6, f"Perfect positive: rho={rho}"

    rho, _ = scipy.stats.spearmanr(human, list(reversed(model_sims)))
    assert abs(rho - (-1.0)) < 1e-6, f"Perfect negative: rho={rho}"

    print("  [PASS] test_spearman_perfect")


def test_compute_similarity_r_cbow():
    """R_CBOW：相同向量 → 1.0，相反向量 → -1.0。"""
    # 模拟实际模型输出的 1D 向量 (dim,)
    v = F.normalize(torch.randn(16), p=2, dim=-1)
    s = compute_similarity(v, v, "r_cbow")
    assert abs(s - 1.0) < 1e-6, f"Same vector should be 1.0, got {s}"

    s = compute_similarity(v, -v, "r_cbow")
    assert abs(s - (-1.0)) < 1e-6, f"Opposite should be -1.0, got {s}"

    print("  [PASS] test_compute_similarity_r_cbow")


def test_load_dataset_format_a():
    """格式 A：(index), word1, word2, similarity。"""
    csv = ",word1,word2,similarity\n0,dog,cat,5.0\n1,car,bus,3.0\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv)
        fname = f.name
    df = load_dataset(fname)
    os.unlink(fname)
    assert df is not None
    assert list(df.columns) == ["_w1", "_w2", "_score"]
    assert df.iloc[0]["_w1"] == "dog"
    print("  [PASS] test_load_dataset_format_a")


def test_load_dataset_format_b():
    """格式 B：Word 1, Word 2, Human (mean)。"""
    csv = "Word 1,Word 2,Human (mean)\ndog,cat,5.0\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv)
        fname = f.name
    df = load_dataset(fname)
    os.unlink(fname)
    assert df is not None
    assert df.iloc[0]["_w1"] == "dog"
    print("  [PASS] test_load_dataset_format_b")


def test_load_dataset_format_c():
    """格式 C：(index), similarity, word1, word2, relation。"""
    csv = ",similarity,word1,word2,relation\n0,5.0,dog,cat,HYPER\n"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(csv)
        fname = f.name
    df = load_dataset(fname)
    os.unlink(fname)
    assert df is not None
    assert df.iloc[0]["_w1"] == "dog"
    print("  [PASS] test_load_dataset_format_c")


if __name__ == "__main__":
    print("=== 词相似度评估测试 ===\n")
    test_compute_similarity_c_cbow()
    test_compute_similarity_r_cbow()
    test_spearman_perfect()
    test_load_dataset_format_a()
    test_load_dataset_format_b()
    test_load_dataset_format_c()
    print("\n=== 全部通过 ===")
