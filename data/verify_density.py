"""
验证密度矩阵的数学性质：Hermitian、半正定、单位迹。

密度矩阵 ρ 的数学要求：
  1. ρ = ρ†（Hermitian）
  2. 所有特征值 ≥ 0（正半定）
  3. Tr(ρ) = 1（单位迹）

验证方法：模拟 CBOW 模型的完整前向流程（embed → normalize → mixture），检查输出。

用法：python data/verify_density.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from layers import ComplexEmbedding, ComplexMixture, ComplexMeasurement

torch.manual_seed(42)

embed = ComplexEmbedding(vocab_size=500, embedding_dim=50)
mix = ComplexMixture()
mea = ComplexMeasurement(embed_dim=50, n_units=10)

# ---- 模拟 CBOW 完整前向 ----
x = embed(torch.randint(0, 500, (4, 8)))              # (B, L, D)
weights_ = torch.linalg.norm(x, dim=-1, keepdim=True)  # (B, L, 1)
weights = weights_.masked_fill(weights_ == 0, -10000)
seq = F.normalize(x, dim=-1, p=2)                      # L2 归一化
rho = mix(seq, torch.softmax(weights, dim=1))           # (B, D, D)
probs = mea(rho)                                        # (B, n_units)

print("=" * 55)
print("密度矩阵数学性质验证")
print("=" * 55)

all_ok = True
for b in range(4):
    r = rho[b]
    # 1. Hermitian: ρ - ρ† = 0
    diff = (r - r.conj().T).abs().max().item()
    ok1 = diff < 1e-5

    # 2. 半正定: 所有特征值 ≥ 0
    eigvals = torch.linalg.eigvalsh(r).real
    min_eig = eigvals.min().item()
    ok2 = min_eig >= -1e-5

    # 3. 单位迹: Tr(ρ) = 1
    trace = r.diag().sum().real.item()
    ok3 = abs(trace - 1.0) < 1e-5

    status = "OK" if (ok1 and ok2 and ok3) else "FAIL"
    if not (ok1 and ok2 and ok3):
        all_ok = False

    print(f"\n  batch {b}: [{status}]")
    print(f"    Hermitian:  max|ρ-ρ†| = {diff:.2e}")
    print(f"    PSD:        min λ     = {min_eig:.6f}")
    print(f"    Unit trace: Tr(ρ)     = {trace:.6f}")

# ---- 测量概率验证 ----
print(f"\n  measurement: probs all >= 0? "
      f"{(probs >= -1e-5).all().item()}")
print(f"  measurement: finite?      {probs.isfinite().all().item()}")

print(f"\n{'=' * 55}")
print("结论:", "全部通过" if all_ok else "存在不满足项")
print("=" * 55)
