"""
词相似度评估脚本。

加载 model.pt + vocab.pt，在 12 个基准数据集上计算 Spearman/Pearson 相关系数，
同时支持最近邻检索。

用法：
  python evaluate.py --model_dir weights/final/c_cbow_300d --model_name c_cbow
  python evaluate.py --model_dir weights/final/c_cbow_300d --model_name c_cbow --nearest spring --top 10
"""

import argparse
import math
import os
import glob

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F

# 论文 Table 1 数据集顺序
PAPER_DATASET_ORDER = {
    "EN-MC-30": "MC-30",
    "EN-RG-65": "RG-65",
    "EN-MTurk-287": "MTurk-287",
    "EN-MTurk-771": "MTurk-771",
    "EN-MEN-TR-3k": "MEN-3k",
    "EN-RW-STANFORD": "RW",
    "SimLex-999": "SIMLEX-999",
    "SimVerb-3500": "SimVerb-3500",
    "EN-WS-353-ALL": "WordSim-353-ALL",
    "EN-WS-353-REL": "WordSim-353-REL",
    "EN-WS-353-SIM": "WordSim-353-SIM",
    "EN-YP-130": "YP-130",
}


def load_model_vectors(model_dir: str, model_name: str) -> tuple[torch.Tensor, dict]:
    """加载模型并提取 L2 归一化词向量。

    Args:
        model_dir: 制品目录（含 model.pt, vocab.pt）。
        model_name: "c_cbow" 或 "r_cbow"。

    Returns:
        vectors: (vocab_size, dim) 归一化词向量。
        word_to_idx: {token: idx} 映射。
    """
    model = torch.load(os.path.join(model_dir, "model.pt"), weights_only=False)
    vocab = torch.load(os.path.join(model_dir, "vocab.pt"), weights_only=False)
    word_to_idx = vocab.get_stoi()

    if model_name == "c_cbow":
        sd = model.state_dict()
        amp = sd["embeddings.amplitude_embed.weight"]
        ph = sd["embeddings.phase_embed.weight"]
        vectors = torch.polar(amp, ph)
    elif model_name == "r_cbow":
        vectors = model.state_dict()["embeddings.weight"]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    vectors = F.normalize(vectors, p=2, dim=-1)
    return vectors, word_to_idx


def compute_similarity(v1: torch.Tensor, v2: torch.Tensor,
                       model_name: str) -> float:
    """计算两个词向量的相似度。

    - C_CBOW: fidelity |⟨ψ₁|ψ₂⟩|²
    - R_CBOW: cosine similarity
    """
    if model_name == "c_cbow":
        return torch.inner(v1, v2.conj()).abs().pow(2).item()
    else:
        return torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)
        ).item()


def evaluate_one(data: pd.DataFrame, vectors: torch.Tensor,
                 word_to_idx: dict, model_name: str) -> tuple[float, float, int]:
    """对单个数据集计算 Spearman ρ 和 Pearson r。

    Returns:
        (spearman_rho, pearson_r, valid_pairs)
    """
    human, model_sim = [], []
    for _, row in data.iterrows():
        w1, w2 = str(row["_w1"]), str(row["_w2"])
        if w1 not in word_to_idx or w2 not in word_to_idx:
            continue
        v1 = vectors[word_to_idx[w1]]
        v2 = vectors[word_to_idx[w2]]
        score = float(row["_score"])
        if not score or math.isnan(score):
            continue
        model_sim.append(compute_similarity(v1, v2, model_name))
        human.append(score)

    if len(human) < 3:
        return float("nan"), float("nan"), len(human)

    rho = scipy.stats.spearmanr(human, model_sim).statistic
    p = scipy.stats.pearsonr(human, model_sim).statistic
    return float(rho), float(p), len(human)


def load_dataset(filepath: str) -> pd.DataFrame | None:
    """加载词相似度数据集，按列名动态匹配 word1/word2/score。

    支持三种格式：
      A: (index), word1, word2, similarity
      B: Word 1, Word 2, Human (mean)
      C: (index), similarity, word1, word2, relation

    Returns:
        DataFrame with columns [_w1, _w2, _score]，或 None（解析失败）。
    """
    sep = "," if filepath.endswith(".csv") else "\t"
    header = 0 if filepath.endswith(".csv") else None
    df = pd.read_csv(filepath, sep=sep, header=header, skip_blank_lines=True)

    cols = [c.strip().lower() for c in df.columns]

    w1_col = next((i for i, c in enumerate(cols) if "word" in c and ("1" in c or "one" in c)), None)
    w2_col = next((i for i, c in enumerate(cols) if "word" in c and ("2" in c or "two" in c)), None)
    score_col = next((i for i, c in enumerate(cols)
                      if any(k in c for k in ["similarity", "human", "score", "mean"])), None)

    if w1_col is None or w2_col is None or score_col is None:
        return None

    # 固定列序：word1, word2, score
    result = pd.DataFrame()
    result["_w1"] = df.iloc[:, w1_col]
    result["_w2"] = df.iloc[:, w2_col]
    result["_score"] = df.iloc[:, score_col]
    return result
    return df


def run_evaluation(model_dir: str, model_name: str, data_dir: str):
    """主评估：按论文 Table 1 顺序输出数据集结果。"""
    vectors, word_to_idx = load_model_vectors(model_dir, model_name)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")) +
                   glob.glob(os.path.join(data_dir, "*.txt")))

    print(f"{'Dataset':<25} {'#Pairs':>7} {'Spearman ρ':>11} {'Pearson r':>10}")
    print("-" * 55)

    # 按文件名建立索引
    file_map = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        file_map[name] = f

    results = {}
    ordered_names = list(PAPER_DATASET_ORDER.keys())  # 论文顺序
    extra_names = sorted(set(file_map.keys()) - set(PAPER_DATASET_ORDER.keys()))  # 剩余放最后

    for name in ordered_names + extra_names:
        if name not in file_map:
            continue
        df = load_dataset(file_map[name])
        if df is None:
            continue
        rho, p, n = evaluate_one(df, vectors, word_to_idx, model_name)
        display_name = PAPER_DATASET_ORDER.get(name, name)
        print(f"{display_name:<25} {n:>7} {rho:>11.4f} {p:>10.4f}")
        results[display_name] = {"spearman": round(rho, 4), "pearson": round(p, 4), "pairs": n}

    avg_rho = np.nanmean([r["spearman"] for r in results.values()])
    print(f"\n{'Avg (Spearman ρ × 100)':<25} {avg_rho * 100:>7.2f}")

    return results


def find_nearest(model_dir: str, model_name: str,
                 word: str, top_n: int = 10):
    """最近邻检索。"""
    vectors, word_to_idx = load_model_vectors(model_dir, model_name)
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    if word not in word_to_idx:
        print(f"Word '{word}' not in vocabulary.")
        return

    query = vectors[word_to_idx[word]]

    sims = []
    for i in range(len(vectors)):
        if i == word_to_idx[word]:
            continue
        s = compute_similarity(vectors[i], query, model_name)
        sims.append((i, s))

    sims.sort(key=lambda x: -x[1])
    print(f"Nearest neighbors of '{word}':")
    for idx, s in sims[:top_n]:
        print(f"  {idx_to_word[idx]:<20} {s:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["c_cbow", "r_cbow"])
    parser.add_argument("--data_dir", type=str,
                        default="dataset/eval_benchmarks")
    parser.add_argument("--nearest", type=str, help="Word to find neighbors for")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if args.nearest:
        find_nearest(args.model_dir, args.model_name, args.nearest, args.top)
    else:
        run_evaluation(args.model_dir, args.model_name, args.data_dir)
