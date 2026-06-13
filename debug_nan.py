"""排查 WS-353-REL 和 WS-353-SIM 评估 NaN 问题。"""
import pandas as pd
import numpy as np
import scipy.stats
from evaluate import load_dataset, load_model_vectors, compute_similarity

MODEL_DIR = "weights/final/r_cbow_50d"
MODEL_NAME = "r_cbow"
DATASETS = ["EN-WS-353-REL", "EN-WS-353-SIM"]

vectors, word_to_idx = load_model_vectors(MODEL_DIR, MODEL_NAME)

for ds in DATASETS:
    print(f"\n{'='*60}")
    print(f"--- {ds} ---")

    # Step 1: 原始 CSV
    raw = pd.read_csv(f"dataset/eval_benchmarks/{ds}.csv")
    score_col = [c for c in raw.columns if c not in ("Unnamed: 0", "word1", "word2")][0]
    print(f"\n[1] Raw CSV: score_col='{score_col}', dtype={raw[score_col].dtype}")
    print(f"    NaN count in raw: {raw[score_col].isna().sum()}")

    # Step 2: load_dataset 处理后
    df = load_dataset(f"dataset/eval_benchmarks/{ds}.csv")
    print(f"\n[2] load_dataset: shape={df.shape}, _score dtype={df['_score'].dtype}")
    print(f"    NaN count in _score: {df['_score'].isna().sum()}")
    print(f"    _score samples: {df['_score'].head(10).tolist()}")

    # Step 3: 逐个提取 human + sims
    human, sims = [], []
    bad_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        w1, w2 = str(row["_w1"]), str(row["_w2"])
        if w1 not in word_to_idx or w2 not in word_to_idx:
            continue
        try:
            s = float(row["_score"])
        except (ValueError, TypeError):
            bad_rows.append((i, row["_score"]))
            continue
        sims.append(compute_similarity(
            vectors[word_to_idx[w1]], vectors[word_to_idx[w2]], MODEL_NAME))
        human.append(s)

    print(f"\n[3] After extraction: |human|={len(human)}, |sims|={len(sims)}")
    print(f"    NaN in human: {np.isnan(human).any()}, NaN in sims: {np.isnan(sims).any()}")
    if bad_rows:
        print(f"    Bad score rows: {bad_rows[:10]}")
    else:
        print("    All scores convertible to float")

    # Step 4: 计算相关系数
    if len(human) >= 3:
        human_a, sims_a = np.array(human), np.array(sims)
        # 剔除 NaN
        mask = ~np.isnan(human_a) & ~np.isnan(sims_a)
        print(f"\n[4] After NaN filter: {mask.sum()} pairs remain")
        if mask.sum() >= 3:
            rho, pval = scipy.stats.spearmanr(human_a[mask], sims_a[mask])
            pr, pp = scipy.stats.pearsonr(human_a[mask], sims_a[mask])
            print(f"    spearman: {rho:.4f} (p={pval:.4f})")
            print(f"    pearson: {pr:.4f} (p={pp:.4f})")
        else:
            print("    Not enough valid pairs after NaN filter")
    else:
        print("    Not enough pairs")
