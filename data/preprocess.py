"""
数据预处理脚本：一次性构建词表并缓存，后续训练直接加载。

用法：
  python -m data.preprocess --dataset WikiText103 --data_dir dataset
  python -m data.preprocess --dataset wikitext-2 --data_dir dataset/WikiText2
"""
import argparse
import os
import sys
import time

import torch

# 确保项目根目录在 sys.path 中（兼容 python data/preprocess.py 和 python -m data.preprocess）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader_torchtext import get_cbow_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="dataset")
    args = parser.parse_args()

    ds_key = args.dataset.lower().replace("-", "_")
    cache_path = os.path.join(args.data_dir, f"{ds_key}_vocab.pt")

    if os.path.exists(cache_path):
        print(f"Vocab cache already exists: {cache_path}")
        return

    print(f"Building vocab for {args.dataset} ...")
    t0 = time.time()
    _, _, vocab = get_cbow_dataloader(args.data_dir, args.dataset, batch_size=128)
    torch.save(vocab, cache_path)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
    print(f"Done. Vocab size={len(vocab)}, time={elapsed}")
    print(f"Cached to: {cache_path}")


if __name__ == "__main__":
    main()
