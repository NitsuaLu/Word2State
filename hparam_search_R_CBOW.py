"""
超参数网格搜索脚本。

在 WikiText2（小数据集）上快速筛选最优超参数，
结果写入 weights/hparam_search.json，不保存模型。

用法：
  python hparam_search.py
"""

import json
import os
import time
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data.loader_torchtext import get_cbow_dataloader
from models import C_CBOW, R_CBOW

# ---- 搜索空间 -----------------------------------------------------------
MODEL_NAMES = ["c_cbow", "r_cbow"]
DIMS = [50, 100, 300]
BATCH_SIZES = [64, 128, 256, 512]
LEARNING_RATES = [0.0025, 0.001, 0.01, 0.1]
EPOCHS = [5, 10, 20]
DS_NAME = "wikitext-2"
DATA_DIR = "dataset/WikiText2"
OUTPUT_FILE = "weights/hparam_search.json"


def _get_lr_scheduler(optimizer, total_epochs):
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_one(model_name, dim, bs, lr, epochs):
    # 固定随机种子，确保每组参数可复现
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, _, vocab = get_cbow_dataloader(DATA_DIR, DS_NAME, bs, True)
    val_dl, _, _ = get_cbow_dataloader(DATA_DIR, DS_NAME, bs, False, vocab)

    ModelClass = C_CBOW if model_name == "c_cbow" else R_CBOW
    model = ModelClass(len(vocab), dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = _get_lr_scheduler(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                val_losses.append(criterion(model(inputs), labels).item())

        scheduler.step()

    val_loss = float(np.mean(val_losses))
    return val_loss, float(np.exp(val_loss))


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 如果已有结果文件，断点续跑
    results = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            results = json.load(f)

    completed = {(r["model"], r["dim"], r["batch_size"], r["lr"], r["epochs"])
                 for r in results}

    total = len(MODEL_NAMES) * len(DIMS) * len(BATCH_SIZES) * len(LEARNING_RATES) * len(EPOCHS)
    skipped = len(completed)

    for model_name, dim, bs in itertools.product(MODEL_NAMES, DIMS, BATCH_SIZES):
        for lr, ep in itertools.product(LEARNING_RATES, EPOCHS):
            key = (model_name, dim, bs, lr, ep)
            if key in completed:
                continue

            # C_CBOW 全 dim 排除 lr=0.1（已确认无法收敛）
            if model_name == "c_cbow" and lr == 0.1:
                continue

            print(f"[{len(results) + 1 + skipped}/{total}] "
                  f"{model_name} dim={dim} bs={bs} lr={lr} ep={ep} ...",
                  end=" ", flush=True)

            val_loss, val_ppl = run_one(model_name, dim, bs, lr, ep)

            entry = {
                "model": model_name,
                "dim": dim,
                "batch_size": bs,
                "lr": lr,
                "epochs": ep,
                "val_loss": round(val_loss, 4),
                "val_ppl": round(val_ppl, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            results.append(entry)
            print(f"loss={val_loss:.4f} ppl={val_ppl:.2f}")

            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
