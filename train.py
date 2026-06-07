"""
训练脚本：YAML 配置驱动，纯 PyTorch。

对齐原代码 train.py + utils/trainer.py + utils/helper.py 的算法逻辑：
  - Adam optimizer，线性 LR 衰减
  - CrossEntropyLoss
  - 每 epoch 训练/验证 + 保存制品

用法：
  python train.py --config config/default.yaml
"""

import argparse
import json
import os
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data.loader_torchtext import get_cbow_dataloader
from models import C_CBOW, R_CBOW


def get_model_class(model_name: str):
    if model_name == "c_cbow":
        return C_CBOW
    elif model_name == "r_cbow":
        return R_CBOW
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def get_lr_scheduler(optimizer: optim.Optimizer, total_epochs: int):
    """线性学习率衰减：(epochs - epoch) / epochs，对齐原代码。"""
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train(config: dict):
    """主训练入口。

    流程：加载数据 → 构建模型 → 训练/验证循环 → 保存制品。
    """
    os.makedirs(config["model_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 数据加载 ----
    dl_kwargs = {
        "data_dir": config["data_dir"],
        "ds_name": config["dataset"],
        "batch_size": config["train_batch_size"],
        "shuffle": config["shuffle"],
    }
    train_dl, _, vocab = get_cbow_dataloader(**dl_kwargs)
    dl_kwargs["batch_size"] = config["val_batch_size"]
    dl_kwargs["shuffle"] = False
    val_dl, _, _ = get_cbow_dataloader(vocab=vocab, **dl_kwargs)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # ---- 模型 ----
    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size, config["embedding_dim"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"])

    train_steps = config.get("train_steps")
    val_steps = config.get("val_steps")
    checkpoint_freq = config.get("checkpoint_frequency")

    loss_history: dict[str, list[float]] = {"train": [], "val": []}

    # ---- 训练循环 ----
    for epoch in range(config["epochs"]):
        # train
        model.train()
        train_running = []
        for i, (inputs, labels) in enumerate(train_dl, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running.append(loss.item())
            if train_steps and i >= train_steps:
                break

        train_loss = float(np.mean(train_running))
        loss_history["train"].append(train_loss)

        # validate
        model.eval()
        val_running = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dl, 1):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running.append(loss.item())
                if val_steps and i >= val_steps:
                    break

        val_loss = float(np.mean(val_running))
        loss_history["val"].append(val_loss)

        lr_scheduler.step()

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Epoch {epoch + 1}/{config['epochs']} | "
              f"Train Loss={train_loss:.4f} PPL={np.exp(train_loss):.2f} | "
              f"Val Loss={val_loss:.4f} PPL={np.exp(val_loss):.2f}")

        if checkpoint_freq and (epoch + 1) % checkpoint_freq == 0:
            ckpt_path = os.path.join(
                config["model_dir"], f"checkpoint_{str(epoch + 1).zfill(3)}.pt")
            torch.save(model, ckpt_path)

    print("Training finished.")

    # ---- 保存制品 ----
    torch.save(model, os.path.join(config["model_dir"], "model.pt"))
    torch.save(vocab, os.path.join(config["model_dir"], "vocab.pt"))
    with open(os.path.join(config["model_dir"], "loss.json"), "w") as f:
        json.dump(loss_history, f)
    with open(os.path.join(config["model_dir"], "config.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"Artifacts saved to: {config['model_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)
