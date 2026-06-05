# Word2State

PyTorch 复现 [Word2State: Modeling Word Representations as States with Density Matrices](https://doi.org/10.23919/cje.2023.00.336)（*Chinese Journal of Electronics*, 2025）。

## 项目说明

原始论文代码依赖已停止维护的 torchtext 等老旧库，与当前 Python/PyTorch 版本不兼容。本项目基于 [word2state](https://github.com/zhangchener/word2state) 进行重构，全程保留原始算法逻辑，不修改设计思路。

## 项目结构

```
.
└── data/                         # 数据预处理模块
    ├── loader.py                 #   正则版（丢弃标点，保留字母数字）
    ├── loader_basic_english.py   #   basic_english 版（标点剥离为独立 token）
    ├── test_regex.py             #   正则版单元测试
    ├── test_basic_english.py     #   basic_english 版单元测试
    └── test_compare.py           #   两版差异对比
```

## 快速开始

### 环境要求

```
torch >= 2.0
numpy >= 1.26
pandas >= 2.2
PyYAML >= 6.0
scikit-learn >= 1.5
scipy >= 1.13
```

### 数据预处理

```python
from data.loader import get_cbow_dataloader
# 或使用 basic_english 版
# from data.loader_basic_english import get_cbow_dataloader

# WikiText2
train_dl, valid_dl, vocab = get_cbow_dataloader(
    "dataset/WikiText2", "wikitext-2", batch_size=128, shuffle=True
)

# WikiText103
train_dl, valid_dl, vocab = get_cbow_dataloader(
    "dataset", "WikiText103", batch_size=128, shuffle=True
)
```

### 运行测试

```bash
python data/test_regex.py             # 正则版测试
python data/test_basic_english.py     # basic_english 版测试
python data/test_compare.py           # 两版差异对比
```

## 参考

- 论文：Zhang C, Li Q, Su Z, et al. *Word2State: Modeling Word Representations as States with Density Matrices*. Chinese Journal of Electronics, 2025.
