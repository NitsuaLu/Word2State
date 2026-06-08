# Word2State

PyTorch 复现 [Word2State: Modeling Word Representations as States with Density Matrices](https://doi.org/10.23919/cje.2023.00.336)（*Chinese Journal of Electronics*, 2025）。

## 项目说明

原始论文代码依赖已停止维护的 torchtext 等老旧库，与当前 Python/PyTorch 版本不兼容。本项目基于 [word2state](https://github.com/zhangchener/word2state) 进行重构，全程保留原始算法逻辑，不修改设计思路。

## 项目结构

```
.
├── data/                                # 数据预处理模块
│   ├── loader.py                        #   正则版（丢弃标点，保留字母数字）
│   ├── loader_basic_english.py          #   basic_english 版（标点剥离为独立 token）
│   └── loader_torchtext.py              #   torchtext 完整对齐版
├── layers/                              # 核心层
│   ├── embedding.py                     #   ComplexEmbedding
│   ├── mixture.py                       #   ComplexMixture
│   └── measurement.py                   #   ComplexMeasurement
├── models/                              # 模型架构
│   └── cbow.py                          #   C_CBOW / R_CBOW
├── config/default.yaml                  # 训练配置
├── train.py                             # 训练流水线
├── hparam_search.py                     # 超参数网格搜索
└── evaluate.py                          # 词相似度评估
```

## 快速开始

### 环境要求

```
torch >= 2.0
numpy >= 1.26
pandas >= 2.2
PyYAML >= 6.0
scipy >= 1.13
```

### 数据预处理

```python
from data.loader import get_cbow_dataloader
# 或使用 torchtext 对齐版
# from data.loader_torchtext import get_cbow_dataloader

# WikiText2
train_dl, valid_dl, vocab = get_cbow_dataloader(
    "dataset/WikiText2", "wikitext-2", batch_size=128, shuffle=True
)

# WikiText103
train_dl, valid_dl, vocab = get_cbow_dataloader(
    "dataset", "WikiText103", batch_size=128, shuffle=True
)
```

### 核心层

```python
from layers import ComplexEmbedding, ComplexMixture

embed = ComplexEmbedding(vocab_size=50000, embedding_dim=300)
mix = ComplexMixture()

x = embed(torch.randint(0, 50000, (2, 8)))   # (2, 8, 300)
rho = mix(x)                                   # (2, 300, 300)
```

### 训练

```bash
python train.py --config config/default.yaml
```

### 超参数搜索

```bash
python hparam_search.py
```

### 词相似度评估

```bash
python evaluate.py --model_dir weights/final/c_cbow_300d --model_name c_cbow
python evaluate.py --model_dir weights/final/c_cbow_300d --model_name c_cbow --nearest spring --top 10
```

## 参考

- 论文：Zhang C, Li Q, Su Z, et al. *Word2State: Modeling Word Representations as States with Density Matrices*. Chinese Journal of Electronics, 2025.
- 原代码：[word2state](https://github.com/zhangchener/word2state)
