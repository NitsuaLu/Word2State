"""
验证 loader.py（regex 版）的各项功能。
用法：python data/test_regex.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.loader import (
    tokenize, Vocab, build_cbow_windows,
    CbowDataset, collate_cbow, get_cbow_dataloader,
)


def test_tokenize():
    """分词器：保留字母数字，丢弃标点。"""
    assert tokenize("hello, world!") == ["hello", "world"]
    assert tokenize("don't stop") == ["don t", "stop"]
    assert tokenize("<unk> test") == ["<unk>", "test"]
    assert tokenize("Valkyria 3 : <unk>") == ["Valkyria", "3", "<unk>"]
    assert tokenize("state-of-the-art @-@") == ["state of the art"]
    assert tokenize("") == []
    print("  [PASS] test_tokenize")


def test_build_cbow_windows():
    """CBOW 滑动窗口：size=9，中间词为 target。"""
    # 恰好 9 词 → 1 个窗口
    w = build_cbow_windows("a b c d e f g h i")
    assert len(w) == 1
    ctx, tgt = w[0]
    assert len(ctx) == 8
    assert tgt == "e"                   # index 4 是中间词（9 词窗口的第 5 个）

    # 10 词 → 2 个窗口
    w = build_cbow_windows("a b c d e f g h i j")
    assert len(w) == 2

    # 不足 9 词 → 0 个窗口
    assert build_cbow_windows("too short") == []
    print("  [PASS] test_build_cbow_windows")


def test_vocab():
    """词表构建与接口。"""
    texts = ["the cat sat on the mat", "the dog ran", "<unk> cat dog"]
    v = Vocab(iter(texts), min_freq=1)

    assert v["<unk>"] == 0
    assert v["the"] != 0
    assert v["xyz_not_a_word"] == 0
    assert v.lookup_token(0) == "<unk>"
    assert len(v) > 3
    print(f"  [PASS] test_vocab (size={len(v)})")


def test_dataloader_wikitext2():
    """WikiText2 完整加载流程。"""
    dl, _, vocab = get_cbow_dataloader(
        "dataset/WikiText2", "wikitext-2", batch_size=128, shuffle=False
    )
    inputs, labels = next(iter(dl))
    assert inputs.shape == (128, 8)
    assert labels.shape == (128,)
    assert inputs.dtype == torch.long
    assert len(vocab) > 1000
    print(f"  [PASS] test_dataloader_wikitext2 (vocab={len(vocab)}, batches={len(dl)})")


def test_dataloader_wikitext103():
    """WikiText103 完整加载流程。"""
    dl, _, vocab = get_cbow_dataloader(
        "dataset", "WikiText103", batch_size=128, shuffle=False
    )
    inputs, labels = next(iter(dl))
    assert inputs.shape == (128, 8)
    assert labels.shape == (128,)
    assert len(vocab) > 10000
    print(f"  [PASS] test_dataloader_wikitext103 (vocab={len(vocab)}, batches={len(dl)})")


if __name__ == "__main__":
    print("=== loader.py (regex 版) 测试 ===\n")
    test_tokenize()
    test_build_cbow_windows()
    test_vocab()
    test_dataloader_wikitext2()
    test_dataloader_wikitext103()
    print("\n=== 全部通过 ===")
