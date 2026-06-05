"""
两版分词器差异对比 + 词表大小对比。
用法：python data/test_compare.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import tokenize as t_regex
from data.loader_basic_english import tokenize as t_be


def test_tokenizer_diff():
    """打印两版分词器行为差异。"""
    cases = [
        "hello, world!",
        "don't stop believin'",
        "state-of-the-art @-@ technology",
        "the (quick) brown fox .",
        "Valkyria 3 : <unk> Chronicles",
    ]
    print("=== 分词器差异 ===\n")
    for s in cases:
        r = t_regex(s)
        b = t_be(s)
        if r != b:
            print(f"  {s}")
            print(f"    regex:          {r}")
            print(f"    basic_english:  {b}")
            print()


def test_vocab_compare_wikitext2():
    """WikiText2 词表大小对比。"""
    from data.loader import get_cbow_dataloader as get_regex
    from data.loader_basic_english import get_cbow_dataloader as get_be

    _, _, vr = get_regex("dataset/WikiText2", "wikitext-2", 128, False)
    _, _, vb = get_be("dataset/WikiText2", "wikitext-2", 128, False)
    print("=== WikiText2 词表对比 ===")
    print(f"  regex 版:          {len(vr)}")
    print(f"  basic_english 版:  {len(vb)}")
    print(f"  差值:              {len(vb) - len(vr)}")
    print()


def test_vocab_compare_wikitext103():
    """WikiText103 词表大小对比。"""
    from data.loader import get_cbow_dataloader as get_regex
    from data.loader_basic_english import get_cbow_dataloader as get_be

    _, _, vr = get_regex("dataset", "WikiText103", 128, False)
    _, _, vb = get_be("dataset", "WikiText103", 128, False)
    print("=== WikiText103 词表对比 ===")
    print(f"  regex 版:          {len(vr)}")
    print(f"  basic_english 版:  {len(vb)}")
    print(f"  差值:              {len(vb) - len(vr)}")
    print(f"  论文报告值:        49622")
    print()


if __name__ == "__main__":
    test_tokenizer_diff()
    test_vocab_compare_wikitext2()
    test_vocab_compare_wikitext103()
    print("=== 完成 ===")
