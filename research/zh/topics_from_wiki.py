"""Build a topic pool by mining noun phrases out of the wiki corpus.

Asking the teacher to list 120 topics produced lecture notes and bullet points
instead ("目录:", "为了确保代码质量"), and 6 batches took 20 minutes. Mining the
corpus is both faster and more diverse: these are subjects real Chinese text is
actually about.
"""

import argparse
import json
import re
import sys
from collections import Counter

import jieba
import jieba.posseg as pseg

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIKI = ROOT / "data" / "zh" / "raw" / "wiki_corpus.txt"
OUT = ROOT / "data" / "zh" / "raw" / "topics.jsonl"

# Keep nouns and proper nouns: a topic has to be a thing, not a verb or a
# particle, or the teacher writes a paragraph about a grammar fragment.
KEEP = {"n", "ns", "nt", "nz", "nr", "nw", "vn", "an", "nz", "j", "l", "i", "eng"}
STOP = set("""东西 问题 情况 方式 时间 这个 那个 一个 我们 他们 可以 没有 其他 其中 由于 因此 例如 一些 这些 那些 什么 怎么 这样 一样 进行 出现 成为 包括 表示 认为 使用 提供 主要 不同 相关 方面 内容 部分 过程 结果 目前 现在 以及 对于 关于 通过 并且 但是 如果 所以 而且 同时 之后 之前 以上 以下 一种 一般 各种 每个 很多 大量 几个 全部 整个 世界 中国 国家 地区 城市 文化 历史 发展 社会 经济 政治 技术 研究 系统 数据 信息 文章 图片 文件 名称 名称 类型 数量 单位 世纪 年代 年 月 日""".split())
BAD = re.compile(r"[0-9a-zA-Z_\-\|=\{\}\[\]<>/\\]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def mine(path, max_chars):
    """Count candidate words with plain cut; posseg is ~50x slower and does not
    scale to tens of millions of characters, so tagging happens once per
    candidate afterwards instead of once per occurrence."""
    counts = Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            n += len(text)
            for word in jieba.cut(text, cut_all=False):
                if not (2 <= len(word) <= 5):
                    continue
                if not CJK.search(word) or BAD.search(word):
                    continue
                if word in STOP:
                    continue
                counts[word] += 1
            if n >= max_chars:
                break
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default=str(WIKI))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--max-chars", type=int, default=30_000_000)
    ap.add_argument("--min-count", type=int, default=8)
    args = ap.parse_args()

    counts = mine(args.wiki, args.max_chars)
    candidates = [(w, c) for w, c in counts.most_common() if c >= args.min_count]
    topics = []
    for w, c in candidates:
        flag = next(iter(pseg.cut(w))).flag
        if flag in KEEP:
            topics.append((w, c))
        if len(topics) >= args.n:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for w, c in topics:
            f.write(json.dumps({"topic": w, "count": c}, ensure_ascii=False) + "\n")
    print(f"{len(topics):,} topics -> {args.out}")
    print("sample:", " ".join(w for w, _ in topics[:40]))


if __name__ == "__main__":
    sys.exit(main())
