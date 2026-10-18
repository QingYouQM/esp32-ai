"""Build a story-topic pool by combining characters with situations.

Abstract wiki topics ("经济", "教育") make terrible story prompts; a story
needs someone doing something. Cross a cast of characters with a deck of
everyday situations and you get thousands of concrete, child-friendly story
seeds in the TinyStories spirit.

Usage:
    uv run python -m research.zh.story_topics --n 6000
"""

import argparse
import json
import random
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "zh" / "raw" / "story_topics.jsonl"

CHARACTERS = [
    "小明", "小红", "小华", "小丽", "小军", "小芳", "小杰", "小雨", "小林", "小云",
    "哥哥", "姐姐", "弟弟", "妹妹", "爷爷", "奶奶", "妈妈", "爸爸", "老师", "校工",
    "医生", "护士", "邮递员", "司机", "厨师", "警察", "消防员", "农夫", "渔夫", "园丁",
    "画家", "面包师", "图书管理员", "兽医", "列车员", "环卫工",
    "小猫", "小狗", "小兔子", "小刺猬", "小松鼠", "小鸭子", "小燕子", "小麻雀",
    "小乌龟", "小鹿", "小狐狸", "小熊", "小蚂蚁", "小蜜蜂", "蝴蝶", "老槐树",
    "一只流浪猫", "一只小山羊", "一只大雁", "一只萤火虫",
]

SITUATIONS = [
    "捡到一只受伤的小鸟", "第一次坐火车", "帮妈妈买菜", "在放学路上",
    "下雨天忘带伞", "生日那天", "迷路了", "交到一个新朋友", "种下一颗种子",
    "放风筝", "去河边玩", "参加运动会", "帮助了别人", "发现一个秘密",
    "丢了心爱的东西", "第一次做饭", "夏天的傍晚", "冬天的第一场雪",
    "搬进新家", "养了一盆花", "在图书馆里", "去果园摘苹果", "半夜听到声音",
    "学会了骑自行车", "拔牙", "看星星", "爬山", "捡到一枚硬币",
    "遇到一位老爷爷", "比赛输了", "考试得了满分", "帮邻居收衣服",
    "在公交车上让座", "堆雪人", "捉迷藏", "去看大海", "参观博物馆",
    "第一天上学", "和好朋友吵架了", "给小鸟做一个窝", "照顾生病的同学",
    "收到一份神秘的礼物", "在田野里放牛", "学游泳", "牙疼的老虎",
    "找夏天", "等一场雨停", "月亮出来了", "风筝断线了", "口袋里的种子发芽了",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    combos = [f"{c}{s}" for c in CHARACTERS for s in SITUATIONS]
    rng = random.Random(args.seed)
    rng.shuffle(combos)
    combos = combos[: args.n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for topic in combos:
            f.write(json.dumps({"topic": topic}, ensure_ascii=False) + "\n")
    print(f"{len(combos):,} story topics -> {args.out}")
    print("sample:", " / ".join(combos[:5]))


if __name__ == "__main__":
    sys.exit(main())
