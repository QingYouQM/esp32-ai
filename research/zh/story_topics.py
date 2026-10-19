"""Build a story-topic pool by combining characters with situations.

Abstract wiki topics ("经济", "教育") make terrible story prompts; a story
needs someone doing something. Cross a cast of characters with a deck of
everyday situations and you get thousands of concrete, child-friendly story
seeds in the TinyStories spirit.

Usage:
    uv run python -m research.zh.story_topics --n 20000 \
        --exclude data/zh/raw/synth_story.jsonl --out data/zh/raw/story_topics_v2.json
"""

import argparse
import json
import os
import random
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "zh" / "raw" / "story_topics.jsonl"

CHARACTERS = [
    # 孩子与家人
    "小明", "小红", "小华", "小丽", "小军", "小芳", "小杰", "小雨", "小林", "小云",
    "小刚", "小强", "小敏", "小燕", "小波", "小涛", "小婷", "小磊", "小静", "小鹏",
    "哥哥", "姐姐", "弟弟", "妹妹", "外公", "外婆", "叔叔", "阿姨", "表哥", "表姐",
    "爷爷", "奶奶", "妈妈", "爸爸", "同桌", "校长",
    # 职业
    "老师", "校工", "医生", "护士", "邮递员", "司机", "厨师", "警察", "消防员",
    "农夫", "渔夫", "园丁", "画家", "面包师", "图书管理员", "兽医", "列车员", "环卫工",
    "宇航员", "飞行员", "船长", "科学家", "记者", "理发师", "裁缝", "木匠", "鞋匠",
    # 动物与自然
    "小猫", "小狗", "小兔子", "小刺猬", "小松鼠", "小鸭子", "小燕子", "小麻雀",
    "小乌龟", "小鹿", "小狐狸", "小熊", "小蚂蚁", "小蜜蜂", "蝴蝶", "老槐树",
    "一只流浪猫", "一只小山羊", "一只大雁", "一只萤火虫",
    "小象", "小河马", "小考拉", "小企鹅", "小青蛙", "小蝌蚪", "小鲤鱼", "小螃蟹",
    "小恐龙", "小机器人", "小木偶", "小鲸鱼", "小海豚", "小水母", "小人鱼",
    "小精灵", "小巫师", "一朵蒲公英", "一颗小星星", "月亮", "太阳", "一朵雪花", "一阵风",
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
    # 第一次与成长
    "第一次坐飞机", "第一次坐公交", "自己走回家", "打碎了花瓶", "弄坏了玩具",
    "换了新书包", "弄丢了作业本", "借钱忘记还", "学会道歉", "照顾弟弟妹妹",
    "登台表演", "画了一幅画", "写了一封信", "寄一张明信片", "加入合唱团",
    "加入足球队",
    # 家务与校园
    "打扫教室", "整理房间", "学洗碗", "植树节那天", "分享零食",
    # 生病与看医生
    "发烧住院", "打疫苗", "看牙医", "配了一副眼镜",
    # 自然观察
    "养蚕宝宝", "观察蜗牛", "抓知了", "看到了彩虹", "早起看日出",
    "秋天的落叶", "春天的野餐", "台风来了", "停电的夜晚", "害怕打雷",
    # 出行与户外
    "去露营", "学钓鱼", "摘草莓", "挖红薯", "喂鸽子", "在海边捡贝壳",
    "第一次坐轮船", "第一次坐地铁", "第一次坐缆车", "在超市里走散了",
    # 节日与美食
    "逛庙会", "看花灯", "看烟花", "守岁", "包饺子", "做月饼", "贴春联",
    # 意外与小麻烦
    "雨天踩水坑", "跑步摔倒了",
]


def load_used(path):
    """Topics already present in an existing corpus file (jsonl or plain lines).

    Regenerating the pool from bigger lists re-produces every old combo; the
    teacher would then rewrite them and the corpus doubles topics instead of
    broadening them.
    """
    used = set()
    if not path or not os.path.exists(path):
        return used
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    used.add(json.loads(line)["topic"])
                    continue
                except (ValueError, KeyError):
                    pass
            used.add(line)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="max topics (0 = all combos)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None,
                    help="file with already-used topics to skip")
    args = ap.parse_args()

    combos = [f"{c}{s}" for c in CHARACTERS for s in SITUATIONS]
    used = load_used(args.exclude)
    if used:
        before = len(combos)
        combos = [c for c in combos if c not in used]
        print(f"excluded {before - len(combos):,} already-used topics")
    rng = random.Random(args.seed)
    rng.shuffle(combos)
    if args.n:
        combos = combos[: args.n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for topic in combos:
            f.write(json.dumps({"topic": topic}, ensure_ascii=False) + "\n")
    print(f"{len(combos):,} story topics -> {args.out}")
    print("sample:", " / ".join(combos[:5]))


if __name__ == "__main__":
    sys.exit(main())
