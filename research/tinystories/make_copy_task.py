"""Generate materials whose stories are guaranteed to quote them.

A 3M core reproduces the shape of a card and none of its content: on twelve
cards taken from its own training set it scores 0% entity and 8% event. The
hand corpus cannot teach copying, because it never rewards it as such -- a story
that merely resembles its card lowers the loss exactly as much as one that
copies it -- and every extra document of the same kind adds more of the same
indifference.

So copying is made the cheapest thing to learn. The material is built from
slots, the story is built from the same slots with the event phrases quoted
verbatim, and the prose between the quotes varies. Two properties hold by
construction, and are asserted per document:

  entity  every name declared in 人物 appears in the story
  event   the story contains the 经过 turn and the 结果 phrase verbatim

Held-out cards are written as KB entries, which is the shape coverage.py reads,
so a synthetic card and a real card are measured by the same code.

    uv run python -m research.tinystories.make_copy_task --count 8000 ^
        --out data/copy-task/pairs.txt --held-out 200 --held-out-dir data/copy-task/entries
"""

import argparse
import json
import random
import sys
from pathlib import Path

from research.kb.render import pair_text
from research.kb.spec import FIELD_MAX, MATERIAL_MAX, STORY_MAX, STORY_MIN

SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华魏陶姜"
GIVEN = "嘉文永福志成明德振华清和守义安顺庆长贵昌平远山石金木"
ROLES = ("木匠", "货郎", "教书先生", "摆渡人", "更夫", "裁缝", "石匠",
         "挑夫", "船工", "账房", "漆匠", "篾匠")
OBJECTS = ("长凳", "茶壶", "渔网", "灯笼", "账本", "药箱", "木箱", "扁担",
           "草帽", "铜锁", "信笺", "菜篮", "竹篓", "酒坛", "秤杆", "面盆")
ERAS = ("大晟", "永昌", "明治", "民国", "光绪", "昭和", "咸丰", "天启")
NUM = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
       "十一", "十二", "十三", "十五", "十七", "十九")
SEASONS = ("开春", "入夏", "深秋", "腊月", "梅雨季节", "头一场雪后", "年关之前")
PARTS = ("天没亮", "晌午刚过", "太阳偏西", "掌灯时分", "后半夜", "清早")
PLACE_HEAD = ("青石", "南门", "东街", "西塘", "北岸", "下河", "柳林", "杏花",
              "沙坡", "后埔", "新安", "安平", "白沙", "后溪", "桥头", "榕树",
              "竹山", "双溪", "太平", "长乐")
PLACE_TAIL = ("镇", "村", "桥", "山", "寺", "馆", "铺", "巷", "街", "庄", "坊")
LUNAR = ("腊月二十三", "三月初七", "八月十五", "谷雨那天", "清明前后", "霜降")

# cause / turn / result. The turn is the pivot: it is quoted verbatim into the
# story, so it must read as a sentence, not as a summary.
SCENES = (
    dict(axis="赶路-失误", title="过桥送物",
         cause="要把修好的{obj}赶在天黑前送到{p2}",
         turn="{p1}的木板早就朽了，走到中间时木板断了，{obj}掉进水里",
         result="{p2}的人留他吃了晚饭，{obj}晾干后摆在门口"),
    dict(axis="看管-他者", title="夜里失守",
         cause="替{p2}看管一屋子的{obj}",
         turn="半夜有人敲门说{p1}那头出了事，他出去看了一眼，回来时门开着，{obj}少了两件",
         result="天亮后他一家一家去问，{obj}再没找回来"),
    dict(axis="寻找-未解", title="找一件旧物",
         cause="要找一件落在{p1}的旧{obj}",
         turn="在{p1}问了七户人家都说没见过，最后一个人在灶台后面指了指",
         result="他抱着{obj}出来，天已经黑透了"),
    dict(axis="等待-天气", title="等一个没来的人",
         cause="要在{p1}等一个从{p2}来的人",
         turn="雨从晌午下到天黑，{p1}的屋檐一直在滴水，那个人一直没有来",
         result="他把{obj}放在石阶上，自己走回去了"),
    dict(axis="手艺-意外", title="交晚了的活",
         cause="要给{p2}打一只{obj}",
         turn="锤子落在手上，血滴在木料上，那一处只好重新刨平",
         result="{obj}晚交了三天，{p2}的人没有说什么"),
    dict(axis="失而复得-误会", title="柜台底下",
         cause="把{obj}寄放在{p1}的铺子里",
         turn="回来时铺子关了门，掌柜说没有这件东西，两个人在门口吵了起来",
         result="第二天{obj}从柜台底下翻了出来，是学徒收错了地方"),
    dict(axis="借还-他者", title="磕掉一个角",
         cause="要向{p2}借{obj}用一天",
         turn="回来的路上{obj}磕在石头上，缺了一个角",
         result="他把{obj}还回去，赔了半吊钱，从此不再开口去借"),
    dict(axis="看火-失误", title="一窑灰",
         cause="在{p1}守着一窑炭",
         turn="后半夜他打了个盹，窑口的火塌下去，一窑炭烧成了灰",
         result="他蹲在窑口直到天亮，一句话也没说"),
    dict(axis="送信-意外", title="拆开过的信",
         cause="要把一封{obj}送到{p2}",
         turn="过{p1}的时候被一队人拦住盘问，{obj}被拆开看过",
         result="信还是送到了，收信的人看完只是点头"),
    dict(axis="学艺-回归", title="卖不出去的第一件",
         cause="跟{p2}的师傅学做{obj}",
         turn="第一件做坏了，师傅让他自己拿去{p1}卖掉",
         result="那件{obj}没卖出去，他一直留在身边"),
    dict(axis="赶集-天气", title="翻在沟里",
         cause="挑着一担{obj}去{p1}赶集",
         turn="半路上起了风，担子被吹翻在沟里，{obj}滚了一地",
         result="他一件件捡回来，到集市时摊子已经收了"),
    dict(axis="代人-反讽", title="认不出的那一件",
         cause="替{p2}的老人看一天{obj}",
         turn="老人回来时{obj}被擦得干干净净，他却认不出自己那一件了",
         result="老人笑着说了声好，把{obj}收进柜子里"),
)

# The scene decides what can be carried, made or missed; a 账本 cannot be
# hammered into shape, and a story that says so teaches a false pairing.
SCENE_OBJS = {
    "过桥送物": ("长凳", "木箱", "面盆", "扁担"),
    "夜里失守": ("灯笼", "铜锁", "酒坛", "草帽"),
    "找一件旧物": ("铜锁", "草帽", "秤杆", "木匣"),
    "等一个没来的人": ("菜篮", "竹篓", "面盆"),
    "交晚了的活": ("长凳", "木箱", "秤杆", "竹篓"),
    "柜台底下": ("竹篓", "菜篮", "铜锁"),
    "磕掉一个角": ("扁担", "秤杆", "面盆", "酒坛"),
    "一窑灰": ("竹篓",),
    "拆开过的信": ("信笺", "账本"),
    "卖不出去的第一件": ("长凳", "木箱", "竹篓", "面盆"),
    "翻在沟里": ("菜篮", "竹篓", "酒坛"),
    "认不出的那一件": ("铜锁", "灯笼", "面盆", "秤杆"),
}
SCENE_ROLES = {
    "过桥送物": ("木匠", "货郎", "挑夫"),
    "夜里失守": ("更夫", "账房", "摆渡人"),
    "找一件旧物": ("货郎", "教书先生", "挑夫"),
    "等一个没来的人": ("摆渡人", "船工", "更夫"),
    "交晚了的活": ("木匠", "篾匠", "石匠", "漆匠"),
    "柜台底下": ("货郎", "账房", "教书先生"),
    "磕掉一个角": ("挑夫", "船工", "篾匠"),
    "一窑灰": ("挑夫", "更夫"),
    "拆开过的信": ("货郎", "账房"),
    "卖不出去的第一件": ("木匠", "篾匠", "漆匠", "石匠"),
    "翻在沟里": ("挑夫", "货郎", "篾匠"),
    "认不出的那一件": ("更夫", "账房", "摆渡人"),
}

OPENERS = (
    "{time}，{name}从{p1}的住处出来，衣襟上还沾着木屑。",
    "{time}，{name}在{p1}的街口站了一会儿，把{obj}换了三次手。",
    "{time}，{name}把{obj}收拾好，锁上门，往{p1}的那头走。",
    "{time}，{name}在灶前烧了一壶水，灌进瓦罐，才动身。",
    "{time}，{name}早早就醒了，坐在门槛上等天亮。",
)
BODIES = (
    "这一趟是为了一件事：{cause}。",
    "他这些年做{role}，手上过的{obj}不少，{cause}却是头一回。",
    "临出门时他又回头看了一眼屋子，{cause}这件事压在心上，走得很慢。",
    "路上没有人，只有几只麻雀从房檐上飞起来；{cause}，他不敢耽搁。",
)
BEATS = (
    "他站在原地没有动，手指在袖口里绞着",
    "他先是想走，后来又回过头来",
    "他蹲下去看了很久，才慢慢站起来",
    "旁边有人问他怎么了，他摇了摇头",
    "风从巷子那头灌过来，吹得他睁不开眼",
)
CLOSERS = (
    "这件事后来没有人再提起",
    "很多年以后他还记得那天的光",
    "第二天他照旧去了{p2}",
    "他把这件事记在心里，没跟人说过",
)


def make_name(rng):
    return rng.choice(SURNAMES) + "".join(rng.choice(GIVEN) for _ in range(rng.choice((1, 1, 1, 2))))


def make_place(rng, taken):
    for _ in range(50):
        p = rng.choice(PLACE_HEAD) + rng.choice(PLACE_TAIL)
        if p not in taken and not any(p in t or t in p for t in taken):
            return p
    raise RuntimeError("no distinct place")


def make_time(rng):
    era, n = rng.choice(ERAS), rng.choice(NUM)
    return rng.choice((
        f"{era}{n}年{rng.choice(SEASONS)}",
        f"{rng.choice(SEASONS)}的{rng.choice(PARTS)}",
        rng.choice(LUNAR),
        f"{era}{n}年，{rng.choice(PARTS)}",
    ))


def build(rng):
    scene = rng.choice(SCENES)
    name = make_name(rng)
    role = rng.choice(SCENE_ROLES[scene["title"]])
    obj = rng.choice(SCENE_OBJS[scene["title"]])
    p1 = make_place(rng, ())
    p2 = make_place(rng, (p1,))
    when = make_time(rng)
    s = dict(name=name, role=role, obj=obj, p1=p1, p2=p2, time=when)

    turn = scene["turn"].format(**s)
    result = scene["result"].format(**s)
    cause = scene["cause"].format(**s)

    beat_a, beat_b = rng.sample(BEATS, 2)   # twice the same beat reads as a bug
    story = (rng.choice(OPENERS).format(**s)
             + rng.choice(BODIES).format(**s, cause=cause)
             + turn + "。"
             + beat_a + "。"
             + beat_b + "。"
             + result + "。"
             + rng.choice(CLOSERS).format(**s) + "。")

    fields = {
        "人物": f"{name}，{p1}的{role}。",
        "时间": f"{when}。",
        "地点": f"{p1}，还有{p2}。",
        "起因": f"{name}{cause}。",
        "经过": f"{name}走到{p1}，{turn}。",
        "结果": f"{result}。",
    }
    # The two gates the hand corpus never enforced on itself.
    assert name in story, "name missing from the story"
    assert turn in story, "the turn is not quoted"
    assert result in story, "the result is not quoted"
    for k, v in fields.items():
        assert len(v) <= FIELD_MAX[k], f"{k} over {FIELD_MAX[k]}: {len(v)}"
    assert sum(len(v) for v in fields.values()) <= MATERIAL_MAX
    assert STORY_MIN <= len(story) <= STORY_MAX, f"story {len(story)} chars"
    return scene, fields, story, when


def entry(idx, scene, fields, when):
    return {
        "id": f"syn-copy-{idx:06d}",
        "title": f"{scene['title']}·{when}"[:24],
        "source": {"work": "synthetic copy task", "url": "", "revision": None,
                   "quotes": ["generated by make_copy_task; no external source"]},
        "fields": {k: {"text": v, "evidence": None, "inferred": True}
                   for k, v in fields.items()},
        "notes": scene["axis"],
        "review": {"status": "draft", "by": None, "at": None},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=8000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", type=int, default=200,
                    help="cards written as KB entries instead of pairs")
    ap.add_argument("--held-out-dir", default="data/copy-task/entries")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out, held = Path(args.out), Path(args.held_out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    held.mkdir(parents=True, exist_ok=True)

    docs, lengths = [], []
    for i in range(args.count + args.held_out):
        scene, fields, story, when = build(rng)
        e = entry(i, scene, fields, when)
        if i < args.held_out:
            (held / f"{e['id']}.json").write_text(
                json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            docs.append(pair_text(e, story))
            lengths.append(len(story))

    out.write_text("".join(docs), encoding="utf-8")
    print(f"{len(docs)} pairs -> {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"{args.held_out} held-out cards -> {held}")
    print(f"story length min {min(lengths)} median {sorted(lengths)[len(lengths) // 2]} "
          f"max {max(lengths)}")
    print(f"scenes used: {len(set(d.split('<|story|>')[0][-60:] for d in docs))} distinct "
          f"material tails (should be ~all distinct)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
