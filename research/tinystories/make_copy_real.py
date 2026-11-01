"""E2 surface-form round: the same copy supervision dressed as a real card.

E1 verdict (runs/e1a-*, runs/e1b-*): with the sampler fixed, a 3M core taught
by make_copy_gen copies unseen name combinations at 98% person / 83% both-spans,
but transfers 0% to the 24 real cards -- it composes "陈集美" out of card
characters instead of copying "陈嘉庚", and runs off into the twelve synthetic
templates. The copy behaviour generalises over name combinations, not over
material surface forms.

The real cards differ from the synthetic corpus in surface statistics, not in
structure: Arabic-digit years ("1913年起"), ranges ("1913年至1952年"),
institutions (学堂、报馆、罐头厂), enumerated 经过 ("划分疫区、限制流动，封锁铁
路"), outcome numbers ("九百九十九年"). None of that ever appears in
make_copy_task's twelve template worlds.

This round keeps the verbatim turn/result quotes (the copy supervision) and
replaces the surface layer with the real cards' statistics, genre-matched to
the overseas-Chinese civic/industry corpus but content-disjoint from it: no
real person, no real place, no real institution. The 24 real cards stay
evaluation-only.

    uv run python -m research.tinystories.make_copy_real \
        --count 8000 --tokenizer data/zh-ground/vocab-16384/tokenizer.json \
        --out data/copyreal/pairs.txt --entries-dir data/copyreal/entries
"""

import argparse
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders

from research.kb.render import pair_text
from research.kb.spec import (
    COMPOUND_SURNAMES,
    FIELD_MAX,
    MATERIAL_MAX,
    STORY_MAX,
    STORY_MIN,
    SURNAMES,
)
from research.tinystories.make_copy_task import BEATS

# Names of the 24 evaluation cards. They must never be generated, so the
# corpus cannot teach any of them by accident.
REAL_NAMES = (
    "陈嘉庚", "陈六使", "黄乃裳", "胡文虎", "伍连德", "张弼士", "叶亚来",
)

# Classical given-name characters: frequent in names, rare in prose, so a
# generated name reads like a person and not like a sentence fragment.
GIVEN_CHARS = (
    "嘉庚文永福振德明志成仁义礼智信忠孝廉安邦定国兴华建业立本维新"
    "泽民恩培元亨瑞祥麟凤鸣岐山川兰芳玉堂锦绳武世昌克绍英杰俊彦才良"
    "弼辅汝淑婉娴静仪佩秋月云鹤松柏龄延年鹤龄松年延龄庆祥瑞霖雨亭"
    "思远致诚景行维则秉衡守正怀瑾握瑜慕韩师韩宗韩少韩慕唐绍唐"
    "竹君石泉砚卿墨卿琴溪渔樵耕读牧歌晓帆济川慕樵啸岑研耕砚农"
    "承传家声显耀富贵荣华前裕后孝悌诗书鸿鹄凌云鹏程飞翔高翱涛"
    "潮汐澜波澎湃涌泓渊澄碧珠珍玲珊瑞琪琦瑶瑾璇玫桂馨香馥郁芬芳"
    "栋梁材森楠杉梧桐榆槐杨枫彬彭泽浩瀚宇寰星辰日月辉光耀照"
    "焕炜烨烽炎灿烁煌熹燃煦暖春晖曦曦昕昀晞晚晴朗畅快意遂成"
)

PLACE_HEAD = ("永安", "宁阳", "潮安", "梅江", "嘉应", "同禾", "云霄", "诏溪",
              "海澄", "漳浦", "南靖", "长泰", "东山", "平和", "连城", "上杭",
              "武平", "清流", "宁化", "将乐", "沙邑", "尤溪", "大田", "德化",
              "永春", "仙游", "惠北", "安溪", "同禾", "金门", "思明", "禾山")
PLACE_TAIL = ("县", "镇", "城", "港", "埠", "街", "村", "社", "圩", "湾")
PLACE_BARE = ("埠头", "矿场", "码头", "坡上", "河口", "沙田", "盐场", "林场")

ERAS = ("光绪", "宣统", "民国")
SEASONS = ("春", "夏", "秋", "冬")

# cause / turn / result. The turn carries a year or a count and is quoted
# verbatim into the story; the result is quoted verbatim as the closure.
# Slots: {name} {p1} {p2} {year} {year2} {n1} {n2} {inst}
SCENES = (
    dict(axis="兴学-落成", title="乡里办学",
         cause="决意在{p1}办一所{inst}",
         turn="{year}年学堂落成，头一年招了{n1}名学生",
         result="乡里的孩子从此不必再走{n2}里路去县城念书"),
    dict(axis="兴学-女学", title="办女学",
         cause="要替{p1}的女孩子办一间女学",
         turn="{year}年女学开学，来报名的有{n1}人",
         result="县里头一间女学就这么立住了"),
    dict(axis="公益-修桥", title="独资修桥",
         cause="见{p1}的石桥年久失修，出资重修",
         turn="桥修到一半遇上山洪，木料冲走了{n1}成",
         result="第二年桥修成，行人不用再绕{p2}的渡口"),
    dict(axis="公益-医院", title="捐建医院",
         cause="在{p1}捐建一所{inst}",
         turn="{year}年医院开诊，头一个月看了{n1}名病人",
         result="穷人来抓药，只收半价"),
    dict(axis="实业-建厂", title="开办工厂",
         cause="在{p2}筹办一间{inst}",
         turn="{year}年工厂开机，雇了{n1}名工人",
         result="三年间厂子扩到{n2}间厂房"),
    dict(axis="实业-报馆", title="创办报馆",
         cause="决定在{p2}办一份报纸",
         turn="报纸印到第{n1}期时，订户过了{n2}千",
         result="同乡要写信回家，总先翻一翻这张报"),
    dict(axis="实业-侨批", title="开办批局",
         cause="在南洋开一间批局，替同乡汇款捎信回家",
         turn="{year}年批局开张，头一年经手汇款{n1}万元",
         result="信和钱从{p2}到{p1}，走的不满一个月"),
    dict(axis="公益-义渡", title="设义渡",
         cause="见{p1}渡口船资太贵，出资设一艘义渡船",
         turn="{year}年起义渡免费，摆渡的老船工由他按月付工钱",
         result="两岸的人过河，从此不花一个铜板"),
    dict(axis="公益-平粜", title="灾年平粜",
         cause="{p1}遭了灾，米价一日三涨",
         turn="他把米行的存米平价放出，一共{n1}石",
         result="灾年里头，{p1}没有饿死人"),
    dict(axis="公益-水利", title="兴修水利",
         cause="出钱修{p1}的旧陂圳",
         turn="{year}年动工，挖了{n1}里新渠",
         result="第二年春天，望天田变成了水浇田"),
    dict(axis="公益-马路", title="修筑马路",
         cause="出资把{p1}到{p2}的土路修成马路",
         turn="{year}年路成，全长{n1}里",
         result="挑担的、推车的，一天能打一个来回"),
    dict(axis="公益-育婴", title="设育婴堂",
         cause="在{p1}设一间育婴堂",
         turn="{year}年堂里收养了{n1}名婴儿",
         result="十年下来，活下来的孩子有{n2}个"),
    dict(axis="兴学-奖学金", title="设奖学金",
         cause="在{p1}的学堂设一笔奖学金",
         turn="{year}年起，每年资助{n1}名学生出洋读书",
         result="学成回来的人，又在{p1}办起了新学堂"),
    dict(axis="实业-罐头", title="菠萝罐头厂",
         cause="在{p2}办一间罐头厂",
         turn="{year}年出第一批货，装了{n1}箱上船",
         result="两年后，码头上的洋行都认这块招牌"),
    dict(axis="实业-碾米", title="新式碾米",
         cause="从南洋运回一架新式碾米机",
         turn="{year}年机器装好，一天能碾{n1}石谷",
         result="附近几个乡的谷子，都拉到{p1}来碾"),
    dict(axis="公益-藏书", title="建藏书楼",
         cause="在{p1}建一座藏书楼",
         turn="{year}年落成，藏书{n1}万卷",
         result="楼门日夜不锁，读书人随到随看"),
    dict(axis="公益-消防", title="置办消防",
         cause="替{p1}的街坊置办一套消防水龙",
         turn="{year}年水龙到位，又招了{n1}名义务消防队员",
         result="此后{p1}再失火，没有烧成过一条街"),
    dict(axis="公益-公园", title="辟公园",
         cause="把{p1}河边的一块荒地辟成公园",
         turn="{year}年开放，园里种了{n1}棵树",
         result="傍晚的时候，河边的空地总有人唱歌"),
    dict(axis="调解-械斗", title="平息械斗",
         cause="{p1}和{p2}为争水源结了怨",
         turn="他两头走了一个月，请两姓父老坐下来谈",
         result="{year}年起，两姓共用水渠，再没有动过手"),
    dict(axis="公益-义庄", title="设义庄",
         cause="在南洋设一间义庄，暂厝同乡的灵柩",
         turn="{year}年义庄建成，头一年运回{n1}口棺柩",
         result="客死他乡的人，都能归葬故土"),
    dict(axis="实业-垦荒", title="招工垦荒",
         cause="在{p2}看中一片荒地，招同乡去垦",
         turn="{year}年头一批{n1}户人家下船",
         result="五年后，荒地成了{n2}亩熟田"),
    dict(axis="实业-造林", title="造林", 
         cause="出钱在{p1}后山造林",
         turn="{year}年种下{n1}万株树苗",
         result="十几年后，后山的泉水又旺了起来"),
    dict(axis="兴学-夜学", title="办夜学",
         cause="见{p1}的做工人家不识字，办一所夜学",
         turn="{year}年开课，白天做工的人晚上来听",
         result="一冬下来，{n1}个工人能自己看报纸了"),
    dict(axis="公益-赠药", title="施药",
         cause="每年入夏，{p1}总有人中暑发痧",
         turn="{year}年起他在街口设一个茶摊，又备下{n1}包药",
         result="喝了凉茶、拿了药的人，分文不花"),
)

INST_BY_SCENE = {
    "乡里办学": ("小学", "学堂", "新式学堂"),
    "办女学": ("女学", "女子学堂"),
    "捐建医院": ("医院", "医馆", "诊所"),
    "开办工厂": ("纱厂", "织布厂", "酱油厂", "机器厂"),
    "创办报馆": ("报馆", "印务局"),
    "开办批局": ("批局", "信局"),
}

OPENERS = (
    "{year}年{season}，{name}从南洋回到{p1}。",
    "{year}年{season}，{name}把{p1}的事一件件安排下来。",
    "那阵子{name}常在{p1}的街上走，看见什么都要停下来问一问。",
    "{year}年，{name}把行里的事交给伙计，自己搬回了{p1}。",
    "{name}在{p2}的生意上了轨道，人却总惦记着{p1}。",
    "那年{season}，{name}从{p2}动身回乡，行李里带的全是图纸和账册。",
    "{year}年{season}，{name}站在{p1}的码头上，数着下船的同乡。",
    "头一回从{p2}回{p1}，{name}在船上就想好了要做的几件事。",
)

BODIES = (
    "{cause}这件事，他心里已经盘算了很多年。",
    "{cause}——话是他自己说的，钱也是他自己出的。",
    "主意定了以后，{name}先请了{p1}的几位父老来商量。",
    "有人劝他省下这笔钱，{name}只是摇头，说事情总要有人起头。",
    "{cause}，账房的先生劝了他三回，他一回也没有听。",
    "{name}算过一笔账：{cause}，这些钱花得比留在行里值。",
    "{p1}的父老起先不信，说这样的人和事只在戏文里有。",
)

CLOSERS = (
    "后来的人提起{p1}，总会先说起这件事。",
    "多少年后，{p1}的老人讲起旧事，还是从这一年讲起。",
    "他后来又做了几件事，但{p1}的人最先记得的，还是这一件。",
    "学堂的钟声、机器的响声，{p1}的人慢慢都听惯了。",
    "账册上这一笔，他记的却是{p1}的名字。",
    "有人问他图什么，他说在外头挣的钱，总要有个去处。",
)

PUNCT_STRIP = "，。；：、！？"


def single_char_cjk(tok):
    bl = decoders.ByteLevel()
    out = []
    for t in tok.get_vocab():
        try:
            d = bl.decode([t])
        except Exception:
            continue
        if len(d) == 1 and "\u4e00" <= d <= "\u9fff":
            out.append(d)
    return out


def build_pools(tok, rng):
    single = set(single_char_cjk(tok))
    given = [c for c in dict.fromkeys(GIVEN_CHARS) if c in single]
    if len(given) < 80:
        raise SystemExit(f"only {len(given)} given chars survive the tokenizer; widen GIVEN_CHARS")
    return given


def make_name(rng, surnames, given):
    while True:
        surname = rng.choice(surnames)
        n = 2 if rng.random() < 0.8 else 1
        name = surname + "".join(rng.choice(given) for _ in range(n))
        if name not in REAL_NAMES:
            return name


def make_place(rng, taken):
    for _ in range(60):
        if rng.random() < 0.15:
            p = rng.choice(PLACE_BARE)
        else:
            p = rng.choice(PLACE_HEAD) + rng.choice(PLACE_TAIL)
        if p not in taken and not any(p in t or t in p for t in taken):
            return p
    raise RuntimeError("no distinct place")


NUMS = ("三", "五", "七", "九", "十", "十二", "十八", "二十", "三十", "四十",
        "五十", "六十", "八十", "百", "三百", "五百", "八百", "一千", "两千")


def make_time(rng):
    year = rng.randint(1885, 1948)
    style = rng.random()
    if style < 0.35:
        return f"{year}年", year, None
    if style < 0.55:
        return f"{year}年{rng.choice(SEASONS)}", year, None
    year2 = min(1949, year + rng.randint(3, 12))
    return f"{year}年至{year2}年", year, year2


def build(rng, name, given, tries=80):
    """A document, resampled scene-by-scene until the story lands 150-200 chars.

    Wrong-length stories are resampled, never dropped: dropping silently
    skews the corpus toward short scenes and, worse, leaves names registered
    as "seen" whose documents were never written.
    """
    for _ in range(tries):
        doc = _build_once(rng, name, given)
        if doc is not None and STORY_MIN <= len(doc[2]) <= STORY_MAX:
            return doc
    raise RuntimeError(f"no in-length story for {name!r}")


def _build_once(rng, name, given):
    scene = rng.choice(SCENES)
    p1 = make_place(rng, ())
    p2 = make_place(rng, (p1,))
    when, year, year2 = make_time(rng)
    inst = rng.choice(INST_BY_SCENE.get(scene["title"], ("学堂",)))
    n1, n2 = rng.choice(NUMS), rng.choice(NUMS)
    s = dict(name=name, p1=p1, p2=p2, year=year, year2=year2 or year,
             n1=n1, n2=n2, inst=inst, season=rng.choice(SEASONS))

    cause = scene["cause"].format(**s)
    turn = scene["turn"].format(**s)
    result = scene["result"].format(**s)

    beat_a, beat_b = rng.sample(BEATS, 2)
    closer = rng.choice(CLOSERS).format(**s)
    story = (rng.choice(OPENERS).format(**s)
             + rng.choice(BODIES).format(**s, cause=cause)
             + turn + "。"
             + beat_a + "。"
             + beat_b + "。"
             + result + "。"
             + (closer if closer.endswith("。") else closer + "。"))

    fields = {
        "人物": f"{name}，{p1}人，从南洋回乡的实业人士。",
        "时间": f"{when}。",
        "地点": f"{p1}，还有{p2}。",
        "起因": f"{name}{cause}。",
        "经过": f"{name}在{p1}着手此事。{turn}。",
        "结果": f"{result}。",
    }
    assert name in story, "name missing from the story"
    assert turn in story, "the turn is not quoted"
    assert result in story, "the result is not quoted"
    for k, v in fields.items():
        assert len(v) <= FIELD_MAX[k], f"{k} over {FIELD_MAX[k]}: {len(v)} chars: {v}"
    assert sum(len(v) for v in fields.values()) <= MATERIAL_MAX
    return scene, fields, story, when, turn, result, p1


def entry(idx, scene, fields, when, turn, result, p1, group, name):
    spans = [when.rstrip("。"), p1, turn, result]
    return {
        "id": f"syn-real-{group}-{idx:04d}",
        "title": f"{scene['title']}·{when}"[:24],
        "source": {"work": "synthetic E2 surface-form round", "url": "",
                   "revision": None,
                   "quotes": ["generated by make_copy_real; no external source"]},
        "fields": {k: {"text": v, "evidence": None, "inferred": True}
                   for k, v in fields.items()},
        "acceptance": {"required_spans": spans},
        "notes": f"{scene['axis']}; group={group}; name={name}",
        "review": {"status": "draft", "by": None, "at": None},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=8000)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--entries-dir", required=True)
    ap.add_argument("--eval-seen", type=int, default=100)
    ap.add_argument("--eval-unseen", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tok = Tokenizer.from_file(args.tokenizer)
    given = build_pools(tok, rng)
    surnames = list(SURNAMES) + list(COMPOUND_SURNAMES)

    out, held = Path(args.out), Path(args.entries_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    (held / "eval-unseen").mkdir(parents=True, exist_ok=True)
    (held / "eval-seen").mkdir(parents=True, exist_ok=True)

    docs, train_names = [], set()
    for _ in range(args.count):
        name = make_name(rng, surnames, given)
        train_names.add(name)
        scene, fields, story, when, turn, result, p1 = build(rng, name, given)
        e = entry(0, scene, fields, when, turn, result, p1, "train", name)
        docs.append(pair_text(e, story))
    corpus = "".join(docs)
    out.write_text(corpus, encoding="utf-8")

    def write_group(group, n, name_of):
        for i in range(n):
            name = name_of()
            scene, fields, story, when, turn, result, p1 = build(rng, name, given)
            e = entry(i, scene, fields, when, turn, result, p1, group, name)
            (held / group / f"{e['id']}.json").write_text(
                json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")

    write_group("eval-seen", args.eval_seen, lambda: rng.choice(sorted(train_names)))

    def unseen():
        # Not in the training name list AND not a substring of the corpus:
        # a two-character name is often the prefix of a three-character
        # training name, and a substring hit would fake a prior.
        while True:
            name = make_name(rng, surnames, given)
            if name not in train_names and name not in corpus:
                return name
    write_group("eval-unseen", args.eval_unseen, unseen)

    real_leak = [n for n in REAL_NAMES if n in corpus]
    print(f"{len(docs)} training pairs -> {out} "
          f"({out.stat().st_size / 1e6:.2f} MB)")
    print(f"distinct training names: {len(train_names)}; "
          f"given-char pool {len(given)}; scenes {len(SCENES)}")
    print(f"real-name leakage into corpus: {real_leak or 'none'}")
    assert not real_leak, "a real evaluation name leaked into training"
    for group in ("eval-seen", "eval-unseen"):
        paths = sorted((held / group).glob("*.json"))
        names = [json.loads(p.read_text(encoding="utf-8"))["notes"].split("name=")[-1]
                 for p in paths]
        leak = sum(n in corpus for n in names)
        real = [n for n in names if n in REAL_NAMES]
        print(f"{group}: {len(paths)} cards, {leak} names verbatim in training, "
              f"real-name hits {real or 'none'}")
        assert group != "eval-unseen" or leak == 0, "unseen split leaked"
        assert not real, "real name generated into evaluation"
    return 0


if __name__ == "__main__":
    sys.exit(main())
