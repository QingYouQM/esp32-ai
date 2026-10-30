"""Write material-and-story pairs in one call, in the direction of inference.

make_pairs.py takes stories that already exist and asks for the card they imply.
That works, but the model writes the story first and the card second, so the
card is a description of prose that ignored it. Here the card is written first
and the story has to follow it, which is the order the device uses and half the
API calls.

Diversity comes from the same axis grid as augment.py: a locked recipe with one
moving assignment is what stopped the first batch collapsing into one mould.

    uv run python -m research.tinystories.make_pairs_direct --limit 20 --dry-run
    uv run python -m research.tinystories.make_pairs_direct --count 14000 --workers 6
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.kb.render import material_text
from research.kb.spec import EOD_MARK, FIELDS, MATERIAL_MARK, STORY_MARK

from .augment import Combinator, call_api, clean, load_docs, read_key
from .make_pairs import extract_json, check_material

SYSTEM = """你是中文短故事作者。先写一张六字段的材料卡，再写一篇完全符合这张卡的故事。

规则：
- 材料卡只写具体、可核对的事实：谁、什么时候、在哪、因为什么、做了什么、结果如何。
- 主角姓名、地点、时间必须原样使用上面给定的值，一个字都不许改。年份是数字，就写数字。
- 故事必须严格服从材料卡：人物一个不多一个不少，地点和时间一致，起因、经过、结果
  都能在故事里找到对应。材料卡里没有的人，故事里不能出现。
- 故事 200 到 260 字，短句，面向小学三四年级读者。少于 150 字或多于 300 字整条作废。
- 字段字数上限：人物 40、时间 20、地点 20、起因 60、经过 90、结果 45。超过上限整条作废。
- 只写外部可见的动作和对话，不写心理活动，不写评价，不写比喻堆砌。
- 不用英文、拼音、括号注释。不要标题。

只输出 JSON，不要 Markdown 代码块，不要解释。键名固定：
{"人物": "...", "时间": "...", "地点": "...", "起因": "...", "经过": "...", "结果": "...", "story": "..."}"""

USER_TEMPLATE = """请写一对。这次的设定：
主角：{who}，姓名固定为「{name}」，材料卡的人物栏和故事正文都必须只用这个名字
地点：{where}
时间：{when}
转折：{turn}
结尾基调：{ending}

先写材料卡，再写故事，输出 JSON。"""

# The corpus has to carry the character inventory the resource set uses. A
# historical card says 陈嘉庚 and 1939年; a corpus of 木匠 and 一个下午 teaches a
# model that can copy neither. So the generator is handed a name, a place and
# sometimes a dated year, and the card has to use them verbatim.
SURNAMES = ("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦许何吕施张孔曹严华金魏陶姜谢邹"
            "喻柏窦章苏潘葛范彭鲁韦马苗方俞任袁柳唐薛雷贺倪汤罗毕郝安常傅康伍余"
            "顾孟黄穆萧尹姚邵汪毛米贝明戴宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾"
            "路娄江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万卢莫房解应丁邓单杭")
GIVEN = ("秀英桂英玉兰凤英秀珍秀英桂兰玉珍秀云秀梅秀芳秀华明华建华志强志明"
         "建国建军国强永强海燕晓燕丽娟雅静静怡子涵雨欣佳怡梓萱浩然宇轩俊杰子轩"
         "文博天成嘉诚志远清和守义春生秋实冬青夏至长安永年")
PLACES = ("青石镇", "白沙渡", "柳溪村", "城西的当铺", "南门口的米行", "镇口的旧木桥",
          "山脚的窑厂", "沿河的码头", "东街的茶馆", "村尾的老槐树", "镇北的学堂",
          "海边的渔港", "石桥下的桥洞", "关帝庙后的空场", "城里的火车站", "坡上的果园")
TIMES = ("一个下午", "天快黑的时候", "第二天清早", "过了三天", "开春", "入伏以后")
# A name belongs to a person. The axis grid includes animals (小刺猬), and
# "小刺猬，姓名固定为梁梓" produces a card nobody can act on.
HUMANS = ("木匠", "货郎", "教书先生", "船工", "绣娘", "学徒", "车夫", "账房先生",
          "渔夫", "更夫", "挑夫", "裁缝", "花匠", "邮差", "面包店老板", "看门人",
          "图书管理员", "送水工", "泥瓦匠", "卖菜的")
ANIMALS = ("兔", "刺猬", "狗", "猫", "鸟", "蚂蚁", "龟", "蜻蜓", "熊", "鱼",
           "松鼠", "狐狸", "鸽", "鹅", "鸭", "鸡", "蛇", "蜗牛", "蜜蜂", "蝴蝶")


def pick_place_year(rng, dated):
    when = (f"{rng.choice([1851, 1874, 1898, 1900, 1911, 1921, 1937, 1939, 1949, 1953])}年"
            if dated else rng.choice(TIMES))
    return rng.choice(PLACES), when


def build_user(axes_at, seed):
    # The axis names are augment.py's, not new ones: one grid for the whole
    # corpus, so the two generators cannot drift into different worlds.
    rng = random.Random(seed)
    name = rng.choice(SURNAMES) + "".join(rng.sample(GIVEN, 1))
    place, when = pick_place_year(rng, dated=(seed % 10 < 3))
    who = axes_at.get("protagonists", "木匠")
    if any(a in who for a in ANIMALS):
        who = rng.choice(HUMANS)
    return USER_TEMPLATE.format(
        who=who,
        name=name,
        where=place,
        when=when,
        turn=axes_at.get("turns", "遇到一件意外的事"),
        ending=axes_at.get("endings", "平静的"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axes", default="research/tinystories/seeds/zh/axes.json")
    ap.add_argument("--out", default="research/tinystories/seeds/zh/pairs.txt")
    ap.add_argument("--count", type=int, default=14000, help="how many to add")
    ap.add_argument("--limit", type=int, default=0, help="stop after N (testing)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "MiniMax-M3"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL",
                                                         "https://api.minimax.cn/v1"))
    ap.add_argument("--api-key", default="")
    ap.add_argument("--key-file", default="data/api_key.txt")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    axes = json.loads(Path(args.axes).read_text(encoding="utf-8"))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = len(load_docs(out_path)) if out_path.exists() else 0
    comb = Combinator(axes, seed=0)
    start = have
    todo = min(args.count, args.limit) if args.limit else args.count

    print(f"already paired {have}; adding up to {todo} "
          f"(combinations available {comb.total():,})")
    if args.dry_run:
        for i in range(start, start + 2):
            print("\n" + build_user(comb.at(i), i))
            print(SYSTEM)
        return 0

    api_key = read_key(args)
    if not api_key:
        raise SystemExit(f"no API key in {args.key_file}, $OPENAI_API_KEY or --api-key")

    log = out_path.with_suffix(".rejects.txt")
    kept, rejects, api_fail = 0, 0, 0
    t0 = time.time()

    def one(i):
        prompt = build_user(comb.at(i), i)
        raw, err = call_api(prompt, args.model, args.base_url, api_key,
                            args.temperature, args.timeout,
                            max_tokens=args.max_tokens, thinking=False,
                            reasoning_effort="low", system=SYSTEM)
        if err:
            return i, None, err, "api"
        try:
            obj = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            return i, None, f"json: {e}", "json"
        if not isinstance(obj, dict):
            return i, None, "not an object", "json"
        story = str(obj.pop("story", "")).strip()
        story = clean(story)
        # The corpus band is 120-300 characters; asking for 180-260 and then
        # refusing 150 rejects work the corpus would have accepted.
        if not 120 <= len(story) <= 300:
            return i, None, f"story length {len(story)}", "check"
        fields = {k: str(obj.get(k, "")).strip() for k in FIELDS}
        ok, why = check_material(fields, story)
        if not ok:
            return i, None, why, "check"
        return i, (fields, story), "", ""

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, start + j) for j in range(todo)]
        for n, fut in enumerate(as_completed(futures), 1):
            i, payload, why, kind = fut.result()
            if why:
                rejects += 1
                if kind == "api":
                    api_fail += 1
                with log.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(f"{i}\t{kind}\t{why[:90]}\n")
                continue
            fields, story = payload
            entry = {"fields": {k: {"text": fields[k]} for k in FIELDS}}
            doc = (f"{MATERIAL_MARK}\n{material_text(entry)}\n{STORY_MARK}\n"
                   f"{story}\n{EOD_MARK}\n")
            with out_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(doc)
            kept += 1
            if n % 20 == 0:
                rate = n / max(time.time() - t0, 1e-6)
                left = (todo - n) / max(rate, 1e-9) / 60
                print(f"  {n}/{todo} kept {kept} rejected {rejects} "
                      f"(api {api_fail}) {rate:.2f}/s eta {left:.0f}min", flush=True)

    print(f"\ndone. kept {kept}, rejected {rejects} (api {api_fail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
