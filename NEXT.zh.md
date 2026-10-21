# 后续计划（2026-10-20）

PROGRESS.zh.md 是滚动实验日志，本文是待办路线图。所有代码位置已于 2026-10-20 核实。

## 0. 当前快照

- 语料：`synth_story.jsonl` 41,433 篇 → corpus.txt 40,995 篇 / 1,010 万字 / **523 万 token**
- 冠军：`runs/ple-zh-story3-s0.pt`，val **3.9831**（最优点 3.9416@3250 权重未保存）
- 训练耗时：5000 步 ≈ 8 分钟；蒸馏吞吐 ≈ 5 段/秒
- 端侧固件：**未动**，还是英文 Once-upon-a-time 配置

## 1. 模型质量：EOT 与串故事（2026-10-21 已结案）

**原现象**：prompt 之后第一个故事基本成篇，但结尾不吐 `<|endoftext|>` 自停，
会接着写第二个不相关的故事。

**根因（已定位，非模型问题）**：模型**学会了**吐 EOT——故事讲完后的 token 序列
就是训练分隔符 `\n EOT \n`（换行后第 2 个 token 即 id 0，6/6 seed 复现）。
串故事纯属推理侧没有停止检查：`src/model.py` 的 `generate()` 原本遇 id 0 不停
（本文旧版"generate() 本身遇 id 0 会停"的说法与代码不符，已修正），固件
`ino` 的 argmax 循环同样没有。两处各加一个停止判断即结案，零训练成本。

**遗留（真·模型问题）**：中段逻辑漂移、偶发语法崩塌（"没有及时表达输液"）、
他/她混用。这些是 523 万 token / 22.9M 参数的天花板，候选路线：换更强教师
（3B/7B）重蒸 41k 篇、语料继续放量（收益递减）。

## 2. 数据：还能榨的量

- 当前池 106 角色 × 106 情境 = 11,130 组合，已用 11,024，只剩 106 个
- **扩法 A（最省事）**：往 `research/zh/story_topics.py` 的 CHARACTERS / SITUATIONS
  继续加词（保持儿童友好、具体可写），然后：

  ```bash
  uv run python -m research.zh.story_topics --exclude data/zh/raw/synth_story.jsonl \
      --out data/zh/raw/story_topics_v3.json --n 0
  uv run python -m research.zh.synth --stage write --mode story \
      --topics data/zh/raw/story_topics_v3.json \
      --out data/zh/raw/synth_story_v3.jsonl \
      --batch-size 192 --variants 3 --max-new-tokens 240
  # 跑完合并进 synth_story.jsonl，删旧产物，重跑 prepare_zh --vocab 16384
  ```

- **扩法 B（上限更高）**：加第三个维度（场景：学校/超市/森林/医院…）或角色两两组合，
  组合空间直接翻几倍；改 story_topics.py 的 combos 生成逻辑即可
- 量化预期：每翻 2.5 倍 ≈ +0.30 nats，**收益递减**；冲到 1000 万 token 前值得再来一轮，
  之后应转向架构/配方而不是堆料

## 3. 训练配方备忘（数据实证）

- **val 见底步数 ≈ 1,250 步 / 208 万 token** → 523 万 token 对应 ~3,250 步（story3 实测吻合）
- batch 16 × seq 512 = 8,192 token/步；步数按上面的比例尺换算，别超见底点
- **必须加 `--save-every 250`**：story3 的最优点权重就是这么丢的
- 完整模板：

  ```bash
  uv run python -m research.tinystories.train --arm ple --corpus zh --vocab 16384 \
    --d-model 192 --n-layers 8 --n-heads 4 --ple-dim 128 --seq-len 512 \
    --target-core 3000000 --batch-size 16 --steps 3250 --tag zh-story4 \
    --save-every 250 --amp
  ```

## 4. 端侧固件适配（未动，位置已核实）

- `firmware/esp32_tinystories/esp32_tinystories.ino:31` `USE_DISPLAY 1 → 0`：
  display.h 只有 6x8 ASCII 点阵（display.h:40 `CW 6`），汉字只能走串口
- `ino:38` `PROMPT_IDS[]` 现在编码英文 "Once upon a time"。2026-10-21 实测：
  无提示词（仅以 EOT id 0 起步）生成质量**优于**带主题 prompt——模型自选
  "小X是……"式舒适开局，叙事弧更完整。可简化为 `{0}`，或写死一个中文开头
  （token ids 从 `data/zh/vocab-16384/tokenizer.json` 编码，可写个小脚本）
- `ino:39` `N_GENERATE = 130`：仅作兜底。argmax 循环已在 `best == 0`（EOT）处
  break（2026-10-21 加），模型自停落在 ~116 token（实测 110~137）；中文故事
  1.73 汉字/token，130 token 装 150~200 汉字宽裕
- `ino:149-150` KV cache 是 `float`（fp32），两片 kcache+vcache 直接让 PSRAM 爆表；
  **必须改 fp16**（int8 也可以试，注意精度）
- `runtime/bpe_tokenizer.h:5` ASCII-only（`>= 0x80` 拒绝）：要端侧中文输入得扩展编码器；
  偷懒方案是 PC 端预编码 prompt、固件里写死 ids
- 导出命令（注意 tokenizer 指向中文 vocab）：

  ```bash
  uv run python -m research.tinystories.export zh-s0 \
    --tokenizer data/zh/vocab-16384/tokenizer.json --out artifacts/zh
  ```

## 5. 杂项

- `scripts/fetch_model.sh` 的 PINNED sha256 是官方英文模型的；本地中文产物不走这条
- 若改 train.py 的 EOT 加权（第 1 节路线 2），先备份现有 train.py，保持英文管线可复现
