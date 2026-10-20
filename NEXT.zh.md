# 后续计划（2026-10-20）

PROGRESS.zh.md 是滚动实验日志，本文是待办路线图。所有代码位置已于 2026-10-20 核实。

## 0. 当前快照

- 语料：`synth_story.jsonl` 41,433 篇 → corpus.txt 40,995 篇 / 1,010 万字 / **523 万 token**
- 冠军：`runs/ple-zh-story3-s0.pt`，val **3.9831**（最优点 3.9416@3250 权重未保存）
- 训练耗时：5000 步 ≈ 8 分钟；蒸馏吞吐 ≈ 5 段/秒
- 端侧固件：**未动**，还是英文 Once-upon-a-time 配置

## 1. 模型质量：EOT 与串故事（最大的一关）

**现象**：prompt 之后第一个故事基本成篇，但结尾不吐 `<|endoftext|>` 自停，
会接着写第二个不相关的故事；中段有逻辑漂移。2.5 倍语料（+0.30 nats）没有治好它。

**候选路线**（按性价比排序）：

1. **工程绕过（推荐做端侧 MVP）**：推理侧固定生成 ~160 token，后处理把输出截到
   最后一个句读号，只展示第一个故事。零训练成本，demo 够用。
2. **训练侧教停**：根因是模型学不会吐 EOT（generate() 本身遇 id 0 会停）。
   可试点：语料里混入"正常故事 + 显式收尾句 + EOT"样本，或最后几百步对 EOT 位置
   加权 loss（要动 train.py，先评估值不值）。
3. **接受现状**：串故事对"端侧能讲中文故事"的 demo 目标无伤大雅。

**决策点**：先按 1 把端侧跑通，再决定要不要投入 2。

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
- `ino:38` `PROMPT_IDS[]` 现在编码英文 "Once upon a time"，换成中文开头
  （token ids 从 `data/zh/vocab-16384/tokenizer.json` 编码，可写个小脚本）
- `ino:39` `N_GENERATE = 200` 是 **token 数**：200 汉字 ÷ 1.73 汉字/token ≈ 120，
  算上串故事加余量，建议 160
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
