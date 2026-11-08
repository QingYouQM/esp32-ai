# 中文模型训练进度

目标：在 ESP32-S3 N16R8 上端侧生成一个 **150~200 汉字**、语句流畅、语义连贯的
中文语言模型。这是原仓库（英文 TinyStories，28.9M 存参 / 559K core）的中文改造。

## 方向（已确认）

- 语料：**纯故事蒸馏合成**（150~200 字短篇故事；维基/百科内容已从管线剔除，2026-10-19 起 story-only）
- 取向：**质量优先**，core 从 559K 扩到 ~3M，代价是推理从 ~10 tok/s 降到 ~5 tok/s
- 词表：自训 ByteLevel BPE，**vocab 16384**

## 已完成

| 项 | 状态 |
|---|---|
| CUDA 环境 | torch 2.14.1+cu130（`pyproject.toml` 把 torch 指向 cu130 源；默认 Windows wheel 是 CPU 版） |
| 教师模型 | Qwen2.5-1.5B-Instruct，3.09GB，缓存在 `models/`（已 gitignore） |
| 公开语料 | 中文维基 245,572 段 / **6000 万字符** → `data/zh/raw/wiki_corpus.txt` |
| 主题池 | **12,000** 个中文主题词（jieba 从维基挖掘）→ `data/zh/raw/topics.jsonl` |
| 合成语料 | **20,610 段** / 15.5 MB → `data/zh/raw/synth.jsonl`（12,000 主题 × 5 种 prompt 变体，保留率 99.3%，已全部完成） |
| 训练链路 | 冒烟 200 步通过：**8 步/秒**，val ppl 16923 → 1624 |
| 吞吐实测 | 合成生成 **4.4 段/秒**（batch 192）；训练不是瓶颈，数据生成才是 |

## 纯故事管线实验结果（2026-10-19）

剔除百科/维基后（synth_story.jsonl 16,604 篇 / 414 万字符，重训 BPE 后 1.75 汉字/token）：

| run | 步数 | best val | final val | 备注 |
|---|---:|---:|---:|---|
| ple-zh-story-s0 | 5000 | **4.3465**@1250 | 6.1347 | 1250 步后纯过拟合（train 0.70 / val ppl 462），存档为 final 权重勿用 |
| ple-zh-story2-s0 | 1500 | 4.2459 | 4.2459 | cosine 衰减终点即最优点，**前冠军** |
| （旧参照）ple-zh-s0 | 5000 | 4.679 | 4.679 | 百科+故事混合语料 |

## 放量后首训（2026-10-20，40,995 篇 / 1010 万字 / 523 万 token）

同架构同配方（batch 16 × seq 512，5000 步，8 分钟）：

| run | best val | final val | 备注 |
|---|---:|---:|---|
| ple-zh-story3-s0 | **3.9416**@3250 | 3.9831 | 较前冠军净赢 **0.30 nats** |

- val 在 3250 步见底（ppl 51.5），之后过拟合抬头：train 一路降到 2.57，val 回升到 3.98
- **教训：--save-every 没开，最优点 3250 的权重没保住**，.pt 存的是终点；下次补齐 save_every=250
- 定性未见质变：故事语感锁定、无百科残留，但依旧串故事（一个 prompt 后面跟着不相关的第二个故事）、不会在结尾吐 EOT、逻辑漂移仍在
- 结论：**2.5 倍数据 ≈ +0.30 nats**，天花板仍受 523 万 token / 22.9M 参数约束；要继续压 val 得继续扩语料，加步数只会加剧过拟合

- 纯故事域匹配净赢旧混合语料 **0.43 nats**（4.679 → 4.2459）
- 生成样本：故事语感已锁定（无百科残留），但逻辑连贯性差、不会在结尾吐 EOT 停止，会连续串多个故事——当时归因于数据墙；放量 2.5 倍后（见上节）此痼疾仍在，非单纯数据量问题
- 下一步真瓶颈：扩 story_topics.py 的 CHARACTERS/SITUATIONS 放量生成（实际旧池 2,800 主题已 100% 用完，每主题 ~6 篇；旧记录"2,400 池已用 59%"有误）

**当前停在正式训练之前，等用户确认再开跑**（已确认方案：先 5000 步验货，
不满意从零重训 10000 步；cosine 调度按总步数衰减，不做中途续训）。

## 语料放量（已完成，2026-10-20）

story_topics.py 池子扩充：CHARACTERS 52→106、SITUATIONS 49→106（共 11,130 组合），
用 `--exclude` 剔除旧 corpus 已用的 2,800 个主题，余 **8,224 个新主题**，蒸馏产出
24,655 篇（保留率 99.9%），与旧的 16,778 篇合并后共 41,433 篇。合并前曾因用户关机
中断于 9,970 篇，按行数断点续跑无损恢复。

打包（`prepare_zh`，清晰洗后）：
- 语料 **40,995 篇 / 1,010 万字**（dropped 438：leak 263、latin-run 17、repeat 158）
- **train 5,208,396 token / val 26,172 token**（旧版仅 208 万，**2.5 倍**）
- 压缩率 1.73 汉字/token（与旧版 1.75 持平，512 上下文装 200 汉字仍宽裕）

下一步：正式训练（训练时长约为旧版的 2.5 倍，先 5000 步验货）。若还想再放量，
story_topics.py 里角色/情境池还能继续加，`--exclude` 机制保证不重复生成。

## 模型配置（`--target-core 3000000` 自动解得 ffn=265）

```
vocab 16384 | d_model 192 | layers 8 | heads 4 | ple_dim 128 | seq_len 512
core   2,995,520
head   3,145,728   (tied, 16384 x 192)
table 16,777,216   (16384 x 8 x 128, 留 flash 稀疏读)
total 22,918,464
```

flash 占用估算 ~11.8MB / 分区 14.87MB，可行。
PSRAM 是硬约束：int8 staged(core+head) ≈ 6.15MB + KV cache 必须 < 8MB，
因此 **端侧 KV cache 必须改 fp16**（fp32 会到 3.15MB，直接爆）。

## 新增/改动的文件

- `research/zh/synth.py` — 蒸馏合成（batch 并行、5 种 prompt 变体、按句截断到 200 字、切掉模型自我重复）
- `research/zh/topics_from_wiki.py` — jieba 主题挖掘（先高频 cut，再对候选词判词性；直接 posseg 跑不动千万字级）
- `research/zh/prepare_zh.py` — 中文语料打包 + 自训 BPE + 汉字/token 压缩率实测
- `research/tinystories/train.py` — 新增 `--corpus zh`（读 `data/zh/`）与 `--amp`（bf16，显存减半）
- `research/tinystories/export.py` — 新增 `--out`（导出到独立目录，不覆盖原英文产物）
- `pyproject.toml` — torch 指向 cu130 index；新增 jieba
- `.gitignore` — 新增 `/models/`

## 恢复命令

```bash
# 1. 继续故事语料放量（如需再扩池：先加 story_topics.py 的 CHARACTERS/SITUATIONS，
#    再生成剔除已用主题的新池，然后 below 命令；jsonl 按行数断点续跑，参数须一致）
uv run python -m research.zh.synth --stage write --mode story \
  --topics data/zh/raw/story_topics_v2.json \
  --out data/zh/raw/synth_story_v2.jsonl \
  --batch-size 192 --variants 3 --max-new-tokens 240

# 2. 打包语料 + 训 BPE（纯故事；首次重跑需先删掉 data/zh/vocab-16384/ 下三个旧产物，
#    否则 corpus.txt / tokenizer.json 会被 "already have" 短路，用的还是百科混合旧料）
#    rm data/zh/raw/corpus.txt data/zh/vocab-16384/{tokenizer.json,train.bin,val.bin}
uv run python -m research.zh.prepare_zh --vocab 16384

# 3. 训练（5000 步约 8 分钟；务必加 --save-every 250 保住最优点权重）
uv run python -m research.tinystories.train --arm ple --corpus zh --vocab 16384 \
  --d-model 192 --n-layers 8 --n-heads 4 --ple-dim 128 --seq-len 512 \
  --target-core 3000000 --batch-size 16 --steps 5000 --tag zh-story3 \
  --save-every 250 --amp

# 4. 导出（--out 隔离中文产物）
uv run python -m research.tinystories.export zh-s0 --tokenizer data/zh/tokenizer.json \
  --out artifacts/zh
```

## 待办

1. ~~合成语料放量~~ 已完成（2026-10-20）：41,433 篇合并，40,995 篇入 train，523 万 token（208 万的 2.5 倍）
2. 训 BPE 时注意压缩率：纯维基实测 **1.14 汉字/token**；混入合成语料后应回升，
   若仍 < 1.2，200 汉字段落会撑爆 512 上下文，需要加大 vocab 或 seq_len
3. 正式训练（core 3M，先 5000 步，用户确认后开跑）
4. **端侧固件适配**（尚未动手）：
   - `firmware/esp32_tinystories/esp32_tinystories.ino:38` `PROMPT_IDS` 换成中文 prompt
   - `ino:39` `N_GENERATE = 200` 是 **token 数不是字数**，按 1.14 汉字/token 需 ~230
   - `ino:149-150` KV cache 改 fp16（否则 PSRAM 爆）
   - `display.h` 只有 6x8 ASCII 点阵，**汉字只能走串口**，OLED 需设 `USE_DISPLAY 0`
   - 若要端侧中文输入：`runtime/bpe_tokenizer.h` 目前 ASCII-only（`>= 0x80` 直接拒绝）
5. `scripts/fetch_model.sh` 的 PINNED sha256 是官方英文模型的；本地训练产物不走这条路径

## 端侧部署（2026-11-08）

story3（ple-zh-story3-s0）已走完整条片上部署管线，等板子即刷：

- 导出：`artifacts/zh/model.bin` 11.87MB / 85 张量（int4 组 128，tied head）
- gate：verify.c PASS（max|diff| 1e-5）；staging_verify.c PASS 0 failures；
  ppl fp32 4.0644 / int8 4.0659（int8 激活代价 +0.0015 nats）
- vocab.h：中文 16384 token / 126KB；prompt「一天，」= {484, 259}
- 固件：PROMPT_IDS 中文、USE_DISPLAY 0、N_GENERATE 230、DEV_SEQ 256、
  STAGE_HEAD_INT8 0；llm.h 新增 Scratch.kv_len（0=全长，host 行为不变）
- **PSRAM 账**（推翻旧"KV fp16"方案）：core+head 全驻留 6.15MB 必爆 8MB，
  最终 core 3.0 驻留 + head 留 flash int4 直读 + KV fp32@256 3.15 + logits 0.06
  ≈ 6.2MB，余 1.6MB
- 修复：export.py / generate_vocab.py 的 global 声明 SyntaxError；
  model.py generate() 补 temperature<=0 贪婪分支；verify.c / ppl.c 补 kv_len 初始化
- 工具链：uv 坏（基础解释器被升级+Dev Drive 信任问题），改用
  .venv python 直调 + PYTHONPATH=src;site-packages；host gate 用
  d:/CodingTrash/tools/w64devkit 的 gcc 16.2
- 部署步骤见 `artifacts/zh/DEPLOY.md`；PC 验收（temperature 0 贪婪，与板端一致）
  故事成篇 ~200 字，天花板仍是语料（他/她混用、指代漂移）

## story4 重训（2026-11-08 晚）

用户拍板 A→B 顺序：先交付 story4 产物，重复惩罚作为后续固件增强。

- 合并语料：synth_story.jsonl(41,433 篇) + synth_story_q35.jsonl(18,288 篇新蒸)
  = 59,721 篇 / 45.4MB，字节级拼接零坏行，备份 synth_story.jsonl.story4-base
- prepare_zh 重建：train 7,205,420 token（旧料 1.38 倍），压缩率 1.70 字/token
  （旧 1.14），新 tokenizer 指纹已随 checkpoint 记录
- 训练：5000 步 491 秒（10.2 步/s），final val 3.0088 / ppl 20.26（story3: 3.94/53.7）
- 导出：artifacts/zh/model.bin 11.87MB；三 gate 全绿（verify 1e-5、staging 0 失败、
  ppl fp32 3.0897 / int8 3.0904，量化代价 +0.0007 nats，int4 代价 ~0.08）
- 验收：主人公漂移已解决（小明→爸爸全程稳定）；贪婪解码有短语复读
  （"闪电划破长空"×3），温度 0.8 下无复读且成篇（估 7.5/10）
- 新 prompt ids：「一天，」= {524, 258}（story4 tokenizer），ino 已同步
- 待办 B：板端+sample.py 同步加重复惩罚（纯解码期，不碰训练/权重）

## B 阶段：重复惩罚上线（2026-11-08 晚）

- 固件：REPETITION_PENALTY_PCT=130（整数百分比，规避预处理器浮点比较），
  16384 位位图去重（2KB 静态 SRAM），argmax 前对已见 token 施 CTRL 惩罚，
  每步惩罚不累积（logits 每步重算）
- PC 对齐：model.py generate() 加 repetition_penalty 参数（逐 unique token，
  >0 除否则乘），sample.py 加 --repetition-penalty 透传
- 效果（story4 + 惩罚 1.30 + 贪婪）："闪电划破长空"×3 → ×1，故事完整成篇，
  自然收尾"终于松了一口气"。该输出即板端将逐字输出的内容
- 纯解码期改动：不碰训练/权重/golden，story4 产物与三 gate 结果不受影响
