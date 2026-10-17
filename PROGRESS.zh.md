# 中文模型训练进度

目标：在 ESP32-S3 N16R8 上端侧生成一个 **150~200 汉字**、语句流畅、语义连贯的
中文语言模型。这是原仓库（英文 TinyStories，28.9M 存参 / 559K core）的中文改造。

## 方向（已确认）

- 语料：**大模型蒸馏合成 + 公开中文语料**混合
- 取向：**质量优先**，core 从 559K 扩到 ~3M，代价是推理从 ~10 tok/s 降到 ~5 tok/s
- 词表：自训 ByteLevel BPE，**vocab 16384**

## 已完成

| 项 | 状态 |
|---|---|
| CUDA 环境 | torch 2.14.1+cu130（`pyproject.toml` 把 torch 指向 cu130 源；默认 Windows wheel 是 CPU 版） |
| 教师模型 | Qwen2.5-1.5B-Instruct，3.09GB，缓存在 `models/`（已 gitignore） |
| 公开语料 | 中文维基 245,572 段 / **6000 万字符** → `data/zh/raw/wiki_corpus.txt` |
| 主题池 | **12,000** 个中文主题词（jieba 从维基挖掘）→ `data/zh/raw/topics.jsonl` |
| 合成语料 | **1,337 段** → `data/zh/raw/synth.jsonl`（脚本断点续跑，中断不丢） |
| 训练链路 | 冒烟 200 步通过：**8 步/秒**，val ppl 16923 → 1624 |
| 吞吐实测 | 合成生成 **4.6 段/秒**（batch 192）；训练不是瓶颈，数据生成才是 |

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
# 1. 继续生成合成语料（143,237 段全跑完约 8.7 小时；可随时中断，下次续跑）
#    建议跑到 2~3 万段就先停，训一版看效果，不够再加料
uv run python -m research.zh.synth --stage write --batch-size 192

# 2. 打包语料 + 训 BPE
uv run python -m research.zh.prepare_zh --vocab 16384 --wiki-chars 40000000

# 3. 训练（2 万步约 42 分钟）
uv run python -m research.tinystories.train --arm ple --corpus zh --vocab 16384 \
  --d-model 192 --n-layers 8 --n-heads 4 --ple-dim 128 --seq-len 512 \
  --target-core 3000000 --batch-size 16 --steps 20000 --tag zh --amp

# 4. 导出（--out 隔离中文产物）
uv run python -m research.tinystories.export zh-s0 --tokenizer data/zh/tokenizer.json \
  --out artifacts/zh
```

## 待办

1. 合成语料放量（当前 1,337 段，目标 2~5 万段）
2. 训 BPE 时注意压缩率：纯维基实测 **1.14 汉字/token**；混入合成语料后应回升，
   若仍 < 1.2，200 汉字段落会撑爆 512 上下文，需要加大 vocab 或 seq_len
3. 正式训练（core 3M）
4. **端侧固件适配**（尚未动手）：
   - `firmware/esp32_tinystories/esp32_tinystories.ino:38` `PROMPT_IDS` 换成中文 prompt
   - `ino:39` `N_GENERATE = 200` 是 **token 数不是字数**，按 1.14 汉字/token 需 ~230
   - `ino:149-150` KV cache 改 fp16（否则 PSRAM 爆）
   - `display.h` 只有 6x8 ASCII 点阵，**汉字只能走串口**，OLED 需设 `USE_DISPLAY 0`
   - 若要端侧中文输入：`runtime/bpe_tokenizer.h` 目前 ASCII-only（`>= 0x80` 直接拒绝）
5. `scripts/fetch_model.sh` 的 PINNED sha256 是官方英文模型的；本地训练产物不走这条路径
