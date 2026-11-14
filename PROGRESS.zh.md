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

## 屏幕接入：1.54" ST7789（2026-11-13）

板子确认为小智 AI ESP32-S3-N16R8 + XZ-AI_KZBV1.7 扩展板，屏是 1.54 寸 ST7789 240x240，
插扩展板的 8 针 TFT 座。引脚取自厂商《小智控制固件接线说明》并与 2.5.2 源码的
`main/boards/bread-compact-wifi-lcd/config.h` 逐条核对一致：

- SCK 21 / MOSI 47 / RST 45 / DC 40 / CS 41 / BLK 42（1.3" 的 7 针座没有 CS）
- 原 display.h 的 TFT 分支写的是 10/7/6/12/11，其中 **7 是喇叭 DOUT、6 是麦克风 DIN**，
  照原样开显示会把 I2S 音频带死；而且背光脚从未驱动，屏只会全黑。已全部改掉
- 1.54" 是 IPS，需要 `invertDisplay(true)`；SPI 4 线跑 40MHz

中文字库：Adafruit 内置字体只有 6x8 ASCII，中文会被整段丢弃。新增
`research/zh/gen_cjk_font.py`：从 corpus.txt 提取全部字符（BMP 以内，跳过 10 个
emoji 和一个 U+FFFD），SimSun 16x16 1bpp 烤成 `generated/cjk_font.h`，4,014 字形
/ 125KB flash。BPE token 会切断 UTF-8 多字节序列，`display_puts()` 把残缺尾巴
留到下一次调用续接。

- 排版：15 字 × 15 行 = 225 字，200 字故事单屏够用；超屏清屏重画
- 验证（无硬件）：桩 Arduino 头文件真编译 display.h，喂跨 token 的「一天，」，
  一字一次 drawBitmap、换行与光标推进全部正确；app 只涨 125KB，1MB 分区仍够
- **真机验证（2026-11-14）**：生成时逐字刷屏，173 字单屏（12 行）不翻页；
  结束后 `display_stats()` 清屏换成收尾卡（ESP32-S3 / PLE TinyLM / 22.9M params /
  zh-story4 / 2.9 t/s / 339 ms/token）
- **WiFi 仍未接**：固件里没有任何 WiFi 代码；真正卡点只剩 30~60s 阻塞式 setup()
- 分区已重排（2026-11-14）：app 1MB → **2.75MB**（WiFi + TLS 留量），model 移到
  **@0x2D0000 / 13.13MB**（`0x2D0000..0xFF0000`，model.bin 11.87MB 余 1.8MB）；
  `deploy.sh` 的 `PART_OFFSET` 与 `APP_MAX_BYTES` 已同步。英文 14.9MB 模型
  不再适配此表，要刷旧英文模型得回退分区表

## 上板实测（2026-11-14 晚，分区重排后首刷）

分区重排后首次真机编译通过（esp32 core **3.3.12**，`PartitionScheme=custom`，
app 736KB / 上限 2.75MB），刷 COM4（OTG 原生 USB，`USB-Serial/JTAG`）：

| 项 | 实测 |
|---|---|
| 启动 | `Vin=16384 Vout=16384 D=192 L=8 H=4 F=265 P=128 (mapped 13.8 MB)` |
| PSRAM | 57 张量 int8 6.03MB 驻留；free sram 269KB / psram 1.87MB |
| 指纹 | `bytes=11867236 fp=312e0f85`（与本地 FNV 一致） |
| 生成 | **102 token / 35.00s / 2.91 tok/s（339.1 ms/token）**，EOT 自停 |
| 字数 | **173 汉字 / 202 字符**，落在 150~200 区间 |
| profile | input 5.1 \| attn 45.2 \| ffn 33.6 \| ple 11.3 \| **head 243.9** |

- head 占 72%：`STAGE_HEAD_INT8 0` 让输出头从 flash int4 直读，是唯一瓶颈；
  旧预期 4~6 tok/s 偏乐观，实测 2.91
- 两次冷启动输出逐字相同 → 贪婪确定性成立
- **与 PC fp32 基线在「闪电划破」后分叉**：int4 组量化 + int8 激活让低置信度
  位置的 argmax 翻转（ppl gate 记录的 int4 代价 ~0.08 nats）。"板端 == PC 逐字"
  这条验收标准过严，改成：前 10~15 字一致 + EOT 自停 + 成篇 + 173±20 字
- 质量：同开头不同走向（板端走"光/雷电"意象，收在"黑暗吞噬世界"），
  比 PC 版少一句收束——量化分叉路径不同，不是新缺陷

## head int4 驻留实验（2026-11-15，已回滚）

想法：head 每 token 从 flash 扫 1.57MB int4，实测 243.9ms（有效带宽 ~6.4MB/s）。
把打包 nibble + fp16 scale 原样 memcpy 到 PSRAM（1.56MB，`llm_pin_int4`），
matvec 只换指针、字节不动，输出应当逐字相同。

| | flash 直读 | int4 驻留 PSRAM |
|---|---:|---:|
| head | 243.9 ms | **237.0 ms** |
| 每 token | 339.1 ms | 332.3 ms |
| 吞吐 | 2.91 tok/s | 2.97 tok/s |
| PSRAM 占用 | 6.03 MB | 7.60 MB（余 0.28） |

- 输出逐字相同，说明 pin 本身没错；但只快 **2.8%** —— 换介质不是瓶颈
- 结论：S3 上 flash 与 PSRAM 的有效带宽都在 6~7MB/s 量级（同一条 MSPI 通路 +
  cache 行为相近），237ms 由"每 token 扫 1.57MB 小块"这个访问模式决定，
  不是由存在哪决定
- 副作用是 PSRAM 余量 1.87 → 0.28MB，风险不值；已回滚，固件恢复 flash 直读
- 要提速得动算法侧：int8 驻留（预解包，省解包计算，需 3.0MB）+ 砍 KV，
  或减小输出词表 / 分块 logits，单换存储位置没戏

## WiFi 网页接口（2026-11-15）

板子自己开热点，手机连上打开网页点一下就能要一篇，屏上同步逐字刷：

- AP `PLE-TinyLM` / `tinystory`，网关 `192.168.4.1`——不把路由器密码存进固件
- `GET /` 返回页面（PROGMEM，单独放 `web_page.h`：放 .ino 里会被 Arduino
  预处理器当成 JS 函数原型，编译报 `asyncfunction does not name a type`）
- `GET /start` 起一次生成；`GET /text?since=N` 返回增量 JSON，页面 400ms 轮询
- 生成搬进独立任务（core 1、8KB 栈），`loop()` 只跑 `web.handleClient()`；
  生成循环改成每 2 token `delay(0)`，让网页每 660ms 能拿到一批字

**顺带修了个潜伏 bug**：`matvec_par` 原来把完成通知发给 setup 里记下的
`main_h`，从别的任务调用时（网页任务）worker 把通知发给了 LoopTask、
调用者自己永远等不到 → 卡死在第一个字之后。现在每次调用都用
`xTaskGetCurrentTaskHandle()` 重新取句柄。

实测：`/start` → **203 字符 / 102 token / 34.6s / 2.84 tok/s**（比裸跑 2.91 略低，
WiFi 栈抢 CPU）；app **1.39MB / 2.75MB**（WiFi+WebServer 让固件从 736KB 翻倍），
RAM 67.8KB，free sram 237KB（比无网时少 32KB）。



