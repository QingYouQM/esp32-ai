# 中文配对语料：进度快照

**当前目标（2026-10-31 用户重新确认）**：生成每篇 150–200 字、语言通顺、忠于输入史料且不违背历史事实的中文故事。C“虚构短故事”定位已撤销，资料卡恢复为事实边界。

## 当前结论

旧模型仍不达标，但不再修改目标来迁就模型。第一阶段已重建交付契约、24 张卡的关键事实锚点和逐篇自动验收，并完成自由生成与事实约束生成基线。自动验收仍不是语义史实证明，发布还需要来源审核和人工逐篇复核。

**2026-11-01 重大修正：此前的 grounding 测量被采样器污染。** model.generate 的 no_repeat_ngram=4 和 repetition_penalty 把 prompt（材料段）也算进重复检测范围，逐字引用材料的任何 4-gram 在第 4 个 token 处被强制 -inf——即验收要求的关键事实锚点在采样器层面被禁止生成。本文 A、B 实验的“自由 entity 0%”“event 0%”数字全部受此影响；约束组能到 100% 恰恰因为强制 logits_processor 最后应用、覆盖了禁令。修复：generate(repeat_scope="generated") 让反重复只看生成部分（src/model.py），kb 工具已全部切换，tests/test_model_head.py 有回归测试。

## E1/E2 单点实验（2026-11-01，修复采样器后）

三轮实验回答“vanilla 3M 能否在未见卡片上复制关键片段”。训练配置全部与 zh-ground 相同（3M core + 16.8M table，2500 步，batch 16，seq 512，mask-material，keep-best）。

| 实验 | 语料 | 评测面 | person | 全 span 保留 |
|---|---|---|---:|---:|
| E1a 纯合成（宽名字空间，组合级切分） | 8000 合成 | 100 张未见名字组合卡 | **98%** | **83%** |
| E1b 同上 + 真语料混合 | 8000 合成 + 14651 真 | 同上 | 100% | 21% |
| E1a/E1b | 同上 | 24 张真卡 | 0% | 0% |
| E2a 纯合成（侨乡实业表面统计：阿拉伯年份、机构名、枚举式经过） | 8000 合成 | 未见组合卡（贪心 20 张） | **20/20** | 5/20 |
| E2a/E2b | 同上 | 24 张真卡 | 0% | 0% |

判读：

- **在训练分布内，复制机制可以学会**：E1a 未见名字组合 98%/83%，是“条件化复制存在”的直接证据。此前“3M 学不会复制”的结论部分是采样器假象。
- **分布外（真卡）两轮都失败**。E2 把表面统计调成侨乡实业风格后，模型在真卡上仍退回合成模板；E2b 的“陈嘉成/陈嘉涌”显示模型抄到“陈嘉”后第三个字被先验接管。
- **数字槽是独立的薄弱环节**：E2a 在合成卡上把“五百口棺柩”写成“四十口”、“五个工人”写成“八十个”——24 个场景共享数字槽，模型学会填合理数字而非复制。
- **停止行为在纯合成模型上退化**：E2a 贪心 0/20 自然停止，写完故事后继续生成下一篇材料。

结论：vanilla 3M 的条件化复制是**模板内泛化**，跨不到真实史料分布。两轮语料干预（宽名字空间、表面统计匹配）均未通过真卡门。按既定路线，下一步是主机端显式 span-copy 机制实验（pointer/copy 头或等价监督），不再做第三轮纯语料调参。

## 新标准已做

- `docs/zh-story-prompt.md`：恢复历史故事契约，每篇 150–200 字，不再允许“只当题材种子”。
- 24 张真卡新增 `acceptance.required_spans`；`validate.py` 拒绝不在六字段材料中的锚点。
- `accept.py`：逐篇检查 EOD、字数、人名、事实锚点、额外数字、复读和重复；记录首次与重试后通过率，失败返回 1。
- 人名正则只产出人工复核警告，避免把“陆续”“查看”等普通词硬判成人名。
- `constrain.py`：从姓名单片约束改为有序事实片段约束，可阻止事实未写完时提前 EOD。
- 相关知识库测试 20 项通过；最终完整 Python 套件 281 项通过；24 条资料包字节往返校验通过。

## 新验收实测：ground-3M

命令：

```bash
uv run python -m research.kb.accept \
  --run runs/ple-zh-ground-s0-best.pt \
  --tokenizer data/zh-ground/vocab-16384/tokenizer.json \
  --attempts 4 --report runs/zh-history-ground-baseline.json
```

| 模式 | 首次通过 | 四次后通过 | 选中故事中位字数 |
|---|---:|---:|---:|
| 自由生成 | 0/24 | **0/24** | 185 |

96 个候选中 95 个自然停止；失败不是“不会收尾”，而是 96/96 缺声明人物、共 490 个事实锚点缺失、36 个长度失败、28 个额外数字。报告在 `runs/zh-history-ground-baseline.json`（本地运行产物）。

事实约束冒烟（前两张卡，单次，`--constrain-facts`）：两张分别为 261、258 字，均触及 token 上限；关键短语虽被强制写入，周边叙事仍混乱。**不跑成全量成功，也不作为交付方案。** 报告在 `runs/zh-history-constrained-smoke.json`。

## 下一道门

E1/E2 已回答“vanilla 能否学会复制”：训练分布内可以（98%/83%），真卡分布两轮语料干预均失败。下一道门是**主机端显式 span-copy 机制实验**：在 PyTorch 侧加 pointer/copy 通路（材料段 attention 权重散射成 copy 分布，与 vocab 分布按门控混合），先在 E1a 同语料上验证机制上限，再测真卡迁移。机制未通过前，不修改固件 tensor 格式，不恢复“出厂候选”说法。

注意：此前所有 grounding 数字需以修复后的采样器重测为准（见上方 2026-11-01 修正）。

---

## 旧实验归档（以下 C 及此前结论不代表当前交付契约）

### 当时已完成

| 项 | 状态 |
|---|---|
| 材料格式守则 | `docs/zh-material-spec.md`（六字段、M1–M9、H1–H3） |
| 资料集工具 | `research/kb/`：schema / render / validate / pack / coverage |
| 华侨资料集 | 7 人 24 条，校验通过，打包 9.3 KiB |
| 配对语料（旧） | `seeds/zh/pairs.txt`，13094 条 / 2.25M token，无姓名无数字 |
| 配对语料（新） | `seeds/zh/pairs2.txt`，1557 条 / 230k token，**有姓名和年份** |
| 语料工具 | `make_pairs`（反推）、`make_pairs_direct`（直出）、`audit_pairs`、`filter_pairs` |
| 训练 | 22M 参数（3M core + 16.8M table），masked loss，keep-best |
| 解码 | 重复惩罚 + 4-gram 封锁（修掉了死循环） |

## 关键数字

| 语料 | val（掩码口径） | 实体覆盖 | 事件覆盖 |
|---|---:|---:|---:|
| 6028 条 / 1.1M token | 5.19 | 0% | 0% |
| 13094 条 / 2.25M token | 3.86 | 0% | 0% |
| 14651 条（旧+新混合） | 4.08 | 0% | 0% |

24 张真资料卡，24 条全部 0%/0%。同分布材料下模型会用“医生、听诊器、石桥”这类实体，但**人名一定自己编**。

## A 已做（2026-10-31）：约束解码

`research/kb/constrain.py`：第一个名字开场，其余名字按预算均分，落在各自的第一个句号之后，槽内逐 token 强制，槽外不动模型。`coverage.py --force-names` 同一张卡跑自由和强制两组，种子相同，只差这一个变量。

| 检查点 | 自由 entity | 强制 entity | 自由 event | 强制 event |
|---|---:|---:|---:|---:|
| ple-zh-13k-s0 | 0% | **100%** | 0% | 0% |
| ple-zh-mix-s0 | 0% | **100%** | 0% | 1% |

24 张卡全部命中，没有一张因预算不足丢掉名字（`--tokens 200`）。

**结论：复制这一层绕过去了，落地这一层没绕过去。**

- 名字塞进去之后，模型**第一句会用**它（"陈嘉庚早来到小工厂要开工"），但当主语，不当人物
- 同一段里继续自造名字：周强、小周、李老师——说明它不是"没记住"，是压根没把材料里的人当成参照
- 事件覆盖仍然是 0–1%，材料里的经过和结果一个字都没进故事
- 自由组也会在某处开始复述材料模板（"人物……时间……地点……"），这是语料串档的旧病，不是 A 引入的

所以 A 买到的是 entity 100%，买不到 grounded。名字现在是贴上去的标签，不是故事里的人。

## 语料 bug（2026-10-31 修）：文档之间没有分界

`prepare.py` 把每篇的 `<|endoftext|>` 全部堆到数组末尾，正文里一个分界都没有。模型因此只学会了"一个故事接下一个故事的材料块"，从没学过停下。

证据（同一份语料、同一个 tokenizer，只差这一行）：

```
zh-mix/train.bin     14,355 个 eod，第一个在第 2,510,317 个 token（共 2,524,672），间隔恒为 1
zh-mixfix/train.bin  14,355 个 eod，第一个在第 160 个 token，中位间隔 173
```

重训 2500 步，其余超参完全相同：

| 检查点 | 停在 EOD | 串成材料块 | 中位故事长度 | val |
|---|---:|---:|---:|---:|
| ple-zh-mix-s0（旧） | 0/24 | 3/24 | 从不停 | 4.08 |
| ple-zh-mixfix-s0（新） | **24/24** | 0/24 | 108 token | 4.12 |

引入点在 `5f58e84`（"parameterise the corpus"）：原来的 `ids.append(eot)` 被改写成 `+ [eot] * len(docs)`。`53e19c7` 的英文结果是旧代码产出的，不受影响；`5f58e84` 之后的中文 run 全部建立在无边界的 bin 上。val 曲线看不出来，所以一直没被发现。

**这不解释 grounding 失败。** 修好之后重测 A：自由 entity 仍 0%，强制 100%，event 0%。

## 合成复制任务（2026-10-31，两轮，失败但定性）

目标：把"抄材料"做成一个 3M 学得会的小任务，不调 API。

**第一轮**：`make_copy_task.py` 用槽位造卡、用同一批槽位造故事，经过的转折句和结果句逐字引用，12 个场景。8000 条混进 14651 条真语料（合成占 38%）。

| 评测面 | entity | event |
|---|---:|---:|
| 200 张留出的合成卡（自由解码） | **100%** | **14%** |
| 24 张真卡 | 0% | 0% |
| 真语料里的金故事（上限） | 100% | 46% |

合成卡上它**确实会抄**了，而且名字偶尔抄错一个字（孔文振→孔文挆）。但真卡上输出直接崩成"南解志"这类自造人名——合成卡命中的是它认识的 12 个模板，真卡一个都不像。

**第二轮**：`ground_stories.py` 把每篇故事的开口换成卡片自己的 `起因` 句（本来就是一个点名带姓的动作句），保住真实行文，只让第一句必须从材料里搬。24 张真卡仍然 **0% / 0%**。

**关键诊断（旧检查点，留出的真语料文档，40 篇）**：

- 自由解码：entity 中位 **0%**，event 中位 9%
- 卡片写"葛明"，模型写"**鲁明**"——它把这篇金故事的情节几乎逐句复现了（"过了三天……桥……学别人的样子"），却把姓换了
- 给 10 字起手句再解码：entity 中位仍是 0%，平均 25%

也就是说：**故事它背下来了，材料里的字符串却绑不进输出。** 不是记不住，不是语料里没有，是条件化通路没建立。合成任务能拿到 100%，靠的是"认出模板 + 填槽"，换成没见过的表面形式就退回故事先验。

这不是容量问题的证明，但也**不是"再加语料就能好"**的问题——两轮都指向同一个地方：需要一个显式的复制监督（span-copy 辅助目标），或者把注意力容量/头数单独拉出来做变量。

## B 已做（2026-10-31）：core 3M → 10M

`--target-core 10000000 --d-model 256 --n-layers 8 --n-heads 8`，解出 ffn 1157，core 9,998,720（+16.8M table）。语料、步数、batch、lr、dropout 与 `ple-zh-ground-s0` 逐项相同（zh-ground / 2500 步 / batch 16 / seq 512 / mask-material），只差容量。9m56s。

| 检查点 | core | val | 自由 ent | 自由 evt | 强制 ent | 强制 evt |
|---|---:|---:|---:|---:|---:|---:|
| ple-zh-ground-s0 | 3.0M | 2.41 | 0% | 0% | 100% | 0% |
| ple-zh-b10-s0 | 10.0M | **2.14** | 0% | 1% | 100% | 3% |

val 掉了 0.27，故事明显更顺，grounding 没动。

- 名字第一次被抄了一半：自由组开始出现"陈嘉"这种两字截断（3M 时是直接换姓，写成"鲁明"），但同一篇里仍旧混出"陈嘉昂"——学到的是"这两个字常在这出现"，不是"把卡片上的串搬过来"
- 强制组仍 24/24 命中 100%，槽外续写照样跑偏："陈嘉庚到了东街，陈嘉昂前进入南上还守夜班"
- event 3% 只由两张卡贡献（创办厦门大学 19%、捐助厦大基金 8%），其余大多为 0；自由组的 19% 那张也和强制组同值，说明是故事先验蒙对的，不是读了材料

**结论：容量翻三倍，买到的是流畅，不是落地。B 排除。**

## C 已采纳（2026-10-31）：改成短故事生成器

A、B 两次否证之后，定位改掉了。契约写在 [docs/zh-story-prompt.md](zh-story-prompt.md)：

- 承诺：120–220 字中文短故事，自己收尾
- 不承诺：复述卡片。卡片是题材种子，不是事实来源
- 验收换成 `research/kb/accept.py` 的五项：stop ≥95%、中位字数 120–220、材料模板复读 0、截断名 ≤4/24、4-gram 重复 ≤5%
- `coverage.py`（entity/event）降级为诊断工具，`constrain.py` 降级为可选风格开关

三项检查点按新标准实测（24 张真卡，`--tokens 200`）：

| 检查点 | core | stop | 中位字数 | 截断名 | 复读 | 出厂 |
|---|---:|---:|---:|---:|---:|---|
| ple-zh-ground-s0 | 3.0M | 100% | 178 | 3/24 | 0/24 | **推荐** |
| ple-zh-mixfix-s0 | 3.0M | 100% | 143 | 0/24 | 0/24 | 备选 |
| ple-zh-b10-s0 | 10.0M | 96% | 191 | 8/24 | 0/24 | 不划算 |

ground 版单一主角、场景连贯；mixfix 版名字干净但多人乱入；10M 多 13 个字、多一倍截断名，设备端还掉到 1–2 tok/s。

**剩下的风险不在代码里**：故事里的人名和事实是编的。设备 UI 必须标"虚构故事，非史实"，否则用户会把陈嘉庚当成在码头搬箱子的人。

## 恢复命令（C 的出厂候选，整条链）

```bash
# 1) 真语料：旧 pairs（无姓名）+ 新 pairs2（有姓名/年份）
cat research/tinystories/seeds/zh/pairs.txt \
    research/tinystories/seeds/zh/pairs2.txt > data/combined.txt

# 2) 第一轮合成复制任务：8000 条模板卡，故事逐字引用材料
uv run python -m research.tinystories.make_copy_task \
    --count 8000 --out data/copy-task/synthetic.txt
cat data/copy-task/synthetic.txt data/combined.txt > data/copy-task/combined.txt

# 3) 第二轮：把每篇故事的开口换成卡片自己的 起因 句（22651 篇 = 8000 合成 + 14651 真）
uv run python -m research.tinystories.ground_stories \
    --in data/copy-task/combined.txt --out data/copy2/combined.txt

# 4) 切分（注意 markers，少了它们 <|material|>/<|story|> 会被切碎）
uv run python -m research.tinystories.prepare --tag zh-ground --vocab 16384 \
    --corpus data/copy2/combined.txt --seq-len 512 \
    --val-fraction 0.02 --markers '<|material|>,<|story|>'

# 5) 训练 3M（出厂候选，约 4 分钟）
uv run python -m research.tinystories.train --arm ple --target-core 3000000 \
    --steps 2500 --batch-size 16 --seq-len 512 --vocab 16384 \
    --data-dir data/zh-ground/vocab-16384 --ple-dim 128 --d-model 128 \
    --n-layers 8 --n-heads 4 --tag zh-ground --dropout 0.1 \
    --keep-best --mask-material

# 6) 出厂验收（C 的五项，发货前必须跑）
uv run python -m research.kb.accept --run runs/ple-zh-ground-s0-best.pt \
    --tokenizer data/zh-ground/vocab-16384/tokenizer.json

# 7) 诊断：读没读材料（长期 0%，不是门槛）
uv run python -m research.kb.coverage --run runs/ple-zh-ground-s0-best.pt \
    --tokenizer data/zh-ground/vocab-16384/tokenizer.json
```

B 的扩容版（已否证，留档）：在第 5 步加 `--target-core 10000000 --d-model 256 --n-heads 8`，
其余不变，tag 用 `zh-b10`，约 10 分钟。

## 还没做的（C 之后）

- 设备端故事页加"虚构故事，非史实"标注 —— 唯一挡不住的风险
- `runtime/` 里没有 `accept.py` 的任何等价物，板子上跑起来没人量 stop 率和字数
- 固件仍按六字段组装 prompt，字段文案还是"人物/时间/地点/起因/经过/结果"，没有按三档约束力区分

## 环境备注

- GPU 8GB，batch 64 × seq 512 × vocab 16384 会 OOM（logits 2.1GB），用 batch 16
- GitHub 直连不稳（`Connection was reset`），走本机 Clash 混合端口：
  `git -c http.proxy=http://127.0.0.1:7897 push mine <branch>`
- `zh-tinystories` 与 `zh-copy-task` 都已推到 `mine` 并设了上游
