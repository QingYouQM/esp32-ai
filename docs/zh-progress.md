# 中文配对语料：进度快照

**时间：2026-10-31（A、B 都已补做并排除，定位改为 C：短故事生成器，见文末）**

## 一句话结论

“给定材料写故事”这个目标**没有达成**，已经放弃，改卖短故事生成器（C）。模型能写通顺的中文短故事，但**不会复制材料里的人名**——连语料里出现过的名字都会自己另编一个。根因是 300 万参数的 core 学不会跨上下文的字符串复制，不是数据量、不是词表、不是数字覆盖。

## 已完成

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

## 恢复命令

```bash
# 语料：直出（有姓名/年份）
uv run python -m research.tinystories.make_pairs_direct \
    --count 20000 --workers 6 --out research/tinystories/seeds/zh/pairs2.txt

# 审查
uv run python -m research.tinystories.audit_pairs research/tinystories/seeds/zh/pairs2.txt

# 切分（记得用新 tag）
uv run python -m research.tinystories.prepare --tag zh-p2 --vocab 16384 \
    --corpus research/tinystories/seeds/zh/pairs2.txt --seq-len 512 \
    --val-fraction 0.02 --markers '<|material|>,<|story|>'

# 训练
uv run python -m research.tinystories.train --arm ple --target-core 3000000 \
    --steps 2500 --batch-size 16 --seq-len 512 --vocab 16384 \
    --data-dir data/zh-p2/vocab-16384 --ple-dim 128 --d-model 128 \
    --n-layers 8 --n-heads 4 --tag zh-p2 --dropout 0.1 \
    --keep-best --mask-material

# 出厂验收（C 的五个指标）——发货前必须跑这个
uv run python -m research.kb.accept --run runs/ple-zh-ground-s0-best.pt \
    --tokenizer data/zh-ground/vocab-16384/tokenizer.json

# 诊断：模型到底读没读材料（长期 0%，不是门槛）
uv run python -m research.kb.coverage --run runs/<ckpt>.pt \
    --tokenizer data/<tag>/vocab-16384/tokenizer.json
```

## 还没做的（C 之后）

- 设备端故事页加"虚构故事，非史实"标注 —— 唯一挡不住的风险
- `runtime/` 里没有 `accept.py` 的任何等价物，板子上跑起来没人量 stop 率和字数
- 固件仍按六字段组装 prompt，字段文案还是"人物/时间/地点/起因/经过/结果"，没有按三档约束力区分

## 环境备注

- GPU 8GB，batch 64 × seq 512 × vocab 16384 会 OOM（logits 2.1GB），用 batch 16
- GitHub 直连不稳，推送经常失败，需代理
- 分支 `zh-tinystories` 领先 `mine/zh-tinystories`，重启后先 `git push`
