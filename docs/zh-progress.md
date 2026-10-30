# 中文配对语料：进度快照

**时间：2026-10-30 15:10（重启前保存）**

## 一句话结论

“给定材料写故事”这个目标**没有达成**。模型能写通顺的中文短故事，但**不会复制材料里的人名**——连语料里出现过的名字都会自己另编一个。根因是 300 万参数的 core 学不会跨上下文的字符串复制，不是数据量、不是词表、不是数字覆盖。

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

## 下一步（三选一，未定）

**A. 约束解码（建议先做，约 40 分钟，不训练）**
把材料里的名字做成解码期硬约束，实体覆盖率可从 0% 到 100%，机械保证。事件仍不可控。

**B. core 加到 10M**
赌复制能力会在某个规模涌现。训练 30 分钟，设备速度掉到 1–2 tok/s，成败未知。

**C. 改定位**
承认是短故事生成器，材料只作题材提示，不承诺忠实。

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

# 衡量“模型到底读没读材料”
uv run python -m research.kb.coverage --run runs/<ckpt>.pt \
    --tokenizer data/<tag>/vocab-16384/tokenizer.json
```

## 环境备注

- GPU 8GB，batch 64 × seq 512 × vocab 16384 会 OOM（logits 2.1GB），用 batch 16
- GitHub 直连不稳，推送经常失败，需代理
- 分支 `zh-tinystories` 领先 `mine/zh-tinystories`，重启后先 `git push`
