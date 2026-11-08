# 中文模型烧录部署（esp32_tinystories + artifacts/zh）

状态：story4 模型与解码表已就绪并通过全部 host gate（2026-11-08），等板子到手按本文执行。

## 产物清单

| 文件 | 用途 |
|---|---|
| `model.bin` | int4 权重，烧进 flash `model` 分区（0x110000），11.87MB / 分区 14.87MB |
| `tokenizer.json` | 导出时校验过的自训 BPE 16384（与 story3 训练指纹一致） |
| `golden.txt` / `golden.npz` | PyTorch 4-bit 模型 last-pos logits 参考 |
| `verify.exe` / `staging_verify.exe` / `ppl_int8.exe` / `ppl_fp32.exe` | host gate 工具（w64devkit gcc 16.2 编译） |

`firmware/esp32_tinystories/generated/vocab.h` 已是中文版（16384 token，126KB blob）。
该目录被 .gitignore 排除，新环境需重新生成：
```
python firmware/esp32_tinystories/tools/generate_vocab.py --tokenizer artifacts/zh/tokenizer.json
```

## 已通过的验收记录（story4，新语料 59,721 篇 / 7.2M token 重训）

- verify.c：C int4 路径 vs PyTorch golden，max|diff|=1e-5，top-1 一致 → PASS
- staging_verify.c：int8 驻留 == int4、钩子分发 57/57、头部校验全绿 → PASS (0 failures)
- ppl（`data/zh/vocab-16384/val.bin`）：fp32 CE 3.0897 / int8 CE 3.0904，量化代价 +0.0007 nats
  （训练 fp32 val 3.0088 / ppl 20.26，int4 代价 ~0.08）
- PC 贪婪解码 + 重复惩罚 1.30（temperature 0，与板端 argmax+penalty 一致）：
  复读消失，故事完整成篇（打雷→害怕→安慰→松口气，~200 字），主人公稳定

## 已做的中文适配

- `PROMPT_IDS = {524, 258}`（「一天，」，story4 新 tokenizer；旧 story3 的 {484,259} 已作废）
- `USE_DISPLAY 0`（display.h 只有 ASCII 点阵，汉字走串口）
- `N_GENERATE = 230`（新语料压缩率 1.70 字/token，200 字 ≈ 118 token，230 是兜底）
- `DEV_SEQ 256`：KV cache 按 256 分配（fp32 下 3.15MB，否则 PSRAM 爆）
- `STAGE_HEAD_INT8 0`：head 不驻留 int8（省 3.15MB，走 flash int4 直读）
- `REPETITION_PENALTY_PCT 130`：CTRL 式重复惩罚（位图去重，argmax 前施压），
  与 `sample.py --repetition-penalty 1.30` 公式逐位一致，保证验收对齐；100=关闭
- EOT(id 0) 停止判断已存在

PSRAM 终账：core int8 3.0MB + KV 3.15MB + logits 0.06MB ≈ 6.2MB / 8MB，余 ~1.6MB。

## 烧录步骤（板子到手后）

需要：Arduino CLI + esp32 core（或 Arduino IDE），esptool（Arduino 工具链自带）。

1. **编译固件**（先编译后刷模型，顺序不能反——部署脚本的设计原则）：
   - Arduino CLI 示例：
     ```
     arduino-cli compile --fqbn esp32:esp32:esp32s3:FlashSize=16M,PSRAM=opi,\
     PartitionScheme=no_extra_ota ... firmware/esp32_tinystories
     ```
   - 注意：必须选用 `firmware/esp32_tinystories/partitions.csv` 的自定义分区
     （app 1MB / model 14.87MB @0x110000 / coredump）。
2. **烧模型分区**：
   ```
   esptool --chip esp32s3 --port <COM口> --baud 921600 write_flash 0x110000 artifacts/zh/model.bin
   ```
3. **烧固件**：
   ```
   arduino-cli upload --fqbn <同编译> -p <COM口> firmware/esp32_tinystories
   ```
4. **串口验收**（115200）：
   - 开机应打印 `model: Vin=16384 Vout=16384 D=192 L=8 H=4 F=265 P=128 (mapped 15.6 MB)`
   - `build: bytes=11867236 fp=<指纹> sram=... psram=6.2xMB`
   - 无 `FATAL:` 行；出现即内存分配失败，把 DEV_SEQ 降到 192 重编
   - `>>> ` 后应直接流出中文故事，约 4-6 tok/s

## 已知边界

- 模型质量：新语料 + 重复惩罚后，主人公漂移与贪婪复读均已解决；
  剩余瑕疵是偶发的语义小跳跃（3M 参数天花板），升级需换更强教师或加参数
- 端侧中文输入未做（`runtime/bpe_tokenizer.h` 仍 ASCII-only）；prompt 烘焙在固件里
- greedy 生成确定性输出，同样 prompt 永远同一篇故事（验收基线即 PC 上的输出）
