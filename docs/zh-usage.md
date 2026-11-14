# 中文故事机使用说明

板子：小智 AI ESP32-S3-N16R8 + XZ-AI V1.7 扩展板 + 1.54" ST7789。
模型：中文 story4，22.9M 参数，一次写一篇约 170 字的短故事。

## 快速上手（30 秒）

1. 上电（USB 或电池），屏幕亮起、串口刷完初始化，约 3 秒
2. 手机 WiFi 列表选 **`PLE-TinyLM`**，密码 **`tinystory`**
3. 浏览器打开 **`http://192.168.4.1/`**
4. 页面会自动生成一篇；点「再来一篇」再要一篇

手机连上后提示"无互联网"是正常的——板子只做热点，不联网。选保持连接即可。

## 你会看到什么

| 位置 | 内容 |
|---|---|
| 屏幕 | 15 字 × 15 行，逐字刷出，写完停在最后一句不覆盖 |
| 网页 | 同一篇文本，边写边出；结束后显示 tok/s 和 ms/token |
| 串口（115200，OTG 口） | 完整启动日志、`--- N tokens in X s ---`、profile |

生成结束的标志是模型自己吐出结束符，不是写满。一次通常 **102 token / 约 35 秒 / 173 汉字 / 2.8 tok/s**。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 提示"无互联网" | 正常。板子不开路由，只提供页面 |
| 页面打不开 | 确认连的是 `PLE-TinyLM` 而不是家里 WiFi；串口应有 `wifi : AP "PLE-TinyLM"` |
| 点按钮提示"板子正忙" | 上一篇还在写，等 35 秒左右 |
| 找不到热点 | 板子刚上电需要 3 秒；或断电重上一次 |
| 屏幕全黑但网页正常 | 背光/排线，检查扩展板 8 针座是否插到底 |
| 网页有字、屏幕有字但缺笔画 | 语料外的字不在 `cjk_font.h` 字库里，重跑 `gen_cjk_font.py` |

## 开发者

- **关掉 WiFi**（回到上电自动生成）：`firmware/esp32_tinystories/esp32_tinystories.ino:43`
  的 `#define ENABLE_WIFI 0`，重新编译
- **改热点名/密码**：同文件 `AP_SSID` / `AP_PASS`（密码至少 8 位，否则 softAP 起不来）
- **改网页**：`firmware/esp32_tinystories/web_page.h`。不要放回 `.ino`——
  Arduino 预处理器会扫描 raw string，把 JS 的 `async function` 当成函数原型
- **只换固件不动模型**：`arduino-cli upload` 即可，`model` 分区（0x2D0000）不受影响，
  生成结果逐字不变
- **完整重刷**：见 `docs/zh-deploy.md`

编译命令（Windows，esp32 core 3.3.12）：

```bash
arduino-cli compile \
  --fqbn 'esp32:esp32:esp32s3:UploadSpeed=921600,USBMode=hwcdc,CDCOnBoot=cdc,UploadMode=default,CPUFreq=240,FlashMode=qio,FlashSize=16M,PartitionScheme=custom,PSRAM=opi,DebugLevel=info' \
  --build-property 'compiler.optimization_flags=-O3' \
  --build-property 'upload.maximum_size=2883584' \
  firmware/esp32_tinystories
```

## 参数速查

| 项 | 值 |
|---|---|
| 热点 / 密码 | `PLE-TinyLM` / `tinystory` |
| 页面地址 | `http://192.168.4.1/` |
| 模型 | story4，core 2,995,520 + table 16,777,216，共 22.9M |
| 分区 | app 2.75MB / model 13.125MB @0x2D0000 |
| 固件 | 1.39MB，RAM 67.8KB，PSRAM 6.03MB |
| 速度 | 2.84 tok/s（340~350 ms/token） |
| 屏幕 | ST7789 240×240，16×16 汉字，15×15 格 |
