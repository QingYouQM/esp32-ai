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

## 板子自带 API（不需要电脑）

固件里已经有 OpenAI 格式的接口，板子自己就是服务端：

| 项 | 值 |
|---|---|
| 地址 | `http://<板子IP>/v1`，模型名 `esp32-zh-story4` |
| 接口 | `/v1/models`、`/v1/chat/completions`、`/v1/completions` |
| 开头 | 取**最后一条 `role:"user"` 的 content**；可写开头，也可直接提问（板端剥壳）；BPE 与 PC `tokenizers` 逐 id 一致 |
| 速度 | 约 35 秒一篇（2.85 tok/s），一次只能跑一个请求，忙时返回 503 |
| 流式 | `stream=true` 可解析，但整篇生成完才一次性发出 |

**接进聊天客户端时**：

- 只认最后一条 user 消息当故事开头，`system` 和历史对话全部忽略。它没有指令跟随
  能力，system 提示对它只是"又一段要续写的文字"，还会挤掉你真正说的那句。
- content 写字符串或 `[{"type":"text","text":"…"}]` 都认；`/v1/completions` 的
  `prompt` 同理（字符串或数组取第一段）。
- **可以直接提问。** 板子会剥掉"请帮我写一个关于…的故事"这类首尾指令壳，把中间的主题当
  开头：`请帮我写一个关于小猫找妈妈的故事` → `小猫找妈妈。`。常见抽象主题（勇敢、友谊、
  下雨天、太空等约 30 个）另有一张场景映射表，`勇敢` → `小明一个人走夜路回家。`。
- 剥完只剩一两个汉字（`小猫`、`勇敢`）就保留主题词加逗号，让模型自己接下去。想完全跳过
  改写，在 URL 后加 `?raw=1`，发什么就原样当开头。
- 开头超过 32 token 只取前 32 个，不报错。`temperature` / `top_p` / `max_tokens`
  收下但忽略：板端固定 top-k 20、温度 0.7 采样，加 1.3 重复惩罚。
- 请求体必须是带 `Content-Length` 的 POST。取不到正文或取不到文字会返回 400 并写明
  原因，不会再偷偷用内置的"一天，"顶替你的输入。
- 一次请求要等 35 秒左右才回第一个字节，客户端的读超时调到 60 秒以上，不然会自己
  掐断连接，看起来就像"发出去没反应"。
- **它不是对话模型。** 发"你好，介绍一下你自己"，得到的是一篇以这句开头的故事。
  要问答只能换模型，改提示词没用。提问只决定它写哪个故事，不会换来一句回答。

**连家里 WiFi**：串口（115200）发一条命令，存进 NVS，之后自动连：

```
W 你家WiFi名 密码
C
```

`C` 回 `<<<STA up ip 192.168.x.x` 就成了；地址也会在开机日志里打印
（`api  : http://192.168.x.x/v1`），mDNS 名 `http://ple-tinystory.local/`。
板子同时保留自己的热点 `PLE-TinyLM`，两种入口都在。

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
