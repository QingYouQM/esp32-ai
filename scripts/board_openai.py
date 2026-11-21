"""OpenAI-compatible front end for the model that runs on the ESP32-S3.

No inference happens here: the prompt is encoded with the same BPE tokenizer the
board was flashed with, the token ids go out over USB serial, and the bytes the
board streams back are relayed as OpenAI chunks. Speed is the board's (~2.8
tok/s, ~35 s per story) and there is exactly one model, so concurrent requests
queue on a lock instead of failing.

    python scripts/board_openai.py --port 8001

    from openai import OpenAI
    c = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="x")
    r = c.chat.completions.create(model="esp32-zh-story4",
                                  messages=[{"role": "user", "content": "一天，"}])

The board side is the serial control channel added to esp32_tinystories.ino:
"P <token ids>" starts a framed run (<<<BEGIN ... <<<END), "S" uses the baked
prompt, "?" reports state. The prompt is the only thing the API can influence:
the board decodes greedily with the firmware's repetition penalty, so
temperature / top_p / max_tokens are accepted and ignored.

GET / serves a phone-sized page so any device on the WiFi can use it without a
client library.
"""

import argparse
import codecs
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import serial
from serial.tools import list_ports
from tokenizers import Tokenizer

FRAME_END = b"<<<END"

PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32-S3 PLE TinyLM</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;padding:18px 16px 48px;background:#0e0e14;color:#e8e8f2;
     font:16px/1.75 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
h1{font-size:19px;margin:0 0 4px}
.sub{color:#85859c;font-size:13px;margin-bottom:16px;line-height:1.6}
label{display:block;font-size:13px;color:#9a9ab2;margin:14px 0 6px}
textarea{width:100%;min-height:84px;resize:vertical;background:#171722;color:#eee;
     border:1px solid #2b2b3d;border-radius:10px;padding:12px;font:inherit}
button{font:600 16px/1 system-ui;padding:12px 18px;border:0;border-radius:10px;
       background:#5073ff;color:#fff;cursor:pointer;margin-top:18px}
button:disabled{background:#2c2c3d;color:#777}
button.ghost{background:#22222f;color:#bbb;margin-left:8px}
#st{color:#8b8ba0;font-size:13px;margin-left:10px}
#out{white-space:pre-wrap;background:#14141d;border:1px solid #23233a;border-radius:12px;
     padding:14px;margin-top:18px;min-height:200px;font-size:17px;line-height:2}
.foot{color:#6d6d84;font-size:12px;margin-top:18px;line-height:1.7}
</style>
<h1>ESP32-S3 · PLE TinyLM</h1>
<div class="sub">板端推理：模型跑在那块 N16R8 上，约 2.8 tok/s、一篇 35 秒。手机上看着字一个个出来就行。</div>

<label>故事开头（板子从这个前缀往下写）</label>
<textarea id="p">一天，</textarea>

<button id="go">生成文章</button>
<button id="cp" class="ghost">复制</button><span id="st"></span>
<div id="out">点「生成文章」。板子只有一个模型，多人同时点会排队。</div>
<div class="foot">板端是贪婪解码 + 1.3 重复惩罚，同一个开头永远同一篇；长度固定到模型自己收尾。
不是对话模型，别用“帮我写点关于猫的”这种指令。</div>

<script>
const $=id=>document.getElementById(id);
let busy=false;

$('go').onclick=async()=>{
  if(busy)return;
  busy=true;$('go').disabled=true;$('out').textContent='';$('st').textContent='板子生成中…约 35 秒';
  const t0=performance.now();
  try{
    const r=await fetch('/v1/chat/completions',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'esp32-zh-story4',stream:true,
        messages:[{role:'user',content:$('p').value}]})
    });
    if(!r.ok)throw new Error(await r.text());
    const rd=r.body.getReader(),dec=new TextDecoder();
    let buf='',chars=0,done=false;
    while(!done){
      const {value,done:d}=await rd.read();
      if(d)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\n\n');buf=parts.pop();
      for(const part of parts){
        const line=part.replace(/^data: /,'').trim();
        if(!line)continue;
        if(line==='[DONE]'){done=true;break;}
        const j=JSON.parse(line),c=j.choices&&j.choices[0];
        const s=c&&c.delta&&c.delta.content;
        if(s){$('out').textContent+=s;chars+=s.length;}
        if(c&&c.finish_reason)done=true;
      }
    }
    $('st').textContent='完成：'+chars+' 字 / '+((performance.now()-t0)/1000).toFixed(1)+' 秒';
  }catch(e){
    $('st').textContent='失败：'+String(e.message).slice(0,90);
  }finally{
    busy=false;$('go').disabled=false;
  }
};

$('cp').onclick=async()=>{
  try{await navigator.clipboard.writeText($('out').textContent);$('st').textContent='已复制';}
  catch{$('st').textContent='浏览器不允许复制';}
};
</script>
'''


def lan_addresses():
    import socket
    addrs = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addrs.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in addrs:
                addrs.append(ip)
    except OSError:
        pass
    return addrs


def as_chat_text(payload):
    """The story prefix: the last user turn, and nothing else.

    System prompts and earlier turns are dropped on purpose. This model only
    continues Chinese TinyStories-style prose - it cannot follow instructions -
    so a system prompt would just burn the board's 32-token prompt budget and
    push the user's actual words out of the window.
    """
    turns = []
    for m in payload.get("messages") or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):  # parts form: [{"type":"text","text":"..."}]
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
        if isinstance(content, str) and content.strip():
            turns.append((str(m.get("role") or ""), content.strip()))
    for role, text in reversed(turns):
        if role == "user":
            return text
    return turns[-1][1] if turns else ""


def find_board_port():
    for p in list_ports.comports():
        if p.vid == 0x303A:            # Espressif USB-Serial/JTAG
            return p.device
    ports = list(list_ports.comports())
    return ports[0].device if len(ports) == 1 else None


class BoardState:
    """One board, one story at a time; everything else queues behind the lock."""

    model_name = "esp32-zh-story4"
    default_prompt = "一天，"

    def __init__(self, port, tokenizer_path, prompt_max=32):
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.prompt_max = prompt_max
        self.device = f"serial:{port}"
        self.lock = threading.Lock()
        self.ser = serial.Serial(port, 115200, timeout=0.05, write_timeout=5)

    # -- wire helpers --------------------------------------------------------
    def _readline(self, timeout):
        buf = bytearray()
        end = time.time() + max(timeout, 0.1)
        while time.time() < end:
            b = self.ser.read(1)
            if not b:
                continue
            if b in b"\r\n":
                if buf:
                    return buf.decode("utf-8", "replace").strip()
                continue
            buf += b
            if len(buf) > 200:
                break
        return buf.decode("utf-8", "replace").strip()

    def _wait_idle(self, timeout):
        """Ask until the board answers <<<IDLE; a booting board ignores us."""
        end = time.time() + timeout
        while time.time() < end:
            self.ser.write(b"?\n")
            line = self._readline(1.0)
            if line.startswith("<<<IDLE"):
                return True
            if line.startswith("<<<BUSY"):
                time.sleep(0.5)
        return False

    def _pump(self, on_delta, timeout=300):
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        buf = b""
        first = True
        end = time.time() + timeout
        while time.time() < end:
            chunk = self.ser.read(64)
            if not chunk:
                continue
            buf += chunk
            i = buf.find(FRAME_END)
            if i >= 0:
                text = dec.decode(buf[:i], final=False).rstrip("\r\n")
                if text:
                    on_delta(text)
                return
            keep = len(FRAME_END) - 1
            if len(buf) > keep:
                body, buf = buf[:-keep], buf[-keep:]
                text = dec.decode(body)
                # The board's USB-CDC turns \n into \r\n, and _readline stops at
                # the \r, so the first byte after <<<BEGIN is a stray newline.
                if first:
                    text = text.lstrip("\r\n")
                    first = not text
                if text:
                    on_delta(text)
        raise TimeoutError("board stopped mid-story")

    def prompt_ids(self, text):
        text = (text or "").strip() or self.default_prompt
        return self.tok.encode(text).ids[: self.prompt_max]

    # -- one story -----------------------------------------------------------
    def run(self, text, max_tokens, temperature, top_k, top_p, penalty, stops, on_delta):
        ids = self.prompt_ids(text)
        if not ids:
            return 0, "stop"
        cmd = ("P " + " ".join(str(i) for i in ids) + "\n").encode("ascii")
        collected = []

        def relay(s):
            collected.append(s)
            on_delta(s)

        with self.lock:
            if not self._wait_idle(120):
                raise TimeoutError("board is busy")
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            line = self._readline(10.0)
            while not line.startswith("<<<BEGIN"):
                if line.startswith("<<<BUSY"):
                    if not self._wait_idle(180):
                        raise TimeoutError("board stayed busy")
                    self.ser.reset_input_buffer()
                    self.ser.write(cmd)
                elif line.startswith("<<<ERR"):
                    raise RuntimeError(line)
                line = self._readline(10.0)
                if not line:
                    raise TimeoutError("board did not start the story")
            self._pump(relay)

        out = "".join(collected)
        return len(self.tok.encode(out).ids), "stop"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: BoardState = None

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message, kind="invalid_request_error"):
        self._send_json(code, {"error": {"message": message, "type": kind, "code": None}})

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            raise ValueError(f"request body is not JSON: {e}")

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.close_connection = True

    def _sse_send(self, obj):
        data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/v1/models", "/models"):
            self._send_json(200, {"object": "list", "data": [{
                "id": self.state.model_name, "object": "model",
                "created": int(time.time()), "owned_by": "esp32",
            }]})
        elif path == "/healthz":
            self._send_json(200, {"ok": True, "backend": self.state.device,
                                  "busy": self.state.lock.locked()})
        else:
            self._error(404, f"no route {path}", "not_found")

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            payload = self._body()
        except ValueError as e:
            return self._error(400, str(e))
        if path == "/v1/chat/completions":
            self._completion(payload, chat=True)
        elif path == "/v1/completions":
            self._completion(payload, chat=False)
        else:
            self._error(404, f"no route {path}", "not_found")

    def _completion(self, payload, chat):
        st = self.state
        text = as_chat_text(payload) if chat else payload.get("prompt", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        if not isinstance(text, str):
            return self._error(400, "prompt/messages must be text")

        # Sampling knobs are accepted for SDK compatibility and ignored on
        # purpose: the board decodes greedily and stops at its own end-of-text
        # token, so the only thing that matters is the prompt.
        def run(on_delta):
            return st.run(text,
                          payload.get("max_tokens"),
                          payload.get("temperature", 0),
                          0, 1.0, 1.3, None,
                          on_delta)

        stream = bool(payload.get("stream"))
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        base = {"id": cid, "created": int(time.time()), "model": st.model_name}

        t0 = time.time()
        if stream:
            self._sse_open()
            if chat:
                self._sse_send(dict(base, object="chat.completion.chunk", choices=[{
                    "index": 0, "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None}]))
                on_delta = lambda s: self._sse_send(dict(base, object="chat.completion.chunk",
                                                         choices=[{"index": 0, "delta": {"content": s},
                                                                   "finish_reason": None}]))
            else:
                on_delta = lambda s: self._sse_send(dict(base, object="text_completion",
                                                         choices=[{"index": 0, "text": s,
                                                                   "logprobs": None,
                                                                   "finish_reason": None}]))
            try:
                n, reason = run(on_delta)
            except Exception as e:
                self.log_message("backend failed mid-stream: %s", e)
                self._sse_send(dict(base, object="chat.completion.chunk" if chat else "text_completion",
                                    choices=[{"index": 0,
                                              **({"delta": {}} if chat else {"text": ""}),
                                              "finish_reason": "stop"}]))
                self._sse_send("[DONE]")
                return
            self._sse_send(dict(base, object="chat.completion.chunk" if chat else "text_completion",
                                choices=[{"index": 0,
                                          **({"delta": {}} if chat else {"text": ""}),
                                          "finish_reason": reason}]))
            self._sse_send("[DONE]")
            dt = max(time.time() - t0, 1e-6)
            self.log_message("%d tokens in %.1fs (%.2f tok/s)", n, dt, n / dt)
            return

        chunks = []
        try:
            n, reason = run(chunks.append)
        except Exception as e:
            return self._error(503, f"board backend failed: {e}", "backend_error")
        out = "".join(chunks)
        prompt_tokens = len(st.prompt_ids(text))
        if chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": out},
                      "logprobs": None, "finish_reason": reason}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": out, "logprobs": None, "finish_reason": reason}
            obj = "text_completion"
        self._send_json(200, dict(base, object=obj, choices=[choice], usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n,
            "total_tokens": prompt_tokens + n,
        }))


def serve():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="serial port; autodetected by Espressif VID 0x303A")
    ap.add_argument("--tokenizer", default=str(ROOT / "artifacts/zh/tokenizer.json"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--http-port", type=int, default=8001)
    ap.add_argument("--prompt-max", type=int, default=32,
                    help="prompt token cap; the firmware buffer is 32")
    args = ap.parse_args()

    port = args.port or find_board_port()
    if not port:
        raise SystemExit("no board found: pass --port COM4 (/dev/ttyACM0)")

    state = BoardState(port, args.tokenizer, args.prompt_max)
    Handler.state = state

    print(f"board   : {port} @115200  (model id: {state.model_name})")
    print(f"prompt  : encoded with {Path(args.tokenizer).name}, max {args.prompt_max} tokens")
    print("backend : the ESP32-S3 itself; one story at a time, ~35 s each")
    if args.host in ("0.0.0.0", "::", ""):
        print("open this on any device on the same WiFi:")
        for ip in lan_addresses() or ["<this-pc-ip>"]:
            print(f"    http://{ip}:{args.http_port}/")
    else:
        print(f"listening on http://{args.host}:{args.http_port}/")

    srv = ThreadingHTTPServer((args.host, args.http_port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    serve()
