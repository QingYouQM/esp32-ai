"""OpenAI-compatible HTTP endpoint for a trained PLE checkpoint.

Speaks enough of the OpenAI wire protocol that the official SDKs work
(client.chat.completions.create, streaming included), but the model is a Chinese
story LM trained on continuous documents: there is no chat template and no
instruction tuning, so the request text is used verbatim as the prefix and the
reply is whatever continuation the model writes. Do not expect it to follow
instructions -- send it an opening and it writes a story.

    python scripts/serve_openai.py --port 8000
    curl http://127.0.0.1:8000/v1/models

Endpoints:
    GET  /                        (phone-friendly page, streams a story)
    GET  /v1/models
    POST /v1/completions
    POST /v1/chat/completions     (stream=true -> server-sent events)
    GET  /healthz

Binds 0.0.0.0 by default so any device on the same WiFi can use it; there is no
authentication, so keep it to a trusted network (--host 127.0.0.1 to disable).

Sampling mirrors src/model.py generate(): CTRL-style repetition penalty over
every unique token in the context, greedy when temperature <= 0 (the device's
acceptance mode), top-k / top-p otherwise.
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, TinyLM

STOP_HTML = "<|endoftext|>"

# Served at "/" so a phone or tablet on the same WiFi needs nothing but a
# browser. It talks to the same /v1/chat/completions endpoint the SDKs use.
PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLE TinyLM</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;padding:18px 16px 48px;background:#0e0e14;color:#e8e8f2;
     font:16px/1.75 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
h1{font-size:19px;margin:0 0 4px}
.sub{color:#85859c;font-size:13px;margin-bottom:16px;line-height:1.6}
label{display:block;font-size:13px;color:#9a9ab2;margin:14px 0 6px}
textarea,input[type=number]{width:100%;background:#171722;color:#eee;
     border:1px solid #2b2b3d;border-radius:10px;padding:12px;font:inherit}
textarea{min-height:84px;resize:vertical}
.row{display:flex;gap:14px;flex-wrap:wrap}
.row>div{flex:1 1 150px;min-width:0}
input[type=range]{width:100%}
button{font:600 16px/1 system-ui;padding:12px 18px;border:0;border-radius:10px;
       background:#5073ff;color:#fff;cursor:pointer;margin-top:18px}
button:disabled{background:#2c2c3d;color:#777}
button.ghost{background:#22222f;color:#bbb;margin-left:8px}
#st{color:#8b8ba0;font-size:13px;margin-left:10px}
#out{white-space:pre-wrap;background:#14141d;border:1px solid #23233a;border-radius:12px;
     padding:14px;margin-top:18px;min-height:200px;font-size:17px;line-height:2}
.foot{color:#6d6d84;font-size:12px;margin-top:18px;line-height:1.7}
</style>
<h1>PLE TinyLM</h1>
<div class="sub">中文故事模型 · 续写式生成 · 同一 WiFi 下的任何设备直接打开这个地址就能用</div>

<label>故事开头（模型从这里往下写）</label>
<textarea id="p">一天，</textarea>

<div class="row">
  <div><label>长度（token）</label>
    <input id="n" type="number" value="230" min="16" max="460"></div>
  <div><label>温度 <span id="tv">0.3</span>（0 = 每次结果相同）</label>
    <input id="t" type="range" min="0" max="1" step="0.1" value="0.3"></div>
</div>

<button id="go">生成文章</button>
<button id="cp" class="ghost">复制</button><span id="st"></span>
<div id="out">点「生成文章」，字会一个个长出来。</div>
<div class="foot">这是纯续写模型：没有对话模板，听不懂“写一篇关于猫的文章”这种指令，只顺着你给的开头写。
若一直卡在“生成中”，去 Windows 防火墙允许 Python 访问专用网络。</div>

<script>
const $=id=>document.getElementById(id);
const MODEL="__MODEL__";
$('t').oninput=()=>$('tv').textContent=$('t').value;
let busy=false;

$('go').onclick=async()=>{
  if(busy)return;
  busy=true;$('go').disabled=true;$('out').textContent='';$('st').textContent='生成中…';
  const t0=performance.now();
  try{
    const r=await fetch('/v1/chat/completions',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        model:MODEL,stream:true,
        messages:[{role:'user',content:$('p').value}],
        max_tokens:Math.max(16,Math.min(460,parseInt($('n').value)||230)),
        temperature:parseFloat($('t').value),
        repetition_penalty:1.3
      })
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
        const j=JSON.parse(line);
        const c=j.choices&&j.choices[0];
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


def pick_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_model(run_path, tokenizer_path, device, check_tokenizer=True):
    ck = torch.load(run_path, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    model.to(device).eval()

    want = ck.get("tokenizer_sha256")
    got = hashlib.sha256(open(tokenizer_path, "rb").read()).hexdigest()
    if check_tokenizer and want and want != got:
        raise SystemExit(
            f"tokenizer mismatch: {tokenizer_path} is sha256 {got[:16]}... but "
            f"{os.path.basename(run_path)} was trained with {want[:16]}..."
        )
    tok = Tokenizer.from_file(tokenizer_path)
    return model, tok, ck


@torch.no_grad()
def iter_tokens(model, prompt_ids, max_new, temperature, top_k, top_p,
                repetition_penalty, stop_token):
    """Yield one sampled token id at a time, so the HTTP layer can stream."""
    cfg = model.cfg
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], device=device)
    window = cfg.seq_len

    for _ in range(max_new):
        logits, _ = model(idx[:, -window:])
        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            # Must stay in lockstep with the firmware (REPETITION_PENALTY_PCT)
            # and with model.generate(): one penalty per unique token.
            for t in torch.unique(idx[0]).tolist():
                if logits[0, t] > 0:
                    logits[0, t] /= repetition_penalty
                else:
                    logits[0, t] *= repetition_penalty

        if temperature <= 0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)  # greedy, as on device
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cut = torch.cumsum(probs, dim=-1) - probs > top_p
                sorted_logits[cut] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)

        tok = int(nxt.item())
        if stop_token is not None and tok == stop_token:
            return
        idx = torch.cat([idx, nxt], dim=1)
        yield tok


def make_stop_matcher(stops):
    stops = [s for s in (stops or []) if s]
    return stops


def first_stop(text, stops):
    """Index of the earliest stop string in text, or -1."""
    hit = -1
    for s in stops:
        i = text.find(s)
        if i >= 0 and (hit < 0 or i < hit):
            hit = i
    return hit


def trim_for_stops(text, stops):
    """Hold back text that could still become a stop string, and cut on a hit."""
    i = first_stop(text, stops)
    if i >= 0:
        return text[:i], True
    hold = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), hold, -1):
            if text.endswith(s[:k]):
                hold = k
                break
    return text[: len(text) - hold] if hold else text, False


class State:
    def __init__(self, model, tok, cfg, model_name, default_prompt, max_new_cap):
        self.model = model
        self.tok = tok
        self.cfg = cfg
        self.model_name = model_name
        self.default_prompt = default_prompt
        self.max_new_cap = max_new_cap
        self.stop_token = tok.token_to_id(STOP_HTML)
        self.lock = threading.Lock()

    def prompt_ids(self, text):
        text = (text or "").strip()
        if not text:
            text = self.default_prompt
        return self.tok.encode(text).ids

    def run(self, text, max_tokens, temperature, top_k, top_p,
            repetition_penalty, stops, on_delta):
        """Blocking generation; calls on_delta(text) for every new fragment.

        Returns (completion_tokens, finish_reason). The model is single-instance,
        so callers serialize on self.lock.
        """
        ids = self.prompt_ids(text)
        if not ids:
            return 0, "stop"
        budget = max(1, min(max_tokens, self.max_new_cap, self.cfg.seq_len - len(ids)))

        def stream():
            # A token can end in the middle of a UTF-8 sequence (the device has
            # the same problem and carries the tail across calls). Decoding a
            # partial sequence yields U+FFFD, so hold a trailing replacement
            # character back: the next token completes it and the prefix diff
            # below then emits the real character instead.
            sent = ""
            for t in iter_tokens(self.model, ids, budget, temperature, top_k, top_p,
                                 repetition_penalty, self.stop_token):
                ids.append(t)
                full = self.tok.decode(ids)
                if full.startswith(sent):
                    fresh = full[len(sent):]
                else:                      # model rewrote earlier text: resync
                    fresh, sent = full, ""
                cut = len(fresh)
                while cut and fresh[cut - 1] == "\ufffd":
                    cut -= 1
                if cut:
                    piece = fresh[:cut]
                    sent += piece
                    yield piece

        prompt_len = len(ids)
        emitted = ""
        cut_hit = False
        with self.lock:
            for delta in stream():
                piece, cut = trim_for_stops(emitted + delta, stops)
                fresh = piece[len(emitted):] if len(piece) > len(emitted) else ""
                emitted = piece
                if fresh:
                    on_delta(fresh)
                if cut:
                    cut_hit = True
                    break
        # Count tokens, not streamed fragments: a token can add no characters
        # of its own when it completes a multi-byte character.
        generated = len(ids) - prompt_len
        reason = "length" if (not cut_hit and generated >= budget) else "stop"
        return generated, reason


def as_chat_text(payload):
    """The story prefix this model continues: the last user turn, nothing else.

    System prompts and earlier turns are dropped on purpose. The model writes
    Chinese TinyStories-style prose and cannot follow instructions, so a system
    prompt would only eat the context window and steer the story off-course.
    """
    turns = []
    for m in payload.get("messages") or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):  # parts form, keep the text fields
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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: State = None  # set by serve()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing ------------------------------------------------------------
    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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

    # -- routes --------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send_html(PAGE.replace("__MODEL__", self.state.model_name))
        elif path in ("/v1/models", "/models"):
            self._send_json(200, {"object": "list", "data": [{
                "id": self.state.model_name, "object": "model",
                "created": int(time.time()), "owned_by": "local",
            }]})
        elif path == "/healthz":
            self._send_json(200, {"ok": True, "device": str(next(self.state.model.parameters()).device)})
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
        if isinstance(text, list):  # completions accepts a list of prompts
            text = text[0] if text else ""
        if not isinstance(text, str):
            return self._error(400, "prompt/messages must be text")

        max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 200)
        temperature = float(payload.get("temperature", 0.8))
        top_k = int(payload.get("top_k") or 40)
        top_p = float(payload.get("top_p", 1.0))
        penalty = float(payload.get("repetition_penalty", 1.3))
        stops = make_stop_matcher(payload.get("stop"))
        stream = bool(payload.get("stream"))

        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        base = {"id": cid, "created": created, "model": st.model_name}

        t0 = time.time()
        if stream:
            self._sse_open()
            if chat:
                first = dict(base, object="chat.completion.chunk", choices=[{
                    "index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}])
                self._sse_send(first)
                on_delta = lambda s: self._sse_send(dict(base, object="chat.completion.chunk", choices=[{
                    "index": 0, "delta": {"content": s}, "finish_reason": None}]))
            else:
                on_delta = lambda s: self._sse_send(dict(base, object="text_completion", choices=[{
                    "index": 0, "text": s, "logprobs": None, "finish_reason": None}]))
            try:
                n, reason = st.run(text, max_tokens, temperature, top_k, top_p,
                                   penalty, stops, on_delta)
            except (BrokenPipeError, ConnectionResetError):
                self.log_message("client went away mid-stream")
                return
            last = dict(base, object="chat.completion.chunk" if chat else "text_completion",
                        choices=[{"index": 0,
                                  **({"delta": {}} if chat else {"text": ""}),
                                  "finish_reason": reason}])
            self._sse_send(last)
            self._sse_send("[DONE]")
            dt = max(time.time() - t0, 1e-6)
            self.log_message("%s %d tokens in %.2fs (%.1f tok/s)", "chat" if chat else "completion",
                             n, dt, n / dt)
            return

        chunks = []
        n, reason = st.run(text, max_tokens, temperature, top_k, top_p,
                           penalty, stops, chunks.append)
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
        dt = max(time.time() - t0, 1e-6)
        self.log_message("%s %d tokens in %.2fs (%.1f tok/s)", "chat" if chat else "completion",
                         n, dt, n / dt)


def lan_addresses():
    """Best-effort list of IPv4 addresses other devices on the WiFi can reach."""
    addrs = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))      # routing lookup only; no packet is sent
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


def serve():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/ple-zh-story4-s0.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "artifacts/zh/tokenizer.json"))
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 = reachable from other devices; 127.0.0.1 = this PC only")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--name", default="ple-zh-story4", help="model id reported to clients")
    ap.add_argument("--default-prompt", default="一天，",
                    help="prefix used when the request carries no text")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="upper bound for a single request (device uses 230)")
    ap.add_argument("--skip-tokenizer-check", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tok, ck = load_model(args.run, args.tokenizer, device,
                                not args.skip_tokenizer_check)
    state = State(model, tok, model.cfg, args.name, args.default_prompt, args.max_tokens)
    Handler.state = state

    b = model.param_budget()
    print(f"{os.path.basename(args.run)}  core={b['core']:,} table={b['table']:,}  "
          f"device={device}  seq_len={model.cfg.seq_len}")
    if args.host in ("0.0.0.0", "::", ""):
        print(f"listening on all interfaces, port {args.port}  (model id: {args.name})")
        print("open this on any device on the same WiFi:")
        for ip in lan_addresses() or ["<this-pc-ip>"]:
            print(f"    http://{ip}:{args.port}/")
        print("no authentication: keep it to a trusted network "
              "(use --host 127.0.0.1 to close it off)")
    else:
        print(f"listening on http://{args.host}:{args.port}  (model id: {args.name})")
    print("single model instance: requests serialize, streaming still starts at token 1")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    serve()
