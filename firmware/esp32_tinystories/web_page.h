// Web page served from PROGMEM. Kept out of the .ino so the Arduino
// preprocessor's ctags pass does not mistake the JavaScript for function
// definitions (it scans raw strings too).
#ifndef WEB_PAGE_H
#define WEB_PAGE_H

static const char PAGE[] PROGMEM = R"HTML(<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLE TinyLM</title>
<style>
*{box-sizing:border-box}
body{margin:0;padding:18px 18px 40px;background:#111118;color:#e9e9f2;
     font:17px/1.75 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
h1{font-size:19px;margin:0 0 2px}
.sub{color:#8b8ba0;font-size:13px;margin-bottom:18px}
button{font:600 17px/1 system-ui;padding:13px 22px;border:0;border-radius:10px;
       background:#5073ff;color:#fff;cursor:pointer}
button:disabled{background:#333346;color:#777}
#st{margin-left:12px;color:#8b8ba0;font-size:13px}
#txt{white-space:pre-wrap;margin-top:20px;font-size:19px;line-height:1.95;min-height:45vh}
</style>
<h1>ESP32-S3 &middot; PLE TinyLM</h1>
<div class="sub">22.9M 参数 &middot; 一篇约 170 字 &middot; 板载 1.54" 屏同步显示</div>
<button id="go" onclick="run()">再来一篇</button><span id="st"></span>
<div id="txt"></div>
<script>
let since=0,busy=false;
async function run(){
  if(busy)return;busy=true;go.disabled=true;since=0;txt.textContent='';st.textContent='\u751f\u6210\u4e2d\u2026';
  try{
    const j=await(await fetch('/start')).json();
    if(!j.ok){busy=false;go.disabled=false;st.textContent='\u677f\u5b50\u6b63\u5fd9';return;}
    tick();
  }catch(e){busy=false;go.disabled=false;st.textContent='\u8fde\u63a5\u4e2d\u65ad';}
}
async function tick(){
  try{
    const j=await(await fetch('/text?since='+since)).json();
    if(j.text){txt.textContent+=j.text;since=j.len;}
    if(j.done){st.textContent=j.tps>0?j.tps.toFixed(2)+' tok/s \u00b7 '+j.ms.toFixed(0)+' ms/token':'';
               busy=false;go.disabled=false;return;}
  }catch(e){}
  setTimeout(tick,400);
}
run();
</script>)HTML";

#endif
