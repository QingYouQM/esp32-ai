// Provisioning page: the board's own access point serves this so a phone can
// pick the home WiFi without a serial cable. Kept out of the .ino for the same
// reason as web_page.h -- the Arduino preprocessor scans raw strings and would
// mistake the JavaScript for function prototypes.
#ifndef WIFI_PAGE_H
#define WIFI_PAGE_H

static const char WIFI_PAGE[] PROGMEM = R"HTML(<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WiFi 设置 · PLE TinyLM</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;padding:18px 16px 48px;background:#0e0e14;color:#e8e8f2;
     font:16px/1.7 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
h1{font-size:19px;margin:0 0 4px}
.sub{color:#85859c;font-size:13px;margin-bottom:16px;line-height:1.6}
.card{background:#14141d;border:1px solid #23233a;border-radius:12px;padding:14px;margin-bottom:16px}
.row{display:flex;justify-content:space-between;gap:12px;font-size:14px;padding:3px 0}
.row span:first-child{color:#8b8ba0;flex:none}
.row span:last-child{text-align:right;word-break:break-all}
label{display:block;font-size:13px;color:#9a9ab2;margin:14px 0 6px}
input{width:100%;background:#171722;color:#eee;border:1px solid #2b2b3d;
      border-radius:10px;padding:12px;font:inherit}
button{font:600 15px/1 system-ui;padding:11px 16px;border:0;border-radius:10px;
       background:#5073ff;color:#fff;cursor:pointer}
button.ghost{background:#22222f;color:#bbb}
button:disabled{background:#2c2c3d;color:#777}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
#nets{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
#nets button{background:#1d1d2b;color:#cdd3f5;font-weight:500;padding:8px 12px;font-size:13px}
#st{color:#8b8ba0;font-size:13px;margin-top:12px;min-height:20px}
a{color:#7f9bff;text-decoration:none}
.foot{color:#6d6d84;font-size:12px;margin-top:18px;line-height:1.7}
</style>
<h1>WiFi 设置</h1>
<div class="sub">先连板子的热点 <b>PLE-TinyLM</b>（密码 tinystory），在这个页面选它要接入的路由器。</div>

<div class="card">
  <div class="row"><span>当前热点</span><span id="apssid">-</span></div>
  <div class="row"><span>热点地址</span><span id="apip">-</span></div>
  <div class="row"><span>已保存网络</span><span id="saved">-</span></div>
  <div class="row"><span>路由器连接</span><span id="conn">-</span></div>
  <div class="row"><span>生成页</span><span id="links">-</span></div>
</div>

<label>WiFi 名称（区分大小写，必须是 2.4GHz）</label>
<input id="ssid" placeholder="例如 TP-LINK_5F2A" autocomplete="off">
<button class="ghost" id="scan" style="margin-top:10px">扫描附近网络</button>
<div id="nets"></div>

<label>密码（同一网络留空 = 沿用已保存的密码）</label>
<input id="pass" type="password" placeholder="WiFi 密码" autocomplete="off">

<div class="actions">
  <button id="save">保存并连接</button>
  <button class="ghost" id="forget">忘记网络</button>
</div>
<div id="st"></div>

<div class="foot">
保存后板子会用 AP+STA 模式连上路由器，自己的热点不会关闭。<br>
如果保存时板子切换到与路由器相同的信道，手机可能掉线——重连 PLE-TinyLM 后刷新本页即可。<br>
密码保存在板子的 NVS 里（明文），忘记网络会把它删掉。
</div>

<script>
const $=id=>document.getElementById(id);
const st=$('st');

function render(j){
  $('apssid').textContent=j.ap_ssid;
  $('apip').textContent='http://'+j.ap_ip+'/wifi';
  $('saved').textContent=j.saved?j.ssid:'（未保存）';
  $('conn').textContent=j.connected?('已连接 · '+j.ip+' · '+j.rssi+' dBm'):'未连接';
  const gen='http://'+(j.connected?j.ip:j.ap_ip)+'/';
  $('links').innerHTML='<a href="'+gen+'">'+gen+'</a>'+(j.connected?' · <a href="http://'+j.mdns+'/">'+j.mdns+'</a>':'');
}

async function refresh(){
  try{ render(await (await fetch('/wifi/status',{cache:'no-store'})).json()); }
  catch(e){ $('conn').textContent='读取状态失败，刷新试试'; }
}

$('scan').onclick=async()=>{
  $('scan').disabled=true; st.textContent='扫描中…（约 3 秒）';
  $('nets').textContent='';
  try{
    const j=await (await fetch('/wifi/scan',{cache:'no-store'})).json();
    const seen=new Set();
    for(const n of j.nets||[]){
      if(seen.has(n.ssid)) continue; seen.add(n.ssid);
      const b=document.createElement('button');
      b.textContent=n.ssid+(n.enc?' 🔒':'');
      b.onclick=()=>{ $('ssid').value=n.ssid; $('pass').focus(); };
      $('nets').appendChild(b);
    }
    st.textContent=(j.nets&&j.nets.length)?'点一个网络名填入':'没扫到网络，手动输入吧';
  }catch(e){ st.textContent='扫描失败：'+e.message; }
  $('scan').disabled=false;
};

$('save').onclick=async()=>{
  const ssid=$('ssid').value.trim();
  if(!ssid){ st.textContent='先填 WiFi 名称'; return; }
  $('save').disabled=true; st.textContent='连接中…最多 12 秒，别关页面';
  try{
    const body=new URLSearchParams({ssid:ssid,pass:$('pass').value});
    const j=await (await fetch('/wifi/save',{method:'POST',body})).json();
    if(j.ok) st.textContent='已连接：'+j.ip+'（也可以用 '+j.ip+' 打开生成页）';
    else st.textContent='没连上：检查名称、密码和是否为 2.4GHz 网络';
  }catch(e){ st.textContent='保存失败：'+e.message; }
  $('pass').value='';
  $('save').disabled=false;
  refresh();
};

$('forget').onclick=async()=>{
  await fetch('/wifi/forget',{method:'POST'});
  st.textContent='已忘记，板子只保留自己的热点';
  $('ssid').value=''; $('pass').value='';
  refresh();
};

refresh();
</script>)HTML";

#endif
