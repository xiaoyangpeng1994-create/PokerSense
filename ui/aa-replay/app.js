"use strict";
let position=0, timer=null, generation=0;
const el=id=>document.getElementById(id);
function stop(){clearInterval(timer);timer=null;el("play").textContent="播放";}
async function show(frame){
  if(!Number.isInteger(frame)||frame<0||frame>1800)return;
  const ticket=++generation;position=frame;
  el("slider").value=frame;el("jump").value=frame;
  el("fields").replaceChildren(); // Never leave previous-frame values visible during load.
  el("frameLabel").textContent=`帧 ${frame} / 1800 · 加载中`;
  el("participation").textContent="参与状态：等待当前帧";
  el("invalidity").textContent="失效状态：等待当前帧";
  el("scene").textContent="";
  try{
    const response=await fetch(`/api/frame/${frame}`,{cache:"no-store"});
    if(!response.ok)throw Error(`HTTP ${response.status}`);
    const data=await response.json();if(ticket!==generation)return;
    el("error").textContent="";
    el("frameLabel").textContent=`帧 ${data.frame} / 1800 · ${(data.timestamp_ms/1000).toFixed(3)} 秒`;
    el("source").textContent=data.source_sha256;el("replay").textContent=data.replay_sha256;
    el("participation").textContent=`参与状态：${data.participation}`;
    el("invalidity").textContent=`失效状态：${data.invalidity.join("；")}`;
    el("scene").textContent=`场景：${data.scene}`;
    for(const field of data.fields){
      const tr=document.createElement("tr");
      const value=field.status==="未实现"?"未实现":field.value===null?"未知／拒识":JSON.stringify(field.value);
      for(const text of [field.name,value,field.status,`${field.confidence}；${field.reason}`]){
        const td=document.createElement("td");td.textContent=text;tr.append(td);
      }
      if(field.status==="未实现")tr.className="missing";
      el("fields").append(tr);
    }
  }catch(error){if(ticket!==generation)return;stop();el("error").textContent=`当前帧加载失败：${error.message}`;el("invalidity").textContent="失效：读取失败，已清空字段";}
}
el("prev").onclick=()=>{stop();show(position-1);};
el("next").onclick=()=>{stop();show(position+1);};
el("slider").oninput=event=>{stop();show(Number(event.target.value));};
el("go").onclick=()=>{stop();show(Number(el("jump").value));};
el("jump").onchange=event=>{stop();show(Number(event.target.value));};
el("play").onclick=()=>{if(timer){stop();return;}el("play").textContent="暂停";timer=setInterval(()=>{if(position===1800){stop();return;}show(position+1);},100);};
document.addEventListener("keydown",event=>{if(event.target.tagName==="INPUT")return;if(event.key==="ArrowRight"||event.key==="ArrowLeft"){event.preventDefault();stop();show(position+(event.key==="ArrowRight"?1:-1));}});
show(0);
