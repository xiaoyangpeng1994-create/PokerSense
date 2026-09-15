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
  el("incomplete").textContent="incomplete_fields：等待当前帧";
  el("strategy").textContent="strategy_eligible = false（加载中）";
  el("zone").textContent="分区：等待当前帧";
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
    el("incomplete").textContent=`源缺失声明 incomplete_fields：${data.incomplete_fields.join(" · ") || "无"}`;
    el("strategy").textContent=`strategy_eligible = ${data.strategy_eligible}（仅离线字段完整性，不授权策略建议）`;
    el("zone").textContent=`分区：${data.frame_zone} · ${data.confirmation_enabled?"确认写入私有台账":"只读；启动时需配置 --ledger"}`;
    for(const field of data.fields){
      const tr=document.createElement("tr");
      const value=field.status==="未实现"?"未实现":field.value===null?"未知／拒识":JSON.stringify(field.value);
      for(const text of [field.name,value,field.status,`${field.confidence}；${field.reason}`]){
        const td=document.createElement("td");td.textContent=text;tr.append(td);
      }
      if(field.status==="未实现")tr.className="missing";
      const edit=document.createElement("td");
      if(field.confirmation_field && data.confirmation_enabled){
        const input=document.createElement("input");input.type="text";
        input.setAttribute("aria-label",`${field.name} 人工值`);
        input.style.width="150px";
        const options={seat_presence:"participating / folded / waiting / spectating / empty / all_in",action:"fold / muck / all_in / check / call / bet / raise / none",actor:"座位 0–8",pot:"非负整数金额",current_bet:"非负整数金额",special_mode:'{"critical_hit":{"enabled":false,"triggered":false},"squid":{"enabled":false,"triggered":false},"insurance":{"enabled":false,"triggered":false}}'};
        input.placeholder=options[field.confirmation_field];input.title=input.placeholder;
        input.onfocus=stop;
        const button=document.createElement("button");button.textContent="确认";
        button.onclick=async()=>{
          stop();button.disabled=true;
          try{
            let value=input.value.trim();
            if(field.confirmation_field==="actor"){
              if(!/^[0-8]$/.test(value))throw Error("请输入座位 0–8");
              value=Number(value);
            }
            if(field.confirmation_field==="special_mode")value=JSON.parse(value);
            const response=await fetch("/api/confirm",{method:"POST",headers:{"Content-Type":"application/json","X-AA-Confirmation":"1"},body:JSON.stringify({frame:data.frame,field:field.confirmation_field,seat:field.seat,human_value:value})});
            const result=await response.json();if(!response.ok)throw Error(result.detail);
            if(position===data.frame && ticket===generation)await show(data.frame);
          }catch(error){if(position===data.frame && ticket===generation)el("error").textContent=`确认失败：${error.message}`;}
          finally{button.disabled=false;}
        };
        edit.append(input,button);
      }
      tr.append(edit);
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
el("reportButton").onclick=async()=>{
  stop();el("report").replaceChildren();
  try{
    const response=await fetch("/api/report",{cache:"no-store"});
    const data=await response.json();if(!response.ok)throw Error(data.detail);
    const table=document.createElement("table");
    for(const cells of [["字段","一致","不一致","可比分母","未知／拒识","确认分母"],...Object.entries(data.fields).map(([name,s])=>[name,s.consistent,s.inconsistent,s.comparable_denominator,s.rejected_or_unknown,s.confirmed_denominator])]){
      const tr=document.createElement("tr");for(const text of cells){const td=document.createElement("td");td.textContent=text;tr.append(td);}table.append(tr);
    }
    const note=document.createElement("p");note.textContent=`确认事件 ${data.confirmation_events}；最新确认 ${data.latest_confirmations}。误报率／漏记率：UNKNOWN（没有完整机会真值分母）。台账 SHA ${data.ledger_sha256}；报表 SHA ${data.report_sha256}`;
    el("report").append(table,note);
  }catch(error){el("report").textContent=`复盘失败：${error.message}`;}
};
show(0);
