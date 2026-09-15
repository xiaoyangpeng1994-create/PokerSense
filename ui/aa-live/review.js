"use strict";
let deskView = "watch", reviewTicket = 0, selectedReview = null, lastMarked = null;
let reviewConfig = null, markBusy = false, humanBusy = false, aiBusy = false;
const reviewStates = {RUNNING:"正在复核", COMPLETE:"AI 候选 · 待确认", ERROR:"复核失败", TIMED_OUT:"复核超时", INTERRUPTED:"请求已中断"};
function showDesk(view) {
  if (!["watch", "review", "settings"].includes(view)) return;
  deskView = view;
  for (const name of ["watch", "review", "settings"]) {
    el(`${name}-view`).hidden = name !== view;
    el(`nav-${name}`).setAttribute("aria-pressed", String(name === view));
  }
  if (view === "review") loadReviewList();
  if (view === "settings") loadReviewConfig();
}
async function deskRequest(path, body) {
  const options = {headers, cache:"no-store"};
  if (body !== undefined) Object.assign(options, {method:"POST", headers:{...headers,"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const abort = new AbortController(), timeout = setTimeout(() => abort.abort(), 10000);
  try {
    const response = await fetch(path, {...options,signal:abort.signal});
    const result = await response.json();
    if (!response.ok) throw Error(typeof result.detail === "string" ? result.detail : "操作未完成");
    return result;
  } finally { clearTimeout(timeout); }
}
function reviewButtons() {
  el("mark-frame").disabled = markBusy || statusData.status !== "RUNNING" || !statusData.payload || !statusData.issue_recording_available;
  el("open-marked").disabled = !lastMarked;
  el("human-save").disabled = !selectedReview || humanBusy;
  const exhausted = reviewConfig && reviewConfig.calls_today >= reviewConfig.daily_limit;
  el("ai-run").disabled = !selectedReview || aiBusy || selectedReview?.ai?.status === "RUNNING" || !reviewConfig?.key_configured || reviewConfig?.busy || exhausted || !el("ai-consent").checked;
}
async function loadReviewConfig() {
  try {
    reviewConfig = await deskRequest("/api/review/config");
    el("api-state").textContent = reviewConfig.key_configured ? "密钥已配置" : "未配置密钥";
    el("api-feedback").textContent = `今日请求 ${reviewConfig.calls_today} / ${reviewConfig.daily_limit} 次（UTC）${reviewConfig.busy ? " · 请求处理中" : ""}`;
    if (!reviewConfig.key_configured && selectedReview) el("ai-feedback").textContent = "先到“设置与分析”填写 DeepSeek API Key。";
  } catch (_) { reviewConfig = null; el("api-feedback").textContent = "无法读取 API 配置，请检查本机服务。"; }
  reviewButtons();
}
async function markFrame() {
  if (markBusy || el("mark-frame").disabled) return;
  markBusy = true; reviewButtons(); el("mark-feedback").textContent = "正在保存这一刻…";
  try {
    const result = await deskRequest("/api/review/mark", {});
    lastMarked = result.issue.issue_id;
    el("mark-feedback").textContent = `已标记来源帧 ${text(result.issue.observation.source_frame)}。继续观察，空闲时点“查看刚才标记”。`;
  } catch (_) { el("mark-feedback").textContent = "保存未确认。请到复查列表核对后再重试，避免重复记录。"; }
  finally { markBusy = false; reviewButtons(); }
}
async function loadReviewList() {
  try {
    const result = await deskRequest("/api/review/issues");
    const nodes = result.items.map(row => {
      const option = document.createElement("option"); option.value = row.issue_id;
      option.textContent = `来源帧 ${text(row.source_frame)} · ${row.saved_at.slice(11,19)} UTC · ${row.human_status ? "已有人工记录" : "待复查"}`;
      return option;
    });
    if (!nodes.length) { const option = document.createElement("option"); option.value=""; option.textContent="暂无记录，请先在观察页标记"; nodes.push(option); }
    el("review-select").replaceChildren(...nodes);
    if (selectedReview && result.items.some(row => row.issue_id === selectedReview.issue.issue_id)) el("review-select").value = selectedReview.issue.issue_id;
    else el("review-select").value = "";
    el("review-count").textContent = `显示 ${result.items.length} 条；属于开发复查记录`;
  } catch (_) { el("review-feedback").textContent = "复查列表读取失败，请检查本机服务。"; }
}
function renderFrozenFields(record) {
  const snapshot = record.issue.observation, row = snapshot.payload || {};
  const pairs = [["我的手牌",row.cards?.hero],["公共牌",row.cards?.board_slots],["底池",row.pot?.value],["行动者",row.current_actor],["街道",row.observed_state_v2?.street_candidate],["各座筹码",row.stacks],["可见投入",row.street_wagers],["动作字样",row.glyphs]];
  el("review-fields").replaceChildren(...pairs.map(([label,value]) => {
    const node=document.createElement("div"), title=document.createElement("strong"), content=document.createElement("span");
    title.textContent=label; content.textContent=text(value); node.append(title,content); return node;
  }));
  el("review-frame").textContent = `来源帧 ${text(snapshot.source_frame)}`;
  el("review-source").textContent = `${record.issue.saved_at} · ${snapshot.source_kind === "capture-card" ? "采集来源" : "开发回放"} · 已冻结，非当前画面`;
  el("review-image").src = `/api/review/issues/${record.issue.issue_id}/image`;
}
function renderAI(report) {
  el("ai-state").textContent = reviewStates[report?.status] || "尚未提交";
  el("ai-result").replaceChildren();
  if (report?.status === "COMPLETE" && report.result) {
    const summary = document.createElement("p"); summary.textContent=report.result.summary;
    const nodes = report.result.findings.map(item => {
      const node=document.createElement("p"); node.className=`ai-finding ${["match","mismatch","uncertain"].includes(item.status) ? item.status : "uncertain"}`;
      node.textContent=`${item.field}：程序 ${item.observed} → 图中候选 ${item.visible}（${{match:"一致",mismatch:"疑似不一致",uncertain:"无法确认"}[item.status] || "待确认"}）`; return node;
    });
    el("ai-result").append(summary,...nodes);
    el("ai-feedback").textContent = "AI 复核完成。请核对原画面，再保存你自己的判断。";
  } else el("ai-feedback").textContent = report?.error || (report?.status === "RUNNING" ? "正在复核这张固定画面；可返回观察页，结果会保留。" : reviewConfig?.key_configured ? "勾选发送范围后，可以提交这一帧。" : "先到设置中填写 API Key。");
  reviewButtons();
}
async function selectReview(issueId) {
  const ticket=++reviewTicket; selectedReview=null;
  el("review-content").hidden=true; el("review-image").removeAttribute("src");
  el("human-note").value=""; el("ai-consent").checked=false;
  el("human-feedback").textContent=""; el("ai-result").replaceChildren(); reviewButtons();
  if (!issueId) return;
  el("review-feedback").textContent="正在打开固定画面…";
  try {
    const result=await deskRequest(`/api/review/issues/${issueId}`);
    if(ticket!==reviewTicket) return;
    selectedReview=result; el("review-content").hidden=false; el("review-select").value=issueId;
    renderFrozenFields(result);
    el("human-verdict").value=result.human?.verdict || "unreadable"; el("human-note").value=result.human?.note || "";
    el("review-feedback").textContent=""; renderAI(result.ai);
    await loadReviewConfig();
  } catch (_) { if(ticket===reviewTicket) el("review-feedback").textContent="该记录未能打开，请重新选择。"; }
  reviewButtons();
}
async function saveHumanReview() {
  if (!selectedReview || humanBusy) return;
  const id=selectedReview.issue.issue_id, ticket=reviewTicket;
  humanBusy=true; reviewButtons();
  try {
    const result=await deskRequest(`/api/review/issues/${id}/human`,{verdict:el("human-verdict").value,note:el("human-note").value,revision:selectedReview.human?.revision || null});
    if(ticket!==reviewTicket) return;
    selectedReview.human=result.human;
    el("human-feedback").textContent="人工复查已保存，原始识别结果保留。";
  } catch(e) { if(ticket===reviewTicket) el("human-feedback").textContent=`未保存：${e.message}。如其他页面已修改，请重新打开记录。`; }
  finally {humanBusy=false;reviewButtons();}
}
async function runAIReview() {
  if(!selectedReview || el("ai-run").disabled) return;
  const id=selectedReview.issue.issue_id,ticket=reviewTicket;
  aiBusy=true;reviewButtons();
  try {
    const report=await deskRequest(`/api/review/issues/${id}/ai`,{consent:el("ai-consent").checked});
    if(ticket!==reviewTicket)return;
    selectedReview.ai=report;renderAI(report);
  } catch(e) {if(ticket===reviewTicket) el("ai-feedback").textContent=`提交未确认：${e.message}。正在核对已有请求，请勿连续重试。`;}
  finally {aiBusy=false;await loadReviewConfig();reviewButtons();}
}
async function refreshAI() {
  if(deskView!=="review" || !selectedReview || document.hidden || humanBusy || aiBusy)return;
  const id=selectedReview.issue.issue_id,ticket=reviewTicket;
  try {
    const result=await deskRequest(`/api/review/issues/${id}`);
    if(ticket!==reviewTicket)return;
    selectedReview.ai=result.ai;renderAI(result.ai);
    // Poll only AI state: never overwrite an unsaved human correction.
    await loadReviewConfig();
  } catch(_) {el("ai-feedback").textContent="读取 AI 状态失败，保存的人工记录不受影响。";}
}
for(const name of ["watch","review","settings"]) el(`nav-${name}`).addEventListener("click",()=>showDesk(name));
el("mark-frame").addEventListener("click",markFrame);
el("open-marked").addEventListener("click",()=>{if(lastMarked){showDesk("review");selectReview(lastMarked);}});
el("review-refresh").addEventListener("click",loadReviewList);
el("review-select").addEventListener("change",()=>selectReview(el("review-select").value));
el("human-review-form").addEventListener("submit",event=>{event.preventDefault();saveHumanReview();});
el("ai-consent").addEventListener("change",reviewButtons);
el("ai-run").addEventListener("click",runAIReview);
el("ai-config-form").addEventListener("submit",async event=>{
  event.preventDefault();const key=el("ai-key").value;el("ai-key").value="";
  el("ai-config-save").disabled=true;
  try {reviewConfig=await deskRequest("/api/review/config",{api_key:key,model:el("ai-model").value,daily_limit:Number(el("ai-limit").value)});await loadReviewConfig();}
  catch(e){el("api-feedback").textContent=`配置未保存：${e.message}`;}
  finally{el("ai-config-save").disabled=false;reviewButtons();}
});
el("ai-clear-key").addEventListener("click",async()=>{
  try {await deskRequest("/api/review/clear-key",{});await loadReviewConfig();}
  catch(_){el("api-feedback").textContent="清除未确认，请检查服务。";}
});
el("review-image").addEventListener("error",()=>{el("review-feedback").textContent="冻结图片读取或校验失败，请勿据此确认。";selectedReview=null;reviewButtons();});
setInterval(reviewButtons,500);setInterval(refreshAI,1800);loadReviewConfig();
