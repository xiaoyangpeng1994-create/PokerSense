"use strict";
let analysisEpoch = 0, acceptedAnalysisId = null, analysisReport = null, analysisInput = null;
let analysisBusy = false, cancelTimer = null, analysisBinding = null;
const analysisLabels = {RUNNING:"正在计算（最多 10 秒）", COMPLETE:"条件计算完成", BLOCKED:"输入或场景不受支持", ERROR:"计算失败", CANCELLED:"分析已取消", TIMED_OUT:"计算超时，未返回局部结果", IDLE:"尚未计算"};
function clearAnalysis(message) {
  acceptedAnalysisId = null; analysisReport = null; analysisBinding = null;
  el("analysis-results").replaceChildren(); el("analysis-raw").textContent = "无当前结果";
  el("analysis-export").disabled = true; el("analysis-status").textContent = message;
}
function invalidateAnalysis(message) {
  clearTimeout(cancelTimer); cancelTimer = null;
  ++analysisEpoch; clearAnalysis(message);
}
async function cancelAnalysis() {
  invalidateAnalysis("输入已变更或分析已取消，请重新计算。");
  try { await fetch("/api/analysis/cancel", {method:"POST", headers:{...headers,"Content-Type":"application/json"},body:"{}"}); }
  catch (_) { /* Old results remain hidden even while offline. */ }
}
function analysisEdited() {
  invalidateAnalysis("手工输入已变更，旧分析已失效。");
  cancelTimer = setTimeout(cancelAnalysis, 350);
}
function exactAmount(value) { return value?.decimal ?? value?.exact ?? text(value); }
function actionName(action) {
  if (typeof action === "string") return translated(action.toLowerCase());
  return `${translated(action?.kind)}${action?.target != null ? ` 至 ${action.target}` : ""}`;
}
function renderAnalysis(report, state) {
  if (acceptedAnalysisId === null) return;
  if (!analysisBinding || state && (analysisBinding.generation !== state.generation || analysisBinding.table_rules_revision !== state.table_rules?.revision)) {
    invalidateAnalysis("来源或本桌规则已变更，旧分析已失效。"); return;
  }
  if (!report || report.job_id !== acceptedAnalysisId || report.binding?.generation !== analysisBinding.generation || report.binding?.table_rules_revision !== analysisBinding.table_rules_revision) {
    invalidateAnalysis("分析已被其他操作替换，请重新计算。"); return;
  }
  el("analysis-status").textContent = `${analysisLabels[report.status] || report.status}${report.error ? `：${report.error}` : ""}`;
  if (report.status !== "COMPLETE") {
    analysisReport = null;
    el("analysis-results").replaceChildren(); el("analysis-export").disabled = true;
    el("analysis-raw").textContent = JSON.stringify(report, null, 2); return;
  }
  analysisReport = report; el("analysis-export").disabled = false;
  const result = report.result, rows = [];
  if (report.kind === "terminal") rows.push(["弃牌", exactAmount(result.fold_ev)], ["跟注", exactAmount(result.call_net_ev)]);
  else for (const row of result.root_actions || []) rows.push([actionName(row.action), exactAmount(row.ev)]);
  const table = document.createElement("table"); table.className = "analysis-table";
  const head = document.createElement("tr"); for (const title of ["动作", "条件净 EV（筹码）"]) { const cell = document.createElement("th"); cell.textContent = title; head.append(cell); } table.append(head);
  for (const row of rows) { const tr = document.createElement("tr"); for (const value of row) { const td = document.createElement("td"); td.textContent = value; tr.append(td); } table.append(tr); }
  const notice = document.createElement("p"); notice.className = "footnote"; notice.textContent = "结果相对当前决策计算，过去投入是沉没成本。仅在所填范围、费用与响应假设下成立；不是整手收益、GTO 或实战胜率。";
  el("analysis-results").replaceChildren(table, notice); el("analysis-raw").textContent = JSON.stringify(report, null, 2);
}
el("analysis-example").addEventListener("click", async () => {
  const cancelling = cancelAnalysis();
  const epoch = analysisEpoch;
  try { await cancelling; if (epoch !== analysisEpoch) return; const r = await fetch(`/api/analysis/example/${el("analysis-kind").value}`); const value = await r.json(); if (epoch !== analysisEpoch) return; if (!r.ok) throw Error(text(value.detail)); el("analysis-input").value = JSON.stringify(value.document, null, 2); el("analysis-use-rules").checked = false; el("analysis-status").textContent = "已载入手工算例，未使用当前牌桌数据。请核对全部输入后计算。"; }
  catch (e) { if (epoch === analysisEpoch) el("analysis-status").textContent = e.message; }
});
el("analysis-start").addEventListener("click", async () => {
  if (analysisBusy) return;
  analysisBusy = true; el("analysis-start").disabled = true;
  clearTimeout(cancelTimer); const cancelling = cancelAnalysis();
  const epoch = analysisEpoch;
  try {
    await cancelling; if (epoch !== analysisEpoch) return;
    const document = JSON.parse(el("analysis-input").value);
    const r = await fetch("/api/analysis", {method:"POST",headers:{...headers,"Content-Type":"application/json"},body:JSON.stringify({kind:el("analysis-kind").value,document,rules_source:el("analysis-use-rules").checked ? "table" : "document",rules_revision:statusData.table_rules?.revision})});
    const result = await r.json(); if (epoch !== analysisEpoch) return; if (!r.ok) throw Error(text(result.detail));
    acceptedAnalysisId = result.job_id; analysisInput = document; analysisBinding = JSON.parse(JSON.stringify(result.binding)); renderAnalysis(result);
  } catch (e) { if (epoch === analysisEpoch) el("analysis-status").textContent = `未开始计算：${e.message}`; }
  finally { analysisBusy = false; el("analysis-start").disabled = false; }
});
el("analysis-cancel").addEventListener("click", cancelAnalysis);
el("analysis-input").addEventListener("input", analysisEdited);
el("analysis-kind").addEventListener("change", analysisEdited);
el("analysis-use-rules").addEventListener("change", analysisEdited);
el("analysis-export").addEventListener("click", () => {
  if (!analysisReport) return;
  const effectiveInput = {...analysisInput,rules:analysisReport.binding.effective_rules};
  const data = JSON.stringify({effective_input:effectiveInput,report:analysisReport},null,2);
  const url = URL.createObjectURL(new Blob([data],{type:"application/json"})); const a = document.createElement("a"); a.href = url; a.download = `aa-conditional-${analysisReport.job_id}.json`; a.click(); URL.revokeObjectURL(url);
});
