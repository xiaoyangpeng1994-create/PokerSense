"use strict";
let analysisEpoch = 0, acceptedAnalysisId = null, analysisReport = null, analysisInput = null;
let analysisBusy = false, cancelTimer = null, analysisBinding = null;
let analysisDraftSource = null, analysisExpectedInput = null;
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
// Any input change goes through one path: drop the identity, hide the result,
// and cancel on a short debounce rather than immediately. The debounce is what
// keeps a stale cancel from landing after a new analysis has already started;
// the panel's own start handler clears it.
function analysisInputChanged(reason) {
  analysisExpectedInput = null; analysisDraftSource = null;
  invalidateAnalysis(reason);
  cancelTimer = setTimeout(cancelAnalysis, 350);
}
function analysisEdited() {
  analysisInputChanged("手工输入已变更，旧分析已失效。");
}
function exactAmount(value) { return value?.decimal ?? value?.exact ?? text(value); }
function actionName(action) {
  if (typeof action === "string") return translated(action.toLowerCase());
  return `${translated(action?.kind)}${["bet", "raise"].includes(action?.kind) && action?.target != null ? ` 至 ${action.target}` : ""}`;
}
const analysisHex = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
function renderAnalysis(report, state) {
  if (acceptedAnalysisId === null) return;
  if (!analysisBinding || state && (analysisBinding.generation !== state.generation || analysisBinding.table_rules_revision !== state.table_rules?.revision)) {
    invalidateAnalysis("来源或本桌规则已变更，旧分析已失效。"); return;
  }
  const expected = analysisBinding.expected_input_sha256;
  if (expected !== null && expected !== undefined && expected !== analysisExpectedInput) {
    invalidateAnalysis("录入表单的输入身份已变更，旧分析已失效，请重新核对后计算。"); return;
  }
  if (!report || report.job_id !== acceptedAnalysisId
      || report.kind !== analysisBinding.kind
      || report.binding?.generation !== analysisBinding.generation
      || report.binding?.table_rules_revision !== analysisBinding.table_rules_revision
      || report.binding?.rules_source !== analysisBinding.rules_source) {
    invalidateAnalysis("分析已被其他操作替换，请重新计算。"); return;
  }
  // The identity of the input has to survive all the way to the rendered report:
  // the digest this panel accepted, the digest the report states and - when the
  // computation came from the entry form - the digest that form verified must be
  // the same non-empty value. Three hashes that merely exist are not a check, and
  // a missing one must not be skipped either.
  const accepted = analysisBinding.input_sha256, reported = report.input_sha256;
  if (!analysisHex(accepted) || !analysisHex(reported) || reported !== accepted
      || (expected !== null && expected !== undefined
          && (!analysisHex(expected) || accepted !== expected))) {
    invalidateAnalysis("这条结果无法与核对过的输入对上（身份缺失或不一致），已失效，请重新核对后计算。"); return;
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
  analysisDraftSource = null;
  analysisExpectedInput = null;
  el("strategy-draft-details").hidden = true;
  el("strategy-draft-origin").textContent = "";
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
  const expectedInput = analysisExpectedInput;
  const expectedPayload = el("analysis-input").value;
  try {
    await cancelling; if (epoch !== analysisEpoch) return;
    if (expectedInput !== analysisExpectedInput
        || el("analysis-input").value !== expectedPayload) {
      el("analysis-status").textContent = "输入在开始计算前被改动，请重新核对后再计算。"; return;
    }
    if (expectedInput !== null
        && (!analysisDraftSource || analysisDraftSource.input_sha256 !== expectedInput)) {
      el("analysis-status").textContent = "录入表单的身份与本场景不匹配：请回到录入区重新核对后计算。"; return;
    }
    if (expectedInput !== null && el("analysis-use-rules").checked) {
      el("analysis-status").textContent = "本桌规则会覆盖这个场景的规则，无法与录入表单核对同一份输入：请取消勾选「使用本桌规则」后重新核对。"; return;
    }
    const document = JSON.parse(el("analysis-input").value);
    const r = await fetch("/api/analysis", {method:"POST",headers:{...headers,"Content-Type":"application/json"},body:JSON.stringify({kind:el("analysis-kind").value,document,rules_source:el("analysis-use-rules").checked ? "table" : "document",rules_revision:statusData.table_rules?.revision})});
    const result = await r.json(); if (epoch !== analysisEpoch) return; if (!r.ok) throw Error(text(result.detail));
    const reported = analysisHex(result.input_sha256) ? result.input_sha256 : null;
    if (expectedInput !== null && reported !== expectedInput) {
      // The document really sent did not hash to the identity the form verified,
      // so this result does not belong to the input the human checked.
      el("analysis-status").textContent = "后端收到的输入与录入表单核对过的输入不是同一份，已拒绝这次计算：请回到录入区重新核对。";
      return;
    }
    acceptedAnalysisId = result.job_id; analysisInput = document;
    analysisBinding = JSON.parse(JSON.stringify(result.binding));
    analysisBinding.kind = result.kind ?? el("analysis-kind").value;
    analysisBinding.expected_input_sha256 = expectedInput;
    analysisBinding.input_sha256 = reported;
    renderAnalysis(result);
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
  const data = JSON.stringify({effective_input:effectiveInput,report:analysisReport,draft_source:analysisDraftSource},null,2);
  const url = URL.createObjectURL(new Blob([data],{type:"application/json"})); const a = document.createElement("a"); a.href = url; a.download = `aa-conditional-${analysisReport.job_id}.json`; a.click(); URL.revokeObjectURL(url);
});

function terminalDraftFromReview(record) {
  const issue=record.issue, snapshot=issue.observation, row=snapshot.payload || {};
  const observed=row.observed_state_v2 || {}, table=issue.table_rules || {};
  const currentLedger=row.hand_phase?.current_ledger;
  const ledgerOK=currentLedger?.status === "OBSERVED_HAND_COMMITMENTS_CANDIDATE" && !!observed.observed_epoch && currentLedger.epoch === observed.observed_epoch && Array.isArray(currentLedger.taint_reasons) && currentLedger.taint_reasons.length === 0;
  const commitments=ledgerOK ? currentLedger.hand_commitments : null;
  const wagers=row.causal_street_wagers_v2 || {};
  const wagerOK=wagers.status === "OBSERVED_STREET_WAGERS_CANDIDATE" && wagers.title_center_ledger_reconciled === true;
  const candidate = value => typeof value === "string" && /^\d+(?:\.\d+)?$/.test(value) ? value : null;
  const rules=table.conditional_analysis_ready === true && table.simulation_rules ? JSON.parse(JSON.stringify(table.simulation_rules)) : {
    schema_version:2,table_size:null,small_blind:null,big_blind:null,ante:null,
    ante_mode:null,straddle_mode:null,straddle_amount:null,rake_percent:null,
    rake_cap_bb:null,rake_application:null,rake_rounding:null,rake_distribution:null,
    minimum_chip:null,verification_status:"simulation",source:"manual-frozen-frame-assumptions-pending"
  };
  const seats=[];
  for(let seat=0;seat<8;seat++) seats.push({seat_id:seat,stack:candidate(row.stacks?.[seat]?.value),street_committed:wagerOK ? candidate(wagers.wagers?.[seat]) : null,hand_committed:candidate(commitments?.[seat]),status:({active:"ACTIVE",folded:"FOLDED",all_in:"ALL_IN"})[observed.participants?.[seat]?.state] || null});
  const document={schema_version:1,mode:"manual_hypothesis",range_assumptions:"待填写对手具体组合与权重；截图候选尚需人工核对，不是已验证范围或实战建议。",hero_seat:4,actor_seat:Number.isInteger(row.current_actor) ? row.current_actor : null,hero_cards:row.cards?.hero || [null,null],board_cards:observed.street_candidate === "river" ? row.cards?.board_slots || [] : [],current_bet:wagerOK ? candidate(wagers.street_price) : null,pot_before:candidate(row.pot?.value),other_fees:null,split_policy:"fractional_equal_split",rules,seats,ranges:[]};
  const missing=["对手具体组合、权重及范围假设", "其他费用（未知保留 null）", "核对实际发牌座位，删除未入局物理座位；不能把空座当弃牌"];
  if(observed.street_candidate!=="river")missing.unshift("当前不是完整河牌，终结计算暂不适用");
  if(row.current_actor!==4)missing.unshift("当前行动者不是 Hero 或未知，终结计算暂不适用");
  if(!table.conditional_analysis_ready)missing.push("桌规不完整或有未支持特殊规则；待明确人工假设");
  if(!wagerOK)missing.push("本街完整投入与当前最高下注额");
  if(!ledgerOK)missing.push("整手投入账本（历史结算账本不能当本手投入）");
  else if(currentLedger.unallocated_difference !== "0")missing.push("投入与显示底池尚有未解释差额，不能推断成抽水");
  if(seats.some(seat=>seat.status===null || seat.stack===null))missing.push("各参与座位状态与剩余筹码");
  missing.push("仅支持 Hero 唯一可行动、2–7名已全下对手；其他场景不能套用");
  return JSON.parse(JSON.stringify({document,missing,source:{issue_id:issue.issue_id,saved_at:issue.saved_at,preview_sha256:issue.preview_sha256,instance_id:snapshot.instance_id,generation:snapshot.generation,source_frame:snapshot.source_frame,sequence:snapshot.sequence,observed_epoch:observed.observed_epoch,table_rules_revision:table.revision,scope:"FROZEN_CANDIDATES_MANUAL_REVIEW_REQUIRED"}}));
}
async function openStrategyDraft(record) {
  await cancelAnalysis();
  analysisExpectedInput = null;
  const draft=terminalDraftFromReview(record); analysisDraftSource=draft.source;
  el("analysis-kind").value="terminal";el("analysis-use-rules").checked=false;
  el("analysis-input").value=JSON.stringify(draft.document,null,2);
  el("strategy-draft-details").hidden=false;
  el("strategy-draft-origin").textContent=`固定来源帧 ${text(draft.source.source_frame)} · ${draft.source.issue_id} · 人工待核草稿，不是当前行动建议`;
  el("strategy-draft-missing").replaceChildren(...draft.missing.map(value=>{const li=document.createElement("li");li.textContent=value;return li;}));
  el("analysis-status").textContent="已带入该帧的可见候选。补齐并核对缺项后可使用已有条件计算；当前草稿尚不能计算。";
  showDesk("settings");el("strategy-tools").open=true;
}
el("strategy-from-review").addEventListener("click",()=>{if(selectedReview)openStrategyDraft(selectedReview);});
el("strategy-mark-current").addEventListener("click",async()=>{
  if(el("strategy-mark-current").disabled)return;
  el("strategy-mark-current").disabled=true;
  try {const record=await deskRequest("/api/review/mark",{});await openStrategyDraft(record);}
  catch(_){el("strategy-live-feedback").textContent="策略草稿未建立，请检查来源或从复查记录重试。";}
});
setInterval(()=>{el("strategy-mark-current").disabled=statusData.status!=="RUNNING" || !statusData.payload || !statusData.issue_recording_available;el("strategy-from-review").disabled=!selectedReview;},800);
