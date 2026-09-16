"use strict";
// Saved analysis records: freeze the CURRENT completed analysis, list what is
// saved, reopen it, and recompute it either under the record's OWN saved rules
// or under the current table rules. Reopening reads history only - it never
// calls the kernel and never rewrites a record with the current table rules.
//
// Every request this panel makes is bound to the target it was issued for: a
// monotonic epoch plus the record id it asked for. A late, out-of-order or
// failed response therefore cannot paint another record's numbers, cannot
// restore the previous record's recompute buttons, and cannot revive a stale
// input. The panel voids its rendered context BEFORE each new request.
let recordsToken = 0, recordsList = [], openedRecord = null, recordsEpoch = 0;
const recordFeedback = (message, isError) => {
  el("records-status").textContent = message;
  el("records-status").className = isError ? "error" : "muted";
};
const STATUS_LABELS = {CURRENT: "当前版本", HISTORICAL_RULES: "历史规则版本",
                       HISTORICAL_IMPLEMENTATION: "历史实现版本",
                       INVALID: "未通过自洽校验",
                       UNVERIFIED_FORMAT: "旧格式未校验"};
const STATUS_CLASSES = {CURRENT: "tag", HISTORICAL_RULES: "tag",
                        HISTORICAL_IMPLEMENTATION: "tag", INVALID: "tag blocked",
                        UNVERIFIED_FORMAT: "tag blocked"};
// The sample type is validated by the server and shown verbatim, so a manual,
// unverified input is never displayed as if it were a real observed hand.
const SOURCE_KIND_LABELS = {
  manual_hypothesis_unverified: "手工录入/带入的假设输入（未经观测验证）",
  linked_review_record: "关联到一条已保存的复查记录",
  recomputed_from_analysis_record: "由另一条分析记录重算而来"};
const RECOMPUTE_LABELS = {saved: "保存时的规则情景", current: "本桌当前规则"};
function recordEl(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}
function recordList(title, items) {
  if (!items || !items.length) return [recordEl("p", "footnote", title + "无")];
  const list = document.createElement("ul");
  for (const item of items) list.append(recordEl("li", null, item));
  return [recordEl("p", "footnote", title), list];
}
function recordTable(headers, rows) {
  const table = recordEl("table", "analysis-table");
  const head = document.createElement("tr");
  for (const title of headers) head.append(recordEl("th", null, title));
  table.append(head);
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of row) tr.append(recordEl("td", null, value));
    table.append(tr);
  }
  return table;
}
function recordDetails(summary, nodes) {
  const details = document.createElement("details");
  details.append(recordEl("summary", null, summary));
  for (const node of nodes) details.append(node);
  return details;
}
function recordVoid(message) {
  // Invalidate every in-flight record request and drop the rendered context
  // BEFORE a new one starts: a superseded response must find nothing to restore.
  ++recordsEpoch;
  openedRecord = null;
  el("records-view").replaceChildren();
  el("records-recompute-saved").disabled = true;
  el("records-recompute-current").disabled = true;
  if (message) recordFeedback(message, false);
}
const PROVENANCE_LABELS = {observed: "结构化快照候选（未人工修改）",
                           human_confirmed: "人工填写/确认", assumed: "假设",
                           unknown: "未知"};
function recordProvenanceRows(facts) {
  const rows = [];
  for (const [key, block] of Object.entries(facts || {})) {
    if (!block || typeof block !== "object" || !("provenance" in block)) continue;
    const value = block.value;
    const shown = value === null || value === undefined ? "（未知）"
      : Array.isArray(value) ? `${value.length} 项`
      : typeof value === "object" ? "结构化" : String(value);
    const candidate = block.candidate === null || block.candidate === undefined
      ? "无" : "已保留原候选";
    rows.push([key, PROVENANCE_LABELS[block.provenance] || block.provenance,
               shown, candidate]);
  }
  return rows;
}
function recordRangeRows(assumptions) {
  const rows = [];
  for (const item of (assumptions && assumptions.ranges) || []) {
    for (const combo of item.combos || []) {
      rows.push([`座位 ${item.seat_id}`, combo.combo, combo.weight]);
    }
  }
  return rows;
}
function recordWeightRows(assumptions) {
  return ((assumptions && assumptions.models) || []).map(row =>
    [`座位 ${row.seat_id}`, translated(row.key), row.weight]);
}
function recordRuleRows(rules) {
  if (!rules || typeof rules !== "object") return [];
  return Object.entries(rules).map(([key, value]) => [
    key, value === null || value === undefined ? "（未设置）"
      : typeof value === "object" ? JSON.stringify(value) : String(value)]);
}
function renderRecord(view, envelope) {
  if (!view) {
    el("records-view").replaceChildren(
      recordEl("p", "error",
               `这份记录不能展示数值：${(envelope.notes || []).join("；")}`));
    return;
  }
  const nodes = [];
  const heading = recordEl("p", "footnote",
    `${view.label || view.record_id} · 保存于 ${view.saved_at} · `
    + `${STATUS_LABELS[view.status] || view.status} · ${view.record_kind}`);
  const tag = recordEl("span", STATUS_CLASSES[view.status] || "tag",
                       STATUS_LABELS[view.status] || view.status);
  nodes.push(heading, tag);
  for (const note of envelope.notes || []) {
    nodes.push(recordEl("p", "footnote", note));
  }
  // What kind of sample this is, in words rather than as a digest.
  nodes.push(recordEl("p", "footnote",
    `来源类型：${view.source_kind_label || SOURCE_KIND_LABELS[view.source_kind]
      || view.source_kind}（${view.source_kind}）`));
  const source = view.source || {};
  if (source.issue_id) {
    nodes.push(recordEl("p", "footnote", `关联的复查记录：${source.issue_id}`));
  }
  if (source.parent_analysis_record_id) {
    nodes.push(recordEl("p", "footnote",
      `重算来源的分析记录：${source.parent_analysis_record_id}`));
  }
  const situation = view.situation;
  nodes.push(recordEl("p", "footnote",
    `Hero 座位 ${situation.hero_seat} · 手牌 ${situation.hero_cards.join(" ")} · `
    + `公共牌 ${situation.board_cards.join(" ")} · `
    + `行动顺序 ${situation.action_order.join(" → ")}`));
  nodes.push(recordTable(["座位", "河牌起点状态", "剩余筹码", "本手已投入"],
    situation.seats.map(row => [`座位 ${row.seat_id}`,
      HAND_STATUS_LABELS[row.status] || row.status, row.stack, row.hand_committed])));
  nodes.push(recordEl("p", "footnote",
    `规则情景：来源 ${view.rules.rules_source} · 核对时修订 `
    + `${String(view.rules.rules_revision).slice(0, 12)} · 当前修订 `
    + `${String(view.rules.current_rules_revision).slice(0, 12)}`
    + (view.historical ? "（保存后桌规已变更，本结果是历史版本）" : "")));
  // The rule VALUES, not only a revision digest.
  const ruleRows = recordRuleRows(view.rules.effective_rules);
  nodes.push(recordDetails(`核对时使用的规则数值（${ruleRows.length} 项）`,
    [recordTable(["规则项", "值"], ruleRows)]));
  nodes.push(recordEl("p", "footnote",
    `容量：声明组合乘积 ${view.capacity.declared_combo_product} · `
    + `本街最高下注 ${view.capacity.current_bet} · 推算底池 ${view.capacity.implied_pot} · `
    + `Hero 应付 ${view.capacity.to_call}（单位 ${view.unit}）`));
  if (view.blockers.length) {
    nodes.push(...recordList("本次核对记录到的阻碍：", view.blockers));
  } else {
    nodes.push(recordEl("p", "footnote", "本次核对没有遗留阻碍。"));
  }
  nodes.push(recordEl("p", "footnote", "服务端实际完成结果（每个动作的本街追加金额与条件净 EV）："));
  nodes.push(recordTable(["动作", "本街追加", "条件净 EV", "精确值"],
    view.actions.map(row => [
      row.kind === "bet" || row.kind === "raise"
        ? `${translated(row.kind)} 至 ${row.target_decimal}`
        : translated(row.kind),
      row.additional_cost, row.ev, row.ev_exact])));
  // The opponent model behind those numbers, so it can be checked rather than
  // taken on trust.
  const rangeRows = recordRangeRows(view.assumptions);
  const weightRows = recordWeightRows(view.assumptions);
  nodes.push(recordDetails(
    `本次使用的对手假设（范围 ${rangeRows.length} 个组合 · 响应权重 ${weightRows.length} 条）`,
    [recordTable(["座位", "组合", "相对权重"], rangeRows),
     recordTable(["座位", "动作", "权重"], weightRows),
     recordEl("p", "footnote",
              `加注尺寸 ${(view.assumptions.aggression_targets || []).join(",") || "（无）"}`
              + ` · 加注次数上限 ${view.assumptions.max_aggressions}`
              + ` · 额外费用 ${JSON.stringify(view.assumptions.other_fees)}`)]));
  if (view.history.length) {
    nodes.push(...recordList("公开历史标签：", view.history.map(row => row.label)));
  } else {
    nodes.push(recordEl("p", "footnote", "公开历史：已确认为「本手没有任何公开行动」。"));
  }
  nodes.push(recordEl("p", "footnote", "事实与来源（不会把未知变成确认值）："));
  nodes.push(recordTable(["事实块", "来源", "值", "原候选"],
                         recordProvenanceRows(view.facts)));
  nodes.push(recordEl("p", "footnote", view.note));
  el("records-view").replaceChildren(...nodes);
}
async function recordsRefresh() {
  const token = ++recordsToken;
  try {
    const response = await fetch("/api/analysis/records", {cache: "no-store"});
    const body = await response.json();
    if (token !== recordsToken) return;
    if (!response.ok) throw Error(text(body.detail));
    recordsList = body.items || [];
    const select = el("records-select");
    // The selection is read HERE, at the moment the answer is applied: a save
    // finishes with an automatic refresh, and the human may already have moved
    // to another record while it was in flight. Their choice must survive.
    const keep = select.value;
    select.replaceChildren();
    if (!recordsList.length) {
      const option = document.createElement("option");
      option.value = ""; option.textContent = "暂无已保存的分析";
      select.append(option);
    }
    for (const row of recordsList) {
      const option = document.createElement("option");
      option.value = row.record_id;
      option.textContent = `${row.label || row.record_id} · ${row.saved_at || "?"}`
        + ` · ${STATUS_LABELS[row.status] || row.status}`;
      option.dataset.permitted = row.display_permitted ? "true" : "false";
      select.append(option);
    }
    if (keep && recordsList.some(row => row.record_id === keep)) select.value = keep;
    el("records-open").disabled = !recordsList.length;
    el("records-count").textContent = `${recordsList.length} 条`;
    recordFeedback(recordsList.length
      ? "已保存的分析记录如下；打开不会重新计算，也不改当前桌规。"
      : "还没有保存过分析：先在上方核对并计算一次，再点「保存本次分析」。", false);
  } catch (error) {
    if (token === recordsToken) recordFeedback(`记录列表读取失败：${error.message}`, true);
  }
}
async function recordsSave() {
  const receipt = typeof handReceipt !== "undefined" ? handReceipt : null;
  const binding = typeof analysisBinding !== "undefined" ? analysisBinding : null;
  const accepted = typeof acceptedAnalysisId !== "undefined" ? acceptedAnalysisId : null;
  if (!receipt || !binding || !accepted) {
    recordFeedback("请先核对牌局并在下方算出完成结果，再保存这次分析。", true);
    return;
  }
  if (!receipt.input_sha256
      || binding.expected_input_sha256 !== receipt.input_sha256
      || binding.input_sha256 !== receipt.input_sha256) {
    recordFeedback("当前结果与核对过的输入身份不一致，已拒绝保存：请重新核对并计算。", true);
    return;
  }
  el("records-save").disabled = true;
  // The save is bound to the analysis it was issued for: if the panel has moved
  // on by the time the answer lands, the receipt must not be reported as if it
  // described the CURRENT result.
  const sentBinding = binding, sentJob = accepted;
  const parent = receipt.scenario_record_id
    || (typeof handSource !== "undefined" ? handSource.parent_analysis_record_id
      : null) || null;
  try {
    const response = await fetch("/api/analysis/records", {
      method: "POST", headers: {...headers, "Content-Type": "application/json"},
      body: JSON.stringify({
        job_id: accepted, input_sha256: receipt.input_sha256,
        facts: receipt.facts, assumptions: receipt.assumptions,
        source: {
          issue_id: handSource.issue_id, saved_at: handSource.saved_at,
          preview_sha256: handSource.preview_sha256,
          source_frame: handSource.source_frame, scope: handSource.scope,
          parent_analysis_record_id: parent,
        },
        label: el("records-label").value.trim(),
      })});
    const body = await response.json();
    const moved = typeof acceptedAnalysisId !== "undefined"
      && (acceptedAnalysisId !== sentJob || analysisBinding !== sentBinding);
    if (!response.ok) throw Error(text(body.detail));
    if (body.display_permitted !== true || !body.record_id) {
      recordFeedback(`保存回执未通过自洽校验，已按失败处理：`
                     + `${(body.notes || []).join("；")}`, true);
      await recordsRefresh();
      return;
    }
    if (moved) {
      recordFeedback(`已保存的是发起保存时的那次分析（记录 ${body.record_id}）；`
                     + "当前面板已经换成了另一次分析，这条记录不属于它。", true);
    } else {
      recordFeedback(body.duplicate_of_existing
        ? `这次分析已经保存过：记录 ${body.record_id}（不会重复新增）。`
        : `已保存本次分析：记录 ${body.record_id}。`, false);
    }
    await recordsRefresh();
  } catch (error) {
    recordFeedback(`保存失败：${error.message}`, true);
  } finally {
    el("records-save").disabled = false;
  }
}
async function recordsOpen() {
  const targetId = el("records-select").value;
  if (!targetId) { recordFeedback("先选择一条已保存的分析。", true); return; }
  // Void first: whatever is on screen belongs to another target now, and its
  // recompute buttons must not survive a failed or superseded open.
  recordVoid("正在打开保存的历史结果……");
  const epoch = recordsEpoch;
  try {
    const response = await fetch(
      `/api/analysis/records/${encodeURIComponent(targetId)}`, {cache: "no-store"});
    const body = await response.json();
    if (epoch !== recordsEpoch) return;
    if (el("records-select").value !== targetId) {
      recordFeedback("选择已经改变，这次打开的回包不再对应当前选择，已忽略。", true);
      return;
    }
    if (!response.ok) throw Error(text(body.detail));
    if (body.record_id !== targetId) {
      recordFeedback("返回的记录与请求的目标不是同一条，这次打开已放弃：不会展示任何数值。",
                     true);
      return;
    }
    openedRecord = body;
    renderRecord(body.view, body);
    el("records-recompute-saved").disabled = !body.display_permitted;
    el("records-recompute-current").disabled = !body.display_permitted;
    recordFeedback(body.display_permitted
      ? "已打开保存的历史结果（未重新计算）。"
      : "这份记录未通过自洽校验，已拒绝展示数值；文件保持原样。", !body.display_permitted);
  } catch (error) {
    if (epoch === recordsEpoch) recordFeedback(`打开失败：${error.message}`, true);
  }
}
async function recordsRecompute(mode) {
  const targetId = el("records-select").value;
  const parent = openedRecord;
  if (!targetId) { recordFeedback("先选择一条已保存的分析。", true); return; }
  if (!parent || parent.record_id !== targetId || !parent.display_permitted) {
    recordFeedback("请先打开这条记录：只有通过自洽校验、且仍被选中的记录可以用来重算。",
                   true);
    return;
  }
  // Void FIRST. The previous hand's fields, opponent assumptions, verified
  // receipt and rendered result are all dropped before the request leaves, so a
  // superseded answer can never be loaded into a form that belongs to A while
  // the receipt still belongs to B.
  recordVoid(`正在按${RECOMPUTE_LABELS[mode]}重算：上一套字段、假设与下方旧结果已全部作废。`);
  try {
    if (typeof handResetForSource === "function") {
      handResetForSource({
        kind: mode === "saved" ? "analysis_record_saved_rules"
                               : "analysis_record_current_rules",
        issue_id: null, saved_at: parent.saved_at ?? null,
        preview_sha256: null, source_frame: null, scope: null,
      }, `正在按${RECOMPUTE_LABELS[mode]}重算记录 ${targetId}：`
         + "上一套字段、假设、结束确认与下方旧结果已全部作废。");
    }
  } catch (_) { /* hand_input.js not loaded: the record panel still stands alone */ }
  const epoch = recordsEpoch;
  // The form the answer will be loaded into: an edit, a clear or a rules change
  // during the flight bumps this, and then the answer is dropped instead of
  // being written over what the human now has in front of them.
  const formToken = typeof handToken !== "undefined" ? handToken : null;
  try {
    const response = await fetch(
      `/api/analysis/records/${encodeURIComponent(targetId)}/scenario`, {cache: "no-store"});
    const body = await response.json();
    if (epoch !== recordsEpoch) return;                        // a newer target owns it
    if (el("records-select").value !== targetId) {
      recordFeedback("选择已经改变，这次重算材料的回包不再对应当前选择，已忽略。", true);
      return;
    }
    if (formToken !== null && handToken !== formToken) {
      recordFeedback("这次重算的回包到达时录入区已经被改动（清空 / 改表单 / 换规则）："
                     + "已丢弃该回包，不会把它载入表单。", true);
      return;
    }
    if (!response.ok) throw Error(text(body.detail));
    if (body.record_id !== targetId
        || !body.facts || !body.assumptions || !body.input) {
      recordFeedback("返回的重算材料与请求的目标不是同一条，这次重算已放弃："
                     + "不会改动表单，也不会展示任何数值。", true);
      return;
    }
    if (typeof handLoadRecordScenario !== "function") {
      recordFeedback("当前页面没有加载录入表单，无法把这条记录载入重算。", true);
      return;
    }
    // The ORIGINAL review link travels with the recompute; the analysis-record
    // id stays in its own field and is never used as an observed-review id.
    const reviewIssueId = ((parent.view || {}).source || {}).issue_id ?? null;
    handLoadRecordScenario(targetId, body, mode, reviewIssueId);
    recordFeedback(`已按${RECOMPUTE_LABELS[mode]}载入记录 ${targetId}；`
                   + `请点「核对输入」用${mode === "saved" ? "记录里的规则情景" : "本桌当前规则"}`
                   + "重新核对，再计算并保存为新记录。"
                   + (mode === "saved"
                      ? "若要改用本桌规则，请在录入区勾选「使用本桌已保存规则」。"
                      : ""), false);
  } catch (error) {
    if (epoch === recordsEpoch) {
      recordFeedback(`重算材料载入失败：${error.message}`, true);
    }
  }
}
el("records-save").addEventListener("click", recordsSave);
el("records-refresh").addEventListener("click", recordsRefresh);
el("records-open").addEventListener("click", recordsOpen);
el("records-select").addEventListener("change", () => {
  // Changing the target voids the open record AND any in-flight open/recompute.
  recordVoid("已切换记录：上一条的数值与重算按钮已失效，请重新打开。");
});
el("records-recompute-saved").addEventListener(
  "click", () => recordsRecompute("saved"));
el("records-recompute-current").addEventListener(
  "click", () => recordsRecompute("current"));
recordsRefresh();
