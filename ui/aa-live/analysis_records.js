"use strict";
// Saved analysis records: freeze the CURRENT completed analysis, list what is
// saved, and reopen it. Reopening reads history only - it never calls the kernel
// and never rewrites a record with the current table rules.
//
// The numbers shown here are the ones the server re-derived from the frozen
// exact values; the panel never reads a saved display string as truth.
let recordsToken = 0, recordsList = [], openedRecord = null;
const recordFeedback = (message, isError) => {
  el("records-status").textContent = message;
  el("records-status").className = isError ? "error" : "muted";
};
const STATUS_LABELS = {CURRENT: "当前版本", HISTORICAL_RULES: "历史规则版本",
                       HISTORICAL_IMPLEMENTATION: "历史实现版本",
                       INVALID: "未通过自洽校验"};
const STATUS_CLASSES = {CURRENT: "tag", HISTORICAL_RULES: "tag",
                        HISTORICAL_IMPLEMENTATION: "tag", INVALID: "tag blocked"};
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
        },
        label: el("records-label").value.trim(),
      })});
    const body = await response.json();
    if (!response.ok) throw Error(text(body.detail));
    recordFeedback(body.duplicate_of_existing
      ? `这次分析已经保存过：记录 ${body.record_id}（不会重复新增）。`
      : `已保存本次分析：记录 ${body.record_id}。`, false);
    await recordsRefresh();
  } catch (error) {
    recordFeedback(`保存失败：${error.message}`, true);
  } finally {
    el("records-save").disabled = false;
  }
}
async function recordsOpen() {
  const recordId = el("records-select").value;
  if (!recordId) { recordFeedback("先选择一条已保存的分析。", true); return; }
  try {
    const response = await fetch(
      `/api/analysis/records/${encodeURIComponent(recordId)}`, {cache: "no-store"});
    const body = await response.json();
    if (!response.ok) throw Error(text(body.detail));
    openedRecord = body;
    renderRecord(body.view, body);
    el("records-recompute-saved").disabled = !body.display_permitted;
    el("records-recompute-current").disabled = !body.display_permitted;
    recordFeedback(body.display_permitted
      ? "已打开保存的历史结果（未重新计算）。"
      : "这份记录未通过自洽校验，已拒绝展示数值；文件保持原样。", !body.display_permitted);
  } catch (error) {
    openedRecord = null; el("records-view").replaceChildren();
    recordFeedback(`打开失败：${error.message}`, true);
  }
}
async function recordsRecompute(mode) {
  const recordId = el("records-select").value;
  if (!recordId) { recordFeedback("先选择一条已保存的分析。", true); return; }
  try {
    const response = await fetch(
      `/api/analysis/records/${encodeURIComponent(recordId)}/scenario`,
      {cache: "no-store"});
    const body = await response.json();
    if (!response.ok) throw Error(text(body.detail));
    if (mode === "saved") {
      // Saved conditions: load the frozen document verbatim, so the sent rules
      // are the saved ones. Nothing about the global table rules changes.
      el("analysis-kind").value = "threeway";
      el("analysis-use-rules").checked = false;
      el("analysis-input").value = JSON.stringify(body.input, null, 2);
      if (typeof analysisEdited === "function") analysisEdited();
      el("analysis-status").textContent =
        `已载入记录 ${recordId} 保存时的输入与规则（修订 `
        + `${String(body.saved_rules_revision).slice(0, 12)}）。`
        + "按「计算条件收益」会用这套保存条件重新计算；不会改动本桌规则。";
    } else {
      // Current rules: put the saved facts back in the form, so re-verifying
      // rebuilds the document against the CURRENT table rules by definition.
      if (typeof handApplyImportedFacts === "function") {
        handApplyImportedFacts(recordId, {
          facts: body.facts, gaps: ["已从保存记录载入事实；请核对后再用当前桌规核对。"],
          source: {issue_id: recordId, saved_at: body.saved_at, scope: null}});
      }
      el("analysis-status").textContent =
        `已把记录 ${recordId} 的事实载入上方表单：请点「核对输入」用当前桌规重新核对，`
        + "再计算；不会自动把桌规改回保存时的版本。";
    }
    recordFeedback(mode === "saved"
      ? "已按保存条件载入；请显式点击计算。"
      : "已载入事实到录入区；请重新核对后再计算。", false);
  } catch (error) {
    recordFeedback(`载入失败：${error.message}`, true);
  }
}
el("records-save").addEventListener("click", recordsSave);
el("records-refresh").addEventListener("click", recordsRefresh);
el("records-open").addEventListener("click", recordsOpen);
el("records-select").addEventListener("change", () => {
  el("records-view").replaceChildren(); openedRecord = null;
  el("records-recompute-saved").disabled = true;
  el("records-recompute-current").disabled = true;
});
el("records-recompute-saved").addEventListener(
  "click", () => recordsRecompute("saved"));
el("records-recompute-current").addEventListener(
  "click", () => recordsRecompute("current"));
recordsRefresh();
