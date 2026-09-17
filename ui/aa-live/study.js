"use strict";
// Offline study result view: one controlled SYNTHETIC example, its saved records,
// and nothing else. It never reads an observed hand, never calls the vision API,
// never opens a capture device and never produces a live action.
//
// Pure helpers are separated from DOM wiring so the flow can be exercised by a
// real interaction test without a browser.
let studyToken = 0, studyExamples = [], studyRecords = [], studyCurrent = null;
let studyWired = false, studyBusy = false;
const studyStatusLabels = {
  PREVIEW_NOT_SAVED: "示例预览 · 未保存",
  CURRENT: "已保存 · 当前版本",
  HISTORICAL_UNVERIFIED: "历史记录 · 来源无法再核对",
  INVALID: "无效记录 · 不可作为结果"
};
const studyOutcomeLabels = {frozen_policy:"冻结策略", check_fold:"过牌/弃牌基线", check_call:"过牌/跟注基线"};
const studyGateLabels = {
  selected_minus_manual_reference: "差 vs 手工参考",
  selected_delta_vs_check_fold: "差 vs 过牌/弃牌",
  selected_delta_vs_check_call: "差 vs 过牌/跟注"
};
const studyReachLabels = {
  REACHED_BY_LEFT_ONLY: "仅冻结策略侧可达",
  REACHED_BY_RIGHT_ONLY: "仅手工参考侧可达",
  REACHED_BY_BOTH: "两侧均可达",
  UNREACHABLE_BY_BOTH: "两侧均不可达"
};
function studyStatusLabel(status) { return studyStatusLabels[status] || text(status); }
function studyAmount(value) { return value ? `${value.display} ${value.unit}` : "未知"; }
function studyExact(value) { return value ? value.exact : "未定义"; }
function studyAcceptsTarget(activeTarget, responseTarget) {
  return typeof activeTarget === "string" && activeTarget !== ""
    && activeTarget === responseTarget;
}
function studyDisplayable(record) {
  return !!record && record.display_permitted === true && !!record.view
    && (record.content_status === "CURRENT"
        || record.content_status === "PREVIEW_NOT_SAVED");
}
function studyTargetOf(record) {
  if (!record) return null;
  return record.record_id || (record.identity && record.identity.example_id) || null;
}
function studyIdentityLabel(record) {
  const identity = record && record.identity;
  if (!identity) return "身份不可用";
  return `示例 ${identity.example_id} · 报告 ${String(identity.report_sha256 || "").slice(0, 12)}…`
    + ` · 策略本 ${Object.values(identity.policy_book_sha256 || {}).map(
        value => String(value).slice(0, 8) + "…").join(" / ")}`;
}
function studyAcceptance(activeToken, responseToken) { return activeToken === responseToken; }
function studyVersionOf(record) {
  return record?.content_status === "SUPERSEDED_SOURCE" ? "HISTORICAL_ONLY" : "CURRENT_ONLY";
}
function studyText(record) {
  if (!record || !record.view) return ["无结果"];
  const view = record.view, source = view.source, lines = [];
  lines.push(`${source.title} · ${source.scope_label}`);
  lines.push(`记录编号：${record.record_id || "未保存"} · 状态：${studyStatusLabel(record.content_status)}`);
  lines.push(`研究报告 ${source.report_sha256}（提交 ${source.parent_head}）`);
  lines.push(`研究协议 ${source.protocol_sha256} · 输入 ${source.input_sha256}`);
  for (const book of source.frozen_books || []) {
    lines.push(`${book.label}：${book.policy_book_sha256} · 规划最优 ${studyAmount(book.planning_best_ev_chips)} · 决策 ${book.decisions} 条`);
  }
  const varied = source.varied_object || {};
  lines.push(`唯一变化对象：座位 ${varied.target_opponent_seat} 的组合 ${(varied.target_combos || []).join("/")}，相对权重因子 ${(varied.relative_factors || []).join(" / ")}`);
  for (const factor of view.factors || []) {
    lines.push(`${factor.sentence}`);
    lines.push(`  历史似然 ${studyAmount(factor.history_likelihood)} · 联合分布 ${factor.joint_assignments} 项 · 非零后验 ${factor.nonzero_posterior_entries} 项`);
    lines.push(`  触发失败闸门：${(factor.failure_gates || []).map(name => studyGateLabels[name] || name).join("、") || "无"}`);
    for (const row of factor.paths || []) {
      lines.push(`  路径差 ${studyAmount(row.difference)}（${studyReachLabels[row.reach_status] || row.reach_status}）${row.history_key}`);
    }
  }
  for (const item of view.missing || []) lines.push(`缺失项：${item}`);
  for (const item of view.claims || []) lines.push(`说明：${item}`);
  return lines;
}
function studyElement(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}
function studyTable(headers, rows) {
  const table = studyElement("table", "analysis-table");
  const head = document.createElement("tr");
  for (const title of headers) head.append(studyElement("th", null, title));
  table.append(head);
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of row) tr.append(studyElement("td", null, value));
    table.append(tr);
  }
  return table;
}
function studyBlock(record) {
  const view = record.view, source = view.source, nodes = [];
  nodes.push(studyElement("p", "footnote", `${source.scope_label} · 记录编号 ${record.record_id || "未保存"} · ${studyStatusLabel(record.content_status)}`));
  if (record.reason) nodes.push(studyElement("p", "footnote", record.reason));
  nodes.push(studyElement("p", "footnote", `来源：${source.title} · 研究报告 ${source.report_sha256.slice(0, 16)}… · 研究协议 ${source.protocol_sha256.slice(0, 16)}… · 输入 ${source.input_sha256.slice(0, 16)}…`));
  const books = (source.frozen_books || []).map(book => `${book.label}（规划最优 ${studyAmount(book.planning_best_ev_chips)}，决策 ${book.decisions} 条）`).join("；");
  nodes.push(studyElement("p", "footnote", `固定策略本（三世界复用，未重规划）：${books}`));
  const varied = source.varied_object || {};
  nodes.push(studyElement("p", "footnote", `唯一变化对象：座位 ${varied.target_opponent_seat} 的组合 ${(varied.target_combos || []).join("/")}，相对权重因子 ${(varied.relative_factors || []).join(" / ")}；其余范围、响应模型、桌规、公牌与行动网格未改。`));
  const context = view.table_context;
  if (context) {
    nodes.push(studyElement("p", "footnote", `实验局面（${context.scope_label}）：Hero 座位 ${context.hero_seat} · 手牌 ${(context.hero_cards || []).join(" ")} · 公牌 ${(context.board_cards || []).join(" ")}`));
    const history = (context.history || []).map(item => `座位${item.actor} ${item.kind === "bet" || item.kind === "raise" ? `${item.kind} ${item.target}` : item.kind}`);
    nodes.push(studyElement("p", "footnote", `公开历史：${history.length ? history.join(" → ") : "无"}；底池 ${studyAmount(context.pot_at_decision)}（各座已投入 ${context.pot_components.committed_before_street} + 本街历史投入 ${context.pot_components.street_wagers_from_history}，${context.pot_components.definition}）；决策点需跟注 ${studyAmount(context.to_call)}`));
    nodes.push(studyElement("p", "footnote", "两套固定策略在根节点的动作（合成假设，不是实战指令）："));
    nodes.push(studyTable(["策略本", "根动作", "本街追加", "金额口径"],
      (context.root_actions || []).map(item => [
        item.label,
        item.action.kind === "bet" || item.action.kind === "raise"
          ? `加注到 ${item.action.target}` : item.action.kind === "call" ? "跟注" : item.action.kind,
        `${item.additional_chips} ${context.unit}`,
        item.amount_reading])));
  }
  const rows = (view.factors || []).map(factor => [
    factor.factor_label + (factor.is_baseline ? "（原基准）" : ""),
    studyAmount((factor.outcomes || []).find(item => item.name === "frozen_policy")?.net_ev_chips),
    studyAmount(factor.reference_frozen_policy),
    studyAmount(factor.deltas.vs_manual_reference),
    studyAmount(factor.deltas.vs_check_fold),
    studyAmount(factor.deltas.vs_check_call),
    studyAmount(factor.history_likelihood)
  ]);
  nodes.push(studyTable(["因子", "冻结策略 EV", "手工参考 EV", "差 vs 手工参考", "差 vs 过牌/弃牌", "差 vs 过牌/跟注", "历史似然"], rows));
  for (const factor of view.factors || []) {
    const details = document.createElement("details");
    details.append(studyElement("summary", null, `${factor.factor_label} 精确分数与主要终局路径`));
    details.append(studyElement("p", "footnote", factor.sentence));
    const exactRows = (factor.outcomes || []).map(item => [item.label, studyExact(item.net_ev_chips), studyAmount(item.net_ev_chips)]);
    exactRows.push(["手工参考策略", studyExact(factor.reference_frozen_policy), studyAmount(factor.reference_frozen_policy)]);
    exactRows.push(["差 vs 手工参考", studyExact(factor.deltas.vs_manual_reference), studyAmount(factor.deltas.vs_manual_reference)]);
    exactRows.push(["fallback 概率", studyExact((factor.outcomes || [])[0]?.fallback), studyAmount((factor.outcomes || [])[0]?.fallback)]);
    details.append(studyTable(["项目（精确分数仅供核对）", "精确分数", "易读值"], exactRows));
    if ((factor.paths || []).length) {
      details.append(studyElement("p", "footnote", `主要终局路径贡献差（共 ${factor.rows_total} 条终局路径，此处只列非零的 ${factor.paths.length} 条）：`));
      details.append(studyTable(["路径（公开历史）", "归属", "贡献差"], (factor.paths || []).map(row => [
        row.history_key, studyReachLabels[row.reach_status] || row.reach_status, studyAmount(row.difference)])));
    }
    nodes.push(details);
  }
  const missing = studyElement("ul", null);
  for (const item of view.missing || []) missing.append(studyElement("li", null, item));
  nodes.push(studyElement("p", "footnote", "缺失项（不能编造参数补齐）："), missing);
  const claims = studyElement("ul", null);
  for (const item of view.claims || []) claims.append(studyElement("li", null, item));
  nodes.push(studyElement("p", "footnote", "固定说明（由数值生成，不由模型临时解释）："), claims);
  return nodes;
}
function studyNotice(record) {
  return [
    studyElement("p", "footnote", `记录编号 ${(record && record.record_id) || "未保存"} · ${studyStatusLabel(record && record.content_status)}`),
    studyElement("p", "error", (record && record.reason) || "该记录没有可展示的已验证结果。"),
    studyElement("p", "footnote", `身份：${studyIdentityLabel(record)}`),
    studyElement("p", "footnote", "原始文件已保留；这里不展示任何数值，也不允许保存或比较。")
  ];
}
function renderStudy(record, token, expectedTarget) {
  if (!studyAcceptance(studyToken, token)) return false;
  if (!studyAcceptsTarget(expectedTarget, studyTargetOf(record))) {
    invalidateStudy("回包与请求的目标不一致，结果已丢弃。", true);
    return false;
  }
  const content = el("study-content");
  el("study-source-tag").textContent = studyStatusLabel(record.content_status);
  if (!studyDisplayable(record)) {
    studyCurrent = null;
    content.replaceChildren(...studyNotice(record));
    content.hidden = false;
    el("study-raw").textContent = JSON.stringify({
      source_type: record.source_type, content_status: record.content_status,
      display_permitted: record.display_permitted, reason: record.reason,
      identity: record.identity}, null, 2);
    el("study-save").disabled = true;
    return true;
  }
  studyCurrent = record;
  content.replaceChildren(...studyBlock(record));
  content.hidden = false;
  el("study-raw").textContent = JSON.stringify({source_type: record.source_type, content_status: record.content_status, identity: record.identity, view: record.view}, null, 2);
  return true;
}
function studyFeedback(message, isError) {
  el("study-feedback").textContent = message;
  el("study-error").hidden = !isError;
  if (isError) el("study-error").textContent = message;
}
function invalidateStudy(message, isError = false) {
  ++studyToken;  // anything already in flight is stale from here on
  studyCurrent = null;
  const content = el("study-content");
  content.replaceChildren(); content.hidden = true;
  el("study-raw").textContent = "无结果";
  el("study-save").disabled = true;
  el("study-source-tag").textContent = "尚未载入";
  if (message) studyFeedback(message, isError);
}
async function studyRequest(path, body) {
  const options = {headers, cache: "no-store"};
  if (body !== undefined) Object.assign(options, {method:"POST", headers:{...headers, "Content-Type":"application/json"}, body:JSON.stringify(body)});
  const response = await fetch(path, options);
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw Error(typeof result.detail === "string" ? result.detail : "研究记录请求未完成");
  return result;
}
async function studyInit() {
  try {
    const data = await studyRequest("/api/study/examples");
    studyExamples = data.items || [];
    const select = el("study-example");
    select.replaceChildren();
    for (const item of studyExamples) {
      const option = document.createElement("option");
      option.value = item.example_id;
      option.textContent = `${item.title}（合成示例 · 非真实牌局）${item.available ? "" : " · 当前不可用"}`;
      select.append(option);
    }
    el("study-load").disabled = studyExamples.filter(item => item.available).length === 0;
    studyFeedback("已登记 " + studyExamples.length + " 个合成研究示例；载入只读预览，不会带入任何观测牌局。", false);
  } catch (error) { studyFeedback(`示例列表未载入：${error.message}`, true); }
}
async function studyLoadExample() {
  return studyLoadExampleFor(el("study-example").value);
}
async function studyLoadExampleFor(exampleId) {
  if (!exampleId) return;
  // Invalidate first: the previous content must not survive the new request,
  // and the token guard keeps a slow earlier response from painting over it.
  invalidateStudy("正在读取新的示例来源，旧结果已失效。", false);
  const token = studyToken;
  try {
    const preview = await studyRequest(`/api/study/examples/${encodeURIComponent(exampleId)}/view`);
    if (!studyAcceptance(studyToken, token)) return;
    if (!renderStudy(preview, token, exampleId)) return;
    if (studyDisplayable(preview)) {
      el("study-save").disabled = false;
      studyFeedback("已载入只读预览（未保存）。核对来源与数值后保存为独立记录。", false);
    } else {
      studyFeedback(`示例未提供可展示结果：${preview.reason || studyStatusLabel(preview.content_status)}`, true);
    }
  } catch (error) {
    if (studyAcceptance(studyToken, token)) {
      invalidateStudy(`示例未载入：${error.message}`, true);
    }
  }
}
async function studySave() {
  const exampleId = el("study-example").value;
  if (!exampleId || studyBusy) return;
  studyBusy = true; el("study-save").disabled = true;
  invalidateStudy("正在保存为独立记录，旧结果已失效。", false);
  const token = studyToken;
  try {
    const saved = await studyRequest("/api/study/records", {example_id: exampleId});
    if (!studyAcceptance(studyToken, token)) return;
    if (!renderStudy(saved, token, saved.record_id)) return;
    if (studyDisplayable(saved)) {
      studyFeedback(`已保存为独立研究记录 ${saved.record_id}（来源类型 ${saved.source_type}，未绑定任何观测牌局）。`, false);
    } else {
      studyFeedback(`保存在磁盘上，但没有可展示的已验证结果：${saved.reason || studyStatusLabel(saved.content_status)}`, true);
    }
    await studyRefreshRecords();
  } catch (error) {
    if (studyAcceptance(studyToken, token)) {
      invalidateStudy(`未保存：${error.message}`, true);
      el("study-save").disabled = false;
    }
  } finally { studyBusy = false; }
}
async function studyRefreshRecords() {
  try {
    const data = await studyRequest("/api/study/records");
    studyRecords = data.items || [];
    const select = el("study-record-select");
    select.replaceChildren();
    if (!studyRecords.length) {
      const option = document.createElement("option");
      option.value = ""; option.textContent = "暂无已保存的研究记录";
      select.append(option);
    }
    for (const item of studyRecords) {
      const option = document.createElement("option");
      option.value = item.record_id;
      option.textContent = `${item.record_id} · ${studyStatusLabel(item.content_status)}`;
      select.append(option);
    }
    el("study-record-open").disabled = studyRecords.length === 0;
    el("study-record-count").textContent = `共 ${studyRecords.length} 条（最多 ${data.max_records} 条）`;
    return studyRecords;
  } catch (error) { studyFeedback(`记录列表未载入：${error.message}`, true); return []; }
}
async function studyOpenRecord(recordId) {
  const target = recordId || el("study-record-select").value;
  if (!target) return;
  invalidateStudy(`正在打开记录 ${target}，旧结果已失效。`, false);
  const token = studyToken;
  try {
    const record = await studyRequest(`/api/study/records/${encodeURIComponent(target)}`);
    if (!studyAcceptance(studyToken, token)) return;
    if (!renderStudy(record, token, target)) return;
    if (studyDisplayable(record)) {
      el("study-save").disabled = false;
      studyFeedback(`已重开 ${record.record_id}：内容、标签与来源身份与保存时一致。`, false);
    } else {
      studyFeedback(`${studyStatusLabel(record.content_status)}：${record.reason || "该记录没有可展示的已验证结果。"}`, true);
    }
  } catch (error) {
    if (studyAcceptance(studyToken, token)) {
      invalidateStudy(`记录未打开：${error.message}`, true);
    }
  }
}
if (typeof document !== "undefined" && typeof el === "function" && el("study-load")) {
  el("study-load").addEventListener("click", studyLoadExample);
  el("study-save").addEventListener("click", studySave);
  el("study-record-refresh").addEventListener("click", studyRefreshRecords);
  el("study-record-open").addEventListener("click", () => studyOpenRecord());
  el("study-example").addEventListener("change", () =>
    invalidateStudy("示例选择已变化，旧结果已失效。请重新载入。", false));
  el("study-record-select").addEventListener("change", () =>
    invalidateStudy("已切换记录条目，请点「重开结果」重新读取。", false));
  el("study-tools").addEventListener("toggle", () => {
    if (!el("study-tools").open) {
      invalidateStudy("面板已关闭，当前结果已清空。", false);
    }
  });
  studyWired = true;
  studyInit().then(studyRefreshRecords);
}
