"use strict";
// Plain hand-entry form: it only builds and pre-checks an input, then hands the
// document to the existing conditional-analysis chain. It never computes a
// policy, never reads an observed hand by itself and never fills an unknown.
//
// Row controls (not CSV) are the normal path; the CSV box is an import/export
// helper. Every edit invalidates both the verified input and any analysis result
// that was produced from a previous input.
let handToken = 0, handBuilt = null, handWired = false, handExpectedInput = null;
const handStatus = message => { el("hand-input-tag").textContent = message; };
const handFeedback = (message, isError) => {
  el("hand-status").textContent = message;
  el("hand-status").className = isError ? "error" : "muted";
};
function handElement(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}
const HAND_BLOCKS = {
  seats: ["座位", [
    {key: "seat_id", label: "座位号", type: "number"},
    {key: "status", label: "河牌起点状态", type: "select", options: [
      ["ACTIVE", "参与"], ["FOLDED", "已弃牌"], ["ALL_IN", "全下"]]},
    {key: "stack", label: "剩余筹码", type: "text"},
    {key: "hand_committed", label: "本手已投入", type: "text"}], "hand-rows-seats"],
  history: ["公开历史（按发生顺序）", [
    {key: "actor", label: "座位号", type: "number"},
    {key: "kind", label: "动作", type: "select", options: [
      ["check", "过牌"], ["fold", "弃牌"], ["call", "跟注"],
      ["bet", "下注"], ["raise", "加注"]]},
    {key: "target", label: "本街累计金额", type: "text"}], "hand-rows-history"],
  ranges: ["对手范围（每行一个组合）", [
    {key: "seat_id", label: "座位号", type: "number"},
    {key: "combo", label: "组合（如 JhJd）", type: "text"},
    {key: "weight", label: "相对权重", type: "text"}], "hand-rows-ranges"],
  weights: ["响应权重（每行一个动作）", [
    {key: "seat_id", label: "座位号", type: "number"},
    {key: "key", label: "动作", type: "select", options: [
      ["check", "过牌"], ["fold", "弃牌"], ["call", "跟注"],
      ["bet", "下注"], ["raise", "加注"]]},
    {key: "weight", label: "权重", type: "text"}], "hand-rows-weights"],
};
function handField(field, value) {
  if (field.type === "select") {
    const select = handElement("select");
    for (const [key, label] of field.options) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = label;
      select.append(option);
    }
    select.value = value || field.options[0][0];
    return select;
  }
  const input = handElement("input");
  input.type = field.type === "number" ? "number" : "text";
  input.value = value === undefined || value === null ? "" : String(value);
  return input;
}
function handAddRow(block, values = {}, used = false) {
  const fields = HAND_BLOCKS[block][1], container = HAND_BLOCKS[block][2];
  const row = handElement("div", "rules-grid");
  row.dataset.block = block;
  // A freshly added row is a placeholder, not input: only a row the human has
  // touched counts, so seeding the empty form never fabricates a seat.
  row.dataset.used = used ? "true" : "false";
  for (const field of fields) {
    const label = handElement("label", null, field.label);
    label.append(handField(field, values[field.key]));
    row.append(label);
  }
  const remove = handElement("button", "secondary", "删除此行");
  remove.type = "button";
  remove.addEventListener("click", () => { row.remove(); handCareful(); });
  row.append(remove);
  for (const node of row.querySelectorAll("input,select")) {
    node.addEventListener("input", handTouched(row));
    node.addEventListener("change", handTouched(row));
  }
  el(container).append(row);
  return row;
}
function handTouched(row) {
  return () => { row.dataset.used = "true"; handCareful(); };
}
function handRows(block) {
  const fields = HAND_BLOCKS[block][1], container = HAND_BLOCKS[block][2];
  const rows = [];
  for (const row of el(container).children) {
    const nodes = row.querySelectorAll("input,select");
    if (nodes.length !== fields.length) continue;
    if (row.dataset.used !== "true") continue;
    const values = {};
    fields.forEach((field, index) => { values[field.key] = nodes[index].value; });
    if (Object.values(values).every(value => String(value).trim() === "")) continue;
    rows.push(values);
  }
  return rows;
}
function handClearRows(block) { el(HAND_BLOCKS[block][2]).replaceChildren(); }
function handFillRows(block, rows) {
  handClearRows(block);
  for (const values of rows) handAddRow(block, values, true);
}
function handConfirm(value) { return {value, provenance: "human_confirmed", candidate: null}; }
function handUnknown(candidate = null) { return {value: null, provenance: "unknown", candidate}; }
function handCards(text, count) {
  const values = String(text || "").trim().split(/[\s,]+/).filter(Boolean);
  return values.length === count ? values : null;
}
function handFeeBlock() {
  const choice = el("hand-fees").value;
  if (choice === "unknown") return {value: null, provenance: "unknown"};
  if (choice === "confirmed_zero") return {value: "0", provenance: "human_confirmed"};
  return {value: "0", provenance: "assumed"};
}
function handFacts() {
  const hero = handCards(el("hand-hero").value, 2);
  const board = handCards(el("hand-board").value, 5);
  if (!hero) throw Error("Hero 手牌需要两张牌（如 Qs Qd）");
  if (!board) throw Error("公共牌需要五张牌");
  const seats = handRows("seats").map(values => ({
    seat_id: Number(values.seat_id), status: values.status,
    stack: values.stack, hand_committed: values.hand_committed}));
  if (!seats.length) throw Error("请用「添加一行」录入座位（每行一个座位）");
  const history = handRows("history").map(values => ({
    actor: Number(values.actor), kind: values.kind, target: values.target}));
  const order = String(el("hand-order").value || "").split(",")
    .map(value => value.trim()).filter(Boolean).map(Number);
  return {
    source: null, source_kind: "manual_form",
    ended_hand_confirmed: handConfirm(el("hand-ended").checked),
    hero_seat: el("hand-hero-seat").value === "" ? handUnknown()
      : handConfirm(Number(el("hand-hero-seat").value)),
    hero_cards: handConfirm(hero), board_cards: handConfirm(board),
    action_order: handConfirm(order), seats: handConfirm(seats),
    history: handConfirm(history),
    pot_display: el("hand-pot").value.trim() ? handConfirm(el("hand-pot").value.trim())
      : handUnknown(),
    table_rules: handUnknown(), observed_at: null,
  };
}
function handGroupRanges(rows) {
  // The per-row controls are flat; the kernel takes one range per seat. Grouping
  // by the seat the human typed is a mechanical fold, not an inference: a
  // duplicate combination inside a seat is still rejected downstream.
  const bySeat = new Map();
  for (const row of rows) {
    const seat = Number(row.seat_id);
    if (!bySeat.has(seat)) bySeat.set(seat, []);
    bySeat.get(seat).push({combo: row.combo, weight: row.weight});
  }
  return [...bySeat.entries()].map(([seat_id, combos]) => ({seat_id, combos}));
}
function handAssumptions() {
  return {
    range_source: "manual_unvalidated",
    ranges: handGroupRanges(handRows("ranges")),
    models: handRows("weights").map(values => ({
      seat_id: Number(values.seat_id), key: values.key, weight: values.weight})),
    aggression_targets: String(el("hand-targets").value || "").split(",")
      .map(value => value.trim()).filter(Boolean),
    max_aggressions: Number(el("hand-max-agg").value || 1),
    other_fees: handFeeBlock(),
  };
}
function handBindAnalysisInput(value) {
  // The analysis panel owns this binding; the form only sets it. If the panel
  // script is not on the page, the form still works on its own.
  try { analysisExpectedInput = value; return true; } catch (_) { return false; }
}
function handFailAnalysis(reason) {
  // Reuse the existing analysis invalidation path (and its debounced cancel)
  // instead of firing a second cancel that could land after a new start.
  handExpectedInput = null;
  try { analysisInputChanged(reason); return true; } catch (_) { return false; }
}
function handInvalidate(message) {
  ++handToken; handBuilt = null; handExpectedInput = null;
  el("hand-compute").disabled = true;
  el("hand-gaps").replaceChildren(); el("hand-capacity").replaceChildren();
  el("hand-identity").textContent = "无结果";
  handStatus("未核对");
  handFailAnalysis("上方牌局输入已变更：下方按旧输入算出的结果已失效。");
  if (message) handFeedback(message, false);
}
function handCareful() { handInvalidate("输入已变更，请重新核对后再计算。"); }
function handList(title, items, className) {
  const nodes = [];
  if (!items.length) return nodes;
  nodes.push(handElement("p", "footnote", title));
  const list = document.createElement("ul");
  if (className) list.className = className;
  for (const item of items) list.append(handElement("li", null, item));
  nodes.push(list);
  return nodes;
}
function handTable(headers, rows) {
  const table = handElement("table", "analysis-table");
  const head = document.createElement("tr");
  for (const title of headers) head.append(handElement("th", null, title));
  table.append(head);
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of row) tr.append(handElement("td", null, value));
    table.append(tr);
  }
  return table;
}
const HAND_STATUS_LABELS = {ACTIVE: "参与", FOLDED: "已弃牌", ALL_IN: "全下"};
function handRender(result, facts) {
  const amounts = result.amounts, labels = amounts.labels, room = result.capacity;
  const rows = amounts.rows.map(row => [
    `座位 ${row.seat_id}（${HAND_STATUS_LABELS[row.status] || row.status}）`,
    row.stack || "未填", row.hand_committed,
    row.status === "ACTIVE" ? row.street_wager : "—",
    row.status === "ACTIVE" ? (row.to_call === null ? "0" : row.to_call) : "—",
  ]);
  el("hand-capacity").replaceChildren(
    handElement("p", "footnote",
                `你声明的组合乘积 ${room.declared_combo_product} · `
                + `合法联合组合数 ${room.legal_joint_combos}（上限 ${room.legal_joint_limit}）；`
                + "仅算你给出的组合，不会补全范围"),
    handElement("p", "footnote",
                `金额口径（本次输入，单位 ${amounts.unit}）：${labels.stack} · `
                + `${labels.hand_committed} · ${labels.street_wager} · ${labels.to_call}`),
    handTable(["各座金额口径", labels.stack, labels.hand_committed,
               labels.street_wager, labels.to_call], rows),
    handElement("p", "footnote",
                `当前最高下注 ${amounts.current_bet} · 推算底池 ${amounts.implied_pot} · `
                + `Hero 若跟注需追加 ${amounts.to_call}`),
    handElement("p", "footnote", "内核会计算的全部合法根动作与本次追加金额："),
    handTable(["动作", `${labels.raise_to} / 追加`, "口径"],
              amounts.root_actions.map(action => [
                action.kind === "raise" || action.kind === "bet"
                  ? `加注到 ${action.target}`
                  : {check: "过牌", fold: "弃牌", call: "跟注"}[action.kind],
                action.additional_chips, action.reading])));
  el("hand-identity").textContent = JSON.stringify({
    facts_sha256: result.hashes.facts_sha256,
    assumptions_sha256: result.hashes.assumptions_sha256,
    input_sha256: result.hashes.input_sha256,
    implementation_version: result.hashes.implementation_version,
    kernel: result.hashes.kernel, scope: result.hashes.scope,
    provenance: Object.fromEntries(Object.entries(facts).filter(
      ([, value]) => value && typeof value === "object" && "provenance" in value
    ).map(([key, value]) => [key, value.provenance])),
  }, null, 2);
}
async function handVerify() {
  handFailAnalysis("开始核对：下方旧结果已失效。");
  const token = ++handToken;
  let facts, assumptions;
  try {
    facts = handFacts(); assumptions = handAssumptions();
  } catch (error) {
    handFeedback(error.message, true);
    el("hand-compute").disabled = true;
    return null;
  }
  el("hand-gaps").replaceChildren();
  if (!el("hand-use-rules").checked) {
    handFeedback("本轮只支持使用「本桌规则」里已保存的完整规则；"
                 + "请先勾选该选项并在设置页补齐桌规（未知项不能自动补零）。", true);
    el("hand-compute").disabled = true;
    return null;
  }
  try {
    const response = await fetch("/api/hand-input/build", {
      method: "POST", headers: {...headers, "Content-Type": "application/json"},
      body: JSON.stringify({facts, assumptions, rules_source: "table",
        rules_revision: statusData.table_rules?.revision})});
    const result = await response.json();
    if (token !== handToken) return null;
    if (!response.ok) {
      handFeedback(`未通过核对：${text(result.detail)}`, true);
      el("hand-compute").disabled = true;
      return null;
    }
    if (!result.ok) {
      el("hand-gaps").replaceChildren(...handList("还缺什么（不会自动补齐）：",
                                                  result.reasons, "blockers"));
      handFeedback(`输入尚不能被内核接受（${result.reasons.length} 项）。`, true);
      el("hand-compute").disabled = true;
      handBuilt = null;
      return null;
    }
    handBuilt = result.document;
    handExpectedInput = result.hashes.input_sha256;
    handRender(result, facts);
    el("hand-compute").disabled = false;
    handStatus("已核对");
    handFeedback("输入通过支持性检查；按「按上述假设计算」会用现有内核针对本次输入计算。",
                 false);
    return result;
  } catch (error) {
    if (token === handToken) handFeedback(`核对失败：${error.message}`, true);
    el("hand-compute").disabled = true;
    return null;
  }
}
function handCompute() {
  if (!handBuilt || !handExpectedInput) { handFeedback("请先核对输入。", true); return; }
  // The manual form owns the panel now: the previous draft provenance is dropped
  // and this input's identity travels with the exported report instead.
  if (!handBindAnalysisInput(handExpectedInput)) {
    handFeedback("当前页面没有加载条件分析面板，无法把输入交给内核。", true);
    return;
  }
  analysisDraftSource = {
    kind: "hand_input_form",
    input_sha256: handExpectedInput,
    scope: "MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE",
    rules_revision: statusData.table_rules?.revision ?? null,
    note: "手工录入的已结束牌局 + 人工假设；不是观测验证样本",
  };
  el("analysis-kind").value = "threeway";
  el("analysis-use-rules").checked = false;
  el("analysis-input").value = JSON.stringify(handBuilt, null, 2);
  handFeedback("已把本次输入交给下方条件分析；结果只针对这次输入。", false);
  el("analysis-start").click();
}
function handClear() {
  for (const id of ["hand-hero", "hand-board", "hand-hero-seat", "hand-order",
                    "hand-pot", "hand-targets", "hand-csv"]) {
    el(id).value = "";
  }
  el("hand-max-agg").value = "2";
  el("hand-ended").checked = false;
  el("hand-fees").value = "unknown";
  for (const block of Object.keys(HAND_BLOCKS)) handClearRows(block);
  handInvalidate("已清空录入。");
}
async function handImportRecord() {
  const record = typeof selectedReview !== "undefined" ? selectedReview : null;
  if (!record || !record.issue) {
    handFeedback("请先在「复查记录」里选中一条记录，再点带入。", true);
    return null;
  }
  const issueId = record.issue.issue_id;
  try {
    const response = await fetch(
      `/api/hand-input/facts/${encodeURIComponent(issueId)}`);
    const body = await response.json();
    if (!response.ok) throw Error(text(body.detail));
    const facts = body.facts;
    if (facts.hero_cards.value) el("hand-hero").value = facts.hero_cards.value.join(" ");
    if (facts.board_cards.value) el("hand-board").value = facts.board_cards.value.join(" ");
    if (facts.pot_display.value) el("hand-pot").value = facts.pot_display.value;
    if (facts.hero_seat.value !== null) el("hand-hero-seat").value = facts.hero_seat.value;
    if (facts.action_order.value) {
      el("hand-order").value = facts.action_order.value.join(",");
    }
    if (facts.seats.value) handFillRows("seats", facts.seats.value);
    el("hand-gaps").replaceChildren(...handList(
      `已带入记录 ${issueId} 的结构化候选；下面这些必须人工确认后才能计算：`,
      body.gaps, "blockers"));
    handStatus("已带入候选");
    handFeedback("带入只填有证据的字段；未确认的字段保持未知，不会猜。", false);
    return body;
  } catch (error) {
    handFeedback(`带入失败：${error.message}`, true);
    return null;
  }
}
function handCsvApply() {
  const content = el("hand-csv").value;
  if (!content.trim()) { handFeedback("先在高级区粘贴 CSV。", true); return; }
  const wanted = el("hand-csv-target").value;
  const fields = HAND_BLOCKS[wanted][1].map(field => field.key);
  const rows = [];
  for (const line of content.split("\n")) {
    if (!line.trim()) continue;
    const parts = line.split(",").map(value => value.trim());
    if (parts.length !== fields.length) {
      handFeedback(`CSV 每行需要 ${fields.length} 项（${fields.join(",")}）。`, true);
      return;
    }
    rows.push(Object.fromEntries(fields.map((key, index) => [key, parts[index]])));
  }
  handFillRows(wanted, rows);
  handFeedback(`已把 ${rows.length} 行 CSV 转成逐行控件，请核对后再计算。`, false);
}
if (typeof document !== "undefined" && typeof el === "function"
    && el("hand-build")) {
  el("hand-build").addEventListener("click", handVerify);
  el("hand-compute").addEventListener("click", handCompute);
  el("hand-clear").addEventListener("click", handClear);
  el("hand-import-record").addEventListener("click", handImportRecord);
  el("hand-csv-apply").addEventListener("click", handCsvApply);
  const HAND_NEW_ROW = {
    seats: ["hand-add-seat", {status: "ACTIVE"}],
    history: ["hand-add-history", {kind: "check"}],
    ranges: ["hand-add-range", {weight: "1"}],
    weights: ["hand-add-weight", {key: "check", weight: "1"}],
  };
  for (const [block, [buttonId, seed]] of Object.entries(HAND_NEW_ROW)) {
    const button = el(buttonId);
    if (button) button.addEventListener("click", () => handAddRow(block, seed));
  }
  handAddRow("seats", {status: "ACTIVE"});
  handAddRow("history", {kind: "check"});
  handAddRow("ranges", {weight: "1"});
  handAddRow("weights", {key: "check", weight: "1"});
  for (const id of ["hand-hero", "hand-board", "hand-hero-seat", "hand-order",
                    "hand-pot", "hand-targets", "hand-max-agg"]) {
    el(id).addEventListener("input", handCareful);
  }
  el("hand-ended").addEventListener("change", handCareful);
  el("hand-use-rules").addEventListener("change", handCareful);
  el("hand-fees").addEventListener("change", handCareful);
  handWired = true;
  handFeedback("尚未核对输入。未知项留空即可；核对后会给出还缺什么。", false);
}
