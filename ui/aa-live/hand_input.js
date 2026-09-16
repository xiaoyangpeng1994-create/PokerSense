"use strict";
// Plain hand-entry form: it only builds and pre-checks an input, then hands the
// document to the existing conditional-analysis chain. It never computes a
// policy, never reads an observed hand by itself and never fills an unknown.
let handToken = 0, handBuilt = null, handWired = false;
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
function handCards(text, count) {
  const values = String(text || "").trim().split(/[\s,]+/).filter(Boolean);
  return values.length === count ? values : null;
}
function handRows(text, width, label) {
  const rows = [];
  for (const line of String(text || "").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(",").map(value => value.trim());
    if (parts.length !== width) throw Error(`${label}每行需要 ${width} 项：${trimmed}`);
    rows.push(parts);
  }
  return rows;
}
function handConfirm(value) { return {value, provenance: "human_confirmed", candidate: null}; }
function handUnknown(candidate = null) { return {value: null, provenance: "unknown", candidate}; }
function handFacts() {
  const hero = handCards(el("hand-hero").value, 2);
  const board = handCards(el("hand-board").value, 5);
  if (!hero) throw Error("Hero 手牌需要两张牌（如 Qs Qd）");
  if (!board) throw Error("公共牌需要五张牌");
  const seats = handRows(el("hand-seats").value, 4, "座位");
  if (!seats.length) throw Error("至少填写三名 ACTIVE 座位");
  const facts = {
    source: null, source_kind: "manual_form",
    ended_hand_confirmed: handConfirm(el("hand-ended").checked),
    hero_seat: el("hand-hero-seat").value === "" ? handUnknown()
      : handConfirm(Number(el("hand-hero-seat").value)),
    hero_cards: handConfirm(hero), board_cards: handConfirm(board),
    action_order: handConfirm(String(el("hand-order").value || "").split(",")
      .map(value => value.trim()).filter(Boolean).map(Number)),
    seats: handConfirm(seats.map(parts => ({
      seat_id: Number(parts[0]), status: parts[1].toUpperCase(),
      stack: parts[2], hand_committed: parts[3]}))),
    history: handConfirm(handRows(el("hand-history").value, 3, "公开历史").map(
      parts => ({actor: Number(parts[0]), kind: parts[1].toLowerCase(),
                 target: parts[2]}))),
    pot_display: el("hand-pot").value.trim() ? handConfirm(el("hand-pot").value.trim())
      : handUnknown(),
    table_rules: handUnknown(), observed_at: null,
  };
  return facts;
}
function handAssumptions() {
  const ranges = new Map(), models = new Map();
  for (const parts of handRows(el("hand-ranges").value, 3, "对手范围")) {
    const seat = Number(parts[0]);
    if (!ranges.has(seat)) ranges.set(seat, []);
    ranges.get(seat).push({combo: parts[1], weight: parts[2]});
  }
  for (const parts of handRows(el("hand-weights").value, 3, "响应权重")) {
    const seat = Number(parts[0]);
    if (!models.has(seat)) models.set(seat, {});
    models.get(seat)[parts[1].toLowerCase()] = parts[2];
  }
  return {
    range_source: "manual_unvalidated",
    ranges: [...ranges].map(([seat, combos]) => ({seat_id: seat, combos})),
    models: [...models].map(([seat, weights]) => ({seat_id: seat, weights})),
    aggression_targets: String(el("hand-targets").value || "").split(",")
      .map(value => value.trim()).filter(Boolean),
    max_aggressions: Number(el("hand-max-agg").value || 1),
    other_fees: "0",
  };
}
function handInvalidate(message) {
  ++handToken; handBuilt = null;
  el("hand-compute").disabled = true;
  el("hand-gaps").replaceChildren(); el("hand-capacity").replaceChildren();
  el("hand-identity").textContent = "无结果";
  handStatus("未核对");
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
function handRender(result, facts) {
  const rows = [
    ["对手组合乘积", String(result.capacity.declared_combo_product)],
    ["合法联合组合数", String(result.capacity.legal_joint_combos)],
    ["当前最高下注额", `${result.capacity.current_bet} ${result.capacity.unit}`],
    ["Hero 应付额", `${result.capacity.to_call} ${result.capacity.unit}`],
    ["推算底池", `${result.capacity.implied_pot} ${result.capacity.unit}`],
  ];
  const table = document.createElement("table");
  table.className = "analysis-table";
  const head = document.createElement("tr");
  for (const title of ["容量与金额（本次输入）", "值"]) {
    head.append(handElement("th", null, title));
  }
  table.append(head);
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of row) tr.append(handElement("td", null, value));
    table.append(tr);
  }
  el("hand-capacity").replaceChildren(table);
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
      body: JSON.stringify({facts, assumptions,
        rules_source: el("hand-use-rules").checked ? "table" : "document",
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
    handRender(result, facts);
    el("hand-compute").disabled = false;
    handStatus("已核对");
    handFeedback("输入通过支持性检查；按「按上述假设计算」会用现有内核针对本次输入计算。",
                 false);
    return result;
  } catch (error) {
    if (token === handToken) handFeedback(`核对失败：${error.message}`, true);
    return null;
  }
}
function handCompute() {
  if (!handBuilt) { handFeedback("请先核对输入。", true); return; }
  el("analysis-kind").value = "threeway";
  el("analysis-use-rules").checked = false;
  el("analysis-input").value = JSON.stringify(handBuilt, null, 2);
  handFeedback("已把本次输入交给下方条件分析；结果只针对这次输入。", false);
  el("analysis-start").click();
}
function handClear() {
  for (const id of ["hand-hero", "hand-board", "hand-hero-seat", "hand-order",
                    "hand-pot", "hand-targets", "hand-seats", "hand-history",
                    "hand-ranges", "hand-weights"]) {
    el(id).value = "";
  }
  el("hand-max-agg").value = "2";
  el("hand-ended").checked = false;
  handInvalidate("已清空录入。");
}
if (typeof document !== "undefined" && typeof el === "function"
    && el("hand-build")) {
  el("hand-build").addEventListener("click", handVerify);
  el("hand-compute").addEventListener("click", handCompute);
  el("hand-clear").addEventListener("click", handClear);
  for (const id of ["hand-hero", "hand-board", "hand-hero-seat", "hand-order",
                    "hand-pot", "hand-targets", "hand-max-agg", "hand-seats",
                    "hand-history", "hand-ranges", "hand-weights"]) {
    el(id).addEventListener("input", handCareful);
  }
  el("hand-ended").addEventListener("change", handCareful);
  el("hand-use-rules").addEventListener("change", handCareful);
  handWired = true;
  handFeedback("尚未核对输入。未知项留空即可；核对后会给出还缺什么。", false);
}
