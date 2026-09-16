"use strict";
// Plain hand-entry form: it only builds and pre-checks an input, then hands the
// document to the existing conditional-analysis chain. It never computes a
// policy, never reads an observed hand by itself and never fills an unknown.
//
// Row controls (not CSV) are the normal path; the CSV box is an import/export
// helper. EVERY mutation of the form - a keystroke, adding or deleting a row, a
// programmatic import, a CSV apply, clearing, changing the selected source -
// voids the verified receipt and the downstream result through the single
// `handInvalidate` path, so what is displayed and what would be computed can
// never belong to different hands.
//
// Facts keep their provenance: a value brought in from a structured snapshot
// stays `observed` with its original candidate until the human edits it, at which
// point it becomes `human_confirmed` and the candidate is kept for comparison.
let handToken = 0, handBuilt = null, handWired = false, handExpectedInput = null;
let handSource = handBlankSource(), handOrigin = {};
// The receipt records WHICH table-rules version the verified document was built
// against, so a compute can never pair an old document with a newer revision.
let handReceipt = null;
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
function handBlankSource() {
  return {kind: "manual_form", issue_id: null, saved_at: null,
          preview_sha256: null, source_frame: null, scope: null};
}
function handSeatId(value, where) {
  const raw = String(value === null || value === undefined ? "" : value).trim();
  if (!raw) throw Error(`${where}座位号不能为空：空值不会被当成座位 0`);
  if (!/^\d+$/.test(raw)) throw Error(`${where}座位号必须是 0–7 的整数（当前「${raw}」）`);
  return Number(raw);
}
function handSetOrigin(key, provenance, candidate) {
  handOrigin[key] = {provenance, candidate: candidate ?? null};
}
function handOriginOf(key) {
  return handOrigin[key] || {provenance: "human_confirmed", candidate: null};
}
function handFact(key, value) {
  // `value` already carries the empty-vs-filled decision.
  const origin = handOriginOf(key);
  if (value === null || value === undefined || value === "") {
    return handUnknown(origin.candidate);
  }
  // An imported block stated as unknown never fills a control, so a value that is
  // present here can only have come from the human.
  const provenance = origin.provenance === "unknown" ? "human_confirmed"
    : origin.provenance;
  return {value, provenance, candidate: origin.candidate};
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
    {key: "target", label: "本街累计金额（仅下注/加注需要填）", type: "text"},
    {key: "call_amount", label: "跟注追加额（可选核对）", type: "text"}],
   "hand-rows-history"],
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
// The kernel's own sentinel for an action that carries no size. The human never
// types it: it is generated here and shown as read-only.
const HAND_SIZE_LESS = ["check", "fold", "call"];
const HAND_ROW_HOOKS = {
  history: row => {
    const nodes = row.querySelectorAll("input,select");
    const kind = nodes[1], target = nodes[2], callAmount = nodes[3];
    const sync = () => {
      const sizeLess = HAND_SIZE_LESS.includes(kind.value);
      target.disabled = sizeLess;
      target.value = sizeLess ? "0" : (target.value === "0" ? "" : target.value);
      callAmount.disabled = kind.value !== "call";
      if (kind.value !== "call") callAmount.value = "";
    };
    kind.addEventListener("change", sync);
    sync();
  },
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
  if (HAND_ROW_HOOKS[block]) HAND_ROW_HOOKS[block](row);
  for (const node of row.querySelectorAll("input,select")) {
    node.addEventListener("input", handTouched(row, block));
    node.addEventListener("change", handTouched(row, block));
  }
  el(container).append(row);
  return row;
}
function handTouched(row, block) {
  return () => {
    row.dataset.used = "true";
    // A human touch makes the whole block human-confirmed; the original observed
    // candidate is preserved so the difference stays visible.
    const origin = handOriginOf(block);
    handSetOrigin(block, "human_confirmed", origin.candidate);
    handCareful();
  };
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
function handClearRows(block) {
  el(HAND_BLOCKS[block][2]).replaceChildren();
  delete handOrigin[block];
}
function handFillRows(block, rows, provenance, candidate) {
  // Replacing a block must not throw away the evidence the snapshot already gave
  // for it: unless the caller supplies a candidate, the retained one survives.
  const retained = handOriginOf(block).candidate;
  handClearRows(block);
  for (const values of rows) handAddRow(block, values, true);
  if (rows.length) {
    handSetOrigin(block, provenance || "human_confirmed",
                  candidate === undefined ? retained : candidate);
  }
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
function handSplit(text) {
  return String(text || "").split(",").map(value => value.trim()).filter(Boolean);
}
function handFacts() {
  const hero = handCards(el("hand-hero").value, 2);
  const board = handCards(el("hand-board").value, 5);
  if (!hero) throw Error("Hero 手牌需要两张牌（如 Qs Qd）");
  if (!board) throw Error("公共牌需要五张牌");
  const seatRows = handRows("seats");
  if (!seatRows.length) throw Error("请用「添加一行」录入座位（每行一个座位）");
  const seats = seatRows.map(values => ({
    seat_id: handSeatId(values.seat_id, "座位表里的"),
    status: values.status, stack: values.stack,
    hand_committed: values.hand_committed}));
  const historyRows = handRows("history");
  const noHistory = el("hand-no-history").checked;
  if (noHistory && historyRows.length) {
    throw Error("你既勾选了「本手没有任何公开行动」，又填了公开历史：请二选一");
  }
  const history = historyRows.map(values => {
    const kind = String(values.kind || "").trim().toLowerCase();
    const entry = {
      actor: handSeatId(values.actor, "公开历史里的"),
      kind,
      target: HAND_SIZE_LESS.includes(kind) ? "0"
        : String(values.target || "").trim(),
    };
    const declared = String(values.call_amount || "").trim();
    if (kind === "call" && declared) entry.call_amount = declared;
    return entry;
  });
  const order = handSplit(el("hand-order").value).map(value =>
    handSeatId(value, "行动顺序里的"));
  return {
    source: handSource.issue_id, source_kind: handSource.kind,
    ended_hand_confirmed: handConfirm(el("hand-ended").checked),
    hero_seat: handFact("hand-hero-seat",
                        String(el("hand-hero-seat").value).trim() === "" ? ""
                          : Number(String(el("hand-hero-seat").value).trim())),
    hero_cards: handFact("hand-hero", hero), board_cards: handFact("hand-board", board),
    action_order: order.length ? handFact("hand-order", order)
      : handUnknown(handOriginOf("hand-order").candidate),
    seats: handFact("seats", seats),
    history: historyRows.length ? handFact("history", history)
      : (noHistory ? handConfirm([]) : handUnknown(handOriginOf("history").candidate)),
    pot_display: handFact("hand-pot", String(el("hand-pot").value).trim()),
    table_rules: handUnknown(), observed_at: handSource.saved_at ?? null,
  };
}
function handGroupRanges(rows) {
  // The per-row controls are flat; the kernel takes one range per seat. Grouping
  // by the seat the human typed is a mechanical fold, not an inference: a
  // duplicate combination inside a seat is still rejected downstream.
  const bySeat = new Map();
  for (const values of rows) {
    const seat = handSeatId(values.seat_id, "范围表里的");
    if (!bySeat.has(seat)) bySeat.set(seat, []);
    bySeat.get(seat).push({combo: values.combo, weight: values.weight});
  }
  return [...bySeat.entries()].map(([seat_id, combos]) => ({seat_id, combos}));
}
function handAssumptions() {
  return {
    range_source: "manual_unvalidated",
    ranges: handGroupRanges(handRows("ranges")),
    models: handRows("weights").map(values => ({
      seat_id: handSeatId(values.seat_id, "响应权重表里的"),
      key: values.key, weight: values.weight})),
    aggression_targets: handSplit(el("hand-targets").value),
    max_aggressions: Number(el("hand-max-agg").value || 1),
    other_fees: handFeeBlock(),
  };
}
function handBindAnalysisInput(value) {
  // The analysis panel owns this binding; the form only sets it. If the panel
  // script is not on the page, the form still works on its own.
  try { analysisExpectedInput = value; return true; } catch (_) { return false; }
}
function handDropReceipt() {
  handBuilt = null; handExpectedInput = null; handReceipt = null;
  handRulesInFlight = null;
  el("hand-compute").disabled = true;
}
function handFailAnalysis(reason) {
  // Reuse the existing analysis invalidation path (and its debounced cancel)
  // instead of firing a second cancel that could land after a new start.
  handDropReceipt();
  try { analysisInputChanged(reason); return true; } catch (_) { return false; }
}
function handMarkRevised() {
  // A human edit on an imported hand does not erase where it came from; it marks
  // the chain so the export never passes an edited import off as pure manual entry.
  if (handSource.issue_id !== null && handSource.revised_by_human !== true) {
    handSource = {...handSource, revised_by_human: true};
  }
}
const HAND_VOIDED = "上方牌局输入已变更：下方按旧输入算出的结果已失效。";
function handInvalidate(message) {
  ++handToken;
  handDropReceipt();
  el("hand-gaps").replaceChildren(); el("hand-capacity").replaceChildren();
  el("hand-identity").textContent = "无结果";
  handStatus("未核对");
  handFailAnalysis(HAND_VOIDED);
  if (message) handFeedback(message, false);
}
function handCareful() {
  handMarkRevised();
  handInvalidate("输入已变更，请重新核对后再计算。");
}
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
const HAND_ORIGIN_LABELS = {observed: "结构化快照候选（未人工修改）",
                            human_confirmed: "人工填写/确认",
                            assumed: "假设"};
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
  const provenance = {};
  for (const [key, block] of Object.entries(facts)) {
    if (block && typeof block === "object" && "provenance" in block) {
      provenance[key] = {provenance: block.provenance,
                         candidate: block.candidate === undefined ? null : block.candidate};
    }
  }
  el("hand-identity").textContent = JSON.stringify({
    facts_sha256: result.hashes.facts_sha256,
    assumptions_sha256: result.hashes.assumptions_sha256,
    input_sha256: result.hashes.input_sha256,
    implementation_version: result.hashes.implementation_version,
    kernel: result.hashes.kernel, scope: result.hashes.scope,
    verified_rules: handReceipt ? {
      rules_source: handReceipt.rules_source,
      rules_revision: handReceipt.rules_revision,
      effective_rules_sha256: handReceipt.effective_rules_sha256,
    } : result.verified_rules ?? null,
    source: handSource, provenance,
    note: "任一行被人工修改后，该块整体记为人工确认；原候选保留在 candidate 里",
  }, null, 2);
}
async function handVerify() {
  handFailAnalysis("开始核对：下方旧结果已失效。");
  const token = ++handToken;
  handRulesInFlight = statusData.table_rules?.revision ?? null;
  let facts, assumptions;
  try {
    facts = handFacts(); assumptions = handAssumptions();
  } catch (error) {
    handRulesInFlight = null;
    handFeedback(error.message, true);
    return null;
  }
  el("hand-gaps").replaceChildren();
  if (!el("hand-use-rules").checked) {
    handRulesInFlight = null;
    handFeedback("本轮只支持使用「本桌规则」里已保存的完整规则；"
                 + "请先勾选该选项并在设置页补齐桌规（未知项不能自动补零）。", true);
    return null;
  }
  try {
    const response = await fetch("/api/hand-input/build", {
      method: "POST", headers: {...headers, "Content-Type": "application/json"},
      body: JSON.stringify({facts, assumptions, rules_source: "table",
        rules_revision: handRulesInFlight})});
    const result = await response.json();
    if (token !== handToken) return null;
    handRulesInFlight = null;
    if (!response.ok) {
      handFeedback(`未通过核对：${text(result.detail)}`, true);
      return null;
    }
    if (!result.ok) {
      el("hand-gaps").replaceChildren(...handList("还缺什么（不会自动补齐）：",
                                                  result.reasons, "blockers"));
      handFeedback(`输入尚不能被内核接受（${result.reasons.length} 项）。`, true);
      return null;
    }
    handBuilt = result.document;
    handExpectedInput = result.hashes.input_sha256;
    handReceipt = {
      input_sha256: result.hashes.input_sha256,
      rules_source: result.verified_rules?.rules_source ?? result.rules_source ?? null,
      rules_revision: result.verified_rules?.rules_revision ?? null,
      effective_rules_sha256: result.verified_rules?.effective_rules_sha256 ?? null,
    };
    if (!handReceipt.rules_revision) {
      // Without the revision the receipt was built against, no compute can prove
      // which rules it used; refuse instead of pretending.
      handDropReceipt();
      handFeedback("核对回执缺少本次使用的桌规版本，无法安全计算：请重新核对。", true);
      return null;
    }
    handRender(result, facts);
    el("hand-compute").disabled = false;
    handStatus("已核对");
    handFeedback("输入通过支持性检查；按「按上述假设计算」会用现有内核针对本次输入计算。",
                 false);
    return result;
  } catch (error) {
    handRulesInFlight = null;
    if (token === handToken) handFeedback(`核对失败：${error.message}`, true);
    return null;
  }
}
function handCompute() {
  if (!handBuilt || !handExpectedInput || !handReceipt) {
    handFeedback("请先核对输入。", true); return;
  }
  // Re-check the receipt's rules version at compute time: the page may have seen a
  // newer one since the verify, and an unverified pairing must never be sent.
  if (handReceipt.rules_revision !== (statusData.table_rules?.revision ?? null)) {
    handInvalidate("本桌规则版本与核对时不一致：请重新核对后再计算（不会用新规则套旧输入）。");
    return;
  }
  // The manual form owns the panel now: the previous draft provenance is dropped
  // and this input's identity travels with the exported report instead.
  if (!handBindAnalysisInput(handExpectedInput)) {
    handFeedback("当前页面没有加载条件分析面板，无法把输入交给内核。", true);
    return;
  }
  try { analysisExpectedRulesRevision = handReceipt.rules_revision; }
  catch (_) { /* analysis.js not loaded */ }
  const provenance = {...handOrigin};
  analysisDraftSource = {
    kind: "hand_input_form",
    input_sha256: handExpectedInput,
    scope: "MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE",
    rules_revision: handReceipt.rules_revision,
    rules_source: handReceipt.rules_source,
    effective_rules_sha256: handReceipt.effective_rules_sha256,
    form_source: {...handSource},
    provenance,
    note: "手工录入/带入的已结束牌局 + 人工假设；不是观测验证样本，也不代表实战建议",
  };
  el("analysis-kind").value = "threeway";
  el("analysis-use-rules").checked = false;
  el("analysis-input").value = JSON.stringify(handBuilt, null, 2);
  handFeedback("已把本次输入交给下方条件分析；结果只针对这次输入。", false);
  el("analysis-start").click();
}
function handResetForSource(source, message) {
  // Replacing the source voids every fact that belonged to the previous one: the
  // old fields, the old end-of-hand confirmation and the old opponent
  // assumptions must not be inherited silently.
  for (const id of ["hand-hero", "hand-board", "hand-hero-seat", "hand-order",
                    "hand-pot", "hand-targets"]) {
    el(id).value = "";
  }
  el("hand-max-agg").value = "2";
  el("hand-ended").checked = false;
  el("hand-no-history").checked = false;
  el("hand-fees").value = "unknown";
  for (const block of Object.keys(HAND_BLOCKS)) handClearRows(block);
  handOrigin = {};
  handSource = {...source};
  handInvalidate(message);
}
function handClear() {
  el("hand-csv").value = "";
  handResetForSource(handBlankSource(), "已清空录入与假设：来源恢复为手工输入。");
}
// The revision an in-flight build request used, so a rules change is noticed even
// before the receipt for it exists.
let handRulesInFlight = null;
function handRulesChanged(reason) {
  // The table rules are being changed or were changed somewhere else. A verified
  // receipt names the revision it was built against, so it cannot survive that:
  // void it and the result, but KEEP the typed hand so the human only has to
  // re-verify. This always goes through handInvalidate so the token advances and
  // a build request that is still in flight can never restore "verified".
  handInvalidate(reason);
}
function handRulesRevisionSeen() {
  // Called from the page's status poll: notice a rules revision this page has not
  // reacted to yet and void the receipt (or the in-flight build) before it can be
  // computed.
  const current = statusData.table_rules?.revision ?? null;
  const known = handReceipt ? handReceipt.rules_revision : handRulesInFlight;
  if (known === null || known === undefined || known === current) return;
  handInvalidate("本桌规则已在本机之外变更为新版本：上方输入的旧核对已失效，请重新核对后再计算。");
}
function handSourceChanged(issueId) {
  // The review desk selected another record (or cleared the selection). A receipt
  // and a result never survive a source change; facts that came from a DIFFERENT
  // record cannot be kept as if they still belonged to the new selection. A purely
  // hand-typed form keeps its values - nothing about it came from a record.
  const next = issueId === undefined
    ? ((typeof selectedReview !== "undefined" && selectedReview && selectedReview.issue)
        ? selectedReview.issue.issue_id : null)
    : issueId;
  const cameFromAnotherRecord = handSource.issue_id !== null
    && handSource.issue_id !== next;
  if (cameFromAnotherRecord) {
    handResetForSource(
      {...handBlankSource(), kind: next ? "selected_record_not_imported" : "manual_form",
       issue_id: next},
      next ? `已切换复查记录为 ${next}，而上一套字段来自 ${handSource.issue_id}：`
              + "旧字段、结束确认与对手假设已全部作废；请重新带入或重新录入。"
           : `已取消复查记录选择，而上一套字段来自 ${handSource.issue_id}：`
              + "旧字段、结束确认与对手假设已全部作废。");
    return;
  }
  handSource = {...handSource, kind: next ? "selected_record_not_imported" : "manual_form",
                issue_id: next};
  handInvalidate(next
    ? `已切换复查记录为 ${next}：请点「带入」重新载入有证据的字段。`
    : "已取消复查记录选择：按「带入」前不会使用任何记录里的字段。");
}
function handImportFailed(issueId, message) {
  // The record stays on the source so the failure is visible, but nothing of it
  // is treated as a confirmed fact.
  handSource = {...handSource, kind: "import_failed", issue_id: issueId};
  handFeedback(`带入失败：${message}`, true);
}
const HAND_FACT_LABELS = {hero_cards: "Hero 手牌", board_cards: "公共牌",
                          pot_display: "显示底池", hero_seat: "Hero 座位",
                          action_order: "行动顺序", seats: "各座位"};
function handApplyImportedFacts(issueId, body) {
  const facts = body.facts || {};
  const applied = [], missing = [];
  const scalars = [
    ["hand-hero", "hero_cards", value => value.join(" ")],
    ["hand-board", "board_cards", value => value.join(" ")],
    ["hand-pot", "pot_display", value => String(value)],
    ["hand-hero-seat", "hero_seat", value => String(value)],
    ["hand-order", "action_order", value => value.join(",")],
  ];
  // Retain EVERY returned block's provenance and candidate first, then decide
  // whether a control can be filled. A block the record states as unknown still
  // hands its original evidence to the human; keeping the candidate is not the
  // same as treating it as a confirmed fact.
  for (const [id, key, render] of scalars) {
    const block = facts[key];
    const usable = block && block.value !== null && block.value !== undefined
      && String(block.value).trim() !== "";
    if (usable) {
      el(id).value = render(block.value);
      handSetOrigin(id, block.provenance || "observed", block.candidate);
      applied.push(HAND_FACT_LABELS[key]);
    } else {
      if (block) handSetOrigin(id, "unknown", block.candidate);
      missing.push(HAND_FACT_LABELS[key]);
    }
  }
  if (facts.seats && facts.seats.value) {
    handFillRows("seats", facts.seats.value, facts.seats.provenance || "observed",
                 facts.seats.candidate);
    applied.push(HAND_FACT_LABELS.seats);
  } else {
    if (facts.seats) handSetOrigin("seats", "unknown", facts.seats.candidate);
    missing.push(HAND_FACT_LABELS.seats);
  }
  // The snapshot never supplies a confirmed public history, so it is always a gap
  // for the human - but its candidate must survive into the next build.
  if (facts.history) {
    handSetOrigin("history", "unknown", facts.history.candidate);
    if (facts.history.value) {
      handFillRows("history", facts.history.value, facts.history.provenance
                   || "observed", facts.history.candidate);
      applied.push(HAND_FACT_LABELS.history);
    } else {
      missing.push(HAND_FACT_LABELS.history);
    }
  } else {
    missing.push(HAND_FACT_LABELS.history);
  }
  handSource = {...handSource, kind: "saved_observed_snapshot", issue_id: issueId,
                saved_at: (body.source && body.source.saved_at) || handSource.saved_at,
                preview_sha256: (body.source && body.source.preview_sha256) || null,
                source_frame: (body.source && body.source.source_frame) || null,
                scope: (body.source && body.source.scope) || null};
  const gaps = [...(body.gaps || [])];
  for (const name of missing) {
    gaps.push(`${name}：该记录没有可用候选或仍是未知，需要人工录入/确认`);
  }
  el("hand-gaps").replaceChildren(...handList(
    `已带入记录 ${issueId}；只有下面这些有证据，其余仍是未知，`
    + "必须人工补齐并确认后才能计算（未知项的原候选已保留，供你核对时对照）：",
    gaps, "blockers"));
  handStatus("已带入候选");
  handFeedback(`已带入 ${applied.length} 项有证据的字段（来源标记为候选）；`
               + "带入不等于已核对：请补齐未知项、勾选「本手已结束」并选择费用状态，"
               + "再点「核对输入」。", false);
}
async function handImportRecord() {
  const record = typeof selectedReview !== "undefined" ? selectedReview : null;
  if (!record || !record.issue) {
    handFeedback("请先在「复查记录」里选中一条记录，再点带入。", true);
    return null;
  }
  const issueId = record.issue.issue_id;
  const revision = statusData.table_rules?.revision ?? null;
  // Void first: the displayed form and any result must never describe the
  // previous hand while the new one is being fetched.
  handResetForSource({kind: "saved_observed_snapshot_pending", issue_id: issueId,
                      saved_at: record.issue.saved_at ?? null,
                      preview_sha256: record.issue.preview_sha256 ?? null,
                      source_frame: null, scope: null},
                     `正在从记录 ${issueId} 带入：上一来源的字段、结束确认、`
                     + "对手假设与下方旧结果已全部作废。");
  const token = handToken;
  try {
    const response = await fetch(
      `/api/hand-input/facts/${encodeURIComponent(issueId)}`);
    const body = await response.json();
    if (token !== handToken) return null;   // a newer change owns the form now
    if ((statusData.table_rules?.revision ?? null) !== revision) {
      handImportFailed(issueId, "本桌规则在带入过程中变更；请重新核对规则后再带入。");
      return null;
    }
    if (!response.ok) throw Error(text(body.detail));
    if (!body.source || body.source.issue_id !== issueId) {
      handImportFailed(issueId, "返回的记录与请求的目标不是同一条，这次带入已放弃。");
      return null;
    }
    handApplyImportedFacts(issueId, body);
    return body;
  } catch (error) {
    if (token !== handToken) return null;
    handImportFailed(issueId, error.message);
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
  // Replacing a block voids the receipt and the result exactly like typing does.
  handMarkRevised();
  handInvalidate(`已用 CSV 替换「${HAND_BLOCKS[wanted][0]}」整块：旧核对与旧结果已失效。`);
  handFillRows(wanted, rows, "human_confirmed", null);
  handFeedback(`已把 ${rows.length} 行 CSV 转成逐行控件并作废旧核对，请核对后再计算。`,
               false);
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
    el(id).addEventListener("input", () => {
      const origin = handOriginOf(id);
      handSetOrigin(id, "human_confirmed", origin.candidate);
      handCareful();
    });
  }
  for (const id of ["hand-ended", "hand-use-rules", "hand-no-history"]) {
    el(id).addEventListener("change", handCareful);
  }
  el("hand-fees").addEventListener("change", handCareful);
  handWired = true;
  handFeedback("尚未核对输入。未知项留空即可；核对后会给出还缺什么。", false);
}
