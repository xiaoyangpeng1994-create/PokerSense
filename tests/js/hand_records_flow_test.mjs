// U2-R1 harness: compute THIS input -> save -> REOPEN -> recompute BOTH ways ->
// save the new result -> restart the service -> reopen again.
//
// The shipped scripts are evaluated UNCHANGED in the page's own order
// (app.js -> controls.js -> analysis.js -> hand_input.js -> analysis_records.js)
// in a DOM stub, against the real backend and the real kernel.
//
// Three phases, three REAL server processes sharing one records directory:
//   run1: save the table rules; compute and save TWO different synthetic inputs
//         (B uses different combos, weights and cards from A); run the whole
//         counter-example set (out-of-order and forged responses, failed open,
//         list-refresh selection, the tamper matrix over the real get/list/
//         scenario paths); then open A and RECOMPUTE IT WITH THE CURRENT RULES,
//         re-verify, compute on the real kernel and save a NEW record C.
//   run2: against a NEW process on the SAME records directory, reopen the saved
//         records WITHOUT any recomputation, then open A again and RECOMPUTE IT
//         UNDER ITS OWN SAVED RULES, re-verify, compute and save a NEW record D.
//   run3: against a THIRD process, reopen everything and prove both recomputed
//         results survived the restart unchanged and with no analytical call.
//
// Usage: node hand_records_flow_test.mjs <base-url> <run1|run2|run3> <handoff>

import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import { createDom, fire, click } from "./dom_stub.mjs";

const base = (process.argv[2] || "").replace(/\/$/, "");
const phase = process.argv[3] || "";
const handoff = process.argv[4] || "";
if (!base.startsWith("http://127.0.0.1:") || !["run1", "run2", "run3"].includes(phase)
    || !handoff) {
  console.log(JSON.stringify({name: "arguments", ok: false, detail: base}));
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));
const read = name => readFileSync(join(here, "..", "..", "ui", "aa-live", name),
                                  "utf8");
const checks = [];
const check = (name, ok, detail = "") => {
  const item = {name, ok: !!ok, detail: String(detail)};
  checks.push(item);
  console.log(JSON.stringify(item));
  return item;
};

const dom = createDom();
const calls = [];
// Request control channel: a held request only answers when the test releases it,
// so out-of-order and late responses are produced deterministically against the
// REAL server instead of being simulated.
const held = [];
function hold(match) {
  const entry = {match, release: null, forge: null, path: null};
  entry.gate = new Promise(resolve => { entry.release = resolve; });
  held.push(entry);
  return entry;
}
const REQUEST_TIMEOUT_MS = 15000;
async function fetchStub(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  calls.push({path, method, body: options.body ?? null});
  const index = held.findIndex(entry => entry.match(path, method));
  if (index >= 0) {
    const [entry] = held.splice(index, 1);
    await entry.gate;
    if (entry.forge) return entry.forge(path);
  }
  // A page request against an endpoint that never finishes must reject (the
  // page's own try/catch then reports it) instead of hanging the whole run.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(base + path,
                                 {...options, signal: controller.signal});
    const body = await response.text();
    return {ok: response.ok, status: response.status, text: async () => body,
            json: async () => JSON.parse(body)};
  } finally {
    clearTimeout(timer);
  }
}
const timers = new Map();
let timerId = 0;
const context = vm.createContext({
  document: dom.document, console, JSON, Number, String, Object, Array, Error,
  Math, Date, Map, Set, Boolean, Promise,
  fetch: fetchStub, AbortController,
  URL: {createObjectURL: () => "blob:stub", revokeObjectURL() {}},
  Blob: class {}, showDesk() {}, deskRequest: async () => null,
  setTimeout: callback => { timers.set(++timerId, callback); return timerId; },
  clearTimeout: id => timers.delete(id),
  setInterval: () => 0, clearInterval: () => {},
});
for (const name of ["app.js", "controls.js", "analysis.js", "hand_input.js",
                    "analysis_records.js"]) {
  vm.runInContext(read(name), context, {filename: name});
}
const run = expression => vm.runInContext(expression, context);
// Feature probe for the older head: a binding this round introduced must produce
// a failed CHECK, not a ReferenceError that aborts the whole run and prints
// "failed checks: []" - which would make the red evidence worthless.
function safeRun(expression, fallback = null) {
  try { return run(expression); } catch (_) { return fallback; }
}
const node = id => run(`el(${JSON.stringify(id)})`);
const exposed = name => run(`typeof ${name}`) !== "undefined";
const feedback = () => node("records-status").textContent;
const viewText = () => node("records-view").textContent;
// `hand-input-tag` carries the form's own status word ("未核对" / "已带入候选" /
// "待重新核对"); `hand-status` carries the longer sentence next to it.
const handTag = () => node("hand-input-tag").textContent;
const handText = () => node("hand-status").textContent;
const loaded = () => handTag().includes("待重新核对");
async function waitUntil(predicate, timeout = 20000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise(resolve => setTimeout(resolve, 60));
  }
  return false;
}
const api = async (path, options = {}) => {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(base + path,
                                 {...options, signal: controller.signal});
    const text = await response.text();
    // A refusal is not always JSON (an unhandled error can answer with plain
    // text and an unterminated body), so a broken answer must produce a failed
    // CHECK, not a parse crash that swallows every later check.
    let body = null;
    try { body = text ? JSON.parse(text) : null; } catch (_) { body = null; }
    return {ok: response.ok, status: response.status, body, raw: text};
  } catch (error) {
    const kind = (error && error.constructor && error.constructor.name) || "Error";
    return {ok: false, status: 0, body: null,
            raw: `${kind}: ${error && error.message}`};
  } finally {
    clearTimeout(timer);
  }
};
const serverRules = async () => (await api("/api/rules")).body;
const serverRevision = async () => (await serverRules()).revision;

// --- page fixtures ------------------------------------------------------
node("mode").options = [{disabled: false}, {disabled: false}];
const RULE_VALUES = {
  table_label: "U2 记录校验", dealt_players: "6", small_blind: "2",
  big_blind: "4", ante: "4", straddle_mode: "none", straddle_amount: "0",
  rake_percent: "0", rake_cap_bb: "0", minimum_chip: "1",
  rake_application: "all_pots", rake_rounding: "exact",
  rake_distribution: "proportional_all_pots", insurance: "off", bomb: "off",
  mushroom: "off"};
async function saveRulesForm(overrides = {}) {
  await waitUntil(() => run("rulesRevision") !== null, 8000);
  for (const [key, value] of Object.entries({...RULE_VALUES, ...overrides})) {
    node(`rule-${key}`).value = value;
  }
  await fire(node("rules-form"), "submit", {preventDefault() {}});
  await waitUntil(async () => /^[a-f0-9]{64}$/.test(run("rulesRevision"))
    && run("rulesRevision") === await serverRevision()
    && await serverRevision() === run("statusData").table_rules?.revision);
  return run("rulesRevision");
}

const SEATS = [
  {seat_id: 0, status: "ACTIVE", stack: "200", hand_committed: "20"},
  {seat_id: 1, status: "ACTIVE", stack: "200", hand_committed: "20"},
  {seat_id: 2, status: "ACTIVE", stack: "200", hand_committed: "20"},
  {seat_id: 3, status: "FOLDED", stack: "200", hand_committed: "10"},
  {seat_id: 4, status: "FOLDED", stack: "200", hand_committed: "10"},
  {seat_id: 5, status: "FOLDED", stack: "200", hand_committed: "10"}];
const HISTORY = [{actor: 1, kind: "bet", target: "20"},
                 {actor: 2, kind: "call", target: "0"}];
const RANGES_A = [
  {seat_id: 1, combo: "JhJd", weight: "1"}, {seat_id: 1, combo: "TcTd", weight: "3"},
  {seat_id: 2, combo: "7c7s", weight: "1"}, {seat_id: 2, combo: "KhTh", weight: "3"}];
const WEIGHTS_A = [
  {seat_id: 1, key: "check", weight: "1"}, {seat_id: 1, key: "bet", weight: "2"},
  {seat_id: 1, key: "call", weight: "9"}, {seat_id: 1, key: "fold", weight: "1"},
  {seat_id: 2, key: "check", weight: "1"}, {seat_id: 2, key: "call", weight: "9"},
  {seat_id: 2, key: "fold", weight: "1"}];
// B is deliberately a DIFFERENT opponent model, not only different cards: a
// recompute that mixed A's ranges or weights with another hand's facts would
// still be visible against these numbers.
const RANGES_B = [
  {seat_id: 1, combo: "AhAd", weight: "2"}, {seat_id: 1, combo: "KsKc", weight: "2"},
  {seat_id: 2, combo: "QhQs", weight: "1"}, {seat_id: 2, combo: "8h8d", weight: "4"}];
const WEIGHTS_B = [
  {seat_id: 1, key: "check", weight: "2"}, {seat_id: 1, key: "bet", weight: "5"},
  {seat_id: 1, key: "call", weight: "4"}, {seat_id: 1, key: "fold", weight: "2"},
  {seat_id: 2, key: "check", weight: "3"}, {seat_id: 2, key: "call", weight: "5"},
  {seat_id: 2, key: "fold", weight: "2"}];
function typeHand(overrides = {}) {
  node("hand-hero").value = overrides.hero ?? "Qs Qd";
  node("hand-board").value = overrides.board ?? "2c 4d 7h 9s Jc";
  node("hand-hero-seat").value = "0";
  node("hand-order").value = "1,2,0";
  node("hand-pot").value = overrides.pot ?? "130";
  node("hand-targets").value = "20,40,80";
  node("hand-max-agg").value = "2";
  node("hand-ended").checked = true;
  node("hand-use-rules").checked = true;
  node("hand-no-history").checked = false;
  node("hand-fees").value = "confirmed_zero";
  run(`handFillRows("seats", ${JSON.stringify(SEATS)})`);
  run(`handFillRows("history", ${JSON.stringify(overrides.history ?? HISTORY)})`);
  run(`handFillRows("ranges", ${JSON.stringify(overrides.ranges ?? RANGES_A)})`);
  run(`handFillRows("weights", ${JSON.stringify(overrides.weights ?? WEIGHTS_A)})`);
  for (const handler of node("hand-hero").handlers.input || []) handler({});
}
async function verifyAndCompute() {
  await click(node("hand-build"));
  if (run("handBuilt") === null) {
    // Not a check: a diagnosis so a failing run says WHY the build was refused.
    console.log("# build refused: " + (node("hand-status").textContent + " | "
      + node("hand-gaps").textContent).replace(/\s+/g, " ").slice(0, 240));
    return null;
  }
  await click(node("hand-compute"));
  const job = run("acceptedAnalysisId");
  for (let index = 0; index < 160; index += 1) {
    await run("poll()");
    const analysis = run("statusData").analysis;
    if (job !== null && analysis && analysis.job_id === job
        && analysis.status !== "RUNNING" && analysis.status !== "IDLE") {
      return analysis;
    }
    await new Promise(resolve => setTimeout(resolve, 120));
  }
  return run("statusData").analysis;
}
// Record every value the status line takes: the list refresh overwrites it almost
// immediately, so polling the current text alone would miss the save outcome.
const textHistory = {};
function watchText(id) {
  const target = node(id);
  let current = "";
  textHistory[id] = [];
  Object.defineProperty(target, "textContent", {
    get() { return current; },
    set(value) { current = String(value); textHistory[id].push(current); },
  });
}
watchText("records-status");
const saveOutcome = /已保存本次分析|已经保存过|保存失败|请先核对|不一致|拒绝|不属于它|按失败处理/;
async function saveCurrent(label) {
  node("records-label").value = label;
  const before = textHistory["records-status"].length;
  await fire(node("records-save"), "click", {});
  await waitUntil(() => textHistory["records-status"].length > before
    && saveOutcome.test(
      textHistory["records-status"][textHistory["records-status"].length - 1]));
  const fresh = textHistory["records-status"].slice(before);
  return fresh.find(value => saveOutcome.test(value)) || fresh.join(" / ");
}
async function openRecord(recordId) {
  const select = node("records-select");
  select.value = recordId;
  await fire(node("records-open"), "click", {});
  await waitUntil(() => !/正在打开/.test(feedback()) && viewText().length > 0);
  return viewText();
}
const recordIds = async () => (((await api("/api/analysis/records")).body || {})
  .items || []).map(row => row.record_id);
async function storedView(recordId) {
  const opened = await api(`/api/analysis/records/${recordId}`);
  const view = (opened.body || {}).view;
  if (!view) return null;
  return {hero: view.situation.hero_cards.join(" "),
          seat1: JSON.stringify(view.assumptions.ranges.find(r => r.seat_id === 1)
                                .combos),
          seat2: JSON.stringify(view.assumptions.ranges.find(r => r.seat_id === 2)
                                .combos),
          bet_weight: view.assumptions.models.find(
            row => row.seat_id === 1 && row.key === "bet").weight,
          call_ev: (view.actions.find(row => row.kind === "call") || {}).ev,
          raise_ev: (view.actions.find(row => row.kind === "raise") || {}).ev,
          raise_exact: (view.actions.find(row => row.kind === "raise") || {}).ev_exact,
          evs: view.actions.map(row => row.ev),
          rake: view.rules.effective_rules.rake_percent,
          cap: view.rules.effective_rules.rake_cap_bb,
          rules_source: view.rules.rules_source,
          effective_rules: view.rules.effective_rules,
          rules_revision: view.rules.rules_revision,
          parent: (view.source || {}).parent_analysis_record_id || null,
          issue_id: (view.source || {}).issue_id || null,
          source_kind: view.source_kind,
          job_id: opened.body.identity.job_id,
          status: opened.body.status};
}
// A DIRECT kernel call for one document: the stored record's numbers must equal
// what the untouched entry point produces for that same input. POSTs need the
// page's own header pair (aa_server refuses a bare POST).
async function directAnalysis(document, rulesSource) {
  const rules = await serverRules();
  const started = await api("/api/analysis", {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-AA-Live": "1"},
    body: JSON.stringify({kind: "threeway", document, rules_source: rulesSource,
                          rules_revision: rules.revision})});
  if (!started.ok) {
    return {status: `HTTP_${started.status}`,
            error: started.body && started.body.detail};
  }
  const job = started.body.job_id;
  for (let index = 0; index < 200; index += 1) {
    const analysis = (await api("/api/status")).body.analysis;
    if (analysis && analysis.job_id === job && analysis.status !== "RUNNING"
        && analysis.status !== "IDLE") {
      return analysis;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  return {status: "TIMEOUT"};
}
function sameView(left, right) {
  return !!left && !!right && JSON.stringify(left) === JSON.stringify(right);
}
// Key-order-insensitive comparison, so a rule object that differs only in the
// order its keys were serialised is still recognised as the same rules.
const canonical = value => JSON.stringify(value, (key, item) =>
  (item && typeof item === "object" && !Array.isArray(item))
    ? Object.fromEntries(Object.entries(item).sort(
      ([left], [right]) => (left < right ? -1 : 1)))
    : item);

// --- phases 2 and 3: the record only ever survives a real restart -----------
if (phase === "run2" || phase === "run3") {
  const saved = JSON.parse(readFileSync(handoff, "utf8"));
  for (const [index, id] of saved.ids.entries()) {
    const text = await openRecord(id);
    const expected = saved.views[index];
    check(`${phase}_record_${index + 1}_reopens_with_the_same_numbers`,
        text.includes(expected.hero) && text.includes(expected.call_ev)
        && text.includes(expected.raise_ev) && text.includes(expected.raise_exact),
        `hero=${expected.hero} call=${expected.call_ev}`);
  }
  if (phase === "run3") {
    const listed = await api("/api/analysis/records");
    check("run3_every_record_survived_the_second_restart",
        listed.body.items.length === saved.ids.length
        && saved.ids.every(id => listed.body.items.some(row => row.record_id === id)),
        `after=${listed.body.items.length} before=${saved.ids.length}`);
    check("run3_every_record_still_passes_its_self_check",
        listed.body.items.every(row => row.display_permitted === true),
        String(listed.body.items.map(row => row.status).join(",")));
    const now = [];
    for (const id of saved.ids) now.push(await storedView(id));
    let identical = true;
    for (const [index] of saved.ids.entries()) {
      if (!sameView(now[index], saved.views[index])) identical = false;
    }
    check("run3_each_record_keeps_its_identity_and_numbers", identical,
        JSON.stringify(now.map(entry => entry && `${entry.status}/${entry.job_id}`)));
    check("run3_every_recomputed_record_is_readable_as_itself",
        now.length === saved.ids.length
        && now.filter(entry => entry && entry.parent === saved.a).length === 3
        && now.filter(entry => entry && entry.parent === saved.a
                    && entry.rake === "0.05").length === 2
        && now.filter(entry => entry && entry.parent === saved.a
                    && entry.rake === "0").length === 1,
        JSON.stringify(now.map(entry => entry && `${entry.parent}/${entry.rake}`)));
    check("run3_nothing_was_recomputed_on_any_reopen",
        calls.filter(call => call.path === "/api/analysis"
               && call.method === "POST").length === 0,
        `${calls.length} requests, none analytical`);
    const failed = checks.filter(item => !item.ok);
    console.log(JSON.stringify({
      name: "verdict", ok: failed.length === 0,
      passed: checks.length - failed.length, failed: failed.length,
      requests: calls.length, phase, base}));
    process.exit(failed.length === 0 ? 0 : 1);
  }
  // ---- run2: recompute A under ITS OWN saved rules ------------------------
  const parentId = saved.a;
  const parentBefore = await storedView(parentId);
  const revisionBefore = await serverRevision();
  const opened = await api(`/api/analysis/records/${parentId}`);
  check("run2_the_parent_record_is_readable_before_the_saved_conditions_run",
      opened.body.display_permitted === true, `status=${opened.body.status}`);
  // A fresh page has no receipt; the reopen+recompute must create one by itself.
  check("run2_a_fresh_page_starts_without_a_receipt",
      run("handReceipt") === null && run("acceptedAnalysisId") === null,
      "no receipt, no accepted job");
  await openRecord(parentId);
  await fire(node("records-recompute-saved"), "click", {});
  const loadedNow = await waitUntil(loaded);
  check("run2_saved_conditions_recompute_loads_the_frozen_facts_and_assumptions",
      loadedNow && run("handReceipt") === null && safeRun("handScenario") !== null
      && node("hand-hero").value === parentBefore.hero
      && node("hand-use-rules").checked === false
      && JSON.stringify(safeRun("handScenario").rules.rake_percent) === '"0"',
      `hero=${node("hand-hero").value} rules=`
      + `${safeRun("handScenario") && safeRun("handScenario").rules.rake_percent}`);
  check("run2_saved_conditions_recompute_did_not_touch_the_global_rules",
      await serverRevision() === revisionBefore,
      `revision=${String(await serverRevision()).slice(0, 8)}`);
  const savedCompute = await verifyAndCompute();
  check("run2_the_saved_conditions_input_reverifies_and_computes_on_the_kernel",
      savedCompute?.status === "COMPLETE",
      `status=${savedCompute?.status} error=${savedCompute?.error}`);
  const savedMessage = await saveCurrent("按保存条件重算 A");
  const afterSaved = await recordIds();
  check("run2_the_saved_conditions_result_saves_as_a_NEW_record",
      /已保存本次分析/.test(savedMessage) && afterSaved.length === saved.ids.length + 1
      && !saved.ids.includes(afterSaved[0]),
      `${savedMessage.slice(0, 40)} ids=${afterSaved.length}`);
  const childId = afterSaved[0];
  const child = await storedView(childId);
  const parentAfter = await storedView(parentId);
  check("run2_the_child_record_names_its_parent_and_keeps_the_review_link",
      child && child.parent === parentId
      && child.issue_id === parentBefore.issue_id
      && child.source_kind === "recomputed_from_analysis_record"
      && child.job_id !== parentBefore.job_id,
      `parent=${child && child.parent} kind=${child && child.source_kind}`);
  const currentRulesSibling = saved.views_by_id[saved.e];
  check("run2_the_child_used_the_PARENT_rules_not_the_current_ones",
      child && child.rake === "0" && child.hero === parentBefore.hero
      && child.bet_weight === parentBefore.bet_weight
      && child.seat1 === parentBefore.seat1 && child.seat2 === parentBefore.seat2
      && currentRulesSibling && child.call_ev !== currentRulesSibling.call_ev
      && child.raise_ev !== currentRulesSibling.raise_ev,
      `child call=${child && child.call_ev} (parent rules) vs `
      + `current-rules call=${currentRulesSibling && currentRulesSibling.call_ev}`);
  check("run2_the_parent_record_is_untouched_by_the_recompute",
      sameView(parentAfter, parentBefore),
      `parent status=${parentAfter && parentAfter.status}`);
  check("run2_the_saved_conditions_recompute_did_not_change_the_table_rules",
      await serverRevision() === revisionBefore,
      `revision=${String(await serverRevision()).slice(0, 8)}`);
  const reopened = await openRecord(childId);
  check("run2_the_new_record_reopens_with_its_own_numbers",
      reopened.includes(child.hero) && reopened.includes(child.raise_ev)
      && reopened.includes("由另一条分析记录重算而来"),
      reopened.replace(/\s+/g, " ").slice(0, 90));
  await openRecord(parentId);
  writeFileSync(handoff, JSON.stringify({
    ...saved, parent: parentId, child_saved: childId,
    ids: [...saved.ids, childId],
    views: [...saved.views, child]}), "utf8");
  const failed = checks.filter(item => !item.ok);
  console.log(JSON.stringify({
    name: "verdict", ok: failed.length === 0,
    passed: checks.length - failed.length, failed: failed.length,
    requests: calls.length, phase, base}));
  process.exit(failed.length === 0 ? 0 : 1);
}

// --- phase 1 ------------------------------------------------------------
check("the_page_scripts_are_loaded",
      exposed("handWired") && exposed("recordsSave") && exposed("recordsRefresh")
      && exposed("handReceipt") && exposed("handScenario")
      && exposed("handLoadRecordScenario"),
      "app + controls + analysis + hand_input + analysis_records");
const R1 = await saveRulesForm({rake_percent: "0"});
const rulesState = await serverRules();
check("the_table_rules_are_saved_through_controls",
      /^[a-f0-9]{64}$/.test(R1) && rulesState.conditional_analysis_ready === true,
      `rev=${String(R1).slice(0, 12)} ready=${rulesState.conditional_analysis_ready} `
      + `pending=${JSON.stringify(rulesState.pending_fields)} `
      + `field=${node("rules-feedback").textContent.slice(0, 80)}`);

// ---- input one ---------------------------------------------------------
typeHand({hero: "Qs Qd", pot: "130"});
const first = await verifyAndCompute();
check("input_one_completes_on_the_real_kernel",
      first?.status === "COMPLETE",
      `status=${first?.status} error=${first?.error}`);
check("the_receipt_keeps_the_facts_this_analysis_used",
      run("handReceipt")?.facts?.hero_cards?.value?.join(" ") === "Qs Qd"
      && run("handReceipt")?.assumptions?.ranges?.length === 2,
      "facts and assumptions travel with the receipt");
const firstSaved = await saveCurrent("合成输入一 QQ");
const firstIds = await recordIds();
check("saving_the_current_analysis_creates_one_record",
      /已保存本次分析/.test(firstSaved) && firstIds.length === 1,
      `${firstSaved.slice(0, 40)} ids=${firstIds.length}`);
const aId = firstIds[0];
const firstText = await openRecord(aId);
check("the_saved_record_shows_this_input_not_a_fixed_sample",
      firstText.includes("Qs Qd") && firstText.includes("64.375")
      && firstText.includes("108.90625") && firstText.includes("515/8"),
      firstText.replace(/\s+/g, " ").slice(0, 90));
check("the_saved_record_labels_its_source_kind",
      firstText.includes("人工填写") && firstText.includes("已保留原候选"),
      "provenance is shown");
check("the_saved_record_states_the_sample_type_in_words",
      firstText.includes("未经观测验证")
      && firstText.includes("manual_hypothesis_unverified"),
      firstText.replace(/\s+/g, " ").match(/来源类型：[^·]{0,40}/)?.[0] || "missing");
check("the_saved_record_shows_the_assumptions_and_the_rule_values",
      firstText.includes("本次使用的对手假设") && firstText.includes("JhJd")
      && firstText.includes("核对时使用的规则数值") && firstText.includes("rake_percent"),
      "ranges, weights and rule values are readable, not only digests");
const aView = await storedView(aId);

// ---- input two (a different hand AND a different opponent model) --------
typeHand({hero: "2h 3d", pot: "130", ranges: RANGES_B, weights: WEIGHTS_B});
const second = await verifyAndCompute();
check("input_two_completes_on_the_real_kernel", second?.status === "COMPLETE",
      `status=${second?.status}`);
const secondSaved = await saveCurrent("合成输入二 23");
const secondIds = await recordIds();
check("the_second_analysis_is_a_second_record",
      /已保存本次分析/.test(secondSaved) && secondIds.length === 2
      && secondIds[0] !== secondIds[1],
      `${secondSaved.slice(0, 40)} ids=${secondIds.length}`);
const bId = secondIds[0];
const secondText = await openRecord(bId);
check("the_second_record_has_the_second_hand_numbers",
      secondText.includes("2h 3d") && secondText.includes("AhAd")
      && secondText.includes("KsKc"),
      secondText.replace(/\s+/g, " ").slice(0, 90));
const bView = await storedView(bId);
check("the_two_records_have_different_facts_AND_different_assumptions",
      aView.hero === "Qs Qd" && bView.hero === "2h 3d"
      && aView.seat1 !== bView.seat1 && aView.bet_weight !== bView.bet_weight
      && firstText !== secondText,
      `A seat1=${aView.seat1} B seat1=${bView.seat1}`);

// ---- a repeated save does not duplicate --------------------------------
const repeatMessage = await saveCurrent("合成输入二 23");
const afterRepeat = await recordIds();
check("a_repeated_save_does_not_create_a_second_record",
      /已经保存过/.test(repeatMessage) && afterRepeat.length === 2,
      repeatMessage.slice(0, 40));

// ---- a stale job cannot be saved ---------------------------------------
typeHand({hero: "6c 6d", pot: "130", ranges: RANGES_B, weights: WEIGHTS_B});
const third = await verifyAndCompute();
check("a_third_analysis_computes_for_the_stale_case",
      third?.status === "COMPLETE", `status=${third?.status}`);
node("hand-pot").value = "170";
for (const handler of node("hand-pot").handlers.input || []) handler({});
const staleMessage = await saveCurrent("不该保存的旧任务");
const afterStale = await recordIds();
check("changing_the_input_makes_the_old_job_unsavable",
      /保存失败|请先核对|拒绝|不一致/.test(staleMessage) && afterStale.length === 2,
      staleMessage.slice(0, 70));

// ---- a rules change does not overwrite the old records -----------------
typeHand({hero: "Qs Qd", pot: "130"});
const fourth = await verifyAndCompute();
check("the_fourth_analysis_computes_before_the_rules_change",
      fourth?.status === "COMPLETE", `status=${fourth?.status}`);
// R2 must carry a NON-ZERO rake cap: with `rake_cap_bb = 0` the cap is 0 chips,
// the rake is capped away entirely, and a "current rules" recompute becomes
// numerically identical to the saved-conditions one - which would make the whole
// difference untestable. Measured: cap 0 reproduces the no-rake EVs exactly,
// cap 2 differs (call 60.15625 vs 64.375).
const R2 = await saveRulesForm({rake_percent: "5", rake_cap_bb: "2"});
check("the_rules_really_changed", R2 !== R1, `${R1.slice(0, 8)} -> ${R2.slice(0, 8)}`);
const afterRules = await recordIds();
check("changing_the_rules_does_not_touch_the_existing_records",
      afterRules.length === 2 && afterRules[0] === bId && afterRules[1] === aId,
      `ids=${afterRules.length}`);
const firstAfterRules = await openRecord(aId);
check("an_older_rules_record_is_still_viewable_as_history",
      firstAfterRules.includes("Qs Qd")
      && (firstAfterRules.includes("历史规则版本")
          || firstAfterRules.includes("历史版本")),
      firstAfterRules.replace(/\s+/g, " ").slice(0, 80));
check("the_historical_record_says_it_is_not_a_current_recomputation",
      firstAfterRules.includes("不是用当前桌规重算"), "labelled history");

// ---- A / P1: the rule-source control must really switch the mode ---------
// Loading the saved conditions puts the form into the RECORD's own rules
// scenario with the control OFF. Ticking that control through its real change
// handler must LEAVE the scenario: otherwise the form would keep sending the old
// document rules while the control says "use this table's rules".
await openRecord(aId);
await fire(node("records-recompute-saved"), "click", {});
const savedLoaded = await waitUntil(loaded);
check("the_saved_conditions_load_selects_the_records_own_rules",
    savedLoaded && safeRun("handScenario") !== null
    && safeRun("handScenario").rules.rake_percent === "0"
    && node("hand-use-rules").checked === false
    && node("hand-hero").value === "Qs Qd",
    `rake=${safeRun("handScenario") && safeRun("handScenario").rules.rake_percent} `
    + `checked=${node("hand-use-rules").checked}`);
await click(node("hand-build"));
check("the_saved_scenario_verifies_as_a_document_rule_input",
    run("handReceipt") !== null && run("handReceipt").rules_source === "document"
    && run("handBuilt").rules.rake_percent === "0",
    `source=${run("handReceipt") && run("handReceipt").rules_source} `
    + `doc_rake=${run("handBuilt") && run("handBuilt").rules.rake_percent}`);
node("hand-use-rules").checked = true;
await fire(node("hand-use-rules"), "change", {});
check("ticking_the_real_rule_source_control_leaves_the_saved_scenario",
    safeRun("handScenario") === null && run("handReceipt") === null
    && run("handBuilt") === null && run("acceptedAnalysisId") === null
    && node("hand-use-rules").checked === true
    && /保存条件模式已退出/.test(node("hand-status").textContent),
    `scenario=${safeRun("handScenario")} `
    + `status=${node("hand-status").textContent.slice(0, 44)}`);
check("the_switch_keeps_the_hand_the_assumptions_and_the_parent_link",
    node("hand-hero").value === "Qs Qd"
    && node("hand-board").value === "2c 4d 7h 9s Jc"
    && run('handRows("ranges")').length === RANGES_A.length
    && run('handRows("ranges")')[0].combo === "JhJd"
    && run('handRows("weights")')[1].weight === "2"
    && node("hand-targets").value === "20,40,80"
    && run("handSource").parent_analysis_record_id === aId
    && run("handSource").issue_id === null,
    JSON.stringify(run("handSource")));
const beforeSwitchBuild = calls.length;
const switched = await verifyAndCompute();
const switchedBuild = calls.slice(beforeSwitchBuild).find(
  call => call.path === "/api/hand-input/build" && call.method === "POST");
const switchedBody = switchedBuild ? JSON.parse(switchedBuild.body) : null;
check("the_switched_verify_asks_the_server_for_the_TABLE_rules",
    switchedBody && switchedBody.rules_source === "table"
    && run("handReceipt") && run("handReceipt").rules_source === "table"
    && run("handBuilt").rules.rake_percent === "0.05",
    `sent=${switchedBody && switchedBody.rules_source} `
    + `doc_rake=${run("handBuilt") && run("handBuilt").rules.rake_percent}`);
check("the_switched_recompute_completes_on_the_real_kernel",
    switched?.status === "COMPLETE", `status=${switched?.status} error=${switched?.error}`);
const switchMessage = await saveCurrent("切本桌规则重算 A");
const afterSwitch = await recordIds();
check("the_switched_recompute_saves_as_a_NEW_record",
    /已保存本次分析/.test(switchMessage) && afterSwitch.length === 3
    && afterSwitch.includes(aId) && afterSwitch[0] !== aId,
    `${switchMessage.slice(0, 40)} ids=${afterSwitch.length}`);
const eId = afterSwitch[0];
const eView = await storedView(eId);
const tableRules = (await serverRules()).simulation_rules;
check("the_switched_record_uses_THIS_table_rules_and_keeps_its_parent",
    eView && eView.parent === aId
    && eView.source_kind === "recomputed_from_analysis_record"
    && canonical(eView.effective_rules) === canonical(tableRules)
    && eView.hero === aView.hero && eView.seat1 === aView.seat1
    && eView.job_id !== aView.job_id,
    `parent=${eView && eView.parent} job=${eView && eView.job_id} `
    + `record_rules=${canonical(eView && eView.effective_rules).slice(0, 150)} `
    + `table_rules=${canonical(tableRules).slice(0, 150)}`);
check("the_switched_rules_actually_changed_the_numbers",
    eView && eView.call_ev !== aView.call_ev && eView.raise_ev !== aView.raise_ev
    && eView.raise_exact !== aView.raise_exact,
    `A call=${aView.call_ev}/raise=${aView.raise_ev} -> `
    + `switched call=${eView && eView.call_ev}/raise=${eView && eView.raise_ev}`);

// Switching DURING an in-flight verify must not let the old answer restore the
// saved-rules mode or hand back a receipt for it.
await openRecord(aId);
await fire(node("records-recompute-saved"), "click", {});
await waitUntil(loaded);
const buildGate = hold((path, method) =>
  path === "/api/hand-input/build" && method === "POST");
fire(node("hand-build"), "click", {});
node("hand-use-rules").checked = true;
await fire(node("hand-use-rules"), "change", {});
buildGate.release();
await new Promise(resolve => setTimeout(resolve, 400));
check("switching_the_rule_source_during_an_in_flight_verify_discards_the_answer",
    safeRun("handScenario") === null && run("handReceipt") === null
    && run("handBuilt") === null && handTag() !== "已核对"
    && node("hand-compute").disabled === true,
    `tag=${handTag()} receipt=${run("handReceipt")}`);

// ---- A / P1: recompute with the CURRENT rules -> new record C ----------
// A live receipt that belongs to ANOTHER hand (a fresh 2h3d analysis under the
// CURRENT rules). Opening A and recomputing it must destroy that receipt, the
// built document and the accepted job BEFORE anything of A is loaded - the old
// defect left all three alive while the form showed A.
typeHand({hero: "2h 3d", pot: "130", ranges: RANGES_B, weights: WEIGHTS_B});
const heldAnalysis = await verifyAndCompute();
check("the_recompute_starts_from_an_existing_receipt",
      heldAnalysis?.status === "COMPLETE" && run("handReceipt") !== null
      && run("acceptedAnalysisId") !== null
      && run("handReceipt").facts.hero_cards.value.join(" ") === "2h 3d",
      `status=${heldAnalysis?.status} receipt_hero=`
      + `${run("handReceipt") && run("handReceipt").facts.hero_cards.value.join(" ")}`);
const aBefore = await storedView(aId);
await openRecord(aId);
await fire(node("records-recompute-current"), "click", {});
const currentLoaded = await waitUntil(loaded);
check("recompute_with_current_rules_voids_the_old_receipt_and_result",
      currentLoaded && run("handReceipt") === null
      && run("handBuilt") === null && run("acceptedAnalysisId") === null
      && safeRun("handScenario") === null,
      `receipt=${run("handReceipt")} built=${run("handBuilt")}`);
check("recompute_with_current_rules_loads_A_facts_not_the_previous_hand",
      node("hand-hero").value === "Qs Qd" && node("hand-board").value
      === "2c 4d 7h 9s Jc" && node("hand-ended").checked === true,
      `hero=${node("hand-hero").value}`);
check("recompute_with_current_rules_loads_A_own_opponent_assumptions",
      node("hand-targets").value === "20,40,80"
      && run("handRows(\"ranges\")").length === RANGES_A.length
      && run("handRows(\"ranges\")")[0].combo === "JhJd"
      && run("handRows(\"weights\")").length === WEIGHTS_A.length
      && run("handRows(\"weights\")")[1].weight === "2",
      JSON.stringify(run('handRows("ranges")').slice(0, 2)));
check("recompute_with_current_rules_is_not_wired_to_an_analysis_record_id",
      run("handSource").issue_id === null
      && run("handSource").parent_analysis_record_id === aId
      && run("handSource").analysis_record_id === aId,
      JSON.stringify(run("handSource")));
check("recompute_with_current_rules_does_not_change_the_table_rules",
      await serverRevision() === R2, `revision=${String(await serverRevision()).slice(0, 8)}`);
const recomputed = await verifyAndCompute();
check("the_current_rules_recompute_completes_on_the_real_kernel",
      recomputed?.status === "COMPLETE",
      `status=${recomputed?.status} error=${recomputed?.error}`);
const currentMessage = await saveCurrent("按当前规则重算 A");
const afterCurrent = await recordIds();
check("the_current_rules_recompute_saves_as_a_NEW_record",
    /已保存本次分析/.test(currentMessage) && afterCurrent.length === 4
    && afterCurrent.includes(aId) && afterCurrent.includes(bId)
    && afterCurrent[0] !== aId && afterCurrent[0] !== bId
    && afterCurrent[0] !== eId,
    `${currentMessage.slice(0, 40)} ids=${afterCurrent.length}`);
const cId = afterCurrent[0];
const cView = await storedView(cId);
check("the_new_current_rules_record_names_its_parent_and_its_own_rules",
    cView && cView.parent === aId && cView.rake === "0.05"
    && cView.source_kind === "recomputed_from_analysis_record"
    && cView.job_id !== aView.job_id && cView.hero === aView.hero
    && cView.seat1 === aView.seat1 && cView.bet_weight === aView.bet_weight,
    `rake=${cView && cView.rake} parent=${cView && cView.parent}`);
check("the_two_current_rules_entries_reach_the_same_rules_by_different_controls",
    cView && eView
    && canonical(cView.effective_rules) === canonical(eView.effective_rules)
    && cView.job_id !== eView.job_id,
    `C job=${cView && cView.job_id} E job=${eView && eView.job_id}`);
check("the_original_A_record_is_unchanged_by_the_recompute",
      sameView(await storedView(aId), aBefore), "A untouched");

// ---- B / P1: request identity, out-of-order and forged responses -------
const gateB = hold((path) => path.endsWith(`/records/${bId}`));
const gateA = hold((path) => path.endsWith(`/records/${aId}`));
node("records-select").value = aId;
fire(node("records-open"), "click", {});
node("records-select").value = bId;
await fire(node("records-select"), "change", {});
fire(node("records-open"), "click", {});
gateB.release();
await run("poll()");
const ordered = await waitUntil(() => viewText().includes("2h 3d"));
gateA.release();
await new Promise(resolve => setTimeout(resolve, 300));
check("an_out_of_order_open_paints_only_the_current_selection",
      ordered && node("records-select").value === bId
      && viewText().includes("2h 3d") && !viewText().includes("Qs Qd"),
      `select=${node("records-select").value} `
      + `text=${viewText().replace(/\s+/g, " ").slice(0, 70)}`);

// A response that names a DIFFERENT record than the one requested.
const forged = hold((path) => path.endsWith(`/records/${bId}`));
forged.forge = () => ({ok: true, status: 200, text: async () => "{}",
                       json: async () => JSON.parse(readFileSync(
                         join(process.env.RECORDS_DIR || ".", "analysis-records",
                              aId, "record.json"), "utf8"))});
node("records-select").value = bId;
fire(node("records-open"), "click", {});
forged.release();
await new Promise(resolve => setTimeout(resolve, 300));
check("a_response_for_another_record_is_refused_without_numbers",
      node("records-select").value === bId && !viewText().includes("Qs Qd")
      && !/本街追加/.test(viewText()) && /不是同一条/.test(feedback()),
      feedback().slice(0, 70));
check("no_recompute_button_survives_a_refused_open",
      node("records-recompute-saved").disabled === true
      && node("records-recompute-current").disabled === true,
      "both recompute buttons are disabled");

// A failed open must also leave no numbers behind.
node("records-select").value = aId;
await fire(node("records-select"), "change", {});
const failGate = hold((path) => path.endsWith(`/records/${aId}`));
failGate.forge = () => ({ok: false, status: 400,
                         text: async () => JSON.stringify({detail: "打不开"}),
                         json: async () => ({detail: "打不开"})});
fire(node("records-open"), "click", {});
failGate.release();
await new Promise(resolve => setTimeout(resolve, 300));
check("a_failed_open_leaves_no_numbers_and_no_recompute",
      !viewText().includes("本街追加") && /打开失败/.test(feedback())
      && node("records-recompute-saved").disabled === true
      && node("records-recompute-current").disabled === true,
      feedback().slice(0, 60));

// A late recompute answer must not repopulate a form the human changed.
await openRecord(aId);
const lateGate = hold((path) => path.endsWith(`/records/${aId}/scenario`));
fire(node("records-recompute-saved"), "click", {});
run("handClear()");
const clearedHero = node("hand-hero").value;
lateGate.release();
await new Promise(resolve => setTimeout(resolve, 400));
check("a_late_recompute_answer_does_not_touch_a_form_cleared_while_it_flew",
      node("hand-hero").value === clearedHero && run("handReceipt") === null
      && safeRun("handScenario") === null && !viewText().includes("本街追加")
      && /清空/.test(node("hand-status").textContent),
      `hero=${JSON.stringify(node("hand-hero").value)} `
      + `status=${node("hand-status").textContent.slice(0, 40)}`);

// A late list refresh must not steal a selection the human just changed.
await openRecord(aId);
const refreshGate = hold((path, method) =>
  path === "/api/analysis/records" && method === "GET");
fire(node("records-refresh"), "click", {});
node("records-select").value = bId;
await fire(node("records-select"), "change", {});
refreshGate.release();
await new Promise(resolve => setTimeout(resolve, 400));
check("a_late_list_refresh_does_not_steal_the_selection",
      node("records-select").value === bId,
      `select=${node("records-select").value}`);

// ---- C / P1: the stored CONTENT is sealed and re-verified ---------------
const victim = aId;
const recordPath = join(process.env.RECORDS_DIR || "", "analysis-records",
                        victim, "record.json");
if (process.env.RECORDS_DIR) {
  const pristine = readFileSync(recordPath, "utf8");
  const TAMPER_CASES = [
    ["both_halves_of_an_ev",
     doc => { doc.report.result.root_actions[2].ev.exact = "999/2";
              doc.report.result.root_actions[2].ev.decimal = "499.5"; }],
    ["a_raise_target_off_the_verified_grid",
     doc => { doc.report.result.root_actions[2].action.target = "999"; }],
    ["a_raise_additional_cost",
     doc => { doc.report.result.root_actions[2].additional_cost = "999"; }],
    ["the_source_link",
     doc => { doc.source.issue_id = "20260101T000000-aaaaaaaaaaaa"; }],
    ["the_job_id", doc => { doc.job.job_id = "job-elsewhere"; }],
    ["the_implied_pot", doc => { doc.capacity.implied_pot = "999999"; }],
    ["the_record_id", doc => { doc.record_id = "20260101T000000-abcdefabcdef"; }],
    ["the_root_action_actor",
     doc => { doc.report.result.root_actions[2].action.actor = 3; }],
  ];
  let refused = 0, preserved = 0;
  for (const [label, mutate] of TAMPER_CASES) {
    const document = JSON.parse(pristine);
    mutate(document);
    writeFileSync(recordPath, JSON.stringify(document, null, 2));
    const opened = await api(`/api/analysis/records/${victim}`);
    const listed = (await api("/api/analysis/records")).body.items
      .find(row => row.record_id === victim);
    const scenario = await api(`/api/analysis/records/${victim}/scenario`);
    if (opened.body.display_permitted === false && opened.body.view === null
        && opened.body.status === "INVALID" && listed.display_permitted === false
        && listed.status === "INVALID" && scenario.ok === false) refused += 1;
    if (readFileSync(recordPath, "utf8").length > 0) preserved += 1;
  }
  check("every_rewritten_content_field_is_refused_on_every_read_path",
      refused === TAMPER_CASES.length,
      `${refused}/${TAMPER_CASES.length} refused by get+list+scenario`);
  check("a_refused_record_is_never_deleted_or_rewritten",
      preserved === TAMPER_CASES.length, `${preserved}/${TAMPER_CASES.length} kept`);
  // A record whose content was never sealed is named, not silently approved.
  const unsealed = JSON.parse(pristine);
  delete unsealed.content_seal;
  writeFileSync(recordPath, JSON.stringify(unsealed, null, 2));
  const legacy = await api(`/api/analysis/records/${victim}`);
  check("a_legacy_record_without_a_content_seal_is_not_silently_accepted",
      legacy.body.display_permitted === false && legacy.body.view === null
      && legacy.body.status === "UNVERIFIED_FORMAT"
      && /旧格式|封存/.test((legacy.body.notes || []).join(" ")),
      `status=${legacy.body.status}`);
  writeFileSync(recordPath, pristine);
  const restored = await api(`/api/analysis/records/${victim}`);
  check("restoring_the_original_file_makes_it_readable_again",
      restored.body.display_permitted === true
      && sameView(await storedView(victim), aBefore),
      `status=${restored.body.status}`);

  // The existing display-only negative stays covered.
  const displayOnly = JSON.parse(pristine);
  displayOnly.report.result.root_actions[1].ev.decimal = "999.999";
  writeFileSync(recordPath, JSON.stringify(displayOnly, null, 2));
  const brokenRow = (await api("/api/analysis/records")).body.items
    .find(row => row.record_id === victim);
  const brokenOpen = await api(`/api/analysis/records/${victim}`);
  check("an_edited_display_value_is_reported_as_invalid",
      brokenRow.display_permitted === false && brokenRow.status === "INVALID"
      && brokenOpen.body.display_permitted === false && brokenOpen.body.view === null,
      `status=${brokenRow.status}`);
  const brokenText = await openRecord(victim);
  check("the_ui_refuses_to_show_numbers_for_a_broken_record",
      brokenText.includes("不能展示数值") && !brokenText.includes("服务端实际完成结果")
      && !brokenText.includes("本街追加"),
      brokenText.replace(/\s+/g, " ").slice(0, 70));
  writeFileSync(recordPath, pristine);

  // ---- B / P2: a valid-JSON but wrong-SHAPED file, through the real routes ---
  const initialList = await api("/api/analysis/records");
  const healthyBefore = (((initialList.body || {}).items) || [])
    .filter(item => item.display_permitted === true).map(item => item.record_id);
  const SHAPE_ID = "20260101T000000-5a5a5a5a5a5a";
  const shapeFolder = join(process.env.RECORDS_DIR, "analysis-records", SHAPE_ID);
  const shapePath = join(shapeFolder, "record.json");
  mkdirSync(shapeFolder, {recursive: true});
  const shapeCases = [["an_array", "[]"], ["null", "null"],
                      ["a_string", '"not-an-object"']];
  let shapeServed = 0, shapeRefused = 0, shapeKept = 0, shapeMissing = [];
  for (const [label, payload] of shapeCases) {
    writeFileSync(shapePath, payload);
    const listed = await api("/api/analysis/records");
    const items = ((listed.body || {}).items) || [];
    const row = items.find(item => item.record_id === SHAPE_ID);
    shapeMissing = healthyBefore.filter(id => !items.some(
      item => item.record_id === id && item.display_permitted === true));
    const healthyIntact = healthyBefore.length >= 2 && shapeMissing.length === 0;
    const opened = await api(`/api/analysis/records/${SHAPE_ID}`);
    const scenario = await api(`/api/analysis/records/${SHAPE_ID}/scenario`);
    if (listed.ok && row && row.display_permitted === false
        && row.status === "INVALID" && healthyIntact) shapeServed += 1;
    if (opened.status === 400 && scenario.status === 400) shapeRefused += 1;
    if (readFileSync(shapePath, "utf8") === payload) shapeKept += 1;
    console.log(`# shape ${label}: list=${listed.status} `
                + `row=${row ? row.status : "missing"} get=${opened.status} `
                + `scenario=${scenario.status} healthy=${healthyIntact} `
                + `missing=${JSON.stringify(shapeMissing)}`);
  }
  check("a_wrong_shaped_file_is_listed_as_invalid_and_does_not_hide_the_others",
      shapeServed === shapeCases.length,
      `${shapeServed}/${shapeCases.length} served; healthy=${healthyBefore.length} `
      + `missing=${JSON.stringify(shapeMissing)}`);
  check("get_and_scenario_answer_a_wrong_shaped_file_with_a_refusal_not_a_crash",
      shapeRefused === shapeCases.length,
      `${shapeRefused}/${shapeCases.length} answered 400 (no 5xx)`);
  check("a_wrong_shaped_file_is_never_deleted_or_rewritten",
      shapeKept === shapeCases.length, `${shapeKept}/${shapeCases.length} kept`);
  const afterShape = await openRecord(aId);
  check("the_records_panel_still_reads_the_healthy_records_after_a_bad_shape",
      afterShape.includes("Qs Qd") && afterShape.includes("本街追加"),
      afterShape.replace(/\s+/g, " ").slice(0, 70));
  rmSync(shapeFolder, {recursive: true, force: true});
} else {
  check("every_rewritten_content_field_is_refused_on_every_read_path", false,
        "RECORDS_DIR not provided");
}

// ---- the stored result equals a DIRECT kernel call on its own input ------
// Last, because this posts a fresh analysis and so replaces the live job.
{
  const openedE = await api(`/api/analysis/records/${eId}`);
  const direct = await directAnalysis(openedE.body.document.input, "table");  const directEvs = ((direct.result || {}).root_actions || [])
    .map(row => row.ev.decimal);
  check("the_switched_record_equals_a_direct_kernel_call_on_its_own_input",
      direct.status === "COMPLETE"
      && JSON.stringify(directEvs) === JSON.stringify(eView.evs)
      && direct.input_sha256 === openedE.body.identity.input_sha256,
      `direct=${JSON.stringify(directEvs)} stored=${JSON.stringify(eView.evs)}`);
}

const finalViews = {[aId]: await storedView(aId), [bId]: await storedView(bId),
                    [eId]: await storedView(eId), [cId]: await storedView(cId)};
writeFileSync(handoff, JSON.stringify({
  ids: [aId, bId, eId, cId],
  views: [finalViews[aId], finalViews[bId], finalViews[eId], finalViews[cId]],
  rules: R1, rules_after: R2, a: aId, b: bId, e: eId, c: cId,
  views_by_id: finalViews}), "utf8");

check("no_innerhtml_used", dom.state.innerHTMLWrites === 0,
      `writes=${dom.state.innerHTMLWrites}`);
const failed = checks.filter(item => !item.ok);
console.log(JSON.stringify({
  name: "verdict", ok: failed.length === 0, passed: checks.length - failed.length,
  failed: failed.length, requests: calls.length, phase, base}));
process.exit(failed.length === 0 ? 0 : 1);
