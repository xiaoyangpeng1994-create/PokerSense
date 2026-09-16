// U2 harness: compute THIS input -> save -> restart the service -> reopen.
//
// The shipped scripts are evaluated UNCHANGED in the page's own order
// (app.js -> controls.js -> analysis.js -> hand_input.js -> analysis_records.js)
// in a DOM stub, against the real backend and the real kernel.
//
// It runs in two phases so a genuine restart can be exercised:
//   run1: save the table rules, compute and save TWO different synthetic inputs,
//         run the counter-examples, and write the record ids plus the numbers it
//         saw to a hand-off file;
//   run2: against a NEW server process on the SAME records directory, reopen both
//         records and check the numbers are identical without any recomputation.
//
// Usage: node hand_records_flow_test.mjs <base-url> <run1|run2> <handoff-file>

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import { createDom, fire, click } from "./dom_stub.mjs";

const base = (process.argv[2] || "").replace(/\/$/, "");
const phase = process.argv[3] || "";
const handoff = process.argv[4] || "";
if (!base.startsWith("http://127.0.0.1:") || !["run1", "run2"].includes(phase)
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
async function fetchStub(path, options = {}) {
  calls.push({path, method: (options.method || "GET").toUpperCase(),
              body: options.body ?? null});
  const response = await fetch(base + path, options);
  const body = await response.text();
  return {ok: response.ok, status: response.status, text: async () => body,
          json: async () => JSON.parse(body)};
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
const node = id => run(`el(${JSON.stringify(id)})`);
const exposed = name => run(`typeof ${name}`) !== "undefined";
const feedback = () => node("records-status").textContent;
const viewText = () => node("records-view").textContent;
async function waitUntil(predicate, timeout = 15000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise(resolve => setTimeout(resolve, 60));
  }
  return false;
}
const api = async (path, options) => {
  const response = await fetch(base + path, options);
  const text = await response.text();
  return {ok: response.ok, status: response.status,
          body: text ? JSON.parse(text) : null};
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
  // controls.js loads the saved rules asynchronously; filling the form before
  // that response lands would be overwritten by it.
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
const RANGES = [
  {seat_id: 1, combo: "JhJd", weight: "1"}, {seat_id: 1, combo: "TcTd", weight: "3"},
  {seat_id: 2, combo: "7c7s", weight: "1"}, {seat_id: 2, combo: "KhTh", weight: "3"}];
const WEIGHTS = [
  {seat_id: 1, key: "check", weight: "1"}, {seat_id: 1, key: "bet", weight: "2"},
  {seat_id: 1, key: "call", weight: "9"}, {seat_id: 1, key: "fold", weight: "1"},
  {seat_id: 2, key: "check", weight: "1"}, {seat_id: 2, key: "call", weight: "9"},
  {seat_id: 2, key: "fold", weight: "1"}];
function typeHand(overrides = {}) {
  node("hand-hero").value = overrides.hero ?? "Qs Qd";
  node("hand-board").value = "2c 4d 7h 9s Jc";
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
  run(`handFillRows("history", ${JSON.stringify(overrides.history ?? [
    {actor: 1, kind: "bet", target: "20"}, {actor: 2, kind: "call", target: "0"}])})`);
  run(`handFillRows("ranges", ${JSON.stringify(RANGES)})`);
  run(`handFillRows("weights", ${JSON.stringify(WEIGHTS)})`);
  for (const handler of node("hand-hero").handlers.input || []) handler({});
}
async function verifyAndCompute() {
  await click(node("hand-build"));
  if (run("handBuilt") === null) {
    // Not a check: a diagnosis so a failing run says WHY the build was refused.
    console.log("# build refused: " + (node("hand-status").textContent + " | "
      + node("hand-gaps").textContent).replace(/\s+/g, " ").slice(0, 200));
    return null;
  }
  await click(node("hand-compute"));
  const job = run("acceptedAnalysisId");
  for (let index = 0; index < 120; index += 1) {
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
const saveOutcome = /已保存本次分析|已经保存过|保存失败|请先核对|不一致|拒绝/;
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
  const before = viewText();
  await fire(node("records-open"), "click", {});
  await waitUntil(() => viewText() !== before || viewText().length > 0);
  return viewText();
}
const recordIds = async () => (await api("/api/analysis/records")).body.items
  .map(row => row.record_id);

// --- phase 2: reopen after the restart ---------------------------------
if (phase === "run2") {
  const saved = JSON.parse(readFileSync(handoff, "utf8"));
  const listed = (await api("/api/analysis/records")).body;
  check("run2_the_records_survive_the_restart",
      listed.items.length === saved.ids.length
      && saved.ids.every(id => listed.items.some(row => row.record_id === id)),
      `after restart=${listed.items.length} before=${saved.ids.length}`);
  check("run2_every_record_still_passes_its_self_check",
      listed.items.every(row => row.display_permitted === true),
      String(listed.items.map(row => row.status).join(",")));
  for (const [index, id] of saved.ids.entries()) {
    const text = await openRecord(id);
    const expected = saved.views[index];
    check(`run2_record_${index + 1}_reopens_with_the_same_numbers`,
        text.includes(expected.hero) && text.includes(expected.call_ev)
        && text.includes(expected.raise_ev) && text.includes(expected.exact_call),
        `hero=${expected.hero} call=${expected.call_ev}`);
  }
  check("run2_reopen_did_not_run_the_kernel",
      calls.filter(call => call.path === "/api/analysis"
             && call.method === "POST").length === 0,
      `${calls.length} requests, none analytical`);
  const failed = checks.filter(item => !item.ok);
  console.log(JSON.stringify({
    name: "verdict", ok: failed.length === 0, passed: checks.length - failed.length,
    failed: failed.length, requests: calls.length, phase, base}));
  process.exit(failed.length === 0 ? 0 : 1);
}

// --- phase 1 ------------------------------------------------------------
check("the_page_scripts_are_loaded",
      exposed("handWired") && exposed("recordsSave") && exposed("recordsRefresh")
      && exposed("handReceipt"),
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
const firstText = await openRecord(firstIds[0]);
check("the_saved_record_shows_this_input_not_a_fixed_sample",
      firstText.includes("Qs Qd") && firstText.includes("64.375")
      && firstText.includes("108.90625") && firstText.includes("515/8"),
      firstText.replace(/\s+/g, " ").slice(0, 90));
check("the_saved_record_labels_its_source_kind",
      firstText.includes("人工填写") && firstText.includes("已保留原候选"),
      "provenance is shown");
const firstView = {
  hero: "Qs Qd", call_ev: "64.375", raise_ev: "108.90625", exact_call: "515/8"};

// ---- input two (a different hand) --------------------------------------
typeHand({hero: "2h 3d", pot: "130"});
const second = await verifyAndCompute();
check("input_two_completes_on_the_real_kernel", second?.status === "COMPLETE",
      `status=${second?.status}`);
const secondSaved = await saveCurrent("合成输入二 23");
const secondIds = await recordIds();
check("the_second_analysis_is_a_second_record",
      /已保存本次分析/.test(secondSaved) && secondIds.length === 2
      && secondIds[0] !== secondIds[1],
      `${secondSaved.slice(0, 40)} ids=${secondIds.length}`);
const secondText = await openRecord(secondIds[0]);
check("the_second_record_has_the_second_hand_numbers",
      secondText.includes("2h 3d") && secondText.includes("-59.675")
      && secondText.includes("-2387/40"),
      secondText.replace(/\s+/g, " ").slice(0, 90));
const secondView = {
  hero: "2h 3d", call_ev: "-20", raise_ev: "-59.675", exact_call: "-20"};
check("the_two_records_are_not_the_same_result",
      firstText !== secondText && firstIds[0] !== secondIds[0],
      "different inputs, different records");

// ---- a repeated save does not duplicate --------------------------------
const repeatMessage = await saveCurrent("合成输入二 23");
const afterRepeat = await recordIds();
check("a_repeated_save_does_not_create_a_second_record",
      /已经保存过/.test(repeatMessage) && afterRepeat.length === 2,
      repeatMessage.slice(0, 40));

// ---- a stale job cannot be saved ---------------------------------------
typeHand({hero: "Ah Ad", pot: "130"});
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
const R2 = await saveRulesForm({rake_percent: "5"});
check("the_rules_really_changed", R2 !== R1, `${R1.slice(0, 8)} -> ${R2.slice(0, 8)}`);
const afterRules = await recordIds();
check("changing_the_rules_does_not_touch_the_existing_records",
      afterRules.length === 2 && afterRules[0] === secondIds[0]
      && afterRules[1] === secondIds[1],
      `ids=${afterRules.length}`);
const firstAfterRules = await openRecord(firstIds[0]);
check("an_older_rules_record_is_still_viewable_as_history",
      firstAfterRules.includes("Qs Qd")
      && (firstAfterRules.includes("历史规则版本")
          || firstAfterRules.includes("历史版本")),
      firstAfterRules.replace(/\s+/g, " ").slice(0, 80));
check("the_historical_record_says_it_is_not_a_current_recomputation",
      firstAfterRules.includes("不是用当前桌规重算"), "labelled history");
// The saved conditions still load, without touching the global rules.
const revisionBeforeRecompute = await serverRevision();
await fire(node("records-recompute-saved"), "click", {});
await waitUntil(() => node("analysis-status").textContent.includes("保存时的输入"));
check("recompute_under_saved_conditions_loads_the_saved_rules",
      node("analysis-input").value.includes("\"river_start\"")
      && node("analysis-use-rules").checked === false,
      node("analysis-status").textContent.slice(0, 60));
check("recompute_under_saved_conditions_does_not_change_the_table_rules",
      await serverRevision() === revisionBeforeRecompute,
      `revision=${String(await serverRevision()).slice(0, 8)}`);
await fire(node("records-recompute-current"), "click", {});
await waitUntil(() => node("analysis-status").textContent.includes("重新核对"));
check("recompute_with_current_rules_loads_the_facts_and_asks_for_a_recheck",
      node("hand-hero").value === "Qs Qd"
      && node("analysis-status").textContent.includes("当前桌规"),
      node("analysis-status").textContent.slice(0, 70));

// ---- tampering is detected through the real get/list path --------------
const victim = firstIds[0];
const recordPath = join(process.env.RECORDS_DIR || "", "analysis-records",
                        victim, "record.json");
if (process.env.RECORDS_DIR) {
  const document = JSON.parse(readFileSync(recordPath, "utf8"));
  const original = document.report.result.root_actions[1].ev.decimal;
  document.report.result.root_actions[1].ev.decimal = "999.999";
  writeFileSync(recordPath, JSON.stringify(document, null, 2));
  const listed = (await api("/api/analysis/records")).body.items;
  const row = listed.find(item => item.record_id === victim);
  const opened = await api(`/api/analysis/records/${victim}`);
  check("an_edited_display_value_is_reported_as_invalid",
      row.display_permitted === false && row.status === "INVALID"
      && opened.body.display_permitted === false && opened.body.view === null,
      `status=${row.status}`);
  const uiText = await openRecord(victim);
  check("the_ui_refuses_to_show_numbers_for_a_broken_record",
      uiText.includes("不能展示数值") && !uiText.includes("服务端实际完成结果")
      && !uiText.includes("本街追加"),
      uiText.replace(/\s+/g, " ").slice(0, 70));
  document.report.result.root_actions[1].ev.decimal = original;
  writeFileSync(recordPath, JSON.stringify(document, null, 2));
  const restored = (await api(`/api/analysis/records/${victim}`)).body;
  check("restoring_the_original_value_makes_it_readable_again",
      restored.display_permitted === true, `status=${restored.status}`);
} else {
  check("an_edited_display_value_is_reported_as_invalid", false,
        "RECORDS_DIR not provided");
}

writeFileSync(handoff, JSON.stringify(
  {ids: [secondIds[0], secondIds[1]], views: [secondView, firstView],
   rules: R1, rules_after: R2}), "utf8");

check("no_innerhtml_used", dom.state.innerHTMLWrites === 0,
      `writes=${dom.state.innerHTMLWrites}`);
const failed = checks.filter(item => !item.ok);
console.log(JSON.stringify({
  name: "verdict", ok: failed.length === 0, passed: checks.length - failed.length,
  failed: failed.length, requests: calls.length, phase, base}));
process.exit(failed.length === 0 ? 0 : 1);
