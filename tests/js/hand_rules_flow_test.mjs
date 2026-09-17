// U1-R3 harness: the table-rules lifecycle against a verified hand receipt.
//
// The shipped scripts are evaluated UNCHANGED in the page's own order
// (app.js -> controls.js -> analysis.js -> hand_input.js) in a DOM stub, against
// the real backend, so the rules save really runs controls.js's submit handler
// and the real POST /api/rules, and the external change really goes through the
// page's own poll().
//
// The matrix it drives:
//   R1 verify -> compute with no change            (positive control)
//   R1 verify -> THIS page saves R2 -> compute     (old receipt disabled)
//   R1 verify -> ANOTHER page saves R2 -> compute  (server refuses the old revision)
//   poll notices the external change               (receipt voided)
//   R1 build in flight -> save R2 -> R1 build lands late (must not restore verified)
//   R2 re-verify -> compute                        (real kernel, versions aligned)
//   reset / save failure / external revision change (no resurrection)
//
// Usage: node hand_rules_flow_test.mjs <base-url>

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import { createDom, fire, click } from "./dom_stub.mjs";

const base = (process.argv[2] || "").replace(/\/$/, "");
if (!base.startsWith("http://127.0.0.1:")) {
  console.log(JSON.stringify({name: "base_url", ok: false, detail: base}));
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
  const url = base + path;
  calls.push({path, method: (options.method || "GET").toUpperCase(),
              body: options.body ?? null});
  const response = await fetch(url, options);
  const body = await response.text();
  return {ok: response.ok, status: response.status, text: async () => body,
          json: async () => JSON.parse(body)};
}
// The page's own timers stay inert so the harness drives poll() explicitly and
// nothing races the assertions.
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
for (const name of ["app.js", "controls.js", "analysis.js", "hand_input.js"]) {
  vm.runInContext(read(name), context, {filename: name});
}
const run = expression => vm.runInContext(expression, context);
const node = id => run(`el(${JSON.stringify(id)})`);
const text = value => value === null || value === undefined || value === "UNKNOWN"
  ? "未知" : typeof value === "object" ? JSON.stringify(value) : String(value);
const feedback = () => node("hand-status").textContent;
const rulesFeedback = () => node("rules-feedback").textContent;
const results = () => node("analysis-results").textContent;
const analysisStatus = () => node("analysis-status").textContent;
const exportEnabled = () => node("analysis-export").disabled === false;
const analysisPosts = () => calls.filter(call => call.path === "/api/analysis"
                                          && call.method === "POST");
const lastAnalysisBody = () => {
  const posts = analysisPosts();
  return posts.length ? JSON.parse(posts[posts.length - 1].body) : null;
};
const exposed = name => run(`typeof ${name}`) !== "undefined";
// Values an older head simply does not have yet must fail a check, not throw.
const receiptRevision = () => exposed("handReceipt")
  ? run("handReceipt")?.rules_revision : null;
const receiptSource = () => exposed("handReceipt")
  ? run("handReceipt")?.rules_source : null;
const receiptRulesDigest = () => exposed("handReceipt")
  ? run("handReceipt")?.effective_rules_sha256 : null;
const hasReceipt = () => exposed("handReceipt") && run("handReceipt") !== null;
async function waitUntil(predicate, timeout = 8000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise(resolve => setTimeout(resolve, 60));
  }
  return false;
}
async function settleAnalysis() {
  const job = run("acceptedAnalysisId");
  let last = null;
  for (let index = 0; index < 120; index += 1) {
    await run("poll()");
    last = run("statusData").analysis;
    if (job !== null && last && last.job_id === job && last.status !== "RUNNING"
        && last.status !== "IDLE") return last;
    await new Promise(resolve => setTimeout(resolve, 120));
  }
  return last;
}

// --- page-level fixtures ------------------------------------------------
node("mode").options = [{disabled: false}, {disabled: false}];
const RULE_VALUES = {
  table_label: "R3 规则校验", dealt_players: "6", small_blind: "2",
  big_blind: "4", ante: "4", straddle_mode: "none", straddle_amount: "0",
  rake_percent: "0", rake_cap_bb: "0", minimum_chip: "1",
  rake_application: "all_pots", rake_rounding: "exact",
  rake_distribution: "proportional_all_pots", insurance: "off", bomb: "off",
  mushroom: "off"};
function fillRuleForm(overrides = {}) {
  for (const [key, value] of Object.entries({...RULE_VALUES, ...overrides})) {
    node(`rule-${key}`).value = value;
  }
}
async function submitRulesForm() {
  await fire(node("rules-form"), "submit", {preventDefault() {}});
}
const serverRules = async () => (await (await fetch(base + "/api/rules")).json());
const serverRevision = async () => (await serverRules()).revision;
// controls.js's submit handler starts saveRules() and returns immediately, so a
// save is only done when the server AND the page agree on the new revision.
async function saveRulesForm(overrides = {}) {
  fillRuleForm(overrides);
  await submitRulesForm();
  // The page's own status must catch up too: saveRules ends with await poll().
  await waitUntil(async () => /^[a-f0-9]{64}$/.test(run("rulesRevision"))
    && run("rulesRevision") === await serverRevision()
    && await serverRevision() === run("statusData").table_rules?.revision);
  return run("rulesRevision");
}
async function resetRulesForm() {
  await fire(node("rules-reset"), "click", {preventDefault() {}});
  await waitUntil(async () => run("rulesRevision") === await serverRevision()
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
const HISTORY = [
  {actor: 1, kind: "bet", target: "20"}, {actor: 2, kind: "call", target: "0"}];
const RANGES = [
  {seat_id: 1, combo: "JhJd", weight: "1"}, {seat_id: 1, combo: "TcTd", weight: "3"},
  {seat_id: 2, combo: "7c7s", weight: "1"}, {seat_id: 2, combo: "KhTh", weight: "3"}];
const WEIGHTS = [
  {seat_id: 1, key: "check", weight: "1"}, {seat_id: 1, key: "bet", weight: "2"},
  {seat_id: 1, key: "call", weight: "9"}, {seat_id: 1, key: "fold", weight: "1"},
  {seat_id: 2, key: "check", weight: "1"}, {seat_id: 2, key: "call", weight: "9"},
  {seat_id: 2, key: "fold", weight: "1"}];
function typeHand() {
  node("hand-hero").value = "Qs Qd";
  node("hand-board").value = "2c 4d 7h 9s Jc";
  node("hand-hero-seat").value = "0";
  node("hand-order").value = "1,2,0";
  node("hand-pot").value = "130";
  node("hand-targets").value = "20,40,80";
  node("hand-max-agg").value = "2";
  node("hand-ended").checked = true;
  node("hand-use-rules").checked = true;
  node("hand-no-history").checked = false;
  node("hand-fees").value = "confirmed_zero";
  run(`handFillRows("seats", ${JSON.stringify(SEATS)})`);
  run(`handFillRows("history", ${JSON.stringify(HISTORY)})`);
  run(`handFillRows("ranges", ${JSON.stringify(RANGES)})`);
  run(`handFillRows("weights", ${JSON.stringify(WEIGHTS)})`);
  for (const handler of node("hand-hero").handlers.input || []) handler({});
}
async function verify() {
  await click(node("hand-build"));
  return run("handBuilt") !== null;
}

// --- 0. wiring ----------------------------------------------------------
check("the_page_scripts_are_loaded",
      exposed("handWired") && exposed("saveRules") && exposed("poll")
      && exposed("handRulesChanged") && exposed("handRulesRevisionSeen"),
      "app.js + controls.js + analysis.js + hand_input.js");
// --- R1: save the initial table rules through controls.js ---------------
const loaded = await waitUntil(() => run("rulesRevision") !== null, 5000);
check("controls_loadRules_ran", loaded, `revision=${String(run("rulesRevision")).slice(0, 12)}`);
const R1 = await saveRulesForm({rake_percent: "0"});
const r1Saved = (await serverRules()).conditional_analysis_ready === true;
check("r1_is_saved_through_the_real_controls_handler",
      r1Saved && /^[a-f0-9]{64}$/.test(R1)
      && run("statusData").table_rules?.revision === R1,
      `R1=${String(R1).slice(0, 12)} ready=${r1Saved} `
      + `page=${String(run("statusData").table_rules?.revision).slice(0, 12)}`);

// --- case 1: R1 verify -> compute with no change -------------------------
typeHand();
check("r1_hand_verifies", await verify(), feedback().slice(0, 60));
check("the_receipt_carries_the_verified_rules_revision",
      receiptRevision() === R1 && receiptSource() === "table"
      && typeof receiptRulesDigest() === "string",
      `receipt=${String(receiptRevision()).slice(0, 12)}`);
await click(node("hand-compute"));
const firstReport = await settleAnalysis();
check("case1_r1_compute_is_accepted_and_aligned",
      firstReport?.status === "COMPLETE" && results().includes("条件净 EV")
      && exportEnabled()
      && lastAnalysisBody().rules_revision === R1
      && run("analysisBinding").expected_rules_revision === R1
      && run("analysisBinding").table_rules_revision === R1,
      `status=${firstReport?.status} sent=${String(lastAnalysisBody()?.rules_revision).slice(0, 12)}`);

// --- case 2: THIS page saves R2 -> the old receipt must die -------------
typeHand();
check("the_hand_verifies_before_the_same_page_rule_save", await verify(),
      feedback().slice(0, 50));
const postsBeforeSave = analysisPosts().length;
const R2 = await saveRulesForm({rake_percent: "3"});
check("the_same_page_rule_save_really_changed_the_server",
      R2 !== R1 && run("statusData").table_rules?.revision === R2,
      `R2=${R2.slice(0, 12)}`);
check("case2_the_same_page_save_voids_the_hand_receipt",
      run("handBuilt") === null && !hasReceipt()
      && run("handExpectedInput") === null
      && node("hand-compute").disabled === true,
      `built=${run("handBuilt")} computeDisabled=${node("hand-compute").disabled}`);
check("case2_the_typed_hand_survives_for_re_verification",
      node("hand-hero").value === "Qs Qd" && node("hand-pot").value === "130"
      && run('handRows("seats").length') === SEATS.length,
      `hero=${node("hand-hero").value} seats=${run('handRows("seats").length')}`);
check("case2_the_shown_result_is_cleared",
      results() === "" && exportEnabled() === false
      && analysisStatus().includes("失效"), analysisStatus().slice(0, 50));
await click(node("hand-compute"));
check("case2_the_old_receipt_cannot_be_computed",
      analysisPosts().length === postsBeforeSave
      && node("hand-compute").disabled === true,
      `posts=${analysisPosts().length} before=${postsBeforeSave}`);

// --- case 3: ANOTHER page saves R2 -> the server must refuse -----------
typeHand();
check("the_hand_verifies_at_r2", await verify(), feedback().slice(0, 50));
check("the_receipt_names_r2", receiptRevision() === R2,
      String(receiptRevision()).slice(0, 12));
// The external page: a plain POST that the local page never sees until it polls.
const external = await (await fetch(base + "/api/rules", {
  method: "POST", headers: {"Content-Type": "application/json", "X-AA-Live": "1"},
  body: JSON.stringify({
    document: {...RULE_VALUES, dealt_players: 6, rake_percent: "5"},
    revision: R2})})).json();
const R3 = await serverRevision();
check("the_other_page_changed_the_rules",
      R3 !== R2 && external.conditional_analysis_ready === true,
      `R3=${R3.slice(0, 12)}`);
check("this_page_has_not_polled_yet", run("statusData").table_rules?.revision === R2,
      `page=${String(run("statusData").table_rules?.revision).slice(0, 12)}`);
const postsBeforeStale = analysisPosts().length;
await click(node("hand-compute"));
await new Promise(resolve => setTimeout(resolve, 700));
check("case3_the_request_carried_the_verified_revision",
      analysisPosts().length === postsBeforeStale + 1
      && lastAnalysisBody().rules_revision === R2,
      `sent=${String(lastAnalysisBody()?.rules_revision).slice(0, 12)}`);
check("case3_the_server_refused_the_stale_rules_version",
      run("acceptedAnalysisId") === null && results() === ""
      && exportEnabled() === false
      && (analysisStatus().includes("未开始计算")
          || analysisStatus().includes("已变更")),
      analysisStatus().slice(0, 80));

// --- case 3b: the page's own poll notices -------------------------------
await run("poll()");
check("case3b_the_poll_voids_the_receipt_for_the_external_change",
      run("statusData").table_rules?.revision === R3
      && run("handBuilt") === null && !hasReceipt()
      && node("hand-compute").disabled === true,
      `page=${String(run("statusData").table_rules?.revision).slice(0, 12)}`);

// The page's own rules revision is stale after an external save; the rules
// feedback tells the human to reload, so the harness uses the shipped loadRules.
await run("loadRules()");
const reloaded = await waitUntil(() => run("rulesRevision") === R3);
check("the_page_reloads_the_rules_after_the_external_change", reloaded,
      `page=${String(run("rulesRevision")).slice(0, 12)}`);

// --- case 4: a late build must not restore "verified" -------------------
typeHand();
const buildPath = "/api/hand-input/build";
const originalFetch = context.fetch;
context.fetch = async (path, options = {}) => {
  if (path === buildPath) await new Promise(resolve => setTimeout(resolve, 800));
  return originalFetch(path, options);
};
const lateBuild = fire(node("hand-build"), "click");
await new Promise(resolve => setTimeout(resolve, 120));
const R4 = await saveRulesForm({rake_percent: "7"});
check("the_rules_changed_while_the_build_was_in_flight", R4 !== R3,
      `R4=${R4.slice(0, 12)}`);
await lateBuild;
check("case4_the_late_build_does_not_restore_verified",
      run("handBuilt") === null && !hasReceipt()
      && node("hand-compute").disabled === true
      && node("hand-input-tag").textContent === "未核对",
      `tag=${node("hand-input-tag").textContent} built=${run("handBuilt")}`);
context.fetch = originalFetch;

// --- case 5: R4 re-verify -> real compute -------------------------------
typeHand();
check("the_hand_re_verifies_at_r4", await verify(), feedback().slice(0, 60));
check("the_receipt_names_r4", receiptRevision() === R4,
      `receipt=${String(receiptRevision()).slice(0, 12)} `
      + `R4=${String(R4).slice(0, 12)}`);
await click(node("hand-compute"));
const r4Report = await settleAnalysis();
check("case5_the_re_verified_input_computes_with_aligned_versions",
      r4Report?.status === "COMPLETE" && results().includes("条件净 EV")
      && exportEnabled() && lastAnalysisBody().rules_revision === R4
      && run("analysisBinding").expected_rules_revision === R4
      && run("analysisBinding").table_rules_revision === R4
      && run("analysisBinding").input_sha256
         === run("analysisBinding").expected_input_sha256,
      `status=${r4Report?.status} sent=${String(lastAnalysisBody()?.rules_revision).slice(0, 12)} `
      + `R4=${String(R4).slice(0, 12)} receipt=${String(receiptRevision()).slice(0, 12)}`);
check("the_export_names_the_verified_rules",
      run("analysisDraftSource")?.rules_revision === R4
      && run("analysisDraftSource")?.rules_source === "table"
      && typeof run("analysisDraftSource")?.effective_rules_sha256 === "string",
      `draft=${String(run("analysisDraftSource")?.rules_revision).slice(0, 12)}`);

// --- case 6a: reset -----------------------------------------------------
typeHand();
check("the_hand_verifies_before_the_reset", await verify(), feedback().slice(0, 50));
const R5 = await resetRulesForm();
check("the_reset_really_changed_the_server", R5, "revision moved");
check("case6_the_reset_voids_the_receipt",
      run("handBuilt") === null && !hasReceipt()
      && node("hand-compute").disabled === true,
      `built=${run("handBuilt")}`);
check("case6_the_reset_keeps_the_typed_hand",
      node("hand-hero").value === "Qs Qd" && run('handRows("seats").length') === SEATS.length,
      `hero=${node("hand-hero").value}`);

// --- case 6b: a failed save must not resurrect the receipt --------------
const R6 = await saveRulesForm({rake_percent: "0"});
check("the_rules_are_valid_again", /^[a-f0-9]{64}$/.test(R6), R6.slice(0, 12));
typeHand();
check("the_hand_verifies_at_r6", await verify(), feedback().slice(0, 50));
run('rulesRevision = "0".repeat(64)');
await submitRulesForm();
await new Promise(resolve => setTimeout(resolve, 300));
check("case6b_the_failed_save_reports_and_keeps_the_receipt_dead",
      rulesFeedback().includes("保存失败") && run("handBuilt") === null
      && !hasReceipt() && node("hand-compute").disabled === true,
      rulesFeedback().slice(0, 60));

// --- the previously verified positive path still works ------------------
run(`rulesRevision = ${JSON.stringify(R6)}`);
typeHand();
check("the_hand_verifies_once_more", await verify(), feedback().slice(0, 50));
await click(node("hand-compute"));
const finalReport = await settleAnalysis();
check("the_positive_path_is_not_all_disabled",
      finalReport?.status === "COMPLETE" && results().includes("条件净 EV")
      && exportEnabled(), `status=${finalReport?.status}`);

check("no_innerhtml_used", dom.state.innerHTMLWrites === 0,
      `writes=${dom.state.innerHTMLWrites}`);

const failed = checks.filter(item => !item.ok);
console.log(JSON.stringify({
  name: "verdict", ok: failed.length === 0, passed: checks.length - failed.length,
  failed: failed.length, requests: calls.length, base,
}));
process.exit(failed.length === 0 ? 0 : 1);
