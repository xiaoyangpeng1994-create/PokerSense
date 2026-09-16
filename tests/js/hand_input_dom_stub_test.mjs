// Real interaction test for ui/aa-live/hand_input.js against a running backend.
//
// The shipped form script is evaluated unchanged inside a minimal DOM stub, then
// driven through the product flow: fill an ended hand with per-row controls,
// verify the input, change it, hit each refusal, and finally hand the built
// document to the existing analysis entry. The analysis start itself is
// observed here; the full 「input → kernel → result → invalidate → recompute」
// chain lives in analysis_binding_flow_test.mjs.
//
// Usage: node hand_input_dom_stub_test.mjs <base-url>

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";
import { createDom, click, fire } from "./dom_stub.mjs";

const base = (process.argv[2] || "").replace(/\/$/, "");
if (!base.startsWith("http://127.0.0.1:")) {
  console.log(JSON.stringify({name: "base_url", ok: false, detail: base}));
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "..", "..", "ui", "aa-live",
                                "hand_input.js"), "utf8");
const checks = [];
// Printed as they happen, so a crash mid-flow still shows what already passed.
const check = (name, ok, detail = "") => {
  const item = {name, ok: !!ok, detail: String(detail)};
  checks.push(item);
  console.log(JSON.stringify(item));
  return item;
};
const dom = createDom();
const {document, el} = dom;
const known = value => value !== null && value !== undefined && value !== "UNKNOWN";
const text = value => !known(value) ? "未知" : typeof value === "object" ? JSON.stringify(value) : String(value);
const calls = [];
async function fetchStub(path, options = {}) {
  calls.push({path, method: (options.method || "GET").toUpperCase()});
  const response = await fetch(base + path, options);
  const body = await response.text();
  return {ok: response.ok, status: response.status, text: async () => body,
          json: async () => JSON.parse(body)};
}
const statusData = {table_rules: {revision: 0}};
// The page also loads analysis.js; the two globals the form writes into it are
// declared here so the assignment behaves exactly like it does in the browser.
const context = vm.createContext({
  document, el, text, headers: {"X-AA-Live": "1"}, statusData, fetch: fetchStub,
  console, JSON, Number, String, Object, Array, Error, Math, Date,
  analysisExpectedInput: null, analysisDraftSource: null,
  invalidateAnalysis() {}, cancelAnalysis() {}, analysisInputChanged() {},
  clearTimeout, setTimeout,
});
vm.runInContext(source, context, {filename: "hand_input.js"});
const run = expression => vm.runInContext(expression, context);
const feedback = () => el("hand-status").textContent;
const gaps = () => el("hand-gaps").textContent;
const capacity = () => el("hand-capacity").textContent;

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

function fill(overrides = {}) {
  el("hand-hero").value = overrides.hero ?? "Qs Qd";
  el("hand-board").value = overrides.board ?? "2c 4d 7h 9s Jc";
  el("hand-hero-seat").value = overrides.heroSeat ?? "0";
  el("hand-order").value = overrides.order ?? "1,2,0";
  el("hand-pot").value = overrides.pot ?? "130";
  el("hand-targets").value = overrides.targets ?? "20,40,80";
  el("hand-max-agg").value = overrides.maxAgg ?? "2";
  el("hand-ended").checked = overrides.ended ?? true;
  el("hand-use-rules").checked = overrides.useRules ?? true;
  el("hand-fees").value = overrides.fees ?? "confirmed_zero";
  run(`handFillRows("seats", ${JSON.stringify(overrides.seats ?? SEATS)})`);
  run(`handFillRows("history", ${JSON.stringify(overrides.history ?? HISTORY)})`);
  run(`handFillRows("ranges", ${JSON.stringify(overrides.ranges ?? RANGES)})`);
  run(`handFillRows("weights", ${JSON.stringify(overrides.weights ?? WEIGHTS)})`);
}

// The existing analysis panel is observed: the form must hand it the document.
let analysisStarts = 0;
el("analysis-start").addEventListener("click", () => { analysisStarts += 1; });

check("form_wiring_ran", run("handWired") === true, "handWired");
check("row_controls_are_seeded",
      run('handAddRow("seats", {status: "ACTIVE"})').dataset.used === "false",
      "a seeded row is a placeholder, not input");
check("an_untouched_seeded_row_is_not_input",
      run('handRows("seats").length') === 0, "no fabricated seat");
// The page normally has the saved table rules from /api/status; mirror that.
const rules = await (await fetch(base + "/api/rules")).json();
statusData.table_rules = {revision: rules.revision};
check("saved_table_rules_are_ready", rules.conditional_analysis_ready === true,
      `revision=${String(rules.revision).slice(0, 12)}`);
fill();
await click(el("hand-build"));
const built = run("handBuilt");
check("verified_input_is_accepted", built !== null
      && el("hand-compute").disabled === false, feedback().slice(0, 60));
check("capacity_is_measured", capacity().includes("合法联合组合数")
      && capacity().includes("4") && capacity().includes("130")
      && capacity().includes("20"),
      capacity().replace(/\s+/g, " ").slice(0, 90));
check("identity_hashes_are_shown",
      el("hand-identity").textContent.includes("facts_sha256")
      && el("hand-identity").textContent.includes("input_sha256")
      && el("hand-identity").textContent.includes("human_confirmed"),
      "hashes");
check("document_is_the_kernel_shape",
      built && built.mode === "manual_hypothesis" && built.range_start === "river_start"
      && built.hero_cards.join(" ") === "Qs Qd" && built.history.length === 2
      && built.ranges.length === 2 && built.models.length === 2,
      JSON.stringify(built && {mode: built.mode, hero: built.hero_cards}));
check("no_policy_value_is_shown_here",
      !capacity().includes("EV") && !feedback().includes("EV"),
      "the form only pre-checks; it computes nothing");

await fire(el("hand-pot"), "input");
check("changing_a_field_invalidates_the_verified_input",
      run("handBuilt") === null && el("hand-compute").disabled === true
      && feedback().includes("重新核对"), feedback().slice(0, 40));

// Adding and editing a row is input too, not just the plain fields.
fill();
await click(el("hand-build"));
await fire(run('handRows("seats") && document.getElementById("hand-rows-seats")')
  .children[0].querySelectorAll("input,select")[2], "input");
check("editing_a_row_invalidates_the_verified_input",
      run("handBuilt") === null && el("hand-compute").disabled === true,
      feedback().slice(0, 40));

fill({pot: "999"});
await click(el("hand-build"));
check("pot_conflict_is_refused_in_chinese",
      gaps().includes("显示底池") && el("hand-compute").disabled === true,
      gaps().replace(/\s+/g, " ").slice(0, 90));
check("refusal_lists_what_is_missing", gaps().includes("还缺什么"), "gap list");

fill({board: "Qs 4d 7h 9s Jc"});
await click(el("hand-build"));
check("duplicate_card_is_refused", feedback().includes("重复")
      && el("hand-compute").disabled === true, feedback().slice(0, 60));

fill({fees: "unknown"});
await click(el("hand-build"));
check("unknown_fees_are_refused_instead_of_zeroed",
      feedback().includes("额外费用") && el("hand-compute").disabled === true,
      feedback().slice(0, 70));

// Seat 2 keeps a live model but gets zero mass on the call the history claims.
fill({weights: WEIGHTS.map(row => row.seat_id === 2 && row.key === "call"
                               ? {...row, weight: "0"} : row)});
await click(el("hand-build"));
check("history_without_probability_is_refused", gaps().includes("概率为 0")
      && el("hand-compute").disabled === true,
      (gaps() || feedback()).replace(/\s+/g, " ").slice(0, 80));

fill({pot: "210", history: [{actor: 1, kind: "bet", target: "60"},
                            {actor: 2, kind: "call", target: "0"}]});
await click(el("hand-build"));
check("history_off_the_declared_grid_is_refused",
      gaps().includes("加注尺寸网格") && el("hand-compute").disabled === true,
      gaps().replace(/\s+/g, " ").slice(0, 80));

fill({order: "1,2,3"});
await click(el("hand-build"));
check("wrong_action_order_is_refused", feedback().includes("行动顺序"),
      feedback().slice(0, 60));

fill({seats: SEATS.slice(0, 2)});
await click(el("hand-build"));
check("non_three_way_is_refused", feedback().includes("三名 ACTIVE")
      || gaps().includes("三名 ACTIVE") || feedback().includes("ACTIVE"),
      feedback().slice(0, 60));

fill({heroSeat: ""});
await click(el("hand-build"));
check("a_missing_hero_seat_is_a_gap_not_a_guess",
      feedback().includes("Hero 座位"), feedback().slice(0, 60));

fill();
await click(el("hand-build"));
await click(el("hand-compute"));
check("compute_hands_the_document_to_the_existing_chain",
      el("analysis-kind").value === "threeway" && analysisStarts === 1
      && el("analysis-input").value.includes("\"river_start\"")
      && el("analysis-input").value.includes("\"table_size\""),
      `starts=${analysisStarts}`);
check("compute_binds_the_input_identity_to_the_panel",
      run("analysisExpectedInput") === run("handExpectedInput")
      && run("analysisDraftSource").kind === "hand_input_form"
      && run("analysisDraftSource").input_sha256 === run("handExpectedInput"),
      String(run("analysisExpectedInput")).slice(0, 16));
check("compute_does_not_edit_json_for_the_user",
      !feedback().includes("JSON"), feedback().slice(0, 40));

check("no_innerhtml_used", dom.state.innerHTMLWrites === 0,
      `writes=${dom.state.innerHTMLWrites}`);
check("only_hand_input_and_analysis_endpoints",
      calls.every(call => call.path.startsWith("/api/hand-input/")),
      `${calls.length} calls: ${[...new Set(calls.map(c => c.path))].join(",")}`);

const failed = checks.filter(item => !item.ok);
for (const item of checks) console.log(JSON.stringify(item));
console.log(JSON.stringify({
  name: "verdict", ok: failed.length === 0, passed: checks.length - failed.length,
  failed: failed.length, requests: calls.length, base,
}));
process.exit(failed.length === 0 ? 0 : 1);
