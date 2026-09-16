// Real interaction test for ui/aa-live/hand_input.js against a running backend.
//
// The shipped form script is evaluated unchanged inside a minimal DOM stub, then
// driven through the product flow: fill an ended hand, verify the input, change
// it, hit each refusal, and finally hand the built document to the existing
// analysis entry. The analysis start itself is observed, not executed here.
//
// Usage: node hand_input_dom_stub_test.mjs <base-url>

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import vm from "node:vm";

const base = (process.argv[2] || "").replace(/\/$/, "");
if (!base.startsWith("http://127.0.0.1:")) {
  console.log(JSON.stringify({name: "base_url", ok: false, detail: base}));
  process.exit(2);
}
const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "..", "..", "ui", "aa-live",
                                "hand_input.js"), "utf8");
const checks = [];
const check = (name, ok, detail = "") => checks.push({name, ok: !!ok, detail: String(detail)});
let innerHTMLWrites = 0;
const elements = new Map();
function makeElement(tag) {
  const node = {
    tagName: tag, children: [], attributes: {}, handlers: {},
    _text: "", className: "", hidden: false, disabled: false, value: "",
    checked: false, open: false,
    get textContent() {
      return this._text + this.children.map(child => child.textContent).join("");
    },
    set textContent(value) { this._text = String(value); this.children = []; },
    append(...nodes) { this._text = ""; for (const item of nodes) this.children.push(item); },
    replaceChildren(...nodes) { this._text = ""; this.children = []; for (const item of nodes) this.children.push(item); },
    addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); },
    click() { for (const handler of this.handlers.click || []) handler({}); },
    setAttribute(name, value) { this.attributes[name] = value; },
  };
  Object.defineProperty(node, "innerHTML", {
    get() { return "<stub>"; },
    set() { innerHTMLWrites += 1; },
  });
  return node;
}
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement("div"));
    return elements.get(id);
  },
  createElement(tag) { return makeElement(tag); },
};
const el = id => document.getElementById(id);
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
const context = vm.createContext({
  document, el, text, headers: {"X-AA-Live": "1"}, statusData, fetch: fetchStub,
  console, JSON, Number, String, Object, Array, Error,
});
vm.runInContext(source, context, {filename: "hand_input.js"});
const run = expression => vm.runInContext(expression, context);
const click = async id => {
  for (const handler of el(id).handlers.click || []) await handler({});
};
const fire = async (id, type) => {
  let ran = 0;
  for (const handler of el(id).handlers[type] || []) { ran += 1; await handler({}); }
  return ran;
};
const feedback = () => el("hand-status").textContent;
const gaps = () => el("hand-gaps").textContent;
const capacity = () => el("hand-capacity").textContent;

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
  el("hand-seats").value = overrides.seats ?? [
    "0,ACTIVE,200,20", "1,ACTIVE,200,20", "2,ACTIVE,200,20",
    "3,FOLDED,200,10", "4,FOLDED,200,10", "5,FOLDED,200,10"].join("\n");
  el("hand-history").value = overrides.history ?? "1,bet,20\n2,call,0";
  el("hand-ranges").value = overrides.ranges ?? [
    "1,JhJd,1", "1,TcTd,3", "2,7c7s,1", "2,KhTh,3"].join("\n");
  el("hand-weights").value = overrides.weights ?? [
    "1,check,1", "1,bet,2", "1,call,9", "1,fold,1",
    "2,check,1", "2,call,9", "2,fold,1"].join("\n");
}

// The existing analysis panel is observed: the form must hand it the document.
let analysisStarts = 0;
el("analysis-start").addEventListener("click", () => { analysisStarts += 1; });

check("form_wiring_ran", run("handWired") === true, "handWired");
// The page normally has the saved table rules from /api/status; mirror that.
const rules = await (await fetch(base + "/api/rules")).json();
statusData.table_rules = {revision: rules.revision};
check("saved_table_rules_are_ready", rules.conditional_analysis_ready === true,
      `revision=${String(rules.revision).slice(0, 12)}`);
fill();
await click("hand-build");
const built = run("handBuilt");
check("verified_input_is_accepted", run("handBuilt !== null")
      && el("hand-compute").disabled === false, feedback().slice(0, 60));
check("capacity_is_measured", capacity().includes("4")
      && capacity().includes("合法联合组合数")
      && capacity().includes("130") && capacity().includes("20"),
      capacity().replace(/\s+/g, " ").slice(0, 90));
check("identity_hashes_are_shown",
      el("hand-identity").textContent.includes("facts_sha256")
      && el("hand-identity").textContent.includes("input_sha256")
      && el("hand-identity").textContent.includes("human_confirmed"),
      "hashes");
check("document_is_the_kernel_shape",
      built && built.mode === "manual_hypothesis" && built.range_start === "river_start"
      && built.hero_cards.join(" ") === "Qs Qd" && built.history.length === 2,
      JSON.stringify(built && {mode: built.mode, hero: built.hero_cards}));

await fire("hand-pot", "input");
check("changing_the_input_invalidates_the_verified_input",
      run("handBuilt") === null && el("hand-compute").disabled === true
      && feedback().includes("重新核对"), feedback().slice(0, 40));

fill({pot: "999"});
await click("hand-build");
check("pot_conflict_is_refused_in_chinese",
      gaps().includes("显示底池") && el("hand-compute").disabled === true,
      gaps().replace(/\s+/g, " ").slice(0, 90));
check("refusal_lists_what_is_missing", gaps().includes("还缺什么"), "gap list");

fill({board: "Qs 4d 7h 9s Jc"});
await click("hand-build");
check("duplicate_card_is_refused", feedback().includes("重复")
      && el("hand-compute").disabled === true, feedback().slice(0, 60));

fill({weights: "1,check,1\n1,bet,2\n1,call,9\n1,fold,1\n2,check,1\n2,fold,1"});
await click("hand-build");
check("history_without_probability_is_refused", gaps().includes("概率为 0")
      && el("hand-compute").disabled === true,
      gaps().replace(/\s+/g, " ").slice(0, 80));

fill({order: "1,2,3"});
await click("hand-build");
check("wrong_action_order_is_refused", feedback().includes("行动顺序"),
      feedback().slice(0, 60));

fill({seat0: null, seats: "0,ACTIVE,200,20\n1,ACTIVE,200,20"});
await click("hand-build");
check("non_three_way_is_refused", feedback().includes("三名 ACTIVE")
      || gaps().includes("三名 ACTIVE") || feedback().includes("队友")
      || feedback().includes("ACTIVE"), feedback().slice(0, 60));

fill();
await click("hand-build");
await click("hand-compute");
check("compute_hands_the_document_to_the_existing_chain",
      el("analysis-kind").value === "threeway" && analysisStarts === 1
      && el("analysis-input").value.includes("\"river_start\"")
      && el("analysis-input").value.includes("\"tablesize\"".replace("tablesize", "table_size")),
      `starts=${analysisStarts}`);
check("compute_does_not_edit_json_for_the_user",
      !feedback().includes("JSON"), feedback().slice(0, 40));

check("no_innerhtml_used", innerHTMLWrites === 0, `writes=${innerHTMLWrites}`);
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
