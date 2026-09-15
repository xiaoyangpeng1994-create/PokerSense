"use strict";
// Deterministic DOM/HTTP scheduling tests; executes the actual three UI scripts.
// Run: node tests/ui/test_aa_analysis_ui.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "../..");

function harness() {
  const nodes = new Map(), timers = new Map();
  let timerId = 0;
  const h = {calls: [], blobs: [], fetchImpl: () => new Promise(() => {})};
  class Element {
    constructor() { this.children = []; this._text = ""; this.value = ""; this.options = [{}, {}]; this.listeners = new Map(); this.attrs = {}; this.disabled = false; }
    set id(value) { nodes.set(value, this); }
    set textContent(value) { this._text = String(value); this.children = []; }
    get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this._text = ""; this.children = children; }
    setAttribute(key, value) { this.attrs[key] = value; }
    removeAttribute(key) { delete this.attrs[key]; }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    dispatch(name) { return this.listeners.get(name)?.({preventDefault() {}, target: this}); }
    click() { return this.dispatch("click"); }
  }
  h.el = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  h.document = {hidden: true, getElementById: h.el, createElement: () => new Element(), addEventListener() {}};
  const context = vm.createContext({
    document: h.document, console, AbortController, Blob,
    URL: {createObjectURL(blob) { h.blobs.push(blob); return "blob:test"; }, revokeObjectURL() {}},
    fetch(url, options) { h.calls.push({url, options}); return h.fetchImpl(url, options); },
    setTimeout(callback) { const id = ++timerId; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); }, setInterval() { return ++timerId; },
  });
  h.run = source => vm.runInContext(source, context);
  h.pendingCancelTimers = () => [...timers.values()].filter(fn => fn.name === "cancelAnalysis").length;
  for (const name of ["app.js", "controls.js", "analysis.js"]) {
    vm.runInContext(fs.readFileSync(path.join(root, "ui/aa-live", name), "utf8"), context, {filename: name});
  }
  h.el("mode").value = "development-replay"; h.el("device").value = "0"; h.el("api").value = "MSMF";
  h.el("analysis-kind").value = "terminal";
  return h;
}

function report(id = "old", generation = 1, revision = "r1", status = "COMPLETE") {
  return {job_id: id, kind: "terminal", status,
    binding: {generation, table_rules_revision: revision, effective_rules: {source: "effective-table-rules", rake_percent: "0.03"}},
    result: status === "COMPLETE" ? {fold_ev: {decimal: "0"}, call_net_ev: {decimal: "256"}} : null};
}
function seed(h, value = report()) {
  h.run(`acceptedAnalysisId=${JSON.stringify(value.job_id)};analysisBinding=${JSON.stringify(value.binding)};analysisInput={mode:"manual_hypothesis",rules:{source:"original-document"}};renderAnalysis(${JSON.stringify(value)});`);
  assert.equal(h.el("analysis-export").disabled, false);
}
function cleared(h) {
  assert.equal(h.el("analysis-export").disabled, true);
  assert.equal(h.el("analysis-results").textContent, "");
  assert.equal(h.run("analysisReport"), null);
  assert.equal(h.run("acceptedAnalysisId"), null);
}
const response = value => Promise.resolve({ok: true, json: async () => value});

async function main() {
  let cases = 0;
  {
    const h = harness();
    assert.equal(h.run('actionName({kind:"fold",target:"0"})'), "弃牌");
    assert.equal(h.run('actionName({kind:"call",target:"0"})'), "跟注");
    assert.equal(h.run('actionName({kind:"raise",target:"40"})'), "加注 至 40");
    cases++;
  }
  // Another observer may change rules/source and replace the job between polls.
  {
    const h = harness(); seed(h); h.document.hidden = false;
    h.run('serverInstance="old-server";serverGeneration=10;sequence=99');
    h.fetchImpl = () => response({status:"STOPPED",instance_id:"new-server",generation:0,sequence:null,
      replay_available:true,capture_available:false,table_rules:{revision:"r1"},analysis:{status:"IDLE",job_id:null}});
    await h.run("poll()"); cleared(h);
    assert.equal(h.run("serverGeneration"),0); assert.equal(h.run("serverInstance"),"new-server"); cases++;
  }
  for (const [packet, current] of [
    [report("new", 2, "r2"), {generation: 2, table_rules: {revision: "r2"}}],
    [report(), {generation: 2, table_rules: {revision: "r1"}}],
    [report(), {generation: 1, table_rules: {revision: "r2"}}],
    [report("new"), {generation: 1, table_rules: {revision: "r1"}}],
  ]) {
    const h = harness(); seed(h);
    h.run(`renderAnalysis(${JSON.stringify(packet)},${JSON.stringify(current)});`);
    cleared(h); cases++;
  }
  {
    const h = harness(); seed(h); h.document.hidden = false;
    h.fetchImpl = () => response({status: "STOPPED", sequence: null, generation: 2,
      replay_available: true, capture_available: false, table_rules: {revision: "r2"}, analysis: report("new", 2, "r2")});
    await h.run("poll()"); cleared(h); cases++;
  }
  {
    const h = harness(); seed(h); h.el("analysis-input").dispatch("input"); cleared(h);
    const message = h.el("analysis-status").textContent;
    h.run('renderAnalysis({status:"IDLE",job_id:null},{generation:1,table_rules:{revision:"r1"}})');
    assert.equal(h.el("analysis-status").textContent, message); cases++;
  }
  for (const action of ["start", "stop"]) {
    const h = harness(); seed(h); let atRequest;
    h.fetchImpl = url => { if (url === `/api/${action}`) atRequest = {disabled: h.el("analysis-export").disabled, rows: h.el("analysis-results").textContent}; return new Promise(() => {}); };
    h.run(`command(${JSON.stringify(action)})`);
    assert.deepEqual(atRequest, {disabled: true, rows: ""}); cleared(h); cases++;
  }
  {
    const h = harness(); seed(h); h.run('rulesRevision="r1"'); let atRequest;
    h.fetchImpl = url => { if (url === "/api/rules") atRequest = h.el("analysis-export").disabled; return new Promise(() => {}); };
    h.run("saveRules(false)"); assert.equal(atRequest, true); cleared(h); cases++;
  }
  {
    const h = harness(); h.run('statusData={generation:1,table_rules:{revision:"r1"}}');
    h.el("analysis-input").value = '{"mode":"manual_hypothesis","rules":{}}';
    h.fetchImpl = url => response(url === "/api/analysis" ? report("running", 1, "r1", "RUNNING") : {});
    await h.el("analysis-start").dispatch("click");
    assert.equal(h.run("analysisReport"), null);
    assert.equal(h.run("analysisBinding.table_rules_revision"), "r1");
    h.run(`renderAnalysis(${JSON.stringify(report("running", 1, "r1", "RUNNING"))},{generation:2,table_rules:{revision:"r1"}})`);
    cleared(h); cases++;
  }
  for (const button of ["analysis-start", "analysis-example"]) {
    const h = harness(); let resolveCancel;
    h.el("analysis-input").value = "edited input";
    h.fetchImpl = url => { assert.equal(url, "/api/analysis/cancel"); return new Promise(resolve => { resolveCancel = resolve; }); };
    const operation = h.el(button).dispatch("click");
    h.el("analysis-input").dispatch("input");
    resolveCancel({ok: true}); await operation;
    assert.equal(h.el("analysis-input").value, "edited input");
    assert.equal(h.calls.filter(call => call.url === "/api/analysis" || call.url.startsWith("/api/analysis/example")).length, 0);
    cleared(h); cases++;
  }
  {
    const h = harness(); h.el("analysis-input").dispatch("input");
    assert.equal(h.pendingCancelTimers(), 1);
    h.el("analysis-start").dispatch("click");
    assert.equal(h.pendingCancelTimers(), 0); cases++;
  }
  {
    const h = harness(); seed(h); h.el("analysis-export").dispatch("click");
    const exported = JSON.parse(await h.blobs.at(-1).text());
    assert.equal(exported.effective_input.rules.source, "effective-table-rules");
    assert.deepEqual(exported.effective_input.rules, exported.report.binding.effective_rules); cases++;
  }
  console.log(`PASS: ${cases} deterministic AA analysis UI cases`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
