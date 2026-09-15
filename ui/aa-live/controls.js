"use strict";
const ruleFields = {
  table_label: ["牌桌备注", "text"], dealt_players: ["本手发牌人数", ["", "6", "7", "8"]],
  small_blind: ["小盲", "decimal"], big_blind: ["大盲", "decimal"], ante: ["每人前注", "decimal"],
  straddle_mode: ["Straddle", {unknown:"未知", none:"没有", mandatory_utg:"UTG 强制", optional_explicit_utg:"UTG 可选（逐手确认）"}],
  straddle_amount: ["Straddle 金额", "decimal"], rake_percent: ["抽水百分比（3 = 3%）", "decimal"],
  rake_cap_bb: ["封顶（大盲倍数）", "decimal"], minimum_chip: ["最小筹码单位", "decimal"],
  rake_application: ["抽水时机", {unknown:"未知", all_pots:"所有底池", postflop_only:"见翻牌后"}],
  rake_rounding: ["抽水舍入", {unknown:"未知", exact:"精确值", floor_to_chip:"向下到筹码单位", ceil_to_chip:"向上到筹码单位"}],
  rake_distribution: ["边池抽水分配", {unknown:"未知", proportional_all_pots:"所有池按比例", main_pot_first:"主池优先"}],
  insurance: ["保险", {unknown:"未知", off:"关闭", on:"开启"}],
  bomb: ["暴击", {unknown:"未知", off:"关闭", on:"开启"}],
  mushroom: ["种蘑菇", {unknown:"未知", off:"关闭", on:"开启"}]
};
let rulesRevision = null;
for (const [key, [title, type]] of Object.entries(ruleFields)) {
  const label = document.createElement("label"); label.textContent = title;
  const input = document.createElement(typeof type === "string" ? "input" : "select");
  input.id = `rule-${key}`;
  if (typeof type === "string") { input.type = "text"; input.maxLength = key === "table_label" ? 100 : 32; if (type === "decimal") input.inputMode = "decimal"; input.placeholder = "未知"; }
  else { const choices = Array.isArray(type) ? Object.fromEntries(type.map(v => [v, v || "未知"])) : type; for (const [value, title] of Object.entries(choices)) { const option = document.createElement("option"); option.value = value; option.textContent = title; input.append(option); } }
  label.append(input); el("rules-fields").append(label);
}
function showRules(result, fill) {
  rulesRevision = result.revision;
  if (fill) for (const key of Object.keys(ruleFields)) el(`rule-${key}`).value = result.document[key] ?? "";
  const missing = result.pending_fields.map(k => ruleFields[k]?.[0] || k);
  el("rules-status").textContent = result.conditional_analysis_ready ? "人工配置完整 · 非视觉核验" : "人工配置待补全";
  el("rules-feedback").textContent = missing.length ? `仍未知：${missing.join("、")}` : result.unsupported_effects.length ? "已保存；当前条件策略不支持开启的特殊机制。" : "已保存人工规则；仍需要完整牌局与适用策略，才能计算行动收益。";
}
async function loadRules() {
  try { const response = await fetch("/api/rules", {cache:"no-store"}); if (!response.ok) throw Error(`HTTP ${response.status}`); showRules(await response.json(), true); }
  catch (e) { el("rules-feedback").textContent = `规则读取失败：${e.message}`; }
}
async function saveRules(reset) {
  if (!rulesRevision) return;
  const document = {};
  for (const [key, [, type]] of Object.entries(ruleFields)) {
    const value = reset ? (key === "table_label" ? "" : typeof type === "object" && !Array.isArray(type) ? "unknown" : "") : el(`rule-${key}`).value.trim();
    document[key] = key === "dealt_players" ? (value ? Number(value) : null) : key === "table_label" || typeof type !== "string" ? value : value || null;
  }
  try { const response = await fetch("/api/rules", {method:"POST", headers:{...headers,"Content-Type":"application/json"}, body:JSON.stringify({document,revision:rulesRevision})}); const result = await response.json(); if (!response.ok) throw Error(text(result.detail)); showRules(result, true); clearCurrent("规则已变更，重新开始后使用新桌配置。"); await poll(); }
  catch (e) { el("rules-feedback").textContent = `保存失败：${e.message}。重新加载页面可读取其他页面的新配置。`; }
}
el("rules-form").addEventListener("submit", event => { event.preventDefault(); saveRules(false); });
el("rules-reset").addEventListener("click", () => saveRules(true));
el("issue-form").addEventListener("submit", async event => {
  event.preventDefault(); el("issue-save").disabled = true;
  try {
    const response = await fetch("/api/issues", {method:"POST", headers:{...headers,"Content-Type":"application/json"}, body:JSON.stringify({note:el("issue-note").value,category:el("issue-category").value})});
    const result = await response.json(); if (!response.ok) throw Error(text(result.detail));
    el("issue-feedback").textContent = `已保存，待复核：${result.directory}`; el("issue-note").value = "";
  } catch (e) { el("issue-feedback").textContent = `未保存：${e.message}`; }
});
setInterval(() => { el("issue-save").disabled = statusData.status !== "RUNNING" || !statusData.payload || !statusData.issue_recording_available; }, 800);
loadRules();
