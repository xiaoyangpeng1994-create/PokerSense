// PokerSense companion panel. Language follows the local system on first run
// and is persisted when the player changes it in Settings.

const SUIT_SYMBOL = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RED_SUITS = new Set(["h", "d"]);
const LANGUAGE_STORAGE_KEY = "pokersense.language";
const FIELD_ORDER = ["hero_cards", "board_cards", "street", "pot", "stacks", "bet_size", "action"];
const TRANSLATIONS = {
  en: {
    board: "Board", hero: "Hero", pot: "Pot", winRate: "Showdown win rate", settings: "Settings",
    chips: "chips", randomBasis: "Uniform random ranges · {count} opponents · actions not modeled",
    equityWaiting: "Waiting for verified cards and all seats.",
    waitingPlayers: "Waiting for at least two confirmed active players.",
    heroInactive: "Hero is not active in this hand.",
    tableRulesPending: "Confirm ante, rake cap and straddle rules before strategy advice.",
    straddleUnsupported: "Straddle strategy is not supported yet. Advice is withheld.",
    simulationRules: "Simulation parameters are selected. Confirm the real table rules before using strategy advice.",
    simulationPreview: "SIMULATION · sample cards and random-range equity only. No capture device is connected.",
    extraEffectsUnsupported: "Extra table effects are recorded, but their strategy model is not implemented.",
    tableRulesInvalid: "Table rules could not be loaded. Check the settings.",
    tableRules: "WPK table rules", rulesMode: "Use", simulationMode: "Simulation assumptions", liveRulesMode: "Confirmed table rules",
    rulesHint: "Defaults include simulation assumptions. Check them against the current table, then save.",
    tableSize: "Table size", smallBlind: "Small blind", bigBlind: "Big blind", ante: "Ante per player", minimumChip: "Smallest chip",
    rakePercent: "Rake (%)", rakeCap: "Rake cap (BB; blank = unknown)", straddle: "Straddle", straddleAmount: "Straddle amount (chips)",
    straddleNone: "None", straddleMandatory: "Mandatory", straddleOptional: "Optional", straddleUnknown: "Unconfirmed",
    extraEffects: "Extra rules / bonus effects (notes only)", saveRules: "Save table rules", rulesSaved: "Saved. Previous strategy results have been cleared.",
    rulesSupportHint: "Saving rules does not imply strategy coverage. Unsupported straddles and bonus effects withhold advice.",
    language: "Language", languageAuto: "System default", languageHint: "Changes are saved automatically.",
    bannerConnecting: "Connecting to the engine…",
    bannerWaiting: "Connected — waiting for the first table frame…",
    bannerDisconnected: "Engine disconnected. Retrying…",
    bannerError: "Cannot reach the engine.",
    close: "Close", notCalibrated: "not calibrated", confidence: "confidence", tie: "tie",
    frame: "frame", noData: "no data", live: "PokerSense · live", connecting: "PokerSense · connecting",
    cannotReach: "PokerSense · cannot reach engine", waiting: "PokerSense · waiting for table",
    connectionError: "PokerSense · connection error", disconnected: "PokerSense · engine disconnected, reconnecting",
    strategyAdvice: "Strategy advice", detailsEvidence: "Details and evidence", source: "Source",
    match: "Match", differences: "Differences", gates: "Safety gates", ev: "EV", sizes: "Sizes", reasons: "Reasons", assumptions: "Assumptions",
    evidence: "Evidence", expires: "Expires", preferred: "preferred", recommended: "Primary action",
    actionMix: "Action mix", adviceUnavailable: "Advice is withheld until the required live inputs are verified.",
    inputSource: "Input", sourceNames: { vision: "vision", manual: "manual", config: "config", derived: "derived", inferred: "inferred" },
    adviceStates: { READY: "Ready", PARTIAL: "Partial", ABSTAIN: "No advice", STALE: "Expired" },
    fields: { hero_cards: "Hero", board_cards: "Board", street: "Street", pot: "Pot", stacks: "Stacks", bet_size: "Bet", action: "Action" },
  },
  zh: {
    board: "公共牌", hero: "底牌", pot: "底池", winRate: "摊牌胜率", settings: "设置",
    chips: "筹码", randomBasis: "随机范围 · {count} 名对手 · 尚未结合行动",
    equityWaiting: "等待确认牌面和全部座位后计算。",
    waitingPlayers: "正在确认牌局，至少需要两名有效在局玩家。",
    heroInactive: "本手已不在局，暂不提供行动建议。",
    tableRulesPending: "前注、抽水封顶与 straddle 规则待确认，暂不输出策略建议。",
    straddleUnsupported: "straddle 策略尚未接通，暂不输出行动建议。",
    simulationRules: "当前为模拟参数，请按真实牌桌确认规则后再使用策略建议。",
    simulationPreview: "模拟演示 · 示例牌面与随机范围胜率，未连接采集卡。",
    extraEffectsUnsupported: "额外规则已记录，但对应策略模型尚未实现。",
    tableRulesInvalid: "牌桌规则读取失败，请检查设置。",
    tableRules: "WPK 牌桌规则", rulesMode: "使用方式", simulationMode: "模拟参数", liveRulesMode: "已按牌桌确认",
    rulesHint: "默认值含模拟假设，请按当前牌桌调整并保存。",
    tableSize: "牌桌人数", smallBlind: "小盲", bigBlind: "大盲", ante: "每人前注", minimumChip: "最小筹码单位",
    rakePercent: "抽水比例（%）", rakeCap: "抽水封顶（BB，留空=未知）", straddle: "Straddle", straddleAmount: "Straddle 金额（筹码）",
    straddleNone: "无", straddleMandatory: "强制", straddleOptional: "可选", straddleUnknown: "尚未确认",
    extraEffects: "额外规则 / 暴击说明（仅记录）", saveRules: "保存牌桌规则", rulesSaved: "已保存，旧策略结果已清除。",
    rulesSupportHint: "保存参数不代表已支持该策略；straddle 和未建模的暴击规则会暂缓行动建议。",
    language: "语言", languageAuto: "跟随系统", languageHint: "更改会自动保存。",
    bannerConnecting: "正在连接引擎…",
    bannerWaiting: "已连接，等待首帧牌桌数据…",
    bannerDisconnected: "引擎已断开，正在重试…",
    bannerError: "无法连接引擎。",
    close: "关闭", notCalibrated: "尚未标定", confidence: "置信度", tie: "平局",
    frame: "帧", noData: "暂无数据", live: "PokerSense · 已连接", connecting: "PokerSense · 正在连接",
    cannotReach: "PokerSense · 无法连接引擎", waiting: "PokerSense · 等待牌桌",
    connectionError: "PokerSense · 连接错误", disconnected: "PokerSense · 引擎已断开，正在重连",
    strategyAdvice: "策略建议", detailsEvidence: "详情与证据", source: "来源",
    match: "匹配", differences: "差异维度", gates: "安全门", ev: "EV", sizes: "尺度", reasons: "原因", assumptions: "假设",
    evidence: "证据", expires: "有效期", preferred: "首选", recommended: "优先行动",
    actionMix: "行动频率", adviceUnavailable: "所需实时输入尚未确认，暂不输出行动建议。",
    inputSource: "输入", sourceNames: { vision: "视觉", manual: "人工", config: "配置", derived: "派生", inferred: "推断" },
    adviceStates: { READY: "可执行建议", PARTIAL: "部分结果", ABSTAIN: "暂不建议", STALE: "已过期" },
    fields: { hero_cards: "底牌", board_cards: "公共牌", street: "街道", pot: "底池", stacks: "筹码", bet_size: "下注", action: "行动" },
  },
};

const els = {
  app: document.getElementById("app"), statusBanner: document.getElementById("status-banner"),
  connDot: document.getElementById("conn-dot"), connLabel: document.getElementById("conn-label"),
  streetBadge: document.getElementById("street-badge"), boardSlots: document.getElementById("board-slots"),
  heroSlots: document.getElementById("hero-slots"), potValue: document.getElementById("pot-value"),
  winRate: document.getElementById("win-rate"), tieRate: document.getElementById("tie-rate"),
  equityBasis: document.getElementById("equity-basis"),
  segWin: document.getElementById("seg-win"), segTie: document.getElementById("seg-tie"),
  equityBar: document.querySelector(".equity-bar"), confidenceBadge: document.getElementById("confidence-badge"),
  confidenceValue: document.getElementById("confidence-value"), confidenceFields: document.getElementById("confidence-fields"),
  footerLeft: document.getElementById("footer-left"), footerRight: document.getElementById("footer-right"),
  settingsButton: document.getElementById("settings-button"), settingsDialog: document.getElementById("settings-dialog"),
  settingsClose: document.getElementById("settings-close"), languageSelect: document.getElementById("language-select"),
  advicePanel: document.getElementById("advice-panel"), adviceStatus: document.getElementById("advice-status"),
  adviceConfidence: document.getElementById("advice-confidence"), adviceActions: document.getElementById("advice-actions"),
  adviceHero: document.getElementById("advice-hero"),
  adviceBadges: document.getElementById("advice-badges"),
  adviceMessage: document.getElementById("advice-message"), adviceMeta: document.getElementById("advice-meta"),
  adviceEvidence: document.getElementById("advice-evidence"),
  adviceEvidenceContent: document.getElementById("advice-evidence-content"),
};

let lastBoardCount = 0;
let lastAnalysis = null;
let status = { message: "connecting", tone: "", raw: false };
let savedLanguagePreference = "auto";

function systemLanguage() {
  const locale = (navigator.languages && navigator.languages[0]) || navigator.language || "en";
  return locale.toLowerCase().startsWith("zh") ? "zh" : "en";
}

function languagePreference() {
  return savedLanguagePreference;
}

function activeLanguage() {
  const value = languagePreference();
  return value === "auto" ? systemLanguage() : value;
}

function t(key) { return TRANSLATIONS[activeLanguage()][key] || key; }

/* ---------- presentation helpers ----------
   These only change how existing data is displayed. They never interpret,
   derive or repair a value; the advice contract stays owned by the backend. */

// The root phase drives loading and empty styling in CSS. It is derived from
// connection state only, never from recognised values.
function setPhase(phase) {
  els.app.dataset.phase = phase;
}

function phaseFor(message, tone) {
  if (tone === "error") return "error";
  if (message === "live") return "live";
  if (message === "waiting") return "waiting";
  if (message === "disconnected") return "disconnected";
  return "connecting";
}

// An empty value is rendered as a styled placeholder so that "no data" can
// never be misread as a measured number.
function setEmptyValue(el, text) {
  el.replaceChildren();
  const placeholder = document.createElement("span");
  placeholder.className = "empty-value";
  placeholder.textContent = text;
  el.appendChild(placeholder);
}

function setStreetBadge(value) {
  const known = Boolean(value);
  els.streetBadge.textContent = known ? value : "—";
  els.streetBadge.dataset.empty = known ? "false" : "true";
}

// The banner repeats the connection state in words: the brand dot alone is not
// accessible and is easy to miss when the panel is idle.
function updateStatusBanner(message, raw) {
  const BANNERS = {
    connecting: { key: "bannerConnecting", tone: "info" },
    waiting: { key: "bannerWaiting", tone: "info" },
    disconnected: { key: "bannerDisconnected", tone: "warn" },
  };
  const entry = BANNERS[message];
  if (!entry && !raw) {
    els.statusBanner.hidden = true;
    return;
  }
  els.statusBanner.hidden = false;
  els.statusBanner.dataset.tone = raw ? "error" : entry.tone;
  els.statusBanner.textContent = raw ? message : t(entry.key);
}

function applyLanguage() {
  document.documentElement.lang = activeLanguage() === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  els.settingsButton.setAttribute("aria-label", t("settings"));
  els.settingsClose.setAttribute("aria-label", t("close"));
  els.languageSelect.value = languagePreference();
  if (lastAnalysis) render(lastAnalysis); else renderEmpty();
  showStatus(status.message, status.tone, status.raw);
}

function makeCardEl(code, isNew) {
  const el = document.createElement("div");
  if (!code) { el.className = "card empty"; return el; }
  const rank = code.slice(0, -1);
  const suit = code.slice(-1);
  el.className = "card " + (RED_SUITS.has(suit) ? "red" : "black") + (isNew ? " new-card" : "");
  const rankEl = document.createElement("div"); rankEl.textContent = rank;
  const suitEl = document.createElement("div"); suitEl.className = "suit"; suitEl.textContent = SUIT_SYMBOL[suit] || suit;
  el.append(rankEl, suitEl);
  return el;
}

function renderCardSlots(container, cards, slotCount, animateFromIndex) {
  container.innerHTML = "";
  for (let i = 0; i < slotCount; i++) {
    const code = cards[i] || null;
    container.appendChild(makeCardEl(code, animateFromIndex !== undefined && i >= animateFromIndex && code));
  }
}

function statusOf(analysis, field) {
  const entry = (analysis.confidence.field_status || []).find((item) => item[0] === field);
  return entry ? entry[1] : "unknown";
}

function renderFieldStatuses(statuses) {
  els.confidenceFields.innerHTML = "";
  for (const field of FIELD_ORDER) {
    const dot = document.createElement("span");
    dot.className = "field-dot";
    dot.title = `${TRANSLATIONS[activeLanguage()].fields[field]}: ${statuses[field] || t("noData")}`;
    dot.dataset.status = statuses[field] || "unknown";
    els.confidenceFields.appendChild(dot);
  }
}

function addMeta(label, value) {
  if (value === null || value === undefined || value === "") return;
  const term = document.createElement("dt"); term.textContent = label;
  const detail = document.createElement("dd"); detail.textContent = value;
  els.adviceMeta.append(term, detail);
}

function renderAdvice(advice, unavailableReason = null) {
  if (!advice) {
    els.advicePanel.hidden = !unavailableReason;
    els.advicePanel.dataset.status = "ABSTAIN";
    els.adviceStatus.textContent = TRANSLATIONS[activeLanguage()].adviceStates.ABSTAIN;
    els.adviceConfidence.textContent = "";
    els.adviceHero.replaceChildren(); els.adviceActions.replaceChildren();
    els.adviceBadges.replaceChildren(); els.adviceMeta.replaceChildren();
    els.adviceEvidenceContent.textContent = ""; els.adviceEvidence.hidden = true;
    const reasonKeys = { hero_not_in_hand: "heroInactive",
      table_rules_unverified: "tableRulesPending",
      straddle_strategy_not_supported: "straddleUnsupported",
      simulation_rules_only: "simulationRules",
      simulation_preview: "simulationPreview",
      extra_table_effects_not_supported: "extraEffectsUnsupported",
      table_rules_invalid: "tableRulesInvalid" };
    els.adviceMessage.textContent = unavailableReason
      ? t(reasonKeys[unavailableReason] || "waitingPlayers") : "";
    return;
  }
  els.advicePanel.hidden = false;
  els.advicePanel.dataset.status = advice.status;
  els.adviceStatus.textContent = TRANSLATIONS[activeLanguage()].adviceStates[advice.status] || advice.status;
  els.adviceConfidence.textContent = `${t("confidence")} ${(advice.confidence * 100).toFixed(0)}%`;
  els.adviceBadges.replaceChildren();
  if (advice.match_kind) {
    const matchBadge = document.createElement("span");
    matchBadge.className = "advice-badge";
    matchBadge.textContent = advice.match_kind;
    els.adviceBadges.appendChild(matchBadge);
  }
  const fieldsBySource = {};
  for (const item of advice.input_provenance || []) {
    (fieldsBySource[item.source] ||= []).push(item.field_name);
  }
  for (const source of Object.keys(fieldsBySource).sort()) {
    const badge = document.createElement("span");
    badge.className = `advice-badge source-${source}`;
    const sourceName = TRANSLATIONS[activeLanguage()].sourceNames[source] || source;
    badge.textContent = `${t("inputSource")}: ${sourceName} · ${fieldsBySource[source].sort().join(", ")}`;
    els.adviceBadges.appendChild(badge);
  }
  els.adviceHero.replaceChildren();
  els.adviceActions.replaceChildren();
  if (advice.show_actions) {
    const primary = advice.actions.find((item) => item.preferred) || advice.actions[0];
    if (primary) {
      const hero = document.createElement("div"); hero.className = "advice-primary";
      const copy = document.createElement("div");
      const label = document.createElement("div"); label.className = "advice-primary-label"; label.textContent = t("recommended");
      const action = document.createElement("div"); action.className = "advice-primary-action"; action.textContent = primary.action.toUpperCase();
      copy.append(label, action);
      const probability = document.createElement("div"); probability.className = "advice-primary-probability";
      probability.textContent = `${(primary.probability * 100).toFixed(1)}%`;
      hero.append(copy, probability); els.adviceHero.appendChild(hero);
    }
    const mixLabel = document.createElement("div"); mixLabel.className = "advice-mix-label"; mixLabel.textContent = t("actionMix");
    els.adviceActions.appendChild(mixLabel);
    for (const item of advice.actions) {
      const row = document.createElement("div"); row.className = "advice-action" + (item.preferred ? " preferred" : "");
      const name = document.createElement("span"); name.className = "advice-action-name"; name.textContent = item.action.toUpperCase();
      const bar = document.createElement("div"); bar.className = "advice-action-bar";
      const fill = document.createElement("div"); fill.className = "advice-action-fill"; fill.style.width = `${item.probability * 100}%`;
      bar.appendChild(fill);
      const probability = document.createElement("span"); probability.className = "advice-action-probability"; probability.textContent = `${(item.probability * 100).toFixed(1)}%`;
      const detail = document.createElement("div"); detail.className = "advice-action-detail";
      const pieces = [];
      if (item.sizes.length) pieces.push(`${t("sizes")}: ${item.sizes.join(" / ")}`);
      if (item.ev !== null) pieces.push(`${t("ev")}: ${item.ev}`);
      detail.textContent = pieces.join(" · ");
      row.append(name, bar, probability);
      if (pieces.length) row.appendChild(detail);
      els.adviceActions.appendChild(row);
    }
  }
  const reasons = [...(advice.rejection_reasons || []), ...(advice.missing_inputs || [])];
  els.adviceMessage.textContent = reasons.length
    ? `${t("reasons")}: ${reasons.join(", ")}`
    : (advice.show_actions ? "" : t("adviceUnavailable"));
  els.adviceMeta.replaceChildren();
  addMeta(t("source"), [advice.strategy_source, advice.strategy_version].filter(Boolean).join(" · "));
  addMeta(t("match"), advice.match_kind ? `${advice.match_kind} · ${(advice.state_match_score * 100).toFixed(0)}%` : null);
  addMeta(t("ev"), advice.ev_gap === null ? null : `Δ ${advice.ev_gap}`);
  const identity = advice.identity || {};
  addMeta("Context", identity.player_count ? `${identity.active_player_count}/${identity.player_count} players · v${identity.state_version}` : null);
  const matchDimensions = (advice.match_dimensions || []).map((item) =>
    `${item.name}: ${item.requested} → ${item.matched} (Δ ${item.distance}/${item.maximum_distance})`
  );
  addMeta(t("differences"), matchDimensions.join(" · "));
  const gateResults = (advice.gate_results || []).map((item) =>
    `${item.name}: ${item.status}${item.reasons.length ? ` (${item.reasons.join(", ")})` : ""}`
  );
  addMeta(t("gates"), gateResults.join(" · "));
  addMeta(t("expires"), advice.expires_at);
  const evidenceLines = [];
  if (advice.assumptions && advice.assumptions.length) evidenceLines.push(`${t("assumptions")}: ${advice.assumptions.join(", ")}`);
  if (advice.evidence && advice.evidence.length) evidenceLines.push(`${t("evidence")}: ${advice.evidence.join("\n")}`);
  if (advice.missing_evidence && advice.missing_evidence.length) evidenceLines.push(`${t("reasons")}: ${advice.missing_evidence.join(", ")}`);
  els.adviceEvidenceContent.textContent = evidenceLines.join("\n\n");
  els.adviceEvidence.hidden = evidenceLines.length === 0 && els.adviceMeta.children.length === 0;
}

function render(analysis) {
  if (pendingRulesRevision && analysis.table_rules_revision !== pendingRulesRevision) {
    analysis = { ...analysis, advice: null, advice_unavailable_reason: "table_rules_unverified" };
  } else if (analysis.table_rules_revision === pendingRulesRevision) pendingRulesRevision = null;
  lastAnalysis = analysis;
  showStatus("live", "live");
  const state = analysis.state;
  setStreetBadge(statusOf(analysis, "street") === "valid" ? state.street : null);
  const boardKnown = statusOf(analysis, "board_cards") === "valid";
  const heroKnown = statusOf(analysis, "hero_cards") === "valid" && state.hero_cards.length === 2;
  const boardCards = boardKnown ? state.board_cards : [];
  renderCardSlots(els.boardSlots, boardCards, 5, lastBoardCount);
  lastBoardCount = boardCards.length;
  renderCardSlots(els.heroSlots, heroKnown ? state.hero_cards : [], 2);
  const potKnown = statusOf(analysis, "pot") === "valid";
  els.potValue.classList.toggle("unknown", !potKnown);
  els.potValue.replaceChildren();
  if (potKnown) {
    els.potValue.append(document.createTextNode(state.pot));
    const unit = document.createElement("span"); unit.className = "unit"; unit.textContent = t("chips");
    els.potValue.append(unit);
  } else setEmptyValue(els.potValue, t("notCalibrated"));
  const winPct = analysis.equity.win_rate * 100;
  const tiePct = analysis.equity.tie_rate * 100;
  const equityAvailable = heroKnown && analysis.equity.available !== false;
  els.equityBasis.textContent = !equityAvailable ? t("equityWaiting")
    : (analysis.equity.basis === "uniform_random_active_opponents"
      ? t("randomBasis").replace("{count}", analysis.equity.opponent_count) : "");
  if (!equityAvailable) {
    els.equityBar.classList.add("idle");
    setEmptyValue(els.winRate, "—");
    els.winRate.className = "win idle";
    els.tieRate.textContent = `${t("tie")} —`;
    els.segWin.style.width = "0%";
    els.segTie.style.width = "0%";
  } else {
    els.equityBar.classList.remove("idle");
    els.winRate.textContent = `${winPct.toFixed(1)}%`;
    els.winRate.className = "win " + (analysis.equity.win_rate >= 0.55 ? "" : analysis.equity.win_rate >= 0.35 ? "mid" : "low");
    els.tieRate.textContent = `${t("tie")} ${tiePct.toFixed(1)}%`;
    els.segWin.style.width = `${winPct}%`; els.segTie.style.width = `${tiePct}%`;
  }
  const confidence = analysis.confidence.overall_confidence;
  els.confidenceValue.textContent = `${t("confidence")} ${(confidence * 100).toFixed(0)}%`;
  els.confidenceBadge.style.background = confidence >= 0.9 ? "var(--good)" : confidence >= 0.6 ? "var(--warn)" : "var(--bad)";
  renderFieldStatuses(Object.fromEntries(analysis.confidence.field_status));
  renderAdvice(analysis.advice, analysis.advice_unavailable_reason);
  els.footerLeft.textContent = `${t("frame")} ${analysis.frame_seq}`;
  els.footerRight.textContent = new Date().toLocaleTimeString(activeLanguage() === "zh" ? "zh-CN" : "en");
}

function renderEmpty() {
  renderCardSlots(els.boardSlots, [], 5); renderCardSlots(els.heroSlots, [], 2);
  setStreetBadge(null);
  setEmptyValue(els.potValue, t("noData")); els.potValue.classList.add("unknown");
  setEmptyValue(els.winRate, "—"); els.winRate.className = "win idle";
  els.tieRate.textContent = `${t("tie")} —`;
  els.equityBasis.textContent = t("equityWaiting");
  els.segWin.style.width = "0%"; els.segTie.style.width = "0%"; els.equityBar.classList.add("idle");
  setEmptyValue(els.confidenceValue, `${t("confidence")} —`);
  els.confidenceBadge.style.background = "var(--text-faint)";
  renderFieldStatuses({});
  setEmptyValue(els.footerLeft, `${t("frame")} —`);
  setEmptyValue(els.footerRight, "—");
  renderAdvice(null);
}

function showStatus(message, tone, raw = false) {
  status = { message, tone, raw };
  // Nothing has been confirmed yet, so the dot pulses instead of claiming a
  // steady state.
  const dotTone = tone || (message === "live" ? "" : "pending");
  els.connDot.className = ("dot " + dotTone).trim();
  els.connLabel.textContent = raw ? message : t(message);
  setPhase(raw ? "error" : phaseFor(message, tone));
  updateStatusBanner(message, raw);
}

function connect() {
  const url = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  showStatus("connecting", "");
  let socket;
  try { socket = new WebSocket(url); } catch (_) { showStatus("cannotReach", "error"); return; }
  socket.onopen = () => showStatus("waiting", "");
  socket.onmessage = (event) => {
    let payload; try { payload = JSON.parse(event.data); } catch (_) { return; }
    if (payload.error) { showStatus(payload.error, "error", true); return; }
    render(payload);
  };
  socket.onerror = () => showStatus("connectionError", "error");
  socket.onclose = () => { showStatus("disconnected", "error"); setTimeout(connect, 3000); };
}

let tableRuleDocument = null;
let pendingRulesRevision = null;
const ruleIds = {
  mode: "rules-mode", table_size: "rules-table-size", small_blind: "rules-small-blind",
  big_blind: "rules-big-blind", ante: "rules-ante", minimum_chip: "rules-minimum-chip",
  rake_percent: "rules-rake-percent", rake_cap_bb: "rules-rake-cap",
  straddle_mode: "rules-straddle-mode", straddle_amount: "rules-straddle-amount",
  extra_effects: "rules-extra-effects",
};
async function loadTableRules() {
  const feedback = document.getElementById("rules-feedback");
  const save = document.getElementById("save-table-rules");
  save.disabled = true;
  try {
    const response = await fetch("/table-rules", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || t("tableRulesInvalid"));
    tableRuleDocument = payload.rules;
    for (const [field, id] of Object.entries(ruleIds)) {
      let value = tableRuleDocument[field];
      if (field === "rake_percent" && value !== null) value = Number(value) * 100;
      document.getElementById(id).value = value ?? "";
    }
    feedback.textContent = "";
    save.disabled = false;
  } catch (error) { feedback.textContent = error.message; }
}
document.getElementById("save-table-rules").addEventListener("click", async () => {
  const feedback = document.getElementById("rules-feedback");
  const button = document.getElementById("save-table-rules");
  if (!tableRuleDocument || !document.querySelector(".settings-dialog form").reportValidity()) return;
  const rules = { ...tableRuleDocument };
  for (const [field, id] of Object.entries(ruleIds)) {
    const value = document.getElementById(id).value.trim();
    rules[field] = field === "table_size" ? Number(value)
      : field === "rake_percent" ? (value ? (Number(value) / 100).toFixed(8) : null)
      : (["mode", "straddle_mode", "extra_effects"].includes(field) ? value : (value || null));
  }
  rules.evidence = { source: "user_settings", mode: rules.mode };
  button.disabled = true;
  try {
    const response = await fetch("/table-rules", { method: "PUT",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(rules) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || t("tableRulesInvalid"));
    tableRuleDocument = payload.rules;
    pendingRulesRevision = payload.revision;
    if (lastAnalysis) {
      lastAnalysis = { ...lastAnalysis, advice: null, advice_unavailable_reason: payload.unavailable_reason || "table_rules_unverified" };
      render(lastAnalysis);
    }
    feedback.textContent = t("rulesSaved");
  } catch (error) { feedback.textContent = error.message; }
  finally { button.disabled = false; }
});
els.settingsButton.addEventListener("click", () => { els.settingsDialog.showModal(); loadTableRules(); });
els.languageSelect.addEventListener("change", () => {
  savedLanguagePreference = els.languageSelect.value;
  // Keep this as an upgrade fallback, but the desktop server is the durable
  // source because WKWebView storage may be ephemeral between app launches.
  try { localStorage.setItem(LANGUAGE_STORAGE_KEY, savedLanguagePreference); } catch (_) {}
  fetch("/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ language: savedLanguagePreference }),
    keepalive: true,
  }).catch(() => {});
  applyLanguage();
});

async function loadLanguagePreference() {
  let fallback = "auto";
  try { fallback = localStorage.getItem(LANGUAGE_STORAGE_KEY) || "auto"; } catch (_) {}
  savedLanguagePreference = ["auto", "en", "zh"].includes(fallback) ? fallback : "auto";
  try {
    const response = await fetch("/settings", { cache: "no-store" });
    const settings = await response.json();
    if (response.ok && ["auto", "en", "zh"].includes(settings.language)) {
      savedLanguagePreference = settings.language;
    }
  } catch (_) {}
  applyLanguage();
}

loadLanguagePreference();
connect();
