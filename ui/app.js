// PokerSense companion panel. Language follows the local system on first run
// and is persisted when the player changes it in Settings.

const SUIT_SYMBOL = { s: "♠", h: "♥", d: "♦", c: "♣" };
const RED_SUITS = new Set(["h", "d"]);
const LANGUAGE_STORAGE_KEY = "pokersense.language";
const FIELD_ORDER = ["hero_cards", "board_cards", "street", "pot", "stacks", "bet_size", "action"];
const TRANSLATIONS = {
  en: {
    board: "Board", hero: "Hero", pot: "Pot", winRate: "Win Rate", settings: "Settings",
    language: "Language", languageAuto: "System default", languageHint: "Changes are saved automatically.",
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
    board: "公共牌", hero: "底牌", pot: "底池", winRate: "胜率", settings: "设置",
    language: "语言", languageAuto: "跟随系统", languageHint: "更改会自动保存。",
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
  connDot: document.getElementById("conn-dot"), connLabel: document.getElementById("conn-label"),
  streetBadge: document.getElementById("street-badge"), boardSlots: document.getElementById("board-slots"),
  heroSlots: document.getElementById("hero-slots"), potValue: document.getElementById("pot-value"),
  winRate: document.getElementById("win-rate"), tieRate: document.getElementById("tie-rate"),
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

function renderAdvice(advice) {
  if (!advice) { els.advicePanel.hidden = true; return; }
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
  lastAnalysis = analysis;
  showStatus("live", "live");
  const state = analysis.state;
  els.streetBadge.textContent = statusOf(analysis, "street") === "valid" ? state.street : "—";
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
    const unit = document.createElement("span"); unit.className = "unit"; unit.textContent = "bb";
    els.potValue.append(unit);
  } else els.potValue.textContent = t("notCalibrated");
  const winPct = analysis.equity.win_rate * 100;
  const tiePct = analysis.equity.tie_rate * 100;
  if (!heroKnown) {
    els.equityBar.classList.add("idle");
    els.winRate.textContent = "—";
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
  renderAdvice(analysis.advice);
  els.footerLeft.textContent = `${t("frame")} ${analysis.frame_seq}`;
  els.footerRight.textContent = new Date().toLocaleTimeString(activeLanguage() === "zh" ? "zh-CN" : "en");
}

function renderEmpty() {
  renderCardSlots(els.boardSlots, [], 5); renderCardSlots(els.heroSlots, [], 2);
  els.streetBadge.textContent = "—"; els.potValue.textContent = "—"; els.potValue.classList.add("unknown");
  els.winRate.textContent = "—"; els.winRate.className = "win idle"; els.tieRate.textContent = `${t("tie")} —`;
  els.segWin.style.width = "0%"; els.segTie.style.width = "0%"; els.equityBar.classList.add("idle");
  els.confidenceValue.textContent = `${t("confidence")} —`; els.confidenceBadge.style.background = "var(--text-faint)";
  renderFieldStatuses({}); els.footerLeft.textContent = `${t("frame")} —`; els.footerRight.textContent = "—";
  renderAdvice(null);
}

function showStatus(message, tone, raw = false) {
  status = { message, tone, raw };
  els.connDot.className = "dot " + (tone || "");
  els.connLabel.textContent = raw ? message : t(message);
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

els.settingsButton.addEventListener("click", () => els.settingsDialog.showModal());
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
