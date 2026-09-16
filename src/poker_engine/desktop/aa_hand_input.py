"""Hand-review input adapter: observed facts stay separate from manual assumptions.

Thin layer only. It never solves anything: it turns an already-authorised
structured snapshot (or a plainly typed ended hand) plus explicit opponent
assumptions into the document the existing ``analyze_threeway_river`` entry point
already accepts, refuses inputs the current kernel does not support, and reports
what is still missing in Chinese.

Nothing here reads media, runs OCR, opens a capture device, fabricates a
commitment ledger or upgrades a manual hypothesis into a live result.
"""

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re


IMPLEMENTATION_VERSION = "aa-hand-input-v1"
KIND = "threeway"
MODE = "manual_hypothesis"
UNIT = "chips"
FACTS_PROVENANCE = ("observed", "human_confirmed", "unknown")
MAX_JOINT_ASSIGNMENTS = 128
MAX_JUSTIFICATIONS = 400
SEAT_STATUSES = ("ACTIVE", "FOLDED", "ALL_IN")
CARD = re.compile(r"^[2-9TJQKA][cdhs]\Z")
COMBO = re.compile(r"^[2-9TJQKA][cdhs][2-9TJQKA][cdhs]\Z")
WEIGHT_KEYS = ("fold", "check", "call", "bet", "raise")

# Kernel validation tokens the adapter can explain without inventing anything.
KERNEL_REASONS = {
    "joint_assignment_budget_exceeded": "联合范围组合数超过当前上限（128），请缩小范围或改用容量任务处理",
    "allin_or_sidepot_boundary_not_supported": "存在全下或边池边界，当前内核不支持",
    "single_pot_river_start_equal_active_contributions_required":
        "河牌开始时三名活跃玩家的已投入必须相等且为单底池",
    "three_active_and_explicit_unique_action_order_required":
        "河牌开始时必须恰好三名活跃玩家，并给出明确的行动顺序",
    "distinct_hero_and_five_river_cards_required": "需要两张手牌与五张互不重复的公共牌",
    "two_river_start_opponent_ranges_required": "需要为两名对手各给出一套具体组合范围",
    "two_own_hand_response_models_required": "需要为两名对手各给出一套响应权重",
    "manual_unvalidated_range_source_required": "范围必须标记为人工假设（这不是已校准范围）",
    "zero_legal_response_mass": "某个对手在合法动作上的响应权重全为 0，请补权重",
    "zero_legal_world_override_mass": "响应覆盖把合法动作的权重清零了",
    "invalid_explicit_computation_budget": "计算预算参数不合法",
    "ordered_chip_aligned_targets_and_declared_raise_cap_required":
        "加注尺寸必须递增、与最小筹码对齐，并声明加注上限",
    "history_must_end_at_nonterminal_Hero_decision": "公开历史必须停在 Hero 的未结束决策点",
    "observed_history_has_zero_response_probability": "公开历史在当前响应假设下概率为 0，请核对历史或权重",
    "world_history_must_end_at_Hero_decision": "公开历史必须停在 Hero 的决策点",
}


class HandInputError(ValueError):
    """A user-facing refusal or gap; never carries hidden local data."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def field(value, provenance, candidate=None):
    """One fact with its provenance; `unknown` can never be silently filled."""
    if provenance not in FACTS_PROVENANCE:
        raise HandInputError("事实来源标记不受支持")
    return {"value": value, "provenance": provenance, "candidate": candidate}


def unknown():
    return field(None, "unknown")


def blank_facts():
    return {
        "source": None,
        "source_kind": "manual_entry",
        "ended_hand_confirmed": field(False, "unknown"),
        "hero_seat": unknown(),
        "hero_cards": unknown(),
        "board_cards": unknown(),
        "action_order": unknown(),
        "seats": unknown(),
        "history": unknown(),
        "pot_display": unknown(),
        "table_rules": unknown(),
        "observed_at": None,
    }


def blank_assumptions():
    return {
        "range_source": "manual_unvalidated",
        "ranges": [],
        "models": [],
        "aggression_targets": [],
        "max_aggressions": 1,
        "other_fees": "0",
    }


def _amount(text, label):
    if not isinstance(text, str) or not text.strip():
        raise HandInputError(f"{label}需要填写十进制金额")
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise HandInputError(f"{label}不是合法数值") from None
    if not value.is_finite() or value < 0:
        raise HandInputError(f"{label}必须是有限非负数")
    return value


def _cards(values, label, count):
    if not isinstance(values, list) or len(values) != count:
        raise HandInputError(f"{label}需要 {count} 张牌")
    cleaned = []
    for item in values:
        text = str(item or "").strip()
        if not CARD.fullmatch(text):
            raise HandInputError(f"{label}含非法牌面「{text}」，请用 c/d/h/s 与 T/J/Q/K/A")
        cleaned.append(text[0].upper() + text[1].lower())
    if len(set(cleaned)) != count:
        raise HandInputError(f"{label}存在重复牌面")
    return cleaned


def _seat_status(text):
    status = str(text or "").strip().upper()
    if status not in SEAT_STATUSES:
        raise HandInputError("座位状态只能是 ACTIVE / FOLDED / ALL_IN")
    return status


def normalise_facts(facts):
    """Validate and normalise the fact block; unknown stays unknown."""
    if not isinstance(facts, dict):
        raise HandInputError("缺少牌局事实")
    if facts.get("ended_hand_confirmed", {}).get("value") is not True:
        raise HandInputError("当前只分析已结束的牌局：请先勾选「本手已结束」")
    seats = facts.get("seats")
    if seats.get("value") is None:
        raise HandInputError("缺少各座位状态与筹码：请补录或从已保存记录带入")
    rows = []
    for item in seats["value"]:
        rows.append({
            "seat_id": int(item["seat_id"]),
            "stack": str(_amount(item.get("stack"), "座位筹码")),
            "hand_committed": str(_amount(item.get("hand_committed"), "座位已投入")),
            "status": _seat_status(item.get("status")),
        })
    hero_cards = _cards(facts["hero_cards"]["value"], "Hero 手牌", 2)
    board_cards = _cards(facts["board_cards"]["value"], "公共牌", 5)
    if set(hero_cards) & set(board_cards):
        raise HandInputError("Hero 手牌与公共牌存在重复牌面")
    order = facts["action_order"]["value"]
    if (not isinstance(order, list) or len(order) != 3
            or any(type(value) is not int for value in order)):
        raise HandInputError("行动顺序必须恰好给出三名玩家的座位号")
    active = [row["seat_id"] for row in rows if row["status"] == "ACTIVE"]
    hero = facts["hero_seat"]["value"]
    if type(hero) is not int or hero not in active:
        raise HandInputError("Hero 必须是一名 ACTIVE 玩家")
    if len(active) != 3:
        raise HandInputError(
            f"河牌开始时必须恰好三名 ACTIVE 玩家（当前 {len(active)} 名）")
    if set(order) != set(active):
        raise HandInputError("行动顺序必须与三名 ACTIVE 玩家的座位号一致")
    levels = {row["hand_committed"] for row in rows if row["status"] == "ACTIVE"}
    if len(levels) != 1:
        raise HandInputError("三名 ACTIVE 玩家的已投入必须相等（单底池，无边池）")
    level = Decimal(next(iter(levels)))
    if any(Decimal(row["hand_committed"]) > level for row in rows):
        raise HandInputError("有座位投入高于活跃玩家，存在边池，当前不支持")
    history = []
    for item in facts["history"]["value"] or []:
        actor = int(item["actor"])
        kind = str(item["kind"]).strip().lower()
        if actor not in [row["seat_id"] for row in rows]:
            raise HandInputError("公开历史里出现了不在本手的座位号")
        if kind not in ("check", "fold", "call", "bet", "raise"):
            raise HandInputError("公开历史含不支持的动作")
        target = _amount(item.get("target", "0"), "公开历史金额")
        history.append({"actor": actor, "kind": kind, "target": str(target)})
    rules = facts["table_rules"]["value"]
    if not isinstance(rules, dict):
        raise HandInputError("缺少本桌规则快照（可先把「本桌规则」填完整）")
    table_size = rules.get("table_size")
    if type(table_size) is not int or not 6 <= table_size <= 8:
        raise HandInputError("当前只支持 6–8 人发牌桌")
    if rules.get("verification_status") != "simulation":
        raise HandInputError("桌规必须保持 simulation 语义（未验证平台规则）")
    if str(rules.get("other_fees", "0")) not in ("0", "0.0"):
        raise HandInputError("未知的额外费用不能补零，也不能当已知；请先确认")
    if len({row["seat_id"] for row in rows}) != len(rows):
        raise HandInputError("座位号重复")
    return {
        "source": facts.get("source"), "source_kind": facts.get("source_kind"),
        "hero_seat": hero, "hero_cards": hero_cards, "board_cards": board_cards,
        "action_order": list(order), "seats": rows, "history": history,
        "pot_display": facts["pot_display"]["value"], "rules": deepcopy(rules),
    }


def normalise_assumptions(assumptions, active_seats, hero_seat):
    if not isinstance(assumptions, dict):
        raise HandInputError("缺少对手假设")
    if assumptions.get("range_source") != "manual_unvalidated":
        raise HandInputError("对手范围必须明确标记为人工未验证假设")
    opponents = sorted(set(active_seats) - {hero_seat})
    ranges = []
    for item in assumptions.get("ranges") or []:
        seat = int(item.get("seat_id"))
        combos = []
        for entry in item.get("combos") or []:
            combo = str(entry.get("combo", "")).strip()
            if not COMBO.fullmatch(combo):
                raise HandInputError(f"座位 {seat} 的范围组合「{combo}」格式非法（形如 JhJd）")
            if combo[0] == combo[2] and combo[1] == combo[3]:
                raise HandInputError("范围组合不能是同一张牌")
            weight = _amount(entry.get("weight", "1"), "组合权重")
            if weight <= 0:
                raise HandInputError("组合权重必须为正数")
            combos.append({"combo": combo[0].upper() + combo[1].lower()
                           + combo[2].upper() + combo[3].lower(),
                           "weight": str(weight)})
        if not combos:
            raise HandInputError(f"座位 {seat} 至少要给出一个合法组合")
        if seat not in opponents:
            raise HandInputError("范围只能给 ACTIVE 的对手座位")
        ranges.append({"seat_id": seat, "combos": combos})
    if sorted(item["seat_id"] for item in ranges) != opponents:
        raise HandInputError("必须为两名 ACTIVE 对手各给出一套具体组合范围")
    models = []
    for item in assumptions.get("models") or []:
        seat = int(item.get("seat_id"))
        weights = item.get("weights") or {}
        if seat not in opponents:
            raise HandInputError("响应权重只能给 ACTIVE 的对手座位")
        table = []
        for key in WEIGHT_KEYS:
            if key in weights:
                value = _amount(weights[key], "响应权重")
                table.append({"key": key, "weight": str(value)})
        if not table or all(Decimal(entry["weight"]) == 0 for entry in table):
            raise HandInputError(f"座位 {seat} 的响应权重不能全为 0")
        models.append({"seat_id": seat, "weights": table})
    if sorted(item["seat_id"] for item in models) != opponents:
        raise HandInputError("必须为两名 ACTIVE 对手各给出一套响应权重")
    targets = []
    for value in assumptions.get("aggression_targets") or []:
        targets.append(str(_amount(value, "加注尺寸")))
    if not targets:
        raise HandInputError("至少声明一个加注尺寸（正数、按最小筹码对齐）")
    if len(set(targets)) != len(targets):
        raise HandInputError("加注尺寸不得重复")
    max_aggressions = assumptions.get("max_aggressions")
    if type(max_aggressions) is not int or not 1 <= max_aggressions <= 3:
        raise HandInputError("加注次数上限只能是 1–3")
    other_fees = str(assumptions.get("other_fees", "0"))
    if other_fees not in ("0", "0.0"):
        raise HandInputError(
            "未知的额外费用不能当已知：当前内核只支持 other_fees = 0，"
            "请把未确认的费用留在 UNKNOWN 并说明，而不是填一个估值")
    return {"range_source": "manual_unvalidated", "ranges": ranges,
            "models": models, "aggression_targets": targets,
            "max_aggressions": max_aggressions,
            "other_fees": other_fees}


def check_support(facts, assumptions):
    """Everything the kernel needs, checked before any computation is started."""
    normalised_facts = normalise_facts(facts)
    normalised = normalise_assumptions(
        assumptions, [row["seat_id"] for row in normalised_facts["seats"]
                      if row["status"] == "ACTIVE"], normalised_facts["hero_seat"])
    reasons = []
    product = 1
    for item in normalised["ranges"]:
        product *= len(item["combos"])
    if product > MAX_JOINT_ASSIGNMENTS:
        reasons.append(
            f"两名对手的组合乘积为 {product}，超过当前联合组合上限 {MAX_JOINT_ASSIGNMENTS}；"
            "请缩小范围（本轮不调大上限）")
    highest = max(Decimal(value) for value in normalised["aggression_targets"])
    grid = {Decimal(value) for value in normalised["aggression_targets"]}
    for item in normalised_facts["history"]:
        if item["kind"] in ("bet", "raise"):
            size = Decimal(item["target"])
            if size not in grid:
                reasons.append(
                    f"公开历史里座位 {item['actor']} 的{item['kind']}金额 {size} "
                    f"不在你声明的加注尺寸网格 "
                    f"{sorted(str(value) for value in grid)} 里；"
                    "请把它加入尺寸网格后再计算（不会自动对齐网格）")
    model_by_seat = {item["seat_id"]: {entry["key"]: Decimal(entry["weight"])
                                       for entry in item["weights"]}
                     for item in normalised["models"]}
    kind_name = {"check": "check（过牌）", "bet": "bet（下注）",
                 "call": "call（跟注）", "fold": "fold（弃牌）",
                 "raise": "raise（加注）"}
    for item in normalised_facts["history"]:
        weights = model_by_seat.get(item["actor"], {})
        if weights.get(item["kind"], Decimal(0)) <= 0:
            reasons.append(
                f"公开历史里座位 {item['actor']} 的 {kind_name[item['kind']]} "
                "在给出的响应假设下概率为 0；请给该对手这一类动作一个正权重"
                "（历史必须在该假设下有正概率）")
    for row in normalised_facts["seats"]:
        if row["status"] == "ACTIVE" and Decimal(row["stack"]) <= highest:
            reasons.append(
                f"座位 {row['seat_id']} 的筹码 {row['stack']} 不大于最大加注尺寸 {highest}，"
                "会触及全下/边池边界，当前不支持")
    board = set(normalised_facts["hero_cards"]) | set(normalised_facts["board_cards"])
    for item in normalised["models"]:
        weights = {entry["key"]: Decimal(entry["weight"])
                   for entry in item["weights"]}
        if weights.get("check", Decimal(0)) + weights.get("bet", Decimal(0)) <= 0:
            reasons.append(
                f"座位 {item['seat_id']} 在「无人下注」节点上的响应权重全为 0："
                "请给 check 或 bet 一个正权重")
        if (weights.get("fold", Decimal(0)) + weights.get("call", Decimal(0))
                + weights.get("raise", Decimal(0))) <= 0:
            reasons.append(
                f"座位 {item['seat_id']} 在「面对下注」节点上的响应权重全为 0："
                "请给 fold / call / raise 一个正权重")
    for item in normalised["ranges"]:
        for entry in item["combos"]:
            combo = entry["combo"]
            cards = {combo[:2], combo[2:]}
            if len(cards) != 2:
                reasons.append(f"范围组合 {combo} 重复使用了同一张牌")
            elif cards & board:
                reasons.append(f"范围组合 {combo} 与已知牌面冲突（牌阻断）")
    street = {row["seat_id"]: Decimal(0) for row in normalised_facts["seats"]}
    current = Decimal(0)
    for item in normalised_facts["history"]:
        if item["kind"] in ("bet", "raise"):
            street[item["actor"]] = Decimal(item["target"])
            current = street[item["actor"]]
        elif item["kind"] == "call":
            street[item["actor"]] = current
    committed = sum((Decimal(row["hand_committed"])
                     for row in normalised_facts["seats"]), Decimal(0))
    street_total = sum(street.values(), Decimal(0))
    implied = committed + street_total
    if implied <= 0:
        reasons.append("底池推算为 0：请核对各座已投入与公开历史金额")
    display = normalised_facts["pot_display"]
    if display is not None:
        shown = _amount(str(display), "显示底池")
        if shown != implied:
            reasons.append(
                f"显示底池 {shown} 与按声明推算的底池 {implied} 不一致（各座已投入 "
                f"{committed} + 本街历史投入 {street_total}）；请核对，不能自动平摊或补零")
    to_call = current - street.get(normalised_facts["hero_seat"], Decimal(0))
    if to_call < 0:
        reasons.append("按公开历史，Hero 的应付额小于已投入：请核对行动顺序与金额")
    return {"facts": normalised_facts, "assumptions": normalised,
            "reasons": reasons, "combo_product": product,
            "current_bet": str(current), "to_call": str(max(to_call, Decimal(0))),
            "implied_pot": str(implied), "street_wagers": {
                str(seat): str(value) for seat, value in street.items()}}


def build_document(facts, assumptions):
    """Build the exact document the existing threeway entry point accepts."""
    checked = check_support(facts, assumptions)
    if checked["reasons"]:
        raise HandInputError("；".join(checked["reasons"]))
    normalised_facts = checked["facts"]
    normalised = checked["assumptions"]
    rules = normalised_facts["rules"]
    seats = []
    for row in normalised_facts["seats"]:
        seats.append({"seat_id": row["seat_id"], "stack": row["stack"],
                      "hand_committed": row["hand_committed"],
                      "status": row["status"]})
    document = {
        "schema_version": 1, "mode": MODE, "range_start": "river_start",
        "range_assumptions": "人工核对/补录的已结束牌局事实；不是观测验证范围",
        "model_assumptions": "人工未验证的对手响应权重假设；不是已校准模型",
        "rules": deepcopy(rules), "seats": seats,
        "hero_seat": normalised_facts["hero_seat"],
        "hero_cards": list(normalised_facts["hero_cards"]),
        "board_cards": list(normalised_facts["board_cards"]),
        "action_order": list(normalised_facts["action_order"]),
        "ranges": [{"seat_id": item["seat_id"],
                    "combos": {entry["combo"]: entry["weight"]
                               for entry in item["combos"]}}
                   for item in normalised["ranges"]],
        "models": [{"seat_id": item["seat_id"], "name": "hand-review-manual-v1",
                    "source": "manual_example",
                    "weights": {entry["key"]: entry["weight"]
                                for entry in item["weights"]},
                    "category_weights": {}, "price_multipliers": []}
                   for item in normalised["models"]],
        "aggression_targets": list(normalised["aggression_targets"]),
        "max_aggressions": normalised["max_aggressions"],
        "history": [{"actor": item["actor"], "kind": item["kind"],
                     "target": item["target"]} for item in normalised_facts["history"]],
        "other_fees": normalised["other_fees"],
    }
    return document, checked


def capacity(facts, assumptions):
    """Honest measurement: declared combos, legal joint combos, declared product."""
    from poker_engine.strategy.range_tracker import enumerate_joint_assignments
    from poker_engine.strategy.contracts import RangeDistribution
    from tools.analyze_terminal_multiway import card

    checked = check_support(facts, assumptions)
    ranges = tuple(RangeDistribution(
        item["seat_id"], {entry["combo"]: Decimal(entry["weight"])
                          for entry in item["combos"]},
        "manual_river_start_assumption", f"manual:seat{item['seat_id']}",
        confidence=0, effective_sample_size=0)
        for item in checked["assumptions"]["ranges"])
    cards = [card(value) for value in (checked["facts"]["hero_cards"]
                                       + checked["facts"]["board_cards"])]
    assignments = enumerate_joint_assignments(ranges, cards,
                                              max_combinations=MAX_JOINT_ASSIGNMENTS)
    return {
        "declared_combo_product": checked["combo_product"],
        "legal_joint_combos": len(assignments),
        "legal_joint_limit": MAX_JOINT_ASSIGNMENTS,
        "to_call": checked["to_call"], "current_bet": checked["current_bet"],
        "implied_pot": checked["implied_pot"],
        "unit": UNIT,
    }


def translate_reasons(reasons):
    """Explain kernel refusals in Chinese without hiding the original token."""
    return [f"{KERNEL_REASONS.get(text, '内核拒绝：' + text)}（{text}）"
            for text in reasons]


def facts_hashes(facts, assumptions, document):
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "facts_sha256": digest(facts),
        "assumptions_sha256": digest(assumptions),
        "input_sha256": digest(document),
        "kernel": "analyze_threeway_river",
        "scope": "MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE",
    }


class AAHandInput:
    """Stateless facade used by the HTTP layer; it never computes a policy."""

    def template(self):
        return {"facts": blank_facts(), "assumptions": blank_assumptions(),
                "rules_defaults": {"table_size": None, "small_blind": None,
                                   "big_blind": None, "ante": None,
                                   "rake_percent": None, "minimum_chip": None},
                "scope": "MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE"}

    def build(self, facts, assumptions):
        checked = check_support(facts, assumptions)
        if checked["reasons"]:
            return {"ok": False, "reasons": checked["reasons"], "document": None,
                    "capacity": None, "hashes": None}
        document, checked = build_document(facts, assumptions)
        return {"ok": True, "reasons": [], "document": document,
                "capacity": capacity(facts, assumptions),
                "hashes": facts_hashes(facts, assumptions, document)}

    def from_record(self, record):
        return review_facts(record)


def facts_from_snapshot(payload, *, source=None, source_kind="saved_record"):
    """Fill only what an authorised structured snapshot actually contains.

    Unknown stays unknown: a missing commitment ledger or an ambiguous
    participant state becomes a Chinese gap, never a zero.
    """
    if not isinstance(payload, dict):
        raise HandInputError("结构化快照为空，无法带入事实")
    facts = blank_facts()
    facts["source"] = source
    facts["source_kind"] = source_kind
    gaps = []
    cards = payload.get("cards") or {}
    hero = cards.get("hero")
    board = cards.get("board_slots")
    if isinstance(hero, list) and len(hero) == 2 and all(hero):
        facts["hero_cards"] = field([str(value) for value in hero], "observed",
                                    candidate=dict(cards))
    else:
        gaps.append("Hero 手牌：快照里没有可用候选，需要人工补录")
    if isinstance(board, list) and len([value for value in board if value]) == 5:
        facts["board_cards"] = field([str(value) for value in board], "observed",
                                     candidate=dict(cards))
    else:
        gaps.append("公共牌：快照里不足 5 张，需要人工补录")
    pot = payload.get("pot") or {}
    if pot.get("value") is not None:
        facts["pot_display"] = field(str(pot["value"]), "observed",
                                     candidate=dict(pot))
    else:
        gaps.append("显示底池：快照里没有候选，需要人工录入")
    ledger = payload.get("hand_ledger_v2") or {}
    commitments = ledger.get("hand_commitments")
    if ledger.get("status") == "HAND_COMMITMENTS_UNKNOWN" or commitments is None:
        gaps.append("各座本手已投入：快照标注为 "
                    f"{ledger.get('status', 'HAND_COMMITMENTS_UNKNOWN')}"
                    + (f"（{ledger['taint_reasons'][0]}）"
                       if ledger.get("taint_reasons") else "")
                    + "，不能当 0，需要人工核对")
    else:
        rows = []
        for seat, value in sorted(commitments.items(), key=lambda item: int(item[0])):
            rows.append({"seat_id": int(seat), "stack": str(
                (payload.get("stacks") or {}).get(seat, {}).get("value", "")),
                "hand_committed": str(value), "status": "ACTIVE"})
        facts["seats"] = field(rows, "observed", candidate=dict(ledger))
    state = payload.get("observed_state_v2") or {}
    participants = state.get("participants") or {}
    states = {seat: str(item.get("state")) for seat, item in participants.items()
              if isinstance(item, dict)} if isinstance(participants, dict) else {}
    active = sorted(int(seat) for seat, value in states.items()
                    if value.lower() in ("active", "dealt_in", "in_hand"))
    ambiguous = sorted(seat for seat, value in states.items()
                       if value.lower() in ("unknown", "waiting"))
    if facts["seats"]["value"] is not None:
        if len(active) != 3 or ambiguous:
            gaps.append(
                f"河牌开始的活跃人数：快照识别到 {len(active)} 名 active"
                + (f"、{len(ambiguous)} 个状态不明确（座位 "
                   f"{'、'.join(ambiguous)}）" if ambiguous else "")
                + "，内核要求恰好 3 名，需要人工确认")
        else:
            order = payload.get("action_order") or active
            facts["action_order"] = field([int(value) for value in order],
                                          "observed", candidate=list(order))
            facts["hero_seat"] = field(int(payload.get("hero_seat", active[0])),
                                       "observed")
    else:
        detail = (f"（快照识别到 {len(active)} 名 active、{len(ambiguous)} 个状态不明确）"
                  if states else "")
        gaps.append("各座状态与筹码：缺少投入账本，无法确定 ACTIVE 身份与单底池条件"
                    + detail)
    actor = payload.get("current_actor")
    if actor is None:
        gaps.append("当前行动者：快照为空（可能已结束或未识别），需要人工确认决策点")
    else:
        if facts["hero_seat"]["value"] is None:
            facts["hero_seat"] = field(int(actor), "observed")
    return facts, gaps


def review_facts(record):
    """Facts for one saved review record, from its structured snapshot only."""
    issue = (record or {}).get("issue") or {}
    snapshot = issue.get("observation") or {}
    payload = snapshot.get("payload") or {}
    facts, gaps = facts_from_snapshot(
        payload, source=issue.get("issue_id"), source_kind="saved_record")
    facts["observed_at"] = issue.get("saved_at")
    facts["ended_hand_confirmed"] = field(
        False, "observed" if issue.get("issue_id") else "unknown",
        candidate={"saved_at": issue.get("saved_at")})
    return {"facts": facts, "gaps": gaps,
            "source": {"issue_id": issue.get("issue_id"),
                       "source_frame": snapshot.get("source_frame"),
                       "preview_sha256": issue.get("preview_sha256"),
                       "scope": "SAVED_STRUCTURED_SNAPSHOT_NO_MEDIA_NO_OCR"},
            "strategy_eligible": False, "advice_emitted": False}
