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
    """Assumptions start unknown: a fee state must be confirmed, never defaulted."""
    return {
        "range_source": "manual_unvalidated",
        "ranges": [],
        "models": [],
        "aggression_targets": [],
        "max_aggressions": 1,
        "other_fees": {"value": None, "provenance": "unknown"},
    }


REQUIRED_FACTS = (
    ("hero_seat", "Hero 座位号未知：请确认河牌开始时 Hero 的座位"),
    ("hero_cards", "Hero 手牌未知：请填写或从结构化快照带入"),
    ("board_cards", "公共牌未知：请填写五张河牌"),
    ("action_order", "行动顺序未知：请填写河牌开始时三名活跃玩家的行动顺序"),
    ("seats", "各座位状态、筹码与已投入未知：请补录或从结构化快照带入"),
    ("history", "公开历史未知：请填写，或明确确认「本手没有任何公开行动」"),
    ("pot_display", "显示底池未知：请填写（底池必须能与投入对账）"),
    ("table_rules", "本桌规则未知：请先在本桌规则里保存完整桌规"),
)


def _require_known(facts):
    for key, message in REQUIRED_FACTS:
        block = facts.get(key)
        if not isinstance(block, dict) or block.get("provenance") == "unknown" \
                or block.get("value") is None:
            raise HandInputError(message)


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
    _require_known(facts)
    rows = []
    for item in facts["seats"]["value"]:
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
    levels = {Decimal(row["hand_committed"]) for row in rows
              if row["status"] == "ACTIVE"}
    if len(levels) != 1:
        raise HandInputError("三名 ACTIVE 玩家的已投入必须相等（单底池，无边池）")
    level = next(iter(levels))
    if any(Decimal(row["hand_committed"]) > level for row in rows):
        raise HandInputError("有座位投入高于活跃玩家，存在边池，当前不支持")
    raw_history = facts["history"]["value"]
    history = []
    for item in raw_history:
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
    seen_seats = set()
    for item in assumptions.get("ranges") or []:
        seat = int(item.get("seat_id"))
        if seat in seen_seats:
            raise HandInputError(f"座位 {seat} 的范围重复给出；请合并为一行")
        seen_seats.add(seat)
        combos = []
        seen_combos = set()
        for entry in item.get("combos") or []:
            combo = str(entry.get("combo", "")).strip()
            if not COMBO.fullmatch(combo):
                raise HandInputError(f"座位 {seat} 的范围组合「{combo}」格式非法（形如 JhJd）")
            if combo[0] == combo[2] and combo[1] == combo[3]:
                raise HandInputError("范围组合不能是同一张牌")
            normalized_combo = (combo[0].upper() + combo[1].lower()
                                + combo[2].upper() + combo[3].lower())
            if normalized_combo in seen_combos:
                raise HandInputError(
                    f"座位 {seat} 的组合 {normalized_combo} 重复；请合并权重而不是重复列出")
            seen_combos.add(normalized_combo)
            weight = _amount(entry.get("weight", "1"), "组合权重")
            if weight <= 0:
                raise HandInputError("组合权重必须为正数")
            combos.append({"combo": normalized_combo, "weight": str(weight)})
        if not combos:
            raise HandInputError(f"座位 {seat} 至少要给出一个合法组合")
        if seat not in opponents:
            raise HandInputError("范围只能给 ACTIVE 的对手座位")
        ranges.append({"seat_id": seat, "combos": combos})
    if sorted(item["seat_id"] for item in ranges) != opponents:
        raise HandInputError("必须为两名 ACTIVE 对手各给出一套具体组合范围")
    models = []
    grouped = {}
    seen_rows = set()
    for row in assumptions.get("models") or []:
        seat = int(row.get("seat_id"))
        key = str(row.get("key", "")).strip().lower()
        if key not in WEIGHT_KEYS:
            raise HandInputError(f"响应权重的动作「{key or '空'}」不受支持")
        if (seat, key) in seen_rows:
            raise HandInputError(
                f"座位 {seat} 的 {key} 权重重复给出；请合并为一行")
        seen_rows.add((seat, key))
        if seat not in opponents:
            raise HandInputError("响应权重只能给 ACTIVE 的对手座位")
        value = _amount(row.get("weight"), "响应权重")
        grouped.setdefault(seat, []).append({"key": key, "weight": str(value)})
    for seat in sorted(grouped):
        table = grouped[seat]
        if all(Decimal(entry["weight"]) == 0 for entry in table):
            raise HandInputError(f"座位 {seat} 的响应权重不能全为 0")
        models.append({"seat_id": seat,
                       "weights": sorted(table, key=lambda entry: entry["key"])})
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
    fees = assumptions.get("other_fees")
    if not isinstance(fees, dict) or fees.get("provenance") not in (
            "human_confirmed", "assumed"):
        raise HandInputError(
            "额外费用未知：请确认本手没有额外费用，或明确选择「零额外费用分析情景（假设）」；"
            "未知费用不会自动当 0")
    if str(fees.get("value")) not in ("0", "0.0"):
        raise HandInputError(
            "额外费用不是 0：当前内核只支持 other_fees = 0；"
            "非零费用请单独作为未支持项说明，不能填估值")
    return {"range_source": "manual_unvalidated", "ranges": ranges,
            "models": models, "aggression_targets": targets,
            "max_aggressions": max_aggressions,
            "other_fees": str(fees.get("value")),
            "other_fees_provenance": fees["provenance"]}


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
        if item["actor"] == normalised_facts["hero_seat"]:
            # Hero's own actions are never an opponent response: the kernel's own
            # legal-transition replay below is what validates them.
            continue
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
    checked = {"facts": normalised_facts, "assumptions": normalised,
               "reasons": list(reasons), "combo_product": product,
               "current_bet": str(current), "to_call": str(max(to_call, Decimal(0))),
               "implied_pot": str(implied), "street_wagers": {
                   str(seat): str(value) for seat, value in street.items()},
               "hero_street_wager": str(street.get(normalised_facts["hero_seat"],
                                                   Decimal(0))),
               "row_amounts": [{
                   "seat_id": row["seat_id"], "status": row["status"],
                   "stack": row["stack"], "hand_committed": row["hand_committed"],
                   "street_wager": str(street.get(row["seat_id"], Decimal(0))),
                   "to_call": str(max(
                       current - street.get(row["seat_id"], Decimal(0)),
                       Decimal(0))) if row["status"] == "ACTIVE" else None,
               } for row in normalised_facts["seats"]],
               "document": None, "root_actions": []}
    # The history is always replayed through the kernel's own transitions, even
    # when a static check already failed, so a wrong order is reported as such.
    document = _document(normalised_facts, normalised)
    checked["document"] = document
    replay = _replay(document, normalised_facts["hero_seat"])
    checked["reasons"] = list(reasons) + replay["reasons"]
    if not checked["reasons"]:
        checked["root_actions"] = replay["actions"]
    return checked


def _document(normalised_facts, normalised):
    """The exact document the existing threeway entry point accepts."""
    rules = normalised_facts["rules"]
    seats = [{"seat_id": row["seat_id"], "stack": row["stack"],
              "hand_committed": row["hand_committed"], "status": row["status"]}
             for row in normalised_facts["seats"]]
    return {
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


def _replay(document, hero_seat):
    """Replay the public history through the kernel's own legal transitions.

    No second set of poker rules: the same `_Tree` / `_Node` the analysis entry
    point uses decides whether each declared action is legal, whether the order
    matches the declared actors, and whether the history ends at a Hero decision.
    """
    from poker_engine.core.errors import InvalidStateError
    from poker_engine.strategy.threeway_river_v1 import _Node, _Tree
    from tools.analyze_threeway_river import scenario_from_dict

    try:
        scenario = scenario_from_dict(document)
        tree = _Tree(scenario, MAX_JOINT_ASSIGNMENTS, 20000)
        node = _Node(tree.active, scenario.action_order,
                     tuple(Decimal(0) for _ in scenario.seats), Decimal(0),
                     scenario.rules.big_blind, 0, ())
        for item in document["history"]:
            legal = tree.legal(node)
            wanted = next((action for action in legal
                           if action.actor == item["actor"]
                           and action.kind == item["kind"]
                           and str(action.target) == item["target"]), None)
            if wanted is None:
                return {"reasons": [
                    f"公开历史第 {document['history'].index(item) + 1} 步"
                    f"（座位 {item['actor']} {item['kind']} {item['target']}）"
                    "在当前合法行动里不存在：请核对行动顺序、金额与行动者"],
                    "actions": []}
            node = tree.advance(node, wanted)
        if not tree.legal(node) or node.pending[0] != hero_seat:
            return {"reasons": ["公开历史必须停在 Hero 的未结束决策点："
                                "请检查是否多写或少写了行动"],
                    "actions": []}
        return {"reasons": [], "actions": [_action_row(tree, node, action)
                                           for action in tree.legal(node)]}
    except (ValueError, TypeError, ArithmeticError, InvalidStateError) as exc:
        return {"reasons": [f"公开历史无法在内核里重放：{exc}"], "actions": []}


def _action_row(tree, node, action):
    hero = tree.s.hero_seat
    wager = node.wagers[tree.index[hero]]
    if action.kind in ("bet", "raise"):
        additional = action.target - wager
        reading = f"加注到 {action.target}（本街追加 {additional}）"
    elif action.kind == "call":
        additional = max(node.current_bet - wager, Decimal(0))
        reading = f"跟注（本街追加 {additional}）"
    else:
        additional = Decimal(0)
        reading = "过牌（无追加）" if action.kind == "check" else "弃牌（无追加）"
    return {"kind": action.kind, "target": str(action.target),
            "additional_chips": str(additional), "reading": reading,
            "raise_to": str(action.target) if action.kind in ("bet", "raise") else None}


def build_document(facts, assumptions):
    """Build the exact document the existing threeway entry point accepts."""
    checked = check_support(facts, assumptions)
    if checked["reasons"]:
        raise HandInputError("；".join(checked["reasons"]))
    return checked["document"], checked


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
                    "capacity": None, "hashes": None, "amounts": None}
        document = checked["document"]
        return {"ok": True, "reasons": [], "document": document,
                "capacity": capacity(facts, assumptions),
                "hashes": facts_hashes(facts, assumptions, document),
                "amounts": {
                    "rows": checked["row_amounts"],
                    "root_actions": checked["root_actions"],
                    "to_call": checked["to_call"],
                    "current_bet": checked["current_bet"],
                    "implied_pot": checked["implied_pot"],
                    "hero_street_wager": checked["hero_street_wager"],
                    "unit": UNIT,
                    "labels": {
                        "stack": "河牌起点剩余筹码",
                        "hand_committed": "本手此前已投入",
                        "street_wager": "本街已投入",
                        "to_call": "若跟注需追加",
                        "raise_to": "加注到的本街总额",
                    }}}

    def from_record(self, record):
        return review_facts(record)


SEAT_STATE_MAP = {"active": "ACTIVE", "folded": "FOLDED", "all_in": "ALL_IN"}


def facts_from_snapshot(payload, *, source=None, source_kind="saved_record"):
    """Fill only facts an authorised structured snapshot states for this point.

    Nothing is inferred: an action-order list is not reconstructed from the seat
    numbers, the Hero seat is not taken from whoever happens to be acting now,
    and a live participant state is not treated as the river-start state. Unknown
    stays `unknown` with a concrete Chinese gap, and every candidate is kept.
    """
    if not isinstance(payload, dict):
        raise HandInputError("结构化快照为空，无法带入事实")
    facts = blank_facts()
    facts["source"] = source
    facts["source_kind"] = source_kind
    gaps = []
    cards = payload.get("cards") or {}
    hero = cards.get("hero")
    if isinstance(hero, list) and len(hero) == 2 and all(hero):
        facts["hero_cards"] = field([str(value) for value in hero], "observed",
                                    candidate=dict(cards))
    else:
        gaps.append("Hero 手牌：快照里没有可用候选，需要人工补录")
    board = cards.get("board_slots")
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
    declared_hero = payload.get("hero_seat")
    if type(declared_hero) is int:
        facts["hero_seat"] = field(declared_hero, "observed",
                                   candidate={"hero_seat": declared_hero})
    else:
        facts["hero_seat"] = field(
            None, "unknown", candidate={"current_actor": payload.get("current_actor")})
        gaps.append("Hero 座位：快照没有明确标注 Hero（不按当前行动者或座位号推断），"
                    "需要人工确认")
    declared_order = payload.get("action_order")
    if (isinstance(declared_order, list) and len(declared_order) == 3
            and all(type(value) is int for value in declared_order)):
        facts["action_order"] = field(list(declared_order), "observed",
                                      candidate=list(declared_order))
    else:
        facts["action_order"] = field(None, "unknown",
                                      candidate={"active_guess_rejected": True})
        gaps.append("河牌起点的行动顺序：快照没有明确给出（不按座位号顺序推断），"
                    "需要人工确认")
    ledger = payload.get("hand_ledger_v2") or {}
    commitments = ledger.get("hand_commitments")
    participants = (payload.get("observed_state_v2") or {}).get("participants") or {}
    states = {}
    if isinstance(participants, dict):
        for seat, item in participants.items():
            if isinstance(item, dict):
                states[int(seat)] = str(item.get("state", "")).lower()
    unambiguous = bool(states) and set(states.values()) <= set(SEAT_STATE_MAP)
    if ledger.get("status") == "HAND_COMMITMENTS_UNKNOWN" or commitments is None:
        gaps.append("各座本手已投入：快照标注为 "
                    f"{ledger.get('status', 'HAND_COMMITMENTS_UNKNOWN')}"
                    + (f"（{ledger['taint_reasons'][0]}）"
                       if ledger.get("taint_reasons") else "")
                    + "，不能当 0，需要人工核对")
    elif not isinstance(commitments, dict) or not commitments:
        gaps.append("各座本手已投入：格式不受支持，需要人工核对")
    elif not unambiguous:
        gaps.append("各座状态：快照的参与状态缺失或含不明确项（在场状态不能直接"
                    "当河牌起点状态），需要人工确认")
    elif set(int(seat) for seat in commitments) != set(states):
        gaps.append("各座状态与投入覆盖的座位不一致，需要人工核对")
    else:
        rows = []
        for seat in sorted(int(seat) for seat in commitments):
            rows.append({
                "seat_id": seat,
                "stack": str((payload.get("stacks") or {}).get(
                    str(seat), {}).get("value", "")),
                "hand_committed": str(commitments[str(seat)]),
                "status": SEAT_STATE_MAP[states[seat]]})
        facts["seats"] = field(rows, "observed", candidate=dict(ledger))
    history = payload.get("action_history_candidate")
    if isinstance(history, list) and history:
        facts["history"] = field(None, "unknown", candidate=history)
        gaps.append("公开行动历史：快照只有未确认候选，需要人工逐条确认后才可计算")
    else:
        gaps.append("公开行动历史：快照没有候选，需要人工录入")
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
