"""Minimal integration of one validated offline study report into the review desk.

The study example is a synthetic, frozen research artifact. It is loaded only
through a controlled registry whose pinned digests must match the repository
bytes, and it is never attached to an observed hand: a saved study record only
ever shows the study's own numbers. Nothing here reads media, opens a capture
device, calls a vision API, or changes observations, human notes or AI
candidates.

Exact values stay exact: every reconciliation is recomputed as a Fraction, and
the decimal readings exist for display only.
"""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid


STUDY_ID = re.compile(r"\d{8}T\d{6}-[a-f0-9]{12}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "aa-study-records-v1"
LINEAGE = "OFFLINE_STUDY_RESULT_VIEW_NO_SOLVER_NO_VISION_NO_DEVICE"
SAFE_PATH = re.compile(r"(?!.*\.\.)[A-Za-z0-9_.\-/]+\Z")
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_WORLDS = 8
MAX_PATHS = 64
MAX_ROWS = 64
MAX_RECENT = 30
EXACT_ENCODING = "exact_rational_string_n_over_d"
OUTCOMES = ("frozen_policy", "check_fold", "check_call")
BOOK_NAMES = ("manual_reference", "training_selected")
CURRENT = "CURRENT"
PREVIEW = "PREVIEW_NOT_SAVED"
HISTORICAL = "HISTORICAL_UNVERIFIED"
INVALID = "INVALID"
IDENTITY_KEYS = (
    "source_type", "example_id", "report_sha256", "protocol_sha256",
    "input_sha256", "study_protocol_sha256", "experiment_id", "parent_head",
    "study_group", "study_world", "policy_book_sha256", "world_scenario_sha256",
    "source_revision", "bound_record_id", "bound_input_sha256", "note")
VIEW_KEYS = ("source", "factors", "table_context", "claims", "missing",
             "decision")
DECISION_KEYS = ("strategy_eligible", "advice_emitted", "live_advice")
CONTEXT_KEYS = ("source_type", "scope_label", "hero_seat", "hero_cards",
                "board_cards", "history", "unit", "rules_summary",
                "pot_at_decision", "pot_components", "to_call", "root_actions",
                "synthetic_assumption", "missing")
HERO_LABEL = {"manual_reference": "手工参考策略本",
              "training_selected": "训练选中策略本"}
BOOK_LABEL = {"manual_reference": "手工参考策略", "training_selected": "训练选中策略"}
UNIT = "chips"
CLAIMS = (
    "本示例是固定输入的合成研究，不是本桌数据，也不是任何一手真实牌局。",
    "条件净 EV 的单位是筹码（chips）；小数仅供参考展示，精确值以分数为准，"
    "对账只用精确分数。",
    "相对权重倍数是该组合权重的乘法因子，不是归一化之后的概率倍数。",
    "结果不是 bb/100、不是盈利率、不是 GTO，也不是实战行动指令。",
    "指定世界内的负差表示该策略差于该条基线；不同因子间符号变化不改变这条事实，"
    "也不能据此外推为总体更差或总体更好。",
    "范围、响应模型与桌规都是手写声明式假设；真实抽水、封顶与摊牌规则仍为 UNKNOWN。",
)
MISSING = (
    "未绑定任何观测牌局记录：本记录不包含真实牌局的牌面、行动或金额。",
    "未观测对手范围：全部组合权重都是手写示例值，不是实测频率。",
    "真实抽水/封顶/摊牌规则 UNKNOWN：本示例不能回填到任何真实牌局。",
    "未评估跨世界总体优劣：只给出指定因子下指定世界的条件期望。",
)


class StudyRecordError(ValueError):
    """Safe, user-facing validation failure with no local path or secret data."""


def now():
    return datetime.now(timezone.utc).isoformat()


def default_examples_root():
    return Path(__file__).resolve().parents[3] / "configs" / "strategy" / "examples"


# Controlled registry. An example is only ever loaded from these repository
# files, and the bytes must match the pinned digest; path strings found inside a
# report are compared for consistency but are never used to open anything.
SUPPORTED_EXAMPLES = (
    {
        "example_id": "range-sensitivity-synthetic-v1",
        "title": "对手范围敏感性对照（合成研究示例）",
        "source_type": "SYNTHETIC_STUDY_EXAMPLE",
        "scope_label": "合成研究示例 · 非本桌 · 非真实对局",
        "summary": "同一本冻结策略下，只改变一名对手的一个组合权重（因子 0.5 / 1 / 2），"
                   "查看三组条件净 EV、相对基线差与主要终局路径贡献。",
        "report": "strategy-diag-c-range-experiment-v1.json",
        "report_sha256": (
            "577d87f23f569f81ad14c54748e33b7af7e228d3d9557f5577fa4f0d253db023"),
        "protocol": "strategy-diag-c-range-protocol-v1.json",
        "protocol_sha256": (
            "7b77d280a05336916f2e9b8230e0f7294df0ea931a9f955094af23dd9012f8ce"),
        "input": "threeway-river-response-manual.json",
        "input_sha256": (
            "4f5d11f7446e77e755bfb4266c79fe32fc9092c0f7fbf630b0e63e4930905a64"),
        "study_protocol": "threeway-validation-protocol-v1.json",
        "study_protocol_sha256": (
            "7a34e781aaa02a2ca8c301957ae0a274a1690b6ee2e826d24ff1758052798759"),
    },
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def display(value):
    """Four-decimal reading of a Fraction; display only, never for checks."""
    with localcontext() as context:
        context.prec = 28
        return f"{Decimal(value.numerator) / Decimal(value.denominator):.4f}"


def amount(value, unit=UNIT):
    """Exact amount envelope; a declared Decimal is converted exactly."""
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise StudyRecordError("金额必须是有限数值")
        value = Fraction(value)
    if not isinstance(value, Fraction):
        raise StudyRecordError("精确数值必须是分数")
    return {"exact": str(value), "display": display(value), "unit": unit}


def fraction(text, label):
    if not isinstance(text, str):
        raise StudyRecordError(f"{label}必须是精确分数字符串")
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise StudyRecordError(f"{label}不是合法的精确分数") from None
    return value


def positive_int(value, label, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise StudyRecordError(f"{label}超出受支持范围")
    return value


def hex_id(value, label, pattern=HEX64):
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise StudyRecordError(f"{label}缺失或不是完整的十六进制身份")
    return value


def safe_name(value, label):
    if not isinstance(value, str) or SAFE_PATH.fullmatch(value) is None or (
            value.startswith("/") or ":" in value or "\\" in value):
        raise StudyRecordError(f"{label}不是受支持的仓库内相对文件名")
    return value


def declared_path(value, label):
    """Accept only a plain repository-relative posix path, never a locator."""
    name = safe_name(value, label) if value == Path(value).as_posix() else None
    if name is None or name.startswith("/") or not name.endswith(".json"):
        raise StudyRecordError(f"{label}不是受支持的仓库内相对路径")
    return name.rsplit("/", 1)[-1]


def parse_declared_amount(value, label):
    """Exact decimal reading of one declared amount string."""
    if not isinstance(value, str):
        raise StudyRecordError(f"{label}必须是十进制字符串")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise StudyRecordError(f"{label}不是合法数值") from None
    if not number.is_finite() or number < 0:
        raise StudyRecordError(f"{label}必须是有限非负数")
    return number


def replay_public_history(history, hero_seat, seat_ids):
    """Street wagers and the current bet implied by the declared public history.

    Display arithmetic only: it re-states what the verified input file declares
    and never calls the solver or the settlement kernel.
    """
    street = {seat: Decimal(0) for seat in seat_ids}
    current_bet = Decimal(0)
    for item in history:
        actor, kind = item["actor"], item["kind"]
        if actor not in street:
            raise StudyRecordError("示例输入的公开历史引用了未知座位")
        if kind in ("bet", "raise"):
            street[actor] = parse_declared_amount(item["target"], "公开历史金额")
            current_bet = street[actor]
        elif kind == "call":
            street[actor] = current_bet
        elif kind != "fold":
            raise StudyRecordError("示例输入的公开历史含不支持的动作")
    return street, current_bet


def parse_action_key(segment):
    """Decode one `actor:kind[:target]` public-history segment."""
    parts = segment.split(":")
    if len(parts) not in (2, 3) or not parts[0].isdigit():
        raise StudyRecordError("终局路径键格式不受支持")
    actor, kind = int(parts[0]), parts[1]
    target = Decimal(0)
    if len(parts) == 3:
        if kind not in ("bet", "raise"):
            raise StudyRecordError("只有下注/加注才带金额")
        try:
            target = Decimal(parts[2])
        except InvalidOperation:
            raise StudyRecordError("终局路径键金额不是合法数值") from None
    return actor, kind, target


def build_table_context(raw_input, report):
    """Minimal synthetic experiment context, derived from verified sources.

    Cards, history, amounts and both books' root actions come from the pinned
    input file and the validated report only; nothing is taken from an observed
    hand, and no policy is re-planned for display.
    """
    document = parse_json(raw_input, "示例输入")
    keys = ("schema_version", "mode", "range_start", "range_assumptions",
            "model_assumptions", "rules", "seats", "hero_seat", "hero_cards",
            "board_cards", "action_order", "ranges", "models",
            "aggression_targets", "max_aggressions", "history", "other_fees")
    exact_keys(document, keys, "示例输入")
    hero_seat = document["hero_seat"]
    hero_cards = document["hero_cards"]
    board_cards = document["board_cards"]
    history = document["history"]
    seats = document["seats"]
    if (not isinstance(hero_cards, list) or len(hero_cards) != 2
            or not isinstance(board_cards, list) or len(board_cards) != 5
            or not all(isinstance(card, str) and len(card) == 2
                       for card in hero_cards + board_cards)):
        raise StudyRecordError("示例输入的手牌或公共牌不受支持")
    seat_ids = [item["seat_id"] for item in seats]
    street, current_bet = replay_public_history(history, hero_seat, seat_ids)
    committed = sum((parse_declared_amount(item["hand_committed"], "各座已投入")
                     for item in seats), Decimal(0))
    street_total = sum(street.values(), Decimal(0))
    pot = committed + street_total
    to_call = current_bet - street[hero_seat]
    if to_call < 0:
        raise StudyRecordError("示例输入的公开历史与 Hero 投入不一致")
    actions = []
    for book in BOOK_NAMES:
        actions.append({
            "book": book,
            "label": BOOK_LABEL[book],
            "policy_book_sha256": next(
                item["policy_book_sha256_before"] for item in report["frozen_books"]
                if item["name"] == book),
            "action": root_action_of(report, book, hero_seat),
            "synthetic_assumption": True,
        })
    for item in actions:
        action = item["action"]
        target = parse_declared_amount(action["target"], "根动作金额")
        if action["kind"] in ("bet", "raise"):
            additional = target - street[hero_seat]
        elif action["kind"] == "call":
            additional = to_call
        else:
            additional = Decimal(0)
        item["raise_to"] = (str(target)
                            if action["kind"] in ("bet", "raise") else None)
        item["additional_chips"] = str(additional)
        item["amount_reading"] = (
            f"加注到 {target}（本街追加 {additional}）"
            if action["kind"] in ("bet", "raise")
            else f"跟注（本街追加 {additional}）" if action["kind"] == "call"
            else f"{action['kind']}（本街追加 {additional}）")
    return {
        "source_type": "SYNTHETIC_STUDY_EXAMPLE",
        "scope_label": "合成实验局面 · 非本桌 · 非真实牌局",
        "hero_seat": hero_seat,
        "hero_cards": list(hero_cards),
        "board_cards": list(board_cards),
        "history": [{"actor": item["actor"], "kind": item["kind"],
                     "target": str(item["target"])} for item in history],
        "unit": UNIT,
        "rules_summary": {
            "table_size": document["rules"]["table_size"],
            "small_blind": document["rules"]["small_blind"],
            "big_blind": document["rules"]["big_blind"],
            "rake_percent": document["rules"]["rake_percent"],
        },
        "pot_at_decision": amount(pot),
        "pot_components": {
            "committed_before_street": str(committed),
            "street_wagers_from_history": str(street_total),
            "definition": "各座已投入 + 公开历史的本街投入（按示例输入声明相加，未调用引擎）",
        },
        "to_call": amount(to_call),
        "root_actions": actions,
        "synthetic_assumption": True,
        "missing": [
            "局面来自固定示例输入，不是任何观测到的牌局。",
            "底池由示例输入声明相加得出，不代表真实抽水或封顶。",
        ],
    }


def root_action_of(report, book, hero_seat):
    """The frozen book's first Hero action, read off its reached paths."""
    choices = set()
    for world in report["worlds"]:
        for path in world["books"][book]["terminal_paths"]:
            if path["status"] != "REACHED":
                continue
            for segment in path["history_key"].split("|"):
                actor, kind, target = parse_action_key(segment)
                if actor == hero_seat:
                    choices.add((kind, str(target)))
                    break
    if len(choices) != 1:
        raise StudyRecordError("冻结策略本的根动作无法唯一确定")
    kind, target = next(iter(choices))
    return {"actor": hero_seat, "kind": kind, "target": target}


def read_text(path, label, maximum=MAX_REPORT_BYTES):
    try:
        raw = path.read_bytes()
    except OSError:
        raise StudyRecordError(f"{label}无法读取") from None
    if len(raw) > maximum:
        raise StudyRecordError(f"{label}超过受支持大小")
    return raw


def parse_json(raw, label):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise StudyRecordError(f"{label}含重复字段")
            result[key] = item
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, ValueError):
        raise StudyRecordError(f"{label}不是合法的 JSON") from None


def exact_keys(document, keys, label):
    if not isinstance(document, dict) or set(document) != set(keys):
        raise StudyRecordError(f"{label}字段不受支持（可能是未知版本或已损坏）")
    return document


def validate_report(report, protocol, inputs):
    """Recompute every reported number as a Fraction; raise on any mismatch.

    `inputs` carries the already-digested bytes of the protocol, the study input
    and the study protocol, so no path from the report is ever opened here.
    """
    exact_keys(report, (
        "schema_version", "experiment_id", "scope", "quantity_encoding",
        "parent_head", "protocol", "input", "study_protocol", "study_group",
        "study_world", "selected_id", "fixed_single_varied_object",
        "frozen_books", "baseline_verification", "worlds", "readings_by_factor",
        "descriptive_direction_over_0.5_1_2", "claims_not_made",
        "strategy_eligible", "advice_emitted"), "研究报告")
    if (report["schema_version"] != SCHEMA_VERSION
            or report["quantity_encoding"] != EXACT_ENCODING
            or report["strategy_eligible"] is not False
            or report["advice_emitted"] is not False
            or not isinstance(report["scope"], str)
            or not report["scope"].startswith("OFFLINE_FIXED_POLICY")
            or not isinstance(report["experiment_id"], str)
            or not report["experiment_id"]):
        raise StudyRecordError("研究报告的版本或范围声明不受支持")
    hex_id(report["parent_head"], "研究提交身份", HEX40)
    for key, keys in (("protocol", ("path", "sha256", "frozen_status")),
                      ("input", ("path", "sha256")),
                      ("study_protocol", ("path", "sha256"))):
        block = exact_keys(report[key], keys, f"研究报告 {key}")
        name = declared_path(block["path"], f"{key} 路径")
        if block["sha256"] != inputs[key + "_sha256"]:
            raise StudyRecordError(f"研究报告声明的 {key} 哈希与仓内文件不一致")
        if name != inputs[key + "_name"]:
            raise StudyRecordError(f"研究报告声明的 {key} 文件不是受支持的文件")
    baseline = exact_keys(report["baseline_verification"],
                          ("declared", "observed", "matches_original_reading",
                           "mismatches"), "基线核对")
    if baseline["matches_original_reading"] is not True or baseline["mismatches"]:
        raise StudyRecordError("研究报告未能精确复现原失败场景读数，不能作为有效结果展示")
    books = report["frozen_books"]
    if not isinstance(books, list) or [item.get("name") for item in books] != list(
            BOOK_NAMES):
        raise StudyRecordError("研究报告的冻结策略本身份不受支持")
    frozen = {}
    for item in books:
        exact_keys(item, ("name", "policy_book_sha256_before",
                          "policy_book_sha256_after", "unchanged_across_worlds",
                          "planning_best_ev_chips", "decisions"), "冻结策略本")
        before = hex_id(item["policy_book_sha256_before"], "策略本身份")
        after = hex_id(item["policy_book_sha256_after"], "策略本身份")
        if (item["unchanged_across_worlds"] is not True or before != after
                or positive_int(item["decisions"], "冻结决策数", 64) < 1):
            raise StudyRecordError("冻结策略本没有被证明在三个世界里保持不变")
        fraction(item["planning_best_ev_chips"], "规划最优 EV")
        frozen[item["name"]] = before
    worlds = report["worlds"]
    if not isinstance(worlds, list) or len(worlds) < 2 or len(worlds) > MAX_WORLDS:
        raise StudyRecordError("研究报告的世界数量超出受支持范围")
    factors = []
    for world in worlds:
        validate_world(world, frozen, inputs["protocol_sha256"])
        factors.append(world["factor"])
    readings = report["readings_by_factor"]
    if not isinstance(readings, dict) or sorted(readings) != sorted(factors):
        raise StudyRecordError("研究报告的因子列表与逐因子读数不一致")
    for world in worlds:
        validate_reading(readings[world["factor"]], world)
    return factors


READING_KEYS = (
    "training_selected_frozen_policy", "manual_reference_frozen_policy",
    "selected_minus_manual_reference", "selected_delta_vs_check_fold",
    "selected_delta_vs_check_call", "history_likelihood")


def validate_reading(reading, world):
    exact_keys(reading, READING_KEYS, "逐因子读数")
    selected = world["books"]["training_selected"]["metrics"]
    reference = world["books"]["manual_reference"]["metrics"]
    for key in READING_KEYS:
        fraction(reading[key], f"逐因子读数 {key}")
    if fraction(reading["training_selected_frozen_policy"], "读数") != fraction(
            selected[0]["net_ev_chips"], "指标"):
        raise StudyRecordError("逐因子读数与训练选中策略的指标不一致")
    if fraction(reading["manual_reference_frozen_policy"], "读数") != fraction(
            reference[0]["net_ev_chips"], "指标"):
        raise StudyRecordError("逐因子读数与手工参考策略的指标不一致")
    if fraction(reading["selected_minus_manual_reference"], "读数") != fraction(
            world["selected_minus_manual_reference_chips"], "世界差"):
        raise StudyRecordError("逐因子读数与世界的策略差读数不一致")
    if fraction(reading["history_likelihood"], "读数") != fraction(
            world["history_likelihood"], "世界似然"):
        raise StudyRecordError("逐因子读数与世界的历史似然不一致")


def validate_world(world, frozen, protocol_sha256):
    exact_keys(world, (
        "factor", "is_declared_baseline_factor", "status", "reasons",
        "world_scenario_sha256", "history_likelihood", "joint_assignments",
        "nonzero_posterior_entries", "root_posterior", "books",
        "selected_minus_manual_reference_chips", "failure_gate_fired",
        "cross_book_branch_difference"), "研究报告世界")
    if world["status"] != "COMPLETE_CONDITIONAL_FIXED_POLICY" or world["reasons"]:
        raise StudyRecordError(
            f"因子 {world['factor']} 的世界未完成（{world['status']}），不能显示为有效比较")
    world_sha = hex_id(world["world_scenario_sha256"], "世界身份")
    fraction(world["history_likelihood"], "公开历史似然")
    exact_keys(world["books"], BOOK_NAMES, "世界内的策略本")
    for name in BOOK_NAMES:
        validate_book(world["books"][name], frozen[name], world_sha, name)
    cross = world["cross_book_branch_difference"]
    exact_keys(cross, (
        "left_policy", "right_policy", "left_policy_book_sha256",
        "right_policy_book_sha256", "distinct_books",
        "world_scenario_sha256", "completeness", "total_ev_difference_chips",
        "contribution_difference_sum_chips", "rows_total",
        "rows_with_nonzero_difference", "zero_difference_rows",
        "reach_status_counts", "rows"), "跨本分支对账")
    if (cross["distinct_books"] is not True
            or cross["left_policy_book_sha256"] != frozen["training_selected"]
            or cross["right_policy_book_sha256"] != frozen["manual_reference"]
            or cross["world_scenario_sha256"] != world_sha):
        raise StudyRecordError("跨本分支对账没有绑定到同一世界与同一对策略本")
    reported = fraction(world["selected_minus_manual_reference_chips"],
                        "选中策略与参考策略差")
    validate_reconciliation(cross, reported, "跨本分支对账")


def validate_book(book, book_sha256, world_sha, name):
    exact_keys(book, (
        "status", "reasons", "elapsed_ms", "perturbed_range",
        "policy_hash_before", "policy_hash_after", "policy_unchanged_during_world",
        "metrics", "delta_vs_check_fold_chips", "delta_vs_check_call_chips",
        "nodes", "trace", "terminal_paths", "reconciliations"), "世界内策略本")
    if book["status"] != "COMPLETE_CONDITIONAL_FIXED_POLICY" or book["reasons"]:
        raise StudyRecordError(f"{name} 在该世界的评估未完成")
    before = hex_id(book["policy_hash_before"], "策略本身份")
    after = hex_id(book["policy_hash_after"], "策略本身份")
    if (before != after or before != book_sha256
            or book["policy_unchanged_during_world"] is not True):
        raise StudyRecordError(
            "策略本没有在评估世界内被证明保持不变（空值或缺失不算证据）")
    metrics = book["metrics"]
    if (not isinstance(metrics, list)
            or [item.get("name") for item in metrics] != list(OUTCOMES)):
        raise StudyRecordError("策略指标缺少 frozen_policy / check_fold / check_call")
    values = {}
    for item in metrics:
        exact_keys(item, ("name", "net_ev_chips", "conditional_net_ev_bb",
                          "probability_of_any_fallback", "expected_fallback_count",
                          "unsupported_histories", "nodes"), "策略指标")
        values[item["name"]] = fraction(item["net_ev_chips"], f"指标 {item['name']}")
        for key in ("conditional_net_ev_bb", "probability_of_any_fallback",
                    "expected_fallback_count"):
            fraction(item[key], f"指标 {item['name']}.{key}")
        if not 0 <= fraction(item["probability_of_any_fallback"], "fallback 概率") <= 1:
            raise StudyRecordError("fallback 概率超出取值范围")
    trace = exact_keys(book["trace"], (
        "terminal_paths", "reachable_paths", "unreachable_paths",
        "reach_probability_sum", "contribution_sum_chips", "fallback_reach_sum",
        "nodes_visited"), "终局路径台账")
    paths = book["terminal_paths"]
    if not isinstance(paths, list) or not 1 <= len(paths) <= MAX_PATHS:
        raise StudyRecordError("终局路径数量超出受支持范围")
    reach_total, contribution_total = Fraction(0), Fraction(0)
    for path in paths:
        exact_keys(path, ("history_key", "status", "fallback_used",
                          "reach_probability", "conditional_terminal_net_ev_chips",
                          "contribution_chips"), "终局路径")
        reach = fraction(path["reach_probability"], "路径到达概率")
        if not 0 <= reach <= 1:
            raise StudyRecordError("路径到达概率超出取值范围")
        contribution = fraction(path["contribution_chips"], "路径加权贡献")
        if reach == 0:
            if (path["status"] != "UNREACHABLE"
                    or path["conditional_terminal_net_ev_chips"] is not None
                    or contribution != 0):
                raise StudyRecordError("不可达路径的条件终局 EV 必须是未定义且贡献为 0")
        else:
            conditional = path["conditional_terminal_net_ev_chips"]
            if path["status"] != "REACHED" or conditional is None:
                raise StudyRecordError("可达路径必须给出条件终局 EV")
            if contribution != reach * fraction(conditional, "条件终局 EV"):
                raise StudyRecordError("路径加权贡献与到达概率×条件终局 EV 不一致")
        reach_total += reach
        contribution_total += contribution
    if reach_total != 1:
        raise StudyRecordError("终局路径到达概率之和不是 1")
    if contribution_total != fraction(trace["contribution_sum_chips"],
                                      "贡献和"):
        raise StudyRecordError("终局路径贡献之和与台账贡献和不一致")
    if (fraction(trace["reach_probability_sum"], "概率和") != reach_total
            or fraction(trace["contribution_sum_chips"], "贡献和") != values[
                "frozen_policy"]
            or fraction(trace["fallback_reach_sum"], "fallback 到达质量") != fraction(
                metrics[0]["probability_of_any_fallback"], "fallback 概率")):
        raise StudyRecordError("终局路径台账没有与冻结策略的指标精确对账")
    if (fraction(book["delta_vs_check_fold_chips"], "相对基线差")
            != values["frozen_policy"] - values["check_fold"]
            or fraction(book["delta_vs_check_call_chips"], "相对基线差")
            != values["frozen_policy"] - values["check_call"]):
        raise StudyRecordError("相对基线差与指标不一致")
    for index, reconciliation in enumerate(book["reconciliations"]):
        exact_keys(reconciliation, (
            "left_policy", "right_policy", "left_policy_book_sha256",
            "right_policy_book_sha256", "distinct_books", "world_scenario_sha256",
            "completeness", "total_ev_difference_chips",
            "contribution_difference_sum_chips", "rows_total",
            "rows_with_nonzero_difference", "zero_difference_rows",
            "reach_status_counts", "rows"), "分支对账")
        if (reconciliation["world_scenario_sha256"] != world_sha
                or reconciliation["left_policy_book_sha256"] != book_sha256):
            raise StudyRecordError("分支对账没有绑定到当前世界与当前策略本")
        expected = values["frozen_policy"] - values[
            "check_fold" if index == 0 else "check_call"]
        validate_reconciliation(reconciliation, expected, "分支对账")


def validate_reconciliation(reconciliation, expected_total, label):
    rows = reconciliation["rows"]
    if not isinstance(rows, list) or len(rows) > MAX_ROWS:
        raise StudyRecordError(f"{label}的差异行超出受支持范围")
    total = Fraction(0)
    seen = set()
    for row in rows:
        exact_keys(row, ("history_key", "reach_status", "contribution_chips_left",
                         "contribution_chips_right",
                         "contribution_chips_difference"), f"{label}差异行")
        if not isinstance(row["history_key"], str) or row["history_key"] in seen:
            raise StudyRecordError(f"{label}的差异行缺少唯一路径标识")
        seen.add(row["history_key"])
        difference = fraction(row["contribution_chips_difference"],
                              f"{label}路径差")
        if (difference != fraction(row["contribution_chips_left"], f"{label}左贡献")
                - fraction(row["contribution_chips_right"], f"{label}右贡献")):
            raise StudyRecordError(f"{label}的逐路径差与两侧贡献不一致")
        if difference == 0:
            raise StudyRecordError(f"{label}的差异行必须是非零贡献差")
        total += difference
    if (total != fraction(reconciliation["contribution_difference_sum_chips"],
                          f"{label}差值和")
            or total != fraction(reconciliation["total_ev_difference_chips"],
                                 f"{label}总差")
            or total != fraction(reconciliation["total_ev_difference_chips"],
                                 f"{label}总差")):
        raise StudyRecordError(f"{label}的差值和没有与报告的 EV 差精确相等")
    if total != expected_total:
        raise StudyRecordError(f"{label}的差值和与指标算出的 EV 差不一致")
    counts = reconciliation["reach_status_counts"]
    if sum(counts.values()) != reconciliation["rows_total"]:
        raise StudyRecordError(f"{label}的到达状态计数与总行数不一致")
    if (reconciliation["rows_with_nonzero_difference"] != len(rows)
            or reconciliation["rows_with_nonzero_difference"]
            + reconciliation["zero_difference_rows"]
            != reconciliation["rows_total"]):
        raise StudyRecordError(f"{label}的非零/零差异行计数不自洽")


def factor_sentence(factor, reading, delta):
    """One fixed sentence per factor, computed from the reported numbers."""
    selected = display(fraction(reading["training_selected_frozen_policy"], "EV"))
    reference = display(fraction(reading["manual_reference_frozen_policy"], "EV"))
    verdict = "劣于" if delta < 0 else "优于"
    return (f"因子 {factor}：冻结策略 {selected} 筹码，手工参考 {reference} 筹码，"
            f"差 {display(delta)} 筹码（{verdict}该基线）。")


def build_view(example, report, raw_input):
    """Numbers-first display model; every sentence is computed from values."""
    readings = report["readings_by_factor"]
    factors = []
    for world in report["worlds"]:
        factor = world["factor"]
        selected = world["books"]["training_selected"]
        reference = world["books"]["manual_reference"]
        delta = fraction(world["selected_minus_manual_reference_chips"], "差")
        paths = [row for row in world["cross_book_branch_difference"]["rows"]]
        factors.append({
            "factor": factor,
            "factor_label": f"×{factor}",
            "is_baseline": world["is_declared_baseline_factor"] is True,
            "world_sha256": world["world_scenario_sha256"],
            "history_likelihood": amount(
                fraction(world["history_likelihood"], "似然"), "概率"),
            "joint_assignments": world["joint_assignments"],
            "nonzero_posterior_entries": world["nonzero_posterior_entries"],
            "perturbed_weights": [
                {"combo": combo, "weight": weight} for combo, weight in
                selected["perturbed_range"]["combo_weights"]],
            "outcomes": [{
                "name": item["name"],
                "label": {"frozen_policy": "冻结策略",
                          "check_fold": "过牌/弃牌基线",
                          "check_call": "过牌/跟注基线"}[item["name"]],
                "net_ev_chips": amount(fraction(item["net_ev_chips"], "EV")),
                "fallback": amount(fraction(
                    item["probability_of_any_fallback"], "fallback"), "概率"),
                "expected_fallback_count": amount(fraction(
                    item["expected_fallback_count"], "次数"), "次"),
                "unsupported_histories": item["unsupported_histories"],
            } for item in selected["metrics"]],
            "reference_frozen_policy": amount(fraction(
                reference["metrics"][0]["net_ev_chips"], "参考 EV")),
            "deltas": {
                "vs_manual_reference": amount(delta),
                "vs_check_fold": amount(fraction(
                    selected["delta_vs_check_fold_chips"], "差")),
                "vs_check_call": amount(fraction(
                    selected["delta_vs_check_call_chips"], "差")),
            },
            "failure_gates": list(world["failure_gate_fired"]),
            "paths": [{
                "history_key": row["history_key"],
                "reach_status": row["reach_status"],
                "difference": amount(fraction(
                    row["contribution_chips_difference"], "路径差")),
                "left_contribution": amount(fraction(
                    row["contribution_chips_left"], "左贡献")),
                "right_contribution": amount(fraction(
                    row["contribution_chips_right"], "右贡献")),
            } for row in paths],
            "rows_total": world["cross_book_branch_difference"]["rows_total"],
            "sentence": factor_sentence(factor, readings[factor], delta),
        })
    return {
        "source": {
            "example_id": example["example_id"],
            "title": example["title"],
            "source_type": example["source_type"],
            "scope_label": example["scope_label"],
            "lineage": LINEAGE,
            "implementation_version": IMPLEMENTATION_VERSION,
            "experiment_id": report["experiment_id"],
            "parent_head": report["parent_head"],
            "study_group": report["study_group"],
            "study_world": report["study_world"],
            "selected_id": report["selected_id"],
            "report_sha256": example["report_sha256"],
            "protocol_sha256": example["protocol_sha256"],
            "input_sha256": example["input_sha256"],
            "study_protocol_sha256": example["study_protocol_sha256"],
            "frozen_books": [{
                "name": item["name"],
                "label": {"manual_reference": "手工参考策略本",
                          "training_selected": "训练选中策略本"}[item["name"]],
                "policy_book_sha256": item["policy_book_sha256_before"],
                "planning_best_ev_chips": amount(fraction(
                    item["planning_best_ev_chips"], "规划 EV")),
                "decisions": item["decisions"],
            } for item in report["frozen_books"]],
            "varied_object": {
                "target_opponent_seat": report["fixed_single_varied_object"][
                    "target_opponent_seat"],
                "target_combos": list(report["fixed_single_varied_object"][
                    "target_combos"]),
                "relative_factors": list(report["fixed_single_varied_object"][
                    "relative_factors"]),
                "rationale": report["fixed_single_varied_object"][
                    "selection_rationale"],
            },
        },
        "factors": factors,
        "table_context": build_table_context(raw_input, report),
        "claims": list(CLAIMS),
        "missing": list(MISSING),
        "decision": {
            "strategy_eligible": False,
            "advice_emitted": False,
            "live_advice": False,
        },
    }


class AAStudyRecordStore:
    """Independent sidecar records for one controlled synthetic study example."""

    def __init__(self, directory, *, examples_root=None,
                 examples=SUPPORTED_EXAMPLES):
        self.directory = Path(directory) if directory else None
        self.examples_root = Path(examples_root) if examples_root else (
            default_examples_root())
        self.examples = tuple(examples)
        self.lock = threading.Lock()

    def _example(self, example_id):
        for item in self.examples:
            if item["example_id"] == example_id:
                return item
        raise StudyRecordError("未登记的研究示例")

    def _root(self):
        if self.directory is None:
            raise StudyRecordError("本次启动没有配置研究记录目录")
        root = self.directory / "study-records"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _folder(self, record_id):
        if not isinstance(record_id, str) or STUDY_ID.fullmatch(record_id) is None:
            raise StudyRecordError("研究记录编号格式无效")
        root = self._root()
        folder = root / record_id
        if folder.parent != root or not folder.is_dir():
            raise StudyRecordError("研究记录不存在")
        return folder

    def _read(self, path, maximum=MAX_REPORT_BYTES):
        if path.is_symlink() or not path.is_file():
            raise StudyRecordError("研究记录文件不可读")
        return read_text(path, "研究记录", maximum)

    def examples_view(self):
        items = []
        for item in self.examples:
            available, reason = True, None
            try:
                self._source_bytes(item)
            except StudyRecordError as exc:
                available, reason = False, str(exc)
            items.append({
                "example_id": item["example_id"],
                "title": item["title"],
                "summary": item["summary"],
                "source_type": item["source_type"],
                "scope_label": item["scope_label"],
                "available": available,
                "reason": reason,
                "report_sha256": item["report_sha256"],
            })
        return {"items": items, "scope": LINEAGE,
                "strategy_eligible": False, "advice_emitted": False}

    def _source_bytes(self, example):
        root = self.examples_root
        raw = {}
        for key in ("report", "protocol", "input", "study_protocol"):
            name = safe_name(example[key], f"{key} 文件名")
            path = root / name
            if path.is_symlink():
                raise StudyRecordError(f"示例文件 {name} 不是普通文件")
            content = self._read(path)
            if digest(content) != example[key + "_sha256"]:
                raise StudyRecordError(
                    f"示例文件 {name} 的哈希与登记的版本不一致，"
                    "不能用它显示为有效结果")
            raw[key] = content
        return raw

    def _validated(self, example, raw):
        report = parse_json(raw["report"], "研究报告")
        protocol = parse_json(raw["protocol"], "研究协议")
        names = {key: Path(example[key]).name for key in (
            "report", "protocol", "input", "study_protocol")}
        inputs = {
            "protocol_sha256": digest(raw["protocol"]),
            "protocol_name": names["protocol"],
            "input_sha256": digest(raw["input"]),
            "input_name": names["input"],
            "study_protocol_sha256": digest(raw["study_protocol"]),
            "study_protocol_name": names["study_protocol"],
        }
        if inputs["protocol_sha256"] != example["protocol_sha256"] or inputs[
                "input_sha256"] != example["input_sha256"] or inputs[
                "study_protocol_sha256"] != example["study_protocol_sha256"]:
            raise StudyRecordError("示例的协议或输入与登记的哈希不一致")
        validate_report(report, protocol, inputs)
        return report, protocol

    def view(self, example_id):
        """Read-only preview of a controlled example; nothing is persisted."""
        example = self._example(example_id)
        raw = self._source_bytes(example)
        report, _ = self._validated(example, raw)
        return {"record_id": None, "source_type": example["source_type"],
                "content_status": PREVIEW,
                "display_permitted": True,
                "reason": None,
                "identity": self._identity(example, report),
                "view": build_view(example, report, raw["input"])}

    def _identity(self, example, report):
        return {
            "source_type": example["source_type"],
            "example_id": example["example_id"],
            "report_sha256": example["report_sha256"],
            "protocol_sha256": example["protocol_sha256"],
            "input_sha256": example["input_sha256"],
            "study_protocol_sha256": example["study_protocol_sha256"],
            "experiment_id": report["experiment_id"],
            "parent_head": report["parent_head"],
            "study_group": report["study_group"],
            "study_world": report["study_world"],
            "policy_book_sha256": {
                item["name"]: item["policy_book_sha256_before"]
                for item in report["frozen_books"]},
            "world_scenario_sha256": [item["world_scenario_sha256"]
                                      for item in report["worlds"]],
            "source_revision": 1,
            "bound_record_id": None,
            "bound_input_sha256": example["input_sha256"],
            "note": "本记录不绑定任何观测牌局记录，也不复制其任何字段。",
        }

    def save(self, example_id):
        example = self._example(example_id)
        raw = self._source_bytes(example)
        report, _ = self._validated(example, raw)
        record_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                     + "-" + uuid.uuid4().hex[:12])
        document = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "saved_at": now(),
            "implementation_version": IMPLEMENTATION_VERSION,
            "identity": self._identity(example, report),
            "view": build_view(example, report, raw["input"]),
            "strategy_eligible": False,
            "advice_emitted": False,
        }
        with self.lock:
            folder = self._folder_for_write(record_id)
            self._write(folder / "record.json", document)
        return self._present(document, CURRENT)

    def _folder_for_write(self, record_id):
        root = self._root()
        folder = root / record_id
        folder.mkdir(exist_ok=False)
        return folder

    def _write(self, path, document):
        raw = json.dumps(document, ensure_ascii=False, allow_nan=False,
                         indent=2).encode("utf-8")
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load(self, record_id):
        folder = self._folder(record_id)
        document = parse_json(self._read(folder / "record.json"), "研究记录")
        exact_keys(document, ("schema_version", "record_id", "saved_at",
                              "implementation_version", "identity", "view",
                              "strategy_eligible", "advice_emitted"),
                   "研究记录")
        if (document["schema_version"] != SCHEMA_VERSION
                or document["record_id"] != record_id
                or not isinstance(document["saved_at"], str)
                or not document["saved_at"]
                or document["implementation_version"] != IMPLEMENTATION_VERSION
                or document["strategy_eligible"] is not False
                or document["advice_emitted"] is not False):
            raise StudyRecordError("研究记录的版本或身份与文件名不一致")
        identity = exact_keys(document["identity"], IDENTITY_KEYS, "研究记录身份")
        if (identity["source_type"] != "SYNTHETIC_STUDY_EXAMPLE"
                or identity["bound_record_id"] is not None
                or identity["source_revision"] != 1
                or not isinstance(identity["experiment_id"], str)
                or not identity["experiment_id"]
                or not isinstance(identity["study_group"], str)
                or not identity["study_group"]
                or not isinstance(identity["study_world"], str)
                or not identity["study_world"]):
            raise StudyRecordError("研究记录身份不受支持")
        for key in ("report_sha256", "protocol_sha256", "input_sha256",
                    "study_protocol_sha256", "bound_input_sha256"):
            hex_id(identity[key], f"研究记录身份 {key}")
        hex_id(identity["parent_head"], "研究记录身份 parent_head", HEX40)
        policies = identity["policy_book_sha256"]
        if (not isinstance(policies, dict) or sorted(policies) != sorted(BOOK_NAMES)
                or any(hex_id(value, "研究记录身份 策略本哈希") != value
                       for value in policies.values())):
            raise StudyRecordError("研究记录身份缺少完整的策略本哈希")
        worlds = identity["world_scenario_sha256"]
        if not isinstance(worlds, list) or len(worlds) < 2:
            raise StudyRecordError("研究记录身份缺少世界哈希")
        for value in worlds:
            hex_id(value, "研究记录身份 世界哈希")
        view = exact_keys(document["view"], VIEW_KEYS, "研究记录内容")
        decision = exact_keys(view["decision"], DECISION_KEYS, "研究记录决策状态")
        if any(decision[key] is not False for key in DECISION_KEYS):
            raise StudyRecordError("研究记录不能被标记为可实战或已产生建议")
        context = exact_keys(view["table_context"], CONTEXT_KEYS, "研究记录实验局面")
        if context["source_type"] != "SYNTHETIC_STUDY_EXAMPLE":
            raise StudyRecordError("研究记录的实验局面不是合成来源")
        return document

    def recent(self):
        root = self._root()
        items = []
        for folder in sorted((path for path in root.iterdir()
                              if path.is_dir() and STUDY_ID.fullmatch(path.name)),
                             reverse=True)[:MAX_RECENT]:
            try:
                document = self._load(folder.name)
            except StudyRecordError as exc:
                items.append(self._invalid_envelope(folder.name, str(exc)))
                continue
            status, reason = self._verify(document)
            envelope = self._envelope(document, status)
            envelope["display_permitted"] = status == CURRENT
            envelope["reason"] = reason
            items.append(envelope)
        return {"items": items, "scope": LINEAGE, "max_records": MAX_RECENT}

    def get(self, record_id):
        self._folder(record_id)  # an unknown record id stays an error
        try:
            document = self._load(record_id)
        except StudyRecordError as exc:
            return self._invalid_envelope(record_id, str(exc))
        status, reason = self._verify(document)
        return self._present(document, status, reason)

    def _invalid_envelope(self, record_id, reason):
        return {
            "record_id": record_id,
            "saved_at": None,
            "content_status": INVALID,
            "display_permitted": False,
            "reason": reason,
            "identity": None,
            "view": None,
            "source_type": None,
            "example_id": None,
            "title": None,
            "source_revision": None,
            "report_sha256": None,
            "scope": LINEAGE,
            "strategy_eligible": False,
            "advice_emitted": False,
        }

    def _verify(self, document):
        """Re-derive the record from the verified source; never trust the file.

        The saved record only ever holds an identity and a display view. Both are
        recomputed here from the pinned, hash-checked source, so a record whose
        numbers, labels, binding or example were edited on disk cannot come back
        as a current result. When the registered source itself is no longer
        available or no longer matches, the record is kept on disk but can only
        be reported as unverified history.
        """
        identity = document["identity"]
        try:
            example = self._example(identity["example_id"])
        except StudyRecordError as exc:
            return INVALID, f"记录绑定的示例未登记：{exc}"
        try:
            raw = self._source_bytes(example)
        except StudyRecordError:
            return HISTORICAL, ("登记的示例源已被替换或不可用：本条记录保留文件，"
                                "但无法再与已验证来源核对，仅作历史留存。")
        try:
            report, _ = self._validated(example, raw)
        except StudyRecordError as exc:
            return INVALID, f"示例来源未通过校验：{exc}"
        if identity != self._identity(example, report):
            return INVALID, "记录身份与已验证来源不一致（可能被改写或来自旧版本）。"
        if document["view"] != build_view(example, report, raw["input"]):
            return INVALID, "记录内容与已验证来源重建的结果不一致（可能被改写或来自旧版本）。"
        return CURRENT, None

    def _envelope(self, document, status):
        identity = document["identity"]
        return {
            "record_id": document["record_id"],
            "saved_at": document["saved_at"],
            "content_status": status,
            "source_type": identity["source_type"],
            "example_id": identity.get("example_id"),
            "title": next((item["title"] for item in self.examples
                           if item["example_id"] == identity.get("example_id")),
                          None),
            "source_revision": identity.get("source_revision"),
            "report_sha256": identity.get("report_sha256"),
        }

    def _present(self, document, status, reason=None):
        envelope = self._envelope(document, status)
        envelope["identity"] = document["identity"]
        envelope["display_permitted"] = status == CURRENT
        envelope["view"] = document["view"] if envelope["display_permitted"] else None
        envelope["reason"] = reason
        envelope["scope"] = LINEAGE
        envelope["strategy_eligible"] = False
        envelope["advice_emitted"] = False
        return envelope

    def close(self):
        """No background work, queues or sockets to release."""
