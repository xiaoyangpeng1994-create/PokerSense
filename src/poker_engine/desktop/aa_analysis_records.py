"""Saved records for ONE actually-completed manual analysis.

Thin adapter, not a platform. It freezes what the existing
``AAConditionalAnalysis`` job already produced plus the verified input the hand
form built, and on every read it re-derives the whole view from those frozen
values instead of trusting anything the file claims about itself.

Rules that keep it honest:

* the server takes the report from the LIVE analysis facade - a client cannot
  report its own EV, and it cannot save a RUNNING / ERROR / CANCELLED / TIMED_OUT
  job or a job a newer input already replaced;
* only ``COMPLETE`` results are storable, and the stored input must be the input
  that job actually hashed;
* reopening never calls a model or the solver and never rewrites the record with
  the current table rules: a record whose rules revision or implementation
  version is older stays viewable as a HISTORICAL result when it is
  self-consistent, and a record that fails a check is refused rather than
  silently repaired or deleted;
* display decimals are re-derived from the exact rationals, so an edited display
  value is detected;
* a repeated save of the same job/input/source returns the same record, while two
  different manual sources stay separate even when their numbers coincide.
"""

from copy import deepcopy
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


SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "aa-analysis-record-v1"
SCOPE = "MANUAL_HYPOTHESIS_OFFLINE_NOT_LIVE_ADVICE"
RECORD_KIND = "manual_hypothesis_analysis"
UNIT = "chips"
MAX_RECORD_BYTES = 8 * 1024 * 1024
DECIMAL_PRECISION = 28

RECORD_ID = re.compile(r"\d{8}T\d{6}-[a-f0-9]{12}\Z")
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
ACTION_KINDS = ("check", "call", "fold", "bet", "raise")
SIZE_LESS_KINDS = ("check", "call", "fold")
ACTION_LABELS = {"check": "过牌", "call": "跟注", "fold": "弃牌",
                 "bet": "下注", "raise": "加注"}
FACT_KEYS = ("hero_seat", "hero_cards", "board_cards", "action_order", "seats",
             "history", "pot_display", "table_rules", "ended_hand_confirmed")
SOURCE_KEYS = ("issue_id", "saved_at", "preview_sha256", "source_frame", "scope")

# Self-consistency outcomes. OK and the two HISTORICAL_* states permit display;
# INVALID never does and the file is left exactly as it was.
OK = "CURRENT"
HISTORICAL_RULES = "HISTORICAL_RULES"
HISTORICAL_IMPLEMENTATION = "HISTORICAL_IMPLEMENTATION"
INVALID = "INVALID"


class AnalysisRecordError(ValueError):
    """A refusal the HTTP layer can show verbatim; never carries private data."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hex_id(value, label):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise AnalysisRecordError(f"{label}必须是 64 位十六进制摘要")
    return value


def read_text(path, label, maximum=MAX_RECORD_BYTES):
    if path.resolve().parent != path.parent or not path.is_file():
        raise AnalysisRecordError(f"{label}路径无效或文件不存在")
    if path.stat().st_size > maximum:
        raise AnalysisRecordError(f"{label}文件过大")
    return path.read_text(encoding="utf-8")


def exact_fraction(value, label):
    """Parse an ``{exact, decimal}`` block's exact rational.

    Fractions are stored as ``"515/8"`` and integers as ``"0"``; anything else is
    refused instead of being coerced.
    """
    if not isinstance(value, dict):
        raise AnalysisRecordError(f"{label}必须是含 exact/decimal 的精确数值块")
    exact = value.get("exact")
    if not isinstance(exact, str) or not exact.strip():
        raise AnalysisRecordError(f"{label}缺少精确值 exact")
    try:
        parsed = Fraction(exact)
    except (ValueError, ZeroDivisionError):
        raise AnalysisRecordError(f"{label}的 exact「{exact}」不是合法有理数") from None
    return parsed


def decimal_of(fraction):
    """The display decimal, regenerated from the exact value."""
    with localcontext() as context:
        context.prec = DECIMAL_PRECISION
        return str(Decimal(fraction.numerator) / Decimal(fraction.denominator))


def checked_amount(value, label):
    """Re-derive one exact/decimal pair and refuse an edited display value."""
    fraction = exact_fraction(value, label)
    expected = decimal_of(fraction)
    shown = value.get("decimal")
    if not isinstance(shown, str) or shown != expected:
        raise AnalysisRecordError(
            f"{label}的显示值「{shown}」与精确值 {fraction} 推不出的一致值「{expected}」"
            "不符：记录已被改动或损坏")
    return fraction, expected


def checked_decimal_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise AnalysisRecordError(f"{label}必须是十进制字符串")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise AnalysisRecordError(f"{label}「{value}」不是合法数值") from None
    if not parsed.is_finite() or parsed < 0:
        raise AnalysisRecordError(f"{label}必须是有限非负数")
    return parsed


class AAAnalysisRecordStore:
    """Sidecar records under ``<records_dir>/analysis-records/<record_id>/``."""

    def __init__(self, directory, *, analysis, rules_revision=None,
                 implementation_version=IMPLEMENTATION_VERSION):
        self.directory = None if directory is None else Path(directory)
        self.analysis = analysis
        self.rules_revision = rules_revision or (lambda: None)
        self.implementation_version = implementation_version
        self.lock = threading.RLock()

    # -- storage ----------------------------------------------------------
    def _root(self):
        if self.directory is None:
            raise AnalysisRecordError("本次启动未配置记录目录")
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    def _folder(self, record_id):
        if not isinstance(record_id, str) or not RECORD_ID.fullmatch(record_id):
            raise AnalysisRecordError("分析记录编号无效")
        root = self._root()
        folder = (root / record_id).resolve()
        if folder.parent != root or not folder.is_dir():
            raise AnalysisRecordError("分析记录不存在")
        return folder

    def _paths(self):
        root = self._root()
        return sorted((path for path in root.iterdir()
                       if path.is_dir() and RECORD_ID.fullmatch(path.name)),
                      key=lambda path: path.name, reverse=True)

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
        try:
            document = json.loads(read_text(folder / "record.json", "分析记录"))
        except json.JSONDecodeError:
            raise AnalysisRecordError("分析记录不是合法 JSON，已拒绝展示") from None
        if not isinstance(document, dict):
            raise AnalysisRecordError("分析记录格式不受支持")
        return document

    # -- save -------------------------------------------------------------
    def _live_report(self, job_id, expected_input_sha256):
        """The report must come from the server's own finished job."""
        if not isinstance(job_id, str) or not job_id:
            raise AnalysisRecordError("请指明要保存的分析任务编号")
        status = self.analysis.status()
        current = status.get("job_id")
        if not current:
            raise AnalysisRecordError("当前没有可保存的分析任务：请先计算一次")
        if current != job_id:
            raise AnalysisRecordError(
                "当前分析任务已被新的输入替换，这次保存被拒绝：请对最新结果重新计算后再保存")
        if status.get("status") != "COMPLETE":
            raise AnalysisRecordError(
                f"只有已完成的分析可以保存（当前状态 {status.get('status')}）")
        report_input = status.get("input_sha256")
        hex_id(report_input, "服务端记录的分析输入摘要")
        if expected_input_sha256 is not None:
            hex_id(expected_input_sha256, "本次核对输入摘要")
            if report_input != expected_input_sha256:
                raise AnalysisRecordError(
                    "服务端这次任务用的输入与本次核对过的输入不是同一份：保存被拒绝")
        if not isinstance(status.get("result"), dict):
            raise AnalysisRecordError("已完成的任务没有可用结果，保存被拒绝")
        return deepcopy(status)

    @staticmethod
    def _require_facts(facts, assumptions):
        if not isinstance(facts, dict) or not isinstance(assumptions, dict):
            raise AnalysisRecordError("保存需要本次的牌局事实与假设")
        for key in FACT_KEYS:
            block = facts.get(key)
            if block is not None and not isinstance(block, dict):
                raise AnalysisRecordError(f"事实块 {key} 格式不受支持")
        return deepcopy(facts), deepcopy(assumptions)

    def _identity(self, *, input_sha256, facts, assumptions, rules_revision,
                  effective_rules_sha256, source, job_id):
        return {
            "input_sha256": input_sha256,
            "facts_sha256": digest(facts),
            "assumptions_sha256": digest(assumptions),
            "rules_revision": rules_revision,
            "effective_rules_sha256": effective_rules_sha256,
            "source_issue_id": source.get("issue_id"),
            "job_id": job_id,
        }

    def _find_existing(self, identity):
        """Idempotency: the same job+input+source is one record, never two."""
        key = (identity["job_id"], identity["input_sha256"],
               identity["source_issue_id"])
        for path in self._paths():
            try:
                document = json.loads(read_text(path / "record.json", "分析记录"))
            except (AnalysisRecordError, json.JSONDecodeError):
                continue
            stored = document.get("identity") or {}
            if (stored.get("job_id"), stored.get("input_sha256"),
                    stored.get("source_issue_id")) == key:
                return document
        return None

    def save(self, *, job_id, expected_input_sha256, facts, assumptions,
             source=None, label=None):
        """Freeze ONE actually-completed analysis. Returns the saved envelope."""
        facts, assumptions = self._require_facts(facts, assumptions)
        report = self._live_report(job_id, expected_input_sha256)
        input_sha256 = report["input_sha256"]
        binding = report.get("binding") or {}
        rules_source = binding.get("rules_source")
        if rules_source not in ("document", "table"):
            raise AnalysisRecordError("这次任务的规则来源不受支持，保存被拒绝")
        facts, document, checked = self._rebuild(facts, assumptions, binding)
        # The frozen input must be byte-for-byte what the server really ran: it is
        # rebuilt through the same trusted path the build route used, so the
        # facts/assumptions and the analysed document cannot drift apart.
        if digest(document) != input_sha256:
            raise AnalysisRecordError(
                "根据本次事实与假设重建的规范输入与服务端任务用的输入不一致：保存被拒绝")
        effective_rules_sha256 = digest(binding.get("effective_rules"))
        source = {key: (source or {}).get(key) for key in SOURCE_KEYS}
        verified_rules = {
            "rules_source": rules_source,
            "rules_revision": binding.get("table_rules_revision"),
            "effective_rules_sha256": effective_rules_sha256,
        }
        identity = self._identity(
            input_sha256=input_sha256, facts=facts, assumptions=assumptions,
            rules_revision=binding.get("table_rules_revision"),
            effective_rules_sha256=effective_rules_sha256,
            source=source, job_id=job_id)
        with self.lock:
            existing = self._find_existing(identity)
            if existing is not None:
                # A retry of the same save is the same record, not a new one.
                envelope = self._present(existing)
                envelope["duplicate_of_existing"] = True
                return envelope
            record_id = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                         + "-" + uuid.uuid4().hex[:12])
            document_record = {
                "schema_version": SCHEMA_VERSION,
                "record_id": record_id,
                "saved_at": now(),
                "implementation_version": self.implementation_version,
                "record_kind": RECORD_KIND,
                "scope": SCOPE,
                "label": (label if isinstance(label, str) and label.strip()
                          else f"手动河牌分析 {record_id}")[:200],
                "source": source,
                "identity": identity,
                "job": {
                    "job_id": job_id, "kind": report.get("kind"),
                    "status": report.get("status"),
                    "input_sha256": input_sha256,
                    "binding": deepcopy(binding),
                    "verified_rules": verified_rules,
                },
                "input": document,
                "facts": facts,
                "assumptions": assumptions,
                "support": {
                    "reasons": list(checked["reasons"]),
                    "combo_product": checked["combo_product"],
                    "to_call": checked["to_call"],
                    "current_bet": checked["current_bet"],
                    "implied_pot": checked["implied_pot"],
                    "hero_street_wager": checked["hero_street_wager"],
                    "root_actions": deepcopy(checked["root_actions"]),
                    "unit": UNIT,
                },
                "amounts": {
                    "rows": deepcopy(checked["row_amounts"]),
                    "root_actions": deepcopy(checked["root_actions"]),
                    "to_call": checked["to_call"],
                    "current_bet": checked["current_bet"],
                    "implied_pot": checked["implied_pot"],
                    "hero_street_wager": checked["hero_street_wager"], "unit": UNIT,
                },
                "capacity": {
                    "declared_combo_product": checked["combo_product"],
                    "legal_joint_limit": 128, "to_call": checked["to_call"],
                    "current_bet": checked["current_bet"],
                    "implied_pot": checked["implied_pot"], "unit": UNIT,
                },
                "report": {"result": deepcopy(report.get("result")),
                           "manual_notlive": report.get("manual_notlive"),
                           "strategy_eligible": False, "advice_emitted": False},
                "strategy_eligible": False,
                "advice_emitted": False,
            }
            folder = self._root() / record_id
            folder.mkdir(exist_ok=False)
            self._write(folder / "record.json", document_record)
            return self._present(document_record)

    def _rebuild(self, facts, assumptions, binding):
        """Rebuild the analysed document from the frozen facts/assumptions.

        Uses the same trusted path the build route uses, with the task's own
        effective rules injected, so the record's input is provably the input the
        kernel really ran - not a client-supplied document.
        """
        from .aa_hand_input import build_document

        rules = binding.get("effective_rules")
        if not isinstance(rules, dict):
            raise AnalysisRecordError("任务绑定里缺少有效规则，保存被拒绝")
        facts = {**facts, "table_rules": {
            "value": deepcopy(rules), "provenance": "human_confirmed",
            "candidate": {"revision": binding.get("table_rules_revision")}}}
        try:
            document, checked = build_document(facts, assumptions)
        except Exception as exc:  # noqa: BLE001 - the reason is shown verbatim
            raise AnalysisRecordError(
                f"本次事实与假设无法复算成内核输入，保存被拒绝：{exc}") from None
        return facts, document, checked

    # -- read -------------------------------------------------------------
    def recent(self):
        rows = []
        for path in self._paths():
            try:
                document = json.loads(read_text(path / "record.json", "分析记录"))
            except (AnalysisRecordError, json.JSONDecodeError):
                rows.append({"record_id": path.name, "status": INVALID,
                             "display_permitted": False,
                             "label": None, "saved_at": None,
                             "notes": ["分析记录无法读取，已拒绝展示数值"]})
                continue
            envelope = self._present(document)
            rows.append({key: envelope.get(key) for key in (
                "record_id", "label", "saved_at", "status", "display_permitted",
                "identity", "record_kind", "implementation_version", "notes")})
        return rows

    def get(self, record_id):
        document = self._load(record_id)
        return self._present(document)

    def scenario(self, record_id):
        """The saved input/facts/assumptions, for an EXPLICIT recompute.

        Returns the frozen material only; it never touches the global table rules
        and never starts a job by itself.
        """
        envelope = self.get(record_id)
        if not envelope["display_permitted"]:
            raise AnalysisRecordError(
                "这份记录没有通过自洽校验，不能作为重算输入：" + "；".join(envelope["notes"]))
        document = envelope["document"]
        return {
            "record_id": record_id,
            "saved_at": document.get("saved_at"),
            "saved_rules_revision": document["identity"].get("rules_revision"),
            "current_rules_revision": self.rules_revision(),
            "facts": deepcopy(document.get("facts")),
            "assumptions": deepcopy(document.get("assumptions")),
            "input": deepcopy(document.get("input")),
            "recompute_source": "saved_conditions",
            "note": "按保存时的条件重算：请重新核对后再计算；不会把全局桌规改回保存时的版本",
        }

    # -- verification -----------------------------------------------------
    def _verify(self, document):
        """Re-derive identity and the whole view; return (notes, view)."""
        notes = []
        if document.get("schema_version") != SCHEMA_VERSION:
            return (["记录 schema 版本不受支持"], None)
        if document.get("record_kind") != RECORD_KIND:
            return (["这不是本次分析的记录（类型不受支持）"], None)
        record_id = document.get("record_id")
        if not isinstance(record_id, str) or not RECORD_ID.fullmatch(record_id):
            return (["记录编号无效"], None)
        identity = document.get("identity")
        if not isinstance(identity, dict):
            return (["记录缺少身份块"], None)
        try:
            for key in ("input_sha256", "facts_sha256", "assumptions_sha256"):
                hex_id(identity.get(key), f"身份 {key}")
            facts = document.get("facts")
            assumptions = document.get("assumptions")
            if not isinstance(facts, dict) or not isinstance(assumptions, dict):
                return (["记录缺少事实或假设块"], None)
            if digest(facts) != identity["facts_sha256"]:
                return (["事实块与记录身份不一致（内容已被改写）"], None)
            if digest(assumptions) != identity["assumptions_sha256"]:
                return (["假设块与记录身份不一致（内容已被改写）"], None)
            sealed = document.get("input")
            if (not isinstance(sealed, dict)
                    or digest(sealed) != identity["input_sha256"]):
                return (["规范输入与记录身份不一致（内容已被改写）"], None)
            job = document.get("job") or {}
            if job.get("input_sha256") != identity["input_sha256"]:
                return (["任务输入摘要与记录身份不一致"], None)
            if job.get("status") != "COMPLETE":
                return ([f"只有 COMPLETE 的分析可以作为结果查看（记录为 {job.get('status')}）"], None)
            binding = job.get("binding")
            if not isinstance(binding, dict):
                return (["记录缺少任务绑定"], None)
            effective_rules = binding.get("effective_rules")
            if digest(effective_rules) != identity.get("effective_rules_sha256"):
                return (["有效规则与记录身份不一致（规则已被改写）"], None)
            if digest(sealed.get("rules")) != identity.get("effective_rules_sha256"):
                return (["规范输入里的规则与记录身份不一致"], None)
            verified = job.get("verified_rules") or {}
            if verified.get("rules_revision") != binding.get("table_rules_revision"):
                return (["核对时的规则修订与任务绑定不一致"], None)
            if verified.get("effective_rules_sha256") != digest(effective_rules):
                return (["核对时的有效规则摘要与任务绑定不一致"], None)
            view = self._view(sealed, facts, assumptions, document)
        except AnalysisRecordError as exc:
            return ([str(exc)], None)

        revision = identity.get("rules_revision")
        if revision != self.rules_revision():
            notes.append(
                "这是保存时桌规版本下的历史结果，不是用当前桌规重算的结果")
            status = HISTORICAL_RULES
        elif document.get("implementation_version") != self.implementation_version:
            status = HISTORICAL_IMPLEMENTATION
        else:
            status = OK
        if document.get("implementation_version") != self.implementation_version:
            notes.append(
                "记录由另一个实现版本写入（"
                f"{document.get('implementation_version')}）；"
                f"当前为 {self.implementation_version}，请按历史版本理解这些数值")
        view["status"] = status
        view["notes"] = notes
        return notes, view

    def _view(self, sealed, facts, assumptions, document):
        """The readable result, with every number re-derived."""
        result = (document.get("report") or {}).get("result")
        if not isinstance(result, dict):
            raise AnalysisRecordError("记录里没有服务端完成报告的结果块")
        actions = []
        seen = set()
        for index, row in enumerate(result.get("root_actions") or []):
            if not isinstance(row, dict):
                raise AnalysisRecordError(f"第 {index + 1} 个根动作格式不受支持")
            action = row.get("action") or {}
            kind = action.get("kind")
            if kind not in ACTION_KINDS:
                raise AnalysisRecordError(f"根动作 {index + 1} 的动作类型「{kind}」不受支持")
            target = checked_decimal_string(
                action.get("target"), f"根动作 {index + 1} 的目标额")
            if kind in SIZE_LESS_KINDS and target != 0:
                raise AnalysisRecordError(
                    f"根动作 {index + 1} 是 {kind}，目标额必须是 0（当前 {target}）")
            if kind in ("bet", "raise") and target <= 0:
                raise AnalysisRecordError(f"根动作 {index + 1} 是 {kind}，目标额必须为正")
            if (kind, str(target)) in seen:
                raise AnalysisRecordError(f"根动作 {index + 1} 与前面的动作重复")
            seen.add((kind, str(target)))
            fraction, shown = checked_amount(row.get("ev"), f"根动作 {index + 1} 的条件净 EV")
            cost = checked_decimal_string(row.get("additional_cost"),
                                          f"根动作 {index + 1} 的追加金额")
            actions.append({"kind": kind, "target": str(target),
                            "target_decimal": str(target),
                            "ev_exact": str(fraction), "ev": shown,
                            "additional_cost": str(cost)})
        if not actions:
            raise AnalysisRecordError("记录里没有可展示的根动作")

        posterior = []
        for index, value in enumerate(result.get("root_posterior") or []):
            fraction, shown = checked_amount(value, f"第 {index + 1} 个先验权重")
            posterior.append({"exact": str(fraction), "decimal": shown})

        support = document.get("support") or {}
        if not isinstance(support, dict) or "root_actions" not in support:
            raise AnalysisRecordError("记录缺少本次核对的服务端支持性检查结果")
        to_call = checked_decimal_string(support.get("to_call"), "核对时的应付额")
        call_actions = [row for row in actions if row["kind"] == "call"]
        if call_actions and Decimal(call_actions[0]["additional_cost"]) != to_call:
            raise AnalysisRecordError(
                "跟注的追加金额与核对时的应付额不一致（记录已被改动）")
        hero_cards = sealed.get("hero_cards")
        board_cards = sealed.get("board_cards")
        if (not isinstance(hero_cards, list) or len(hero_cards) != 2
                or not isinstance(board_cards, list) or len(board_cards) != 5):
            raise AnalysisRecordError("规范输入里的牌面不完整")
        return {
            "record_id": document.get("record_id"),
            "label": document.get("label"),
            "saved_at": document.get("saved_at"),
            "record_kind": document.get("record_kind"),
            "implementation_version": document.get("implementation_version"),
            "scope": document.get("scope"),
            "historical": (document.get("identity", {}).get("rules_revision")
                           != self.rules_revision()),
            "rubric": "synthetic_input_labelled" if not
                      (document.get("source") or {}).get("issue_id")
                      else "linked_to_review_record",
            "situation": {
                "hero_seat": sealed.get("hero_seat"),
                "hero_cards": list(hero_cards), "board_cards": list(board_cards),
                "action_order": list(sealed.get("action_order") or []),
                "seats": [{"seat_id": row["seat_id"], "status": row["status"],
                           "stack": row["stack"],
                           "hand_committed": row["hand_committed"]}
                          for row in sealed.get("seats") or []],
            },
            "history": [{"actor": row["actor"], "kind": row["kind"],
                         "label": f"座位 {row['actor']} "
                                  + ACTION_LABELS.get(row["kind"], row["kind"])
                                  + f" {row['target']}",
                         "target": row["target"]}
                        for row in sealed.get("history") or []],
            "facts": deepcopy(facts),
            "assumptions": deepcopy(assumptions),
            "rules": {
                "rules_source": (document.get("job") or {}).get("binding", {}).get(
                    "rules_source"),
                "rules_revision": document.get("identity", {}).get("rules_revision"),
                "effective_rules": deepcopy(
                    (document.get("job") or {}).get("binding", {}).get(
                        "effective_rules")),
                "current_rules_revision": self.rules_revision(),
            },
            "actions": actions,
            "capacity": deepcopy(document.get("capacity")),
            "amounts": deepcopy(document.get("amounts")),
            "blockers": list(support.get("reasons") or []),
            "posterior": posterior,
            "root_pot": support.get("implied_pot"),
            "to_call": support.get("to_call"),
            "unit": support.get("unit") or UNIT,
            "strategy_eligible": False,
            "advice_emitted": False,
            "note": "历史结果：仅代表保存时的输入、假设与规则版本；不是整手收益、"
                    "GTO 或实战胜率，也不代表当前桌规下的重算。",
        }

    def _present(self, document):
        notes, view = self._verify(document)
        permitted = view is not None
        return {
            "record_id": document.get("record_id"),
            "label": document.get("label"),
            "saved_at": document.get("saved_at"),
            "record_kind": document.get("record_kind"),
            "implementation_version": document.get("implementation_version"),
            "identity": deepcopy(document.get("identity")),
            "status": (view or {}).get("status", INVALID),
            "display_permitted": permitted,
            "notes": notes,
            "view": view if permitted else None,
            "document": document,
        }
