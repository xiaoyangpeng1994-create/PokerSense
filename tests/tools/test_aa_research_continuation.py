"""Portable continuation evidence checks; no training or worker execution.

Repinned mutations are synthetic corruption witnesses, not authenticated new
research. Manifest projections cannot reconstruct the original raw documents;
the raw builder and independently pinned artifact remain the trust boundary.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest

from tools.aa_research_equal_bridge import (
    ResearchArtifact, build_continuation_artifact, load_artifact,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "equal_memory_continuation"
ENTRIES = json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))[
    "artifacts"]
MAIN_NAMES = ("checkpoint", "equal", "catalog", "fixture", "quality", "result")
EXTRA_NAMES = ("parent_manifest", "manifest", "parent_result",
               "parent_checkpoint", "parent_equal", "restore")
CONTINUATION = "COMPLETE_CONTINUATION_NO_PROMOTION"
ZERO_HASH = "0" * 64
NEGATIVE_CASES = ("missing_chain",) + tuple(
    "missing_" + name for name in EXTRA_NAMES) + (
    "chain_schema_bool", "main_canonical_digest", "extra_canonical_digest",
    "frozen_role_pin", "main_pin_copy", "malformed_original_document_pin",
    "wrong_parent_manifest_link", "native_seed", "native_rng_missing",
    "native_iterations_bool", "average_semantics", "equal_checkpoint_binding",
    "equal_tap_count", "equal_row_missing", "parent_native_rng_missing",
    "parent_equal_tap_count", "parent_stop_laundered", "parent_not_run_promoted",
    "original_status_laundered", "case_incomplete", "case_missing",
    "case_duplicate", "case_tap_count", "case_uncommitted", "equal_gate_false",
    "final_point_missing", "final_point_duplicate", "quality_bool",
    "quality_above_gate", "quality_rows_missing", "quality_wrong_variant",
    "quality_exact_bool", "restore_field_false", "restore_counter_changed",
    "manifest_average", "manifest_seed", "board_changed", "range_changed",
    "rules_changed", "encoding_changed", "restore_checkpoint_hash",
    "live_eligible", "overall_label_laundered",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def refresh_canonical_digests(document):
    """Refresh content hashes, never claim to recreate the original raw pins."""
    sources = document["sources"]
    checkpoint = sources["checkpoint"]
    checkpoint["sha256"] = digest(canonical({
        key: value for key, value in checkpoint.items() if key != "sha256"}))
    chain = document.get("continuation")
    if chain is not None:
        extras = chain["sources"]
        parent = extras.get("parent_checkpoint")
        if parent is not None:
            parent["sha256"] = digest(canonical({
                key: value for key, value in parent.items() if key != "sha256"}))
        chain["source_canonical_sha256"] = {
            key: digest(canonical(value)) for key, value in extras.items()}
    document["source_canonical_sha256"] = {
        key: digest(canonical(value)) for key, value in sources.items()}


def selected_quality_records(document):
    sources = document["sources"]
    seed = sources["checkpoint"]["seed"]
    records = [sources["quality"]]
    for name in ("points", "joint_points"):
        records.extend(point for point in sources["result"][name]
                       if (point["seed"], point["iterations"], point["variant"])
                       == (seed, 16384, "equal"))
    assert len(records) == 3
    return records


def mutate(document, case):
    sources = document["sources"]
    chain = document["continuation"]
    extras = chain["sources"]
    result = sources["result"]
    checkpoint = sources["checkpoint"]
    equal = sources["equal"]
    if case == "missing_chain":
        document.pop("continuation")
    elif case.startswith("missing_"):
        extras.pop(case.removeprefix("missing_"))
    elif case == "chain_schema_bool":
        chain["schema_version"] = True
    elif case == "frozen_role_pin":
        chain["frozen_role_file_sha256"]["parent_checkpoint"] = ZERO_HASH
    elif case == "main_pin_copy":
        chain["main_source_file_sha256"]["checkpoint"] = ZERO_HASH
    elif case == "malformed_original_document_pin":
        chain["original_manifest_document_sha256"]["manifest"] = "not-a-digest"
    elif case == "wrong_parent_manifest_link":
        extras["manifest"]["old_manifest_sha256"] = ZERO_HASH
    elif case == "native_seed":
        checkpoint["seed"] += 100
    elif case == "native_rng_missing":
        checkpoint.pop("rng_state")
    elif case == "native_iterations_bool":
        checkpoint["iterations"] = True
    elif case == "average_semantics":
        document["average_statistic"] = "linear_weighted_average_v1"
    elif case == "equal_checkpoint_binding":
        equal["native_checkpoint_sha256"] = ZERO_HASH
    elif case == "equal_tap_count":
        equal["tap_commits"] -= 1
    elif case == "equal_row_missing":
        equal["average"].pop(next(iter(equal["average"])))
    elif case == "parent_native_rng_missing":
        extras["parent_checkpoint"].pop("rng_state")
    elif case == "parent_equal_tap_count":
        extras["parent_equal"]["tap_commits"] -= 1
    elif case == "parent_stop_laundered":
        extras["parent_result"]["status"] = "COMPLETE_SECOND_FIXED_GAME_NO_PROMOTION"
    elif case == "parent_not_run_promoted":
        unrun = next(point for point in extras["parent_result"]["points"]
                     if point["status"] == "NOT_RUN")
        unrun["status"] = "PASS"
    elif case == "original_status_laundered":
        result["original_segment_status"] = CONTINUATION
    elif case == "case_incomplete":
        result["cases"][0]["status"] = "STOP_ERROR_OR_BUDGET"
    elif case == "case_missing":
        result["cases"].pop()
    elif case == "case_duplicate":
        result["cases"][-1] = deepcopy(result["cases"][0])
    elif case == "case_tap_count":
        result["cases"][0]["tap_commits"] -= 1
    elif case == "case_uncommitted":
        result["cases"][0]["committed_iterations"] -= 1
    elif case == "equal_gate_false":
        result["final_variant_gates"]["equal"] = False
    elif case in ("final_point_missing", "final_point_duplicate"):
        point = next(point for point in result["points"]
                     if point["seed"] == checkpoint["seed"]
                     and point["iterations"] == 16384
                     and point["variant"] == "equal")
        if case == "final_point_missing":
            result["points"].remove(point)
        else:
            other = next(i for i, item in enumerate(result["points"])
                         if item is not point)
            result["points"][other] = deepcopy(point)
    elif case.startswith("quality_"):
        for point in selected_quality_records(document):
            if case == "quality_bool":
                point["nash_conv_bb"] = True
            elif case == "quality_above_gate":
                point["nash_conv_bb"] = 0.051
                point["exact"]["nash_conv"] = {
                    "numerator": 51, "denominator": 500}
                point["status"] = "HOLD_QUALITY"
            elif case == "quality_rows_missing":
                point["mean_rows"] = 15
                point["missing_keys"] = [next(iter(sources["catalog"]))]
            elif case == "quality_wrong_variant":
                point["variant"] = "linear"
            elif case == "quality_exact_bool":
                point["exact"]["nash_conv"]["numerator"] = True
            else:
                raise AssertionError(case)
        if case == "quality_above_gate":
            # Synchronize the stated quality and counts so the equal gate,
            # rather than a stale scalar or summary, is what rejects it.
            result["final_variant_gates"]["equal"] = False
            for field in ("counts", "joint_counts"):
                result[field]["PASS"] -= 1
                result[field]["HOLD_QUALITY"] += 1
    elif case == "restore_field_false":
        extras["restore"]["fields_equal"]["rng_state"] = False
    elif case == "restore_counter_changed":
        extras["restore"]["poker_sweeps"] = 1
    elif case == "manifest_average":
        extras["manifest"]["equal_formula"] = "linear weighted average"
    elif case == "manifest_seed":
        extras["manifest"]["seeds"][0] += 100
    elif case == "board_changed":
        sources["fixture"]["board"][0] = "8h"
    elif case == "range_changed":
        sources["fixture"]["active_ranges"]["1"]["STRAIGHT_A"] = ["4h", "4s"]
    elif case == "rules_changed":
        checkpoint["binding"]["rules_fingerprint"] = ZERO_HASH
    elif case == "encoding_changed":
        checkpoint["binding"]["encoder_id"] = "v2"
    elif case == "restore_checkpoint_hash":
        extras["restore"]["native_checkpoint_sha256"] = ZERO_HASH
    elif case == "live_eligible":
        result["live_eligible"] = True
    elif case == "overall_label_laundered":
        result["second_game_final_success"] = True
    elif case not in ("main_canonical_digest", "extra_canonical_digest"):
        raise AssertionError(case)

    refresh_canonical_digests(document)
    if case == "native_seed":
        # Preserve the immediate native/equal link to reach seed-plan checking.
        equal["seed"] = checkpoint["seed"]
        equal["native_checkpoint_sha256"] = checkpoint["sha256"]
        refresh_canonical_digests(document)
    elif case == "main_canonical_digest":
        document["source_canonical_sha256"]["result"] = ZERO_HASH
    elif case == "extra_canonical_digest":
        chain["source_canonical_sha256"]["manifest"] = ZERO_HASH


@pytest.fixture(scope="module", params=ENTRIES,
                ids=lambda entry: f"equal-{entry['seed']}-16384")
def loaded(request):
    entry = request.param
    assert len(ENTRIES) == 3
    assert {item["seed"] for item in ENTRIES} == {
        2026100301, 2026100302, 2026100303}
    assert entry["iterations"] == 16384
    path = FIXTURES / entry["path"]
    return entry, load_artifact(path, expected_sha256=entry["sha256"])


def test_equal_gate_accepts_without_promoting_original_or_overall(loaded):
    entry, artifact = loaded
    document = artifact.document
    sources = document["sources"]
    extras = document["continuation"]["sources"]
    assert sources["checkpoint"]["seed"] == entry["seed"]
    assert sources["checkpoint"]["iterations"] == 16384
    assert sources["result"]["status"] == CONTINUATION
    assert sources["result"]["final_variant_gates"] == {
        "equal": True, "linear": False}
    assert sources["result"]["second_game_final_success"] is False
    assert extras["parent_result"]["status"] == "STOP_ERROR_OR_BUDGET"
    assert extras["parent_result"]["counts"]["NOT_RUN"] == 14
    assert "Gate each variant separately;" in extras["parent_manifest"]["final_gate"]
    assert document["strategy_eligible"] is False
    assert document["advice_emitted"] is False
    assert document["qualification"] == "NOT_PRODUCT_QUALIFIED"
    assert sources["result"]["four_street_eligible"] is False
    raw = canonical(document)
    assert ResearchArtifact(raw, digest(raw)).document == document


def test_portable_fixture_contains_no_personal_paths(loaded):
    entry, artifact = loaded
    serialized = canonical(artifact.document).decode("utf-8")
    assert re.search(r"[A-Za-z]:[\\/]|\\\\|/(?:Users|home|tmp)/", serialized) is None
    assert Path(entry["path"]).name == entry["path"]
    assert "files" not in artifact.document["continuation"]["sources"]["manifest"]
    assert "original_files" not in (
        artifact.document["continuation"]["sources"]["manifest"])


@pytest.mark.parametrize("case", NEGATIVE_CASES)
def test_repinning_corrupted_continuation_evidence_fails_closed(loaded, case):
    _, artifact = loaded
    document = artifact.document
    mutate(document, case)
    raw = canonical(document)
    with pytest.raises(ValueError):
        ResearchArtifact(raw, digest(raw))


def test_legacy_8192_status_relabel_does_not_create_continuation_evidence():
    directory = FIXTURES.parent / "equal_memory_bridge"
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    entry = next(item for item in index["artifacts"]
                 if item["path"] == "second-2026100301-8192.json")
    artifact = load_artifact(directory / entry["path"],
                             expected_sha256=entry["sha256"])
    document = artifact.document
    assert document["sources"]["checkpoint"]["iterations"] == 8192
    document["sources"]["result"]["status"] = CONTINUATION
    refresh_canonical_digests(document)
    raw = canonical(document)
    with pytest.raises(ValueError, match="continuation_"):
        ResearchArtifact(raw, digest(raw))


@pytest.mark.parametrize("bad_pin_group", ["main", "continuation"])
def test_raw_builder_rejects_wrong_pin_before_evidence_parsing(
        tmp_path, bad_pin_group):
    # This is only an early raw-pin refusal test, not a valid raw-chain fixture.
    # The real original-input builder path is exercised by fixture generation.
    path = tmp_path / "empty-object.json"
    raw = b"{}"
    path.write_bytes(raw)
    correct_pin = digest(raw)
    main_pins = {name: correct_pin for name in MAIN_NAMES}
    extra_pins = {name: correct_pin for name in EXTRA_NAMES}
    pins = main_pins if bad_pin_group == "main" else extra_pins
    for name in pins:
        pins[name] = ZERO_HASH
    with pytest.raises(ValueError, match="^source_size_or_digest_mismatch$"):
        build_continuation_artifact(
            {name: path for name in MAIN_NAMES},
            expected_source_sha256=main_pins,
            continuation_paths={name: path for name in EXTRA_NAMES},
            expected_continuation_sha256=extra_pins)
