"""Explicit base-only scope receipt, not a release gate or normal-mode detector."""


def scope_receipt(row):
    modes = row.get("special_modes") or {}
    deferred = []
    if modes.get("insurance") == "VISIBLE":
        deferred.append("insurance")
    if modes.get("mushroom_trigger") in (True, "VISIBLE"):
        deferred.append("mushroom")
    if (modes.get("critical_hit_trigger") in (True, "VISIBLE")
            or (modes.get("critical_hit_title") or {}).get(
                "critical_hit_animation") is True):
        deferred.append("bomb")
    status = "BASE_CANDIDATE_UNVERIFIED"
    if deferred:
        status = "DEFERRED_MODE_OBSERVED"
    elif modes.get("block_state_updates") or not row.get("scene_supported"):
        status = "BLOCKED_OR_UNSUPPORTED_SCENE"
    return {"scope_id": "aa8_base_visual_v1", "status": status,
            "observed_deferred_modes": deferred,
            "deferred_semantics": ["insurance", "mushroom", "bomb"],
            "normal_mode_verified": False, "full_visual_acceptance": False,
            "strategy_eligible": False,
            "unknown_cash_policy": "UNALLOCATED",
            "automatically_exempt_from_acceptance": False}
