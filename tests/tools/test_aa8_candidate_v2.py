from types import SimpleNamespace

from tools.aa8_candidate_v2 import CandidateStateV2
from tools.aa8_holdout_predict import FrozenPredictionState


def test_integrated_ledger_consumes_current_adapter_and_hero_actor(monkeypatch):
    row = {"frame": 100, "scene_supported": True, "current_actor": 4,
           "actor_evidence": {"hero_turn": True}, "pot": {"value": "20"},
           "street_wagers": {"4": "2"}}
    monkeypatch.setattr(FrozenPredictionState, "read", lambda *args: row)
    state = CandidateStateV2.__new__(CandidateStateV2)
    visibility = {"4": {"status": "VISIBLE_COIN_CANDIDATE"}}
    state.context_reader = SimpleNamespace(last_visibility=visibility)
    state.center = SimpleNamespace(recognize=lambda image: {"value": "18"})
    calls = []

    def observe(current):
        calls.append("adapter")
        state.adapter.actions.append({"frame": current["frame"]})
        return {"frame": current["frame"]}

    state.adapter = SimpleNamespace(actions=[], observe=observe)

    def ledger(current, adapter, partition):
        assert calls == ["adapter"]
        assert adapter.actions == [{"frame": 100}]
        assert current["current_actor"] == 4
        assert current["actor_evidence"]["hero_turn"]
        assert partition == {"title": "20", "center": {"value": "18"},
                             "wagers": {"4": "2"}, "visibility": visibility}
        return {"status": "WAGERS_UNKNOWN", "wagers": None}

    state.causal_wagers = SimpleNamespace(observe=ledger)
    state.dealer_reader = SimpleNamespace(read=lambda image: {"dealer_seat": 7})
    state.dealer_evidence = SimpleNamespace(observe=lambda *args, **kwargs: {
        "dealer_seat": 7, "canonical_verified": False})
    state.hand_ledger = SimpleNamespace(observe=lambda *args: {
        "hand_commitments": None, "complete_and_canonical_verified": False})
    result = state.read(None, 100, {})
    assert result["observed_actions_v2"] == [{"frame": 100}]
    assert result["causal_street_wagers_v2"]["wagers"] is None


def test_integrated_unsupported_frame_does_not_reuse_visibility(monkeypatch):
    row = {"frame": 101, "scene_supported": False, "pot": {"value": None},
           "street_wagers": None}
    monkeypatch.setattr(FrozenPredictionState, "read", lambda *args: row)
    state = CandidateStateV2.__new__(CandidateStateV2)
    state.context_reader = SimpleNamespace(last_visibility={"4": "stale"})
    state.center = SimpleNamespace(recognize=lambda image: {"value": None})
    state.adapter = SimpleNamespace(actions=[], observe=lambda row: {})

    def ledger(current, adapter, partition):
        assert partition["visibility"] == {}
        assert partition["wagers"] == {}
        return {"status": "WAGERS_UNKNOWN"}

    state.causal_wagers = SimpleNamespace(observe=ledger)
    state.dealer_reader = SimpleNamespace(read=lambda image: {"dealer_seat": None})
    state.dealer_evidence = SimpleNamespace(observe=lambda *args, **kwargs: {
        "dealer_seat": None, "canonical_verified": False})
    state.hand_ledger = SimpleNamespace(observe=lambda *args: {
        "hand_commitments": None, "complete_and_canonical_verified": False})
    assert state.read(None, 101, {})["wager_visibility_v2"] == {}
