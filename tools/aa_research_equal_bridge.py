"""Explicit test-only exact-memory/equal-average bridge; never a V2 provider.

Input byte hashes and the exported artifact hash must be pinned independently by
the caller. They bind evidence, not its scientific validity. No training, capture,
default registration or live qualification occurs here. A current_context getter
is a caller-owned synthetic state source, not evidence about a real table.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time

from poker_engine.desktop.aa_policy_worker import AAIsolatedPolicyWorker
from poker_engine.desktop.aa_turn_runtime import AATurnWindow, TurnIdentity
from poker_engine.strategy.aa_frozen_policy import (
    action_ids, canonical_hash, validate_distribution,
)
from poker_engine.strategy.aa_mccfr import ExternalSamplingMCCFR, NUMERICAL_SEMANTICS
from poker_engine.strategy.aa_policy_encoding_v2 import encode_decision_v2
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2
from tools.aa_research_continuation import (
    CONTINUATION_STATUS, EXTRA_SOURCE_NAMES, make_chain, validate_chain,
)


KIND = 'POKERSENSE_EQUAL_MEMORY_RESEARCH_BRIDGE_V1'
AVERAGE = 'equal_time_premerge_linear_delta_over_iteration_v1'
ENCODING = 'test-river-v2-exact-plus-own-trace-v1'
TRAINING = 'external_sampling_simple_linear_v1'
BASE = '8f158e252e2f092b8c3706716b700c8a38a56af9'
SOURCE_NAMES = {'checkpoint', 'equal', 'catalog', 'fixture', 'quality', 'result'}
MAX_BYTES = 4 * 1024 * 1024
HEADER = dict(schema_version=1, kind=KIND, training_algorithm=TRAINING,
              average_statistic=AVERAGE, information_encoding=ENCODING,
              simulation_only=True, strategy_eligible=False, advice_emitted=False,
              qualification='NOT_PRODUCT_QUALIFIED', source_head=BASE)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def _parse(raw):
    def invalid(_):
        raise ValueError('nonfinite_json')
    try:
        return json.loads(raw, object_pairs_hook=_object, parse_constant=invalid)
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('invalid_json') from exc


def _read_raw_pinned(path, digest):
    if not _hex(digest):
        raise ValueError('external_sha256_required')
    with Path(path).open('rb') as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or _digest(raw) != digest:
        raise ValueError('source_size_or_digest_mismatch')
    return raw


def _read_pinned(path, digest):
    raw = _read_raw_pinned(path, digest)
    value = _parse(raw)
    if not isinstance(value, dict):
        raise ValueError('object_required')
    return value


def _research_key(observation, binding):
    memory = observation.get('own_memory')
    if not isinstance(memory, list) or not memory:
        raise ValueError('complete_memory_required')
    for item in memory:
        if (not isinstance(item, dict) or set(item) != {
                'public_index', 'visible_observation_sha256', 'action_id'} or
                type(item['public_index']) is not int or item['public_index'] < 0 or
                not _hex(item['visible_observation_sha256']) or
                not isinstance(item['action_id'], str)):
            raise ValueError('invalid_memory_entry')
    exact = encode_decision_v2(observation)['exact_key']
    return canonical_hash(dict(scope_id=binding['scope_id'], encoder_id=ENCODING,
                               exact_key=exact, own_memory=memory))


def _validate(document):
    required = set(HEADER) | {
        'sources', 'source_file_sha256', 'source_canonical_sha256', 'policy'}
    if isinstance(document, dict) and 'continuation' in document:
        required.add('continuation')
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError('unknown_artifact_schema')
    for key, expected in HEADER.items():
        if type(document[key]) is not type(expected) or document[key] != expected:
            raise ValueError('incorrect_research_semantics:' + key)
    sources = document['sources']
    pins = document['source_file_sha256']
    digests = document['source_canonical_sha256']
    if any(not isinstance(x, dict) or set(x) != SOURCE_NAMES
           for x in (sources, pins, digests)):
        raise ValueError('incomplete_source_evidence')
    if any(not _hex(pins[k]) or canonical_hash(sources[k]) != digests[k]
           for k in SOURCE_NAMES):
        raise ValueError('source_evidence_digest_mismatch')
    cp, equal, catalog, fixture, quality, result = (
        sources[k] for k in (
            'checkpoint', 'equal', 'catalog', 'fixture', 'quality', 'result'))
    cp_body = {k: v for k, v in cp.items() if k != 'sha256'}
    if (cp.get('sha256') != canonical_hash(cp_body)
            or cp.get('kind') != 'AA_MCCFR_CHECKPOINT_V1'
            or type(cp.get('schema_version')) is not int or cp['schema_version'] != 1
            or cp.get('algorithm') != TRAINING or cp.get('status') != 'research_only'
            or cp.get('numerical_semantics') != NUMERICAL_SEMANTICS
            or cp.get('encoder_id') != 'research_exact_river_own_visible_memory_v1'
            or cp.get('update_regrets') is not True or cp.get('players') != [1, 2]
            or type(cp.get('iterations')) is not int or cp['iterations'] <= 0
            or type(cp.get('seed')) is not int):
        raise ValueError('invalid_native_checkpoint')
    required = {'regrets', 'average', 'rng_state', 'total_nodes', 'committed_visits'}
    if not required <= set(cp):
        raise ValueError('incomplete_native_checkpoint')
    binding = cp['binding']
    if (binding.get('fixture_sha256') != canonical_hash(fixture)
            or binding.get('encoder_id') != ENCODING
            or binding.get('scope_id') != fixture.get('scope_id')
            or binding.get('strategy_eligible') is not False
            or fixture.get('simulation_only') is not True
            or fixture.get('production_eligible') is not False):
        raise ValueError('fixture_binding_mismatch')
    restored = ExternalSamplingMCCFR.restore(
        cp, expected_binding=binding,
        expected_encoder='research_exact_river_own_visible_memory_v1',
        encoder=lambda value: value, menu=lambda value: value,
        update_regrets=True)
    if canonical_hash(restored.checkpoint()) != canonical_hash(cp):
        raise ValueError('native_checkpoint_roundtrip_mismatch')
    if (equal.get('native_checkpoint_sha256') != cp['sha256']
            or type(equal.get('seed')) is not int or equal['seed'] != cp['seed']
            or type(equal.get('iterations')) is not int
            or equal['iterations'] != cp['iterations']
            or type(equal.get('tap_commits')) is not int
            or equal['tap_commits'] != cp['iterations']):
        raise ValueError('unbound_equal_history')
    if (not isinstance(catalog, dict) or not catalog or
            not isinstance(equal.get('average'), dict) or
            set(equal['average']) != set(cp['average']) or
            not set(equal['average']) <= set(catalog)):
        raise ValueError('average_coverage_mismatch')
    for key, record in catalog.items():
        obs = record['visible_observation']
        rule_fingerprint = AARuleProfileV2.from_dict(obs['rules']).fingerprint
        own_hands = fixture['active_ranges'].get(str(obs.get('actor')), {}).values()
        if (not _hex(key) or _research_key(obs, binding) != key
                or canonical_hash(record['key_preimage']) != key
                or obs.get('board') != fixture['board']
                or obs.get('rules_fingerprint') != binding['rules_fingerprint']
                or rule_fingerprint != binding['rules_fingerprint']
                or list(action_ids(obs)) != record['menu']
                or sorted(obs.get('own_hole', [])) not in
                [sorted(h) for h in own_hands]):
            raise ValueError('catalog_encoding_or_scope_mismatch')
    policy = {}
    for key, values in equal['average'].items():
        if (not isinstance(values, dict) or set(values) != set(catalog[key]['menu'])
                or set(values) != set(cp['average'][key])
                or set(values) != set(cp['regrets'][key])
                or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                       for v in values.values())):
            raise ValueError('invalid_equal_weights')
        total = sum(values[a] for a in sorted(values))
        if not math.isfinite(total):
            raise ValueError('invalid_equal_mass')
        if total:
            row = {a: v / total for a, v in values.items()}
            validate_distribution(row, catalog[key]['menu'])
            policy[key] = row
    if (not isinstance(document['policy'], dict) or
            set(document['policy']) != set(policy)):
        raise ValueError('policy_coverage_mismatch')
    for key, row in document['policy'].items():
        validate_distribution(row, catalog[key]['menu'])
    if canonical_hash(document['policy']) != canonical_hash(policy):
        raise ValueError('policy_not_derived_from_equal_history')
    allowed_statuses = {
        'COMPLETE_PAIRED_DIAGNOSTIC_NO_PROMOTION',
        'COMPLETE_PAIRED_EXTENSION_NO_PROMOTION',
        'COMPLETE_SECOND_FIXED_GAME_NO_PROMOTION', 'STOP_ERROR_OR_BUDGET',
    }
    if result.get('status') == CONTINUATION_STATUS:
        validate_chain(document)
        allowed_statuses = {CONTINUATION_STATUS}
    elif 'continuation' in document:
        raise ValueError('continuation_requires_exact_result_status')
    if (quality.get('seed') != cp['seed'] or
            quality.get('iterations') != cp['iterations']
            or quality.get('variant') != 'equal'
            or quality.get('native_checkpoint_sha256') != cp['sha256']
            or quality.get('mean_rows') != len(policy)
            or quality not in result.get('points', [])
            or not _hex(result.get('manifest_sha256'))
            or result.get('production_eligible') is not False
            or result.get('live_eligible') is not False
            or result.get('status') not in allowed_statuses):
        raise ValueError('historical_evidence_mismatch')
    cases = [c for c in result.get('cases', []) if c.get('seed') == cp['seed']]
    if len(cases) != 1 or cases[0].get('committed_iterations', -1) < cp['iterations']:
        raise ValueError('source_checkpoint_not_committed')
    return binding, policy


def build_artifact(paths, *, expected_source_sha256):
    """Package pinned existing evidence only; never fit or fill a strategy."""
    if set(paths) != SOURCE_NAMES or set(expected_source_sha256) != SOURCE_NAMES:
        raise ValueError('six_pinned_sources_required')
    sources = {k: _read_pinned(paths[k], expected_source_sha256[k])
               for k in SOURCE_NAMES}
    policy = {}
    for key, row in sources['equal']['average'].items():
        total = sum(row[a] for a in sorted(row))
        if total:
            policy[key] = {a: v / total for a, v in row.items()}
    source_digests = {k: canonical_hash(v) for k, v in sources.items()}
    doc = dict(HEADER, sources=sources,
               source_file_sha256=dict(expected_source_sha256),
               source_canonical_sha256=source_digests, policy=policy)
    try:
        _validate(doc)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('malformed_source_evidence') from exc
    return doc


def build_continuation_artifact(paths, *, expected_source_sha256,
                                continuation_paths,
                                expected_continuation_sha256):
    """Validate pinned raw parent/continuation evidence before portable export.

    Raw manifests can contain host paths. Their verified hashes and explicit
    semantic projections remain distinct in the artifact. The caller must
    independently pin the resulting artifact, as for the original bridge.
    """
    if (set(paths) != SOURCE_NAMES or set(expected_source_sha256) != SOURCE_NAMES
            or set(continuation_paths) != EXTRA_SOURCE_NAMES
            or set(expected_continuation_sha256) != EXTRA_SOURCE_NAMES):
        raise ValueError('complete_pinned_continuation_sources_required')
    sources = {key: _read_pinned(paths[key], expected_source_sha256[key])
               for key in SOURCE_NAMES}
    extras = {key: _read_pinned(continuation_paths[key],
                                expected_continuation_sha256[key])
              for key in EXTRA_SOURCE_NAMES}
    policy = {}
    for key, row in sources['equal']['average'].items():
        total = sum(row[action] for action in sorted(row))
        if total:
            policy[key] = {action: value / total for action, value in row.items()}
    doc = dict(HEADER, sources=sources,
               source_file_sha256=dict(expected_source_sha256),
               source_canonical_sha256={key: canonical_hash(value)
                                        for key, value in sources.items()},
               policy=policy)
    try:
        doc['continuation'] = make_chain(
            extras, expected_continuation_sha256,
            sources, expected_source_sha256)
        _validate(doc)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('malformed_continuation_evidence') from exc
    return doc


class ResearchArtifact:
    def __init__(self, raw, expected_sha256):
        if (type(raw) is not bytes or len(raw) > MAX_BYTES
                or not _hex(expected_sha256) or _digest(raw) != expected_sha256):
            raise ValueError('pinned_raw_artifact_required')
        try:
            _validate(_parse(raw))
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError('malformed_artifact') from exc
        self._raw = raw
        self._sha256 = expected_sha256

    @property
    def sha256(self):
        return self._sha256

    @property
    def document(self):
        return _parse(self._raw)


def load_artifact(path, *, expected_sha256):
    return ResearchArtifact(_read_raw_pinned(path, expected_sha256), expected_sha256)


class ResearchEqualBridge:
    """Offline opt-in map adapter; no product registration or authority grant."""
    def __init__(self, artifact, *, enabled=False):
        if not isinstance(artifact, ResearchArtifact) or type(enabled) is not bool:
            raise ValueError('loaded_artifact_and_boolean_opt_in_required')
        if _digest(artifact._raw) != artifact.sha256:
            raise ValueError('artifact_identity_changed')
        doc = artifact.document
        self._binding, policy = _validate(doc)
        self._artifact_sha256 = artifact.sha256
        self._contexts = {k: deepcopy(v['visible_observation'])
                          for k, v in doc['sources']['catalog'].items()}
        self._batch_status = doc['sources']['result']['status']
        self._continued = 'continuation' in doc
        self._policy = deepcopy(policy)
        self._worker = AAIsolatedPolicyWorker(
            policy, seed=doc['sources']['checkpoint']['seed'])
        self.enabled = enabled

    def preload(self):
        if not self.enabled:
            raise ValueError('DISABLED')
        self._worker.preload()

    def close(self):
        self._worker.close()

    @staticmethod
    def _abstain(reason):
        return dict(status='ABSTAIN', reason=reason, action=None,
                    strategy_eligible=False, advice_emitted=False)

    def lookup(self, observation, *, artifact_sha256, simulation=False, identity,
               window, source_at, current_context, clock=time.monotonic):
        if not self.enabled:
            return self._abstain('DISABLED')
        if simulation is not True:
            return self._abstain('SIMULATION_ONLY')
        if artifact_sha256 != self._artifact_sha256:
            return self._abstain('ARTIFACT_IDENTITY_MISMATCH')
        if (not isinstance(identity, TurnIdentity)
                or not isinstance(window, AATurnWindow)
                or not callable(current_context)):
            return self._abstain('INVALID_RUNTIME_CONTEXT')
        try:
            obs = deepcopy(observation)
            key = _research_key(obs, self._binding)
            if (key not in self._contexts or
                    canonical_hash(obs) != canonical_hash(self._contexts[key])):
                return self._abstain('OUTSIDE_RESEARCH_SCOPE')
            if key not in self._policy:
                return self._abstain('POLICY_COVERAGE_MISS')
            state = canonical_hash(dict(artifact=artifact_sha256, observation=obs))
            legal = action_ids(obs)
        except (ValueError, KeyError, TypeError, AttributeError):
            return self._abstain('OUTSIDE_RESEARCH_SCOPE')

        def is_current(binding):
            try:
                current = current_context()
                if not isinstance(current, dict) or set(current) != {
                        'observation', 'identity', 'artifact_sha256', 'source_at'}:
                    return False
                return (current['identity'] == identity
                        and current['source_at'] == source_at
                        and current['artifact_sha256'] == artifact_sha256
                        and canonical_hash(current['observation']) ==
                        canonical_hash(obs)
                        and _research_key(current['observation'], self._binding) == key
                        and binding['state'] == state
                        and binding['policy'] == self._worker.policy_sha256
                        and binding['rules'] == self._binding['rules_fingerprint'])
            except Exception:
                return False

        try:
            result = self._worker.lookup(
                key, identity=identity, state_key=state,
                rules_fingerprint=self._binding['rules_fingerprint'],
                legal_actions=legal, window=window, source_at=source_at,
                is_current=is_current, clock=clock)
        except (ValueError, TypeError, AttributeError):
            return self._abstain('INVALID_RUNTIME_CONTEXT')
        result.update(research_artifact_sha256=artifact_sha256,
                      average_statistic=AVERAGE,
                      source_batch_status=self._batch_status,
                      qualification='NOT_PRODUCT_QUALIFIED')
        if self._continued:
            result['source_parent_batch_status'] = 'STOP_ERROR_OR_BUDGET'
        return result
