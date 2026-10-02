"""Strict, portable provenance for the saved second-river continuation.

The builder receives objects whose original bytes the caller has already pinned.
It checks the original manifests before discarding their local path dictionaries.
Portable projections have their own canonical hashes; they are not the original
manifest documents. The independently pinned final artifact remains the trust
anchor. This evidence does not authenticate a live table or qualify a strategy.
"""

from copy import deepcopy
from fractions import Fraction
import math
import re

from poker_engine.strategy.aa_frozen_policy import canonical_hash
from poker_engine.strategy.aa_mccfr import ExternalSamplingMCCFR, NUMERICAL_SEMANTICS
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2


CONTINUATION_STATUS = 'COMPLETE_CONTINUATION_NO_PROMOTION'
EXTRA_SOURCE_NAMES = {
    'parent_manifest', 'manifest', 'parent_result', 'parent_checkpoint',
    'parent_equal', 'restore',
}
_MAIN_NAMES = {'checkpoint', 'equal', 'catalog', 'fixture', 'quality', 'result'}
_KIND = 'POKERSENSE_EQUAL_CONTINUATION_CHAIN_V1'
_TRUST = 'caller_pinned_artifact_after_original_byte_and_manifest_validation'
_BASE = '8f158e252e2f092b8c3706716b700c8a38a56af9'
_STOP = 'STOP_ERROR_OR_BUDGET'
_ENCODING = 'test-river-v2-exact-plus-own-trace-v1'
_NATIVE_ENCODING = 'research_exact_river_own_visible_memory_v1'
_TRAINING = 'external_sampling_simple_linear_v1'
_AVERAGE = 'equal_time_premerge_linear_delta_over_iteration_v1'
_VARIANTS = ['linear', 'equal']
_POINTS = [4096, 8192, 16384]
_SEEDS = [2026100301, 2026100302, 2026100303]
_FIXTURE = '691945d6cc90c596021c165e2d18515beba155a6eff0cbe38136da5ee6fe5067'
_RULES = '47d4265a9afe6f1a1ba6971fd97d8897ac3cbb7cfd6c1dfa82f21653b7fb2934'
_STATUSES = {
    'PASS', 'HOLD_QUALITY', 'NOT_RUN', 'NOT_SCORED_MISSING_AVERAGE',
    'STARTED_NOT_COMPLETED', 'NOT_SCORED_ERROR_OR_BUDGET',
}
_CP_FIELDS = {
    'schema_version', 'kind', 'players', 'seed', 'iterations', 'total_nodes',
    'regrets', 'average', 'rng_state', 'status', 'algorithm',
    'numerical_semantics', 'binding', 'encoder_id', 'update_regrets',
    'committed_visits', 'sha256',
}
_COMMON_FIELDS = {
    'kind', 'sha256', 'source_head', 'fixture', 'binding', 'seeds', 'variants',
    'quality_limit_bb', 'missing_mean', 'linear_formula', 'equal_formula',
    'final_gate', 'production_eligible', 'live_eligible', 'per_seed_seconds',
    'wall_seconds',
}
_PARENT_FIELDS = _COMMON_FIELDS | {
    'source_sha256', 'checkpoint_iterations',
}
_MANIFEST_FIELDS = _COMMON_FIELDS | {
    'old_manifest_sha256', 'old_source_sha256', 'files_sha256',
    'original_directory_sha256', 'points', 'seed_start_iterations', 'targets',
    'original_status_immutable', 'original_counts', 'original_per_seed_seconds',
    'four_street_eligible', 'attempt_limit', 'retries_allowed',
    'budgets_transferable', 'expected_new_sweeps', 'expected_new_quality_points',
    'expected_joint_points', 'expected_new_mean_rows', 'expected_joint_mean_rows',
}
_ROLE_NAMES = {
    'parent_manifest', 'parent_result', 'parent_checkpoint', 'parent_equal',
    'catalog', 'restore',
}


def _require(condition, reason):
    if not condition:
        raise ValueError('continuation_' + reason)


def _hex(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _same(left, right):
    return canonical_hash(left) == canonical_hash(right)


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _pins(value, names):
    _require(isinstance(value, dict) and set(value) == names
             and all(_hex(v) for v in value.values()), 'invalid_source_pins')


def _portable(value):
    """Fail closed if an unexpected source field contains a local path."""
    if isinstance(value, dict):
        for key, item in value.items():
            _portable(key)
            _portable(item)
    elif isinstance(value, list):
        for item in value:
            _portable(item)
    elif isinstance(value, str):
        _require(not re.search(r'[A-Za-z]:[\\/]|\\\\|/(?:Users|home|tmp)/', value),
                 'nonportable_source_value')


def _project(manifest, fields):
    _require(isinstance(manifest, dict) and fields <= set(manifest),
             'incomplete_manifest')
    projected = {key: deepcopy(manifest[key]) for key in sorted(fields)}
    _portable(projected)
    return projected


def _raw_manifest(manifest, *, parent):
    _require(isinstance(manifest, dict) and _hex(manifest.get('sha256')),
             'invalid_original_manifest')
    body = {k: v for k, v in manifest.items() if k != 'sha256'}
    _require(canonical_hash(body) == manifest['sha256'], 'original_manifest_hash')
    files = manifest.get('files')
    _require(isinstance(files, dict) and files
             and all(isinstance(k, str) and k and _hex(v)
                     for k, v in files.items()), 'invalid_original_file_index')
    index_field = 'source_sha256' if parent else 'files_sha256'
    _require(canonical_hash(files) == manifest.get(index_field),
             'original_file_index_hash')
    if not parent:
        originals = manifest.get('original_files')
        _require(isinstance(originals, dict) and originals
                 and all(isinstance(k, str) and k and _hex(v)
                         for k, v in originals.items())
                 and canonical_hash(originals) ==
                 manifest.get('original_directory_sha256'),
                 'parent_directory_index_hash')
        _require(all(files.get(k) == v for k, v in originals.items()),
                 'parent_directory_not_frozen')


def _role_pin(files, basename):
    matches = [pin for name, pin in files.items()
               if name.replace('\\', '/').rsplit('/', 1)[-1] == basename]
    _require(len(matches) == 1, 'ambiguous_or_missing_frozen_role')
    return matches[0]


def make_chain(extras, extra_file_pins, main_sources, main_file_pins):
    """Validate caller-pinned original inputs and return a path-free proof."""
    try:
        _require(isinstance(extras, dict) and set(extras) == EXTRA_SOURCE_NAMES,
                 'six_extra_sources_required')
        _require(isinstance(main_sources, dict) and set(main_sources) == _MAIN_NAMES,
                 'six_main_sources_required')
        _pins(extra_file_pins, EXTRA_SOURCE_NAMES)
        _pins(main_file_pins, _MAIN_NAMES)
        old, new = extras['parent_manifest'], extras['manifest']
        _raw_manifest(old, parent=True)
        _raw_manifest(new, parent=False)
        _require(all(new['files'].get(k) == v for k, v in old['files'].items()),
                 'parent_frozen_inputs_changed')
        first_seed = old['seeds'][0]
        role_basenames = {
            'parent_manifest': 'frozen-manifest.json',
            'parent_result': 'RESULT.json',
            'parent_checkpoint': f'{first_seed}-last-committed-checkpoint.json',
            'parent_equal': f'{first_seed}-last-equal-sums.json',
            'catalog': 'key-preimages.json',
        }
        roles = {role: _role_pin(new['original_files'], name)
                 for role, name in role_basenames.items()}
        roles['restore'] = _role_pin(new['files'], 'ZERO-STEP-RESTORE.json')
        for role, pin in roles.items():
            expected = (main_file_pins['catalog'] if role == 'catalog'
                        else extra_file_pins[role])
            _require(pin == expected, 'frozen_role_file_hash')
        sources = deepcopy(extras)
        sources['parent_manifest'] = _project(old, _PARENT_FIELDS)
        sources['manifest'] = _project(new, _MANIFEST_FIELDS)
        chain = dict(
            schema_version=1, kind=_KIND, trust_boundary=_TRUST, sources=sources,
            source_file_sha256=deepcopy(extra_file_pins),
            source_canonical_sha256={k: canonical_hash(v)
                                     for k, v in sources.items()},
            original_manifest_document_sha256={
                'parent_manifest': canonical_hash(old),
                'manifest': canonical_hash(new),
            },
            frozen_role_file_sha256=roles,
            main_source_file_sha256=deepcopy(main_file_pins))
        _portable(chain)
        document = dict(sources=main_sources, source_file_sha256=main_file_pins,
                        average_statistic=_AVERAGE, continuation=chain)
        validate_chain(document)
        return chain
    except (KeyError, TypeError, AttributeError, OverflowError, IndexError) as exc:
        raise ValueError('continuation_malformed_original_evidence') from exc


def _checkpoint(cp, binding):
    _require(isinstance(cp, dict) and set(cp) == _CP_FIELDS,
             'incomplete_native_state')
    _require(cp['kind'] == 'AA_MCCFR_CHECKPOINT_V1'
             and type(cp['schema_version']) is int and cp['schema_version'] == 1
             and cp['algorithm'] == _TRAINING and cp['status'] == 'research_only'
             and cp['encoder_id'] == _NATIVE_ENCODING
             and cp['numerical_semantics'] == NUMERICAL_SEMANTICS
             and cp['update_regrets'] is True and _same(cp['players'], [1, 2])
             and _integer(cp['seed']) and _integer(cp['iterations'], minimum=1)
             and _same(cp['binding'], binding), 'native_state_semantics')
    _require(cp['sha256'] == canonical_hash(
        {k: v for k, v in cp.items() if k != 'sha256'}), 'native_state_hash')
    restored = ExternalSamplingMCCFR.restore(
        cp, expected_binding=binding, expected_encoder=_NATIVE_ENCODING,
        encoder=lambda value: value, menu=lambda value: value,
        update_regrets=True)
    _require(_same(restored.checkpoint(), cp), 'native_state_roundtrip')


def _mean_state(equal, cp, catalog, *, parent):
    required = {'average', 'iterations', 'tap_commits'}
    if not parent:
        required |= {'native_checkpoint_sha256', 'seed'}
    _require(isinstance(equal, dict) and set(equal) == required,
             'incomplete_equal_state')
    for field in ('iterations', 'tap_commits'):
        _require(_integer(equal[field]) and equal[field] == cp['iterations'],
                 'equal_counter_mismatch')
    if not parent:
        _require(_integer(equal['seed']) and equal['seed'] == cp['seed']
                 and equal['native_checkpoint_sha256'] == cp['sha256'],
                 'equal_checkpoint_binding')
    keys = set(catalog)
    _require(len(keys) == 16 and all(set(cp[name]) == keys
                                     for name in ('average', 'regrets',
                                                  'committed_visits'))
             and set(equal['average']) == keys, 'incomplete_mean_rows')
    for key, row in equal['average'].items():
        actions = set(catalog[key]['menu'])
        _require(isinstance(row, dict) and set(row) == actions
                 and set(cp['average'][key]) == actions
                 and set(cp['regrets'][key]) == actions
                 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                         for v in row.values())
                 and math.isfinite(sum(row.values())) and sum(row.values()) > 0,
                 'invalid_mean_weights')


def _point_map(points, expected):
    _require(isinstance(points, list) and len(points) == len(expected),
             'point_denominator')
    indexed = {}
    for point in points:
        _require(isinstance(point, dict)
                 and _integer(point.get('seed'))
                 and _integer(point.get('iterations'), minimum=1)
                 and point.get('variant') in _VARIANTS, 'invalid_point_identity')
        key = (point['seed'], point['iterations'], point['variant'])
        _require(key in expected and key not in indexed, 'point_identity_coverage')
        indexed[key] = point
    _require(set(indexed) == expected, 'point_identity_coverage')
    return indexed


def _quality(point, big_blind):
    number = point.get('nash_conv_bb')
    _require(type(number) in (int, float) and math.isfinite(number) and number >= 0
             and type(point.get('mean_rows')) is int and point['mean_rows'] == 16
             and type(point.get('expected_mean_rows')) is int
             and point['expected_mean_rows'] == 16
             and point.get('missing_keys') == []
             and _hex(point.get('native_checkpoint_sha256')), 'unscored_quality')
    exact = point.get('exact')
    _require(isinstance(exact, dict) and exact.get('simulation_only') is True
             and exact.get('strategy_eligible') is False, 'quality_semantics')
    fraction = exact.get('nash_conv')
    _require(isinstance(fraction, dict)
             and set(fraction) == {'numerator', 'denominator'}
             and _integer(fraction['numerator'])
             and _integer(fraction['denominator'], minimum=1), 'quality_fraction')
    score = Fraction(fraction['numerator'], fraction['denominator']) / big_blind
    _require(math.isclose(float(score), number, rel_tol=1e-12, abs_tol=1e-15),
             'quality_fraction_mismatch')
    expected = 'PASS' if score <= Fraction(1, 20) else 'HOLD_QUALITY'
    _require(point.get('status') == expected, 'quality_status_mismatch')
    return expected == 'PASS'


def _counts(points, stated):
    _require(isinstance(stated, dict) and set(stated) == _STATUSES,
             'count_schema')
    actual = {status: sum(p.get('status') == status for p in points)
              for status in _STATUSES}
    _require(_same(actual, stated), 'count_mismatch')


def _manifests(old, new, fixture, binding):
    _require(set(old) == _PARENT_FIELDS and set(new) == _MANIFEST_FIELDS,
             'manifest_projection_schema')
    _require(old['kind'] == 'ONE_SECOND_FIXED_RIVER_PAIRED_CROSS_VALIDATION'
             and new['kind'] == 'ONE_SECOND_GAME_CONTINUATION_NEW_BUDGET'
             and old['source_head'] == new['source_head'] == _BASE
             and _hex(old['sha256']) and _hex(new['sha256'])
             and new['old_manifest_sha256'] == old['sha256']
             and new['old_source_sha256'] == old['source_sha256']
             and all(_hex(new[k]) for k in ('files_sha256',
                                            'original_directory_sha256')),
             'manifest_parent_chain')
    _require(_same(old['fixture'], fixture) and _same(new['fixture'], fixture)
             and _same(old['binding'], binding) and _same(new['binding'], binding)
             and binding.get('fixture_sha256') == canonical_hash(fixture)
             and binding.get('fixture_sha256') == _FIXTURE
             and binding.get('rules_fingerprint') == _RULES
             and binding.get('encoder_id') == _ENCODING
             and binding.get('scope_id') == fixture.get('scope_id')
             and binding.get('strategy_eligible') is False
             and fixture.get('production_eligible') is False
             and fixture.get('simulation_only') is True, 'game_scope_binding')
    seeds = old['seeds']
    _require(_same(seeds, _SEEDS)
             and isinstance(seeds, list) and len(seeds) == 3
             and all(_integer(s) for s in seeds) and len(set(seeds)) == 3
             and _same(new['seeds'], seeds)
             and _same(old['checkpoint_iterations'], _POINTS)
             and _same(new['points'], _POINTS)
             and _same(old['variants'], _VARIANTS)
             and _same(new['variants'], _VARIANTS)
             and _same(new['seed_start_iterations'], [16059, 0, 0])
             and _same(new['targets'], [[16384], _POINTS, _POINTS]),
             'frozen_seed_iteration_plan')
    for field, value in {
        'quality_limit_bb': '1/20',
        'missing_mean': 'NOT_SCORED; no uniform/current-regret fallback or '
                        'normalization repair',
        'linear_formula': 'Native SIMPLE scratch increment weighted by global '
                          'iteration t',
        'equal_formula': 'Reviewed observation-only premerge linear delta/t; '
                         'commit after native success',
        'production_eligible': False, 'live_eligible': False,
    }.items():
        _require(_same(old[field], value) and _same(new[field], value),
                 'frozen_average_or_quality_semantics')
    _require('Gate each variant separately;' in old['final_gate'],
             'parent_variant_gate_missing')
    _require(old['per_seed_seconds'] == 360
             and _same(new['per_seed_seconds'], [90, 480, 480])
             and new['original_per_seed_seconds'] == 360
             and old['wall_seconds'] == new['wall_seconds'] == 1200
             and new['original_status_immutable'] == _STOP
             and new['four_street_eligible'] is False
             and type(new['attempt_limit']) is int and new['attempt_limit'] == 1
             and new['retries_allowed'] is False
             and new['budgets_transferable'] is False, 'frozen_execution_contract')
    for key, expected in {
        'expected_new_sweeps': 33093, 'expected_new_quality_points': 14,
        'expected_joint_points': 18, 'expected_new_mean_rows': 224,
        'expected_joint_mean_rows': 288,
    }.items():
        _require(type(new[key]) is int and new[key] == expected,
                 'manifest_denominator')
    return seeds


def _history(old, new, parent_result, result, big_blind):
    seeds = old['seeds']
    _require(parent_result.get('status') == _STOP
             and parent_result.get('manifest_sha256') == old['sha256']
             and result.get('status') == CONTINUATION_STATUS
             and result.get('manifest_sha256') == new['sha256']
             and result.get('original_segment_status') == _STOP,
             'historical_status_chain')
    for record in (parent_result, result):
        _require(record.get('production_eligible') is False
                 and record.get('live_eligible') is False
                 and type(record.get('attempt_count')) is int
                 and record['attempt_count'] == 1, 'result_qualification')
    _require(result.get('four_street_eligible') is False
             and _same(result.get('original_segment_counts'),
                       parent_result.get('counts'))
             and _same(new['original_counts'], parent_result.get('counts')),
             'parent_history_not_preserved')
    _require(parent_result.get('original_batch_stays') == 'HOLD'
             and _same(parent_result.get('final_variant_gates'),
                       {'linear': False, 'equal': False}),
             'parent_stopped_gate_changed')
    for key, expected in {
        'expected_checkpoints': 18, 'expected_mean_rows': 288, 'mean_rows': 64,
    }.items():
        _require(type(parent_result.get(key)) is int
                 and parent_result[key] == expected, 'parent_denominator')
    cases = parent_result.get('cases')
    _require(isinstance(cases, list) and len(cases) == 1, 'parent_case_denominator')
    case = cases[0]
    _require(_same(case.get('seed'), seeds[0]) and case.get('status') == _STOP
             and _same(case.get('committed_iterations'), 16059)
             and _same(case.get('tap_commits'), 16059)
             and _same(case.get('budget_seconds'), 360)
             and _same(parent_result.get('committed_iterations'), 16059),
             'parent_case_not_retained')
    current_cases = result.get('cases')
    _require(isinstance(current_cases, list) and len(current_cases) == 3,
             'case_denominator')
    case_map = {}
    for current in current_cases:
        seed = current.get('seed')
        _require(_integer(seed) and seed in seeds and seed not in case_map,
                 'case_seed_coverage')
        i = seeds.index(seed)
        _require(current.get('status') == 'COMPLETE'
                 and _same(current.get('start_iterations'),
                           new['seed_start_iterations'][i])
                 and _same(current.get('committed_iterations'), 16384)
                 and _same(current.get('tap_commits'), 16384)
                 and _same(current.get('budget_seconds'), new['per_seed_seconds'][i])
                 and current.get('deadline_overrun') is False,
                 'case_incomplete_or_changed')
        case_map[seed] = current
    all_keys = {(s, n, v) for s in seeds for n in _POINTS for v in _VARIANTS}
    new_keys = {(s, n, v) for i, s in enumerate(seeds)
                for n in new['targets'][i] for v in _VARIANTS}
    parent_points = _point_map(parent_result.get('points'), all_keys)
    points = _point_map(result.get('points'), new_keys)
    joint = _point_map(result.get('joint_points'), all_keys)
    for key, point in parent_points.items():
        if key in new_keys:
            _require(_same(point, dict(seed=key[0], iterations=key[1],
                                       variant=key[2], status='NOT_RUN',
                                       nash_conv_bb=None)), 'parent_not_run_changed')
        else:
            _quality(point, big_blind)
    for point in points.values():
        _quality(point, big_blind)
    for key, point in joint.items():
        expected = deepcopy(points[key] if key in points else parent_points[key])
        expected.update(origin_segment='continuation' if key in points else 'original',
                        original_status=parent_points[key]['status'])
        _require(_same(point, expected), 'joint_history_rewritten')
    _counts(list(parent_points.values()), parent_result.get('counts'))
    _counts(list(points.values()), result.get('counts'))
    _counts(list(joint.values()), result.get('joint_counts'))
    _require(parent_result['counts']['PASS'] == 1
             and parent_result['counts']['HOLD_QUALITY'] == 3
             and parent_result['counts']['NOT_RUN'] == 14,
             'parent_count_history_changed')
    for key, expected in {
        'expected_new_points': 14, 'expected_joint_points': 18,
        'expected_new_sweeps': 33093, 'new_committed_sweeps': 33093,
    }.items():
        _require(type(result.get(key)) is int and result[key] == expected,
                 'result_denominator')
    gates = {variant: all(_quality(joint[(seed, 16384, variant)], big_blind)
                          for seed in seeds) for variant in _VARIANTS}
    _require(_same(result.get('final_variant_gates'), gates)
             and gates['equal'] is True, 'equal_final_variant_gate')
    # The historical extra all-variants label remains data, never authorization.
    _require(type(result.get('second_game_final_success')) is bool
             and result['second_game_final_success'] == all(gates.values()),
             'historical_overall_label_changed')
    return points


def _validate_chain(document):
    chain = document['continuation']
    expected_fields = {
        'schema_version', 'kind', 'trust_boundary', 'sources',
        'source_file_sha256', 'source_canonical_sha256',
        'original_manifest_document_sha256', 'frozen_role_file_sha256',
        'main_source_file_sha256',
    }
    _require(isinstance(chain, dict) and set(chain) == expected_fields
             and type(chain['schema_version']) is int and chain['schema_version'] == 1
             and chain['kind'] == _KIND and chain['trust_boundary'] == _TRUST,
             'chain_schema')
    _portable(chain)
    extras = chain['sources']
    _require(isinstance(extras, dict) and set(extras) == EXTRA_SOURCE_NAMES,
             'extra_source_schema')
    _pins(chain['source_file_sha256'], EXTRA_SOURCE_NAMES)
    _pins(chain['source_canonical_sha256'], EXTRA_SOURCE_NAMES)
    _pins(chain['original_manifest_document_sha256'], {'parent_manifest', 'manifest'})
    _pins(chain['frozen_role_file_sha256'], _ROLE_NAMES)
    _pins(chain['main_source_file_sha256'], _MAIN_NAMES)
    _require(all(canonical_hash(v) == chain['source_canonical_sha256'][k]
                 for k, v in extras.items()), 'portable_source_hash')
    _require(_same(chain['main_source_file_sha256'], document['source_file_sha256']),
             'main_source_pins_changed')
    for role, pin in chain['frozen_role_file_sha256'].items():
        expected = (document['source_file_sha256']['catalog'] if role == 'catalog'
                    else chain['source_file_sha256'][role])
        _require(pin == expected, 'frozen_role_pin_changed')
    sources = document['sources']
    _require(isinstance(sources, dict) and set(sources) == _MAIN_NAMES
             and document.get('average_statistic') == _AVERAGE,
             'main_average_semantics')
    old, new = extras['parent_manifest'], extras['manifest']
    cp, equal = sources['checkpoint'], sources['equal']
    catalog, fixture = sources['catalog'], sources['fixture']
    binding = cp['binding']
    seeds = _manifests(old, new, fixture, binding)
    _require(isinstance(catalog, dict) and len(catalog) == 16, 'catalog_denominator')
    profiles = [AARuleProfileV2.from_dict(row['visible_observation']['rules'])
                for row in catalog.values()]
    _require(all(p.fingerprint == binding.get('rules_fingerprint') for p in profiles),
             'catalog_rule_binding')
    big_blind = Fraction(profiles[0].big_blind)
    points = _history(old, new, extras['parent_result'], sources['result'], big_blind)
    _checkpoint(cp, binding)
    _require(cp['seed'] in seeds and cp['iterations'] == 16384, 'endpoint_identity')
    _mean_state(equal, cp, catalog, parent=False)
    quality = sources['quality']
    _require(_same(quality, points[(cp['seed'], cp['iterations'], 'equal')])
             and quality['native_checkpoint_sha256'] == cp['sha256'],
             'endpoint_quality_binding')
    parent_cp, parent_equal = extras['parent_checkpoint'], extras['parent_equal']
    _checkpoint(parent_cp, binding)
    _require(parent_cp['seed'] == seeds[0] and parent_cp['iterations'] == 16059,
             'restored_parent_identity')
    _mean_state(parent_equal, parent_cp, catalog, parent=True)
    proof = extras['restore']
    proof_fields = {
        'status', 'poker_sweeps', 'quality_evaluations', 'fields_equal',
        'native_checkpoint_sha256', 'native_file_sha256', 'equal_file_sha256',
        'equal_table_sha256', 'iterations', 'tap_commits', 'native_rows',
        'equal_rows', 'equal_binding',
    }
    _require(isinstance(proof, dict) and set(proof) == proof_fields
             and proof['status'] == 'PASS'
             and _same(proof['fields_equal'], {k: True for k in _CP_FIELDS})
             and proof['native_checkpoint_sha256'] == parent_cp['sha256']
             and proof['native_file_sha256'] ==
             chain['source_file_sha256']['parent_checkpoint']
             and proof['equal_file_sha256'] ==
             chain['source_file_sha256']['parent_equal']
             and proof['equal_table_sha256'] == canonical_hash(parent_equal['average'])
             and proof['equal_binding'] == 'Both original file hashes plus original '
             'RESULT case counter; original equal file has no embedded native hash',
             'zero_step_proof_binding')
    for key, expected in {
        'poker_sweeps': 0, 'quality_evaluations': 0, 'iterations': 16059,
        'tap_commits': 16059, 'native_rows': 16, 'equal_rows': 16,
    }.items():
        _require(type(proof[key]) is int and proof[key] == expected,
                 'zero_step_proof_counters')


def validate_chain(document):
    """Recheck all portable semantic bindings; never grant product eligibility."""
    try:
        _validate_chain(document)
    except (KeyError, TypeError, AttributeError, OverflowError, IndexError) as exc:
        raise ValueError('continuation_malformed_portable_evidence') from exc
