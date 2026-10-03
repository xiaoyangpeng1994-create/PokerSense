"""Portable source binding for the original S11 A/B run/save functions.

This module never runs an experiment on import or from its CLI. create_runtime
is an explicit, potentially expensive cloud-only operation. The original
learning/statistic/run/save bodies are AST-equivalent extractions. Only their
host paths, old top-level scheduler and historical budget bookkeeping have
been replaced by this binding. No historical checkpoint is distributed;
historical continuation is unavailable from this repository alone.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

SOURCE_HEAD = '38a97da7f550047a12bfacd2e8174b1c225a0419'
SEEDS = (2026100501, 2026100502, 2026100503)
TARGET = 16384
RESUME_CHECKPOINT = 8192
LIMIT = F(1, 20)
STATES = ('POSITIVE_VALID', 'ZERO_MASS', 'ABSENT', 'INVALID_MENU', 'INVALID_VALUE')


def verify_source_files():
    """Hash-only validation: no poker imports, construction, scoring or writes."""
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / 'SOURCE-MAP.json').read_text('utf-8'))
    for item in manifest['files']:
        path = here / item['path']
        if path.parent != here or not path.is_file():
            raise ValueError('source_file_missing_or_outside_scope')
        raw = path.read_bytes()
        if len(raw) != item['size_bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('source_bytes_changed:' + item['path'])
    return dict(status='SOURCE_BYTES_MATCH', source_head=SOURCE_HEAD,
                files=len(manifest['files']), training_run=False,
                historical_resume_available=False, runtime_execution='NOT_RUN')


def create_runtime(repo_root, output_dir, *, total_budget_seconds,
                   global_save_reserve_seconds, seed_save_reserve_seconds,
                   setup_deadline):
    """Bind extracted functions for a separately approved cloud experiment.

    Caller must freeze a NEW budget/ledger before invoking this function, include
    setup/evaluation/save/cleanup in it, and supply bounded per-seed deadlines.
    This is not the old S11 scheduler and cannot reset an old seed's budget.
    The returned run_seed/restore_only functions retain the original save/restore
    format and limitations. They are not authenticated historical-resume APIs.
    """
    verify_source_files()
    GLOBAL_T0 = time.monotonic()
    for value in (total_budget_seconds, global_save_reserve_seconds,
                  seed_save_reserve_seconds, setup_deadline):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('explicit_positive_finite_budget_required')
    TOTAL_BUDGET = total_budget_seconds
    GLOBAL_SAVE_RESERVE = global_save_reserve_seconds
    SEED_SAVE_RESERVE = seed_save_reserve_seconds
    if TOTAL_BUDGET <= GLOBAL_SAVE_RESERVE or setup_deadline <= GLOBAL_T0:
        raise ValueError('setup_and_cleanup_budget_required')
    repo = Path(repo_root).resolve()
    HERE = Path(output_dir).resolve()
    source_dir = Path(__file__).resolve().parent
    if HERE == repo or repo in HERE.parents:
        raise ValueError('output_must_be_outside_source_checkout')
    HERE.mkdir(parents=True, exist_ok=True)
    if any(HERE.iterdir()):
        raise ValueError('fresh_separate_output_directory_required')
    for path in (repo, repo / 'src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from poker_engine.strategy.aa_frozen_policy import action_ids, validate_distribution
    from poker_engine.strategy.aa_mccfr import (
        ExternalSamplingMCCFR, TrainingBudget, TrainingBudgetExceeded)
    from tools import aa_infoset_exact_br as br
    from research.scenario2.scenario2_factory import load_game
    from research.scenario2.own_reach_collector import compile_plan, collect
    from research.scenario2.paired_observer import PairedObserver
    if importlib.metadata.version('pokerkit') != '0.7.5':
        raise ValueError('pokerkit_version_gate')
    sys.set_int_max_str_digits(200000)
    game = load_game(repo)
    expected = json.loads((source_dir / 'S11-SCENARIO.portable.json').read_text('utf-8'))
    if game['expected_binding']()['fixture_sha256'] != expected['fixture_spec_scenario2_sha256']:
        raise ValueError('scenario_fixture_changed')
    root = game['build_tree'](deadline=setup_deadline)
    plan = compile_plan(root, deadline=setup_deadline)
    menus = {identity: item['signature'][1] for identity, item in plan.information.items()}
    cat = json.loads((source_dir / 'VISIBLE-CATALOG2.json').read_text('utf-8'))
    DENOM = len(menus)
    if DENOM != 1488 or len(cat) != DENOM:
        raise ValueError('frozen_denominator_mismatch')

    def utc():
        return datetime.now(timezone.utc).isoformat()

    def read(path):
        return json.loads(Path(path).read_text('utf-8'))

    def write(name, value):
        path = HERE / name
        pending = path.with_suffix(path.suffix + '.pending')
        pending.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False,
                                      default=str) + '\n', 'utf-8')
        pending.replace(path)

    def fj(v):
        return dict(numerator=str(v.numerator), denominator=str(v.denominator),
                    float=float(v), numerator_digits=len(str(v.numerator)))

    def require(ok, reason):
        if not ok:
            raise AssertionError(reason)

    def global_left():
        return TOTAL_BUDGET - (time.monotonic() - GLOBAL_T0)

    def identity_of(key):
        return tuple(tuple(x) if isinstance(x, list) else x for x in cat[key]['infoset'])

    identity_of_key = {key: identity_of(key) for key in cat}
    key_of_identity = {v: k for k, v in identity_of_key.items()}
    require(len(key_of_identity) == DENOM, 'identity_bijection')
    require(set(identity_of_key.values()) == set(menus), 'tree_catalog_identity_mismatch')

    def menu_of(key):
        return tuple(menus[identity_of_key[key]])

    def checked_encoder(observation):
        key = game['research_key'](observation)
        require(key in cat and observation == cat[key]['visible_observation'],
                'unrecognized_actual_visible_observation')
        return key

    def snapshot(solver):
        prof = {}
        for key, identity in identity_of_key.items():
            row_obj = solver.regrets.get(key)
            menu = menu_of(key)
            row = ExternalSamplingMCCFR._strategy(
                row_obj if row_obj is not None else {a: 0.0 for a in menu})
            exact = {a: F.from_float(v) for a, v in row.items()}
            mass = sum(exact.values(), F(0))
            prof[identity] = {a: v / mass for a, v in exact.items()}
        return prof

    class DualObserver(PairedObserver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.acc_a = {}
            self.acc_b = {}
            self._pending = None
            self.bypass = dict(sweeps_collected=0, discarded=0,
                               zero_mass_rows_skipped=0)

        def iterate(self, factory):
            t = self.iterations + 1          # GLOBAL sweep index
            profile = snapshot(self)
            pending = collect(plan, profile, weight=F(1))
            self._pending = pending
            try:
                result = super().iterate(factory)
            except BaseException:
                self._pending = None
                self.bypass['discarded'] += 1
                raise
            increment = self._pending
            self._pending = None
            for identity, row in increment['delta'].items():
                if sum(row.values(), F(0)) == 0:
                    self.bypass['zero_mass_rows_skipped'] += 1
                    continue
                ta = self.acc_a.setdefault(identity, {})
                tb = self.acc_b.setdefault(identity, {})
                for action, value in row.items():
                    tb[action] = tb.get(action, 0.0) + float(value)
                    ta[action] = ta.get(action, 0.0) + float(t * value)
            self.bypass['sweeps_collected'] += 1
            return result

    def classify(table):
        c = Counter({s: 0 for s in STATES})
        for key in cat:
            menu = menu_of(key)
            row = table.get(key)
            if row is None:
                st = 'ABSENT'
            elif not isinstance(row, dict) or set(row) != set(menu):
                st = 'INVALID_MENU'
            elif any(type(v) not in (float, int) or v != v
                     or v in (float('inf'), float('-inf')) or v < 0
                     for v in row.values()):
                st = 'INVALID_VALUE'
            else:
                st = 'POSITIVE_VALID' if sum(row[a] for a in sorted(row)) > 0 \
                    else 'ZERO_MASS'
            c[st] += 1
        return dict(fixed_denominator=DENOM, counts=dict(c),
                    complete=(c['POSITIVE_VALID'] == DENOM and not c['ABSENT']
                              and not c['ZERO_MASS'] and not c['INVALID_MENU']
                              and not c['INVALID_VALUE']))

    def score(table, variant, sweeps, seed_left):
        entry = dict(variant=variant, sweeps=sweeps, rows=len(table),
                     evaluation_execution='NOT_RUN', status='NOT_SCORED', reason=None,
                     nash_conv_bb=None, gains=None, quality_limit_bb='1/20',
                     seconds=60,
                     metric='nash_conv_bb=(gain_p1+gain_p2)/big_blind(2)')
        if global_left() <= GLOBAL_SAVE_RESERVE or seed_left <= 5.0:
            entry['reason'] = 'BUDGET_EXHAUSTED_BEFORE_SCORING'
            return entry
        cov = classify(table)
        if not cov['complete']:
            entry['reason'] = 'POLICY_NOT_COMPLETE_LEGAL_NORMALIZED'
            return entry
        normalized = {}
        for key, row in table.items():
            total = sum(row.values())
            if not total > 0:
                entry['reason'] = 'ZERO_OR_NONFINITE_MASS_ROW'
                return entry
            normalized[key] = {a: v / total for a, v in row.items()}
        profile = {}
        for key, row in normalized.items():
            validate_distribution(row, menu_of(key))
            exact = {a: F.from_float(p) for a, p in row.items()}
            mass = sum(exact.values(), F(0))
            profile[identity_of_key[key]] = {a: p / mass for a, p in exact.items()}
        require(len(profile) == DENOM, 'profile_row_count')
        entry['evaluation_execution'] = 'RUN'
        started = time.monotonic()
        try:
            actual = br.evaluate(root, profile, seconds=60, max_ops=100000)
            ncb = actual['nash_conv_bb']
            entry.update(status='PASS' if ncb <= LIMIT else 'HOLD_QUALITY',
                         nash_conv_bb=fj(ncb),
                         gains={str(k): fj(v) for k, v in actual['gains'].items()},
                         stats=actual['stats'])
        except BaseException as exc:
            entry.update(status='NOT_SCORED',
                         reason=type(exc).__name__ + ':' + str(exc))
        entry['elapsed_seconds'] = time.monotonic() - started
        return entry

    def by_key(table):
        return {key_of_identity[i]: r for i, r in table.items()}

    def restore_only(restore):
        """Read-only restore: proves the saved bundle is a COMPLETE, resumable
        state without training, saving or overwriting anything."""
        solver = DualObserver.restore(
            restore['native'], expected_binding=game['expected_binding'](),
            expected_encoder=game['ENCODER_ID'], encoder=checked_encoder,
            menu=action_ids, clock=time.monotonic,
            budget=TrainingBudget(max_nodes=114, max_infosets=DENOM,
                                  max_depth=8, seconds=300))
        solver.equal = restore['equal']['equal']
        solver.tap_commits = restore['equal']['tap_commits']
        solver._pending_equal = None
        solver.acc_a = {identity_of_key[k]: dict(v)
                        for k, v in restore['a']['rows'].items()}
        solver.acc_b = {identity_of_key[k]: dict(v)
                        for k, v in restore['b']['rows'].items()}
        solver._pending = None
        chk = solver.checkpoint()
        zero = dict(resumed_at=solver.iterations,
                    total_nodes=solver.total_nodes,
                    digest_matches=chk['sha256'] == restore['native']['sha256'],
                    canonical_json_matches=canon(chk) == canon(restore['native']),
                    equal_matches=solver.equal == restore['equal']['equal'],
                    a_matches=all(solver.acc_a[identity_of_key[k]] == v
                                  for k, v in restore['a']['rows'].items()),
                    b_matches=all(solver.acc_b[identity_of_key[k]] == v
                                  for k, v in restore['b']['rows'].items()))
        zero['all_identical'] = (zero['digest_matches']
                                 and zero['canonical_json_matches']
                                 and zero['equal_matches'] and zero['a_matches']
                                 and zero['b_matches'])
        return solver, zero

    def run_seed(seed, restore, deadline_abs, used_before):
        tag = str(seed)
        started = time.monotonic()
        cpu_start = time.process_time()

        def clock():
            now = time.monotonic()
            if now >= deadline_abs:
                raise TrainingBudgetExceeded('seed_deadline')
            return now

        if restore is None:
            solver = DualObserver((1, 2), seed=seed, encoder=checked_encoder,
                                  menu=action_ids, binding=game['expected_binding'](),
                                  encoder_id=game['ENCODER_ID'], clock=clock,
                                  budget=TrainingBudget(max_nodes=114,
                                                        max_infosets=DENOM,
                                                        max_depth=8, seconds=300))
            require(solver.iterations == solver.tap_commits == 0, 'not_fresh_seed')
            base = 0
            zero = None
        else:
            solver = DualObserver.restore(
                restore['native'], expected_binding=game['expected_binding'](),
                expected_encoder=game['ENCODER_ID'], encoder=checked_encoder,
                menu=action_ids, clock=clock,
                budget=TrainingBudget(max_nodes=114, max_infosets=DENOM,
                                      max_depth=8, seconds=300))
            solver.equal = restore['equal']['equal']
            solver.tap_commits = restore['equal']['tap_commits']
            solver._pending_equal = None
            solver.acc_a = {identity_of_key[k]: dict(v)
                            for k, v in restore['a']['rows'].items()}
            solver.acc_b = {identity_of_key[k]: dict(v)
                            for k, v in restore['b']['rows'].items()}
            solver._pending = None
            chk = solver.checkpoint()
            zero = dict(resumed_at=solver.iterations,
                        total_nodes=solver.total_nodes,
                        digest_matches=chk['sha256'] == restore['native']['sha256'],
                        canonical_json_matches=canon(chk) == canon(restore['native']),
                        equal_matches=solver.equal == restore['equal']['equal'],
                        a_matches=all(solver.acc_a[identity_of_key[k]] == v
                                      for k, v in restore['a']['rows'].items()),
                        b_matches=all(solver.acc_b[identity_of_key[k]] == v
                                      for k, v in restore['b']['rows'].items()))
            zero['all_identical'] = (zero['digest_matches']
                                     and zero['canonical_json_matches']
                                     and zero['equal_matches'] and zero['a_matches']
                                     and zero['b_matches'])
            require(zero['all_identical'], 'resume_state_mismatch')
            base = solver.iterations

        rec = dict(seed=seed, base_sweeps=base, resumed=restore is not None,
                   resume_zero_step=zero, status='STARTED', attempted_sweeps=0,
                   start_utc=utc(), checkpoints=[])
        try:
            while solver.iterations < TARGET:
                left = deadline_abs - time.monotonic()
                if left <= 0.05:
                    rec['status'] = 'STOP_BUDGET'
                    rec['termination_reason'] = 'seed_deadline_before_next_sweep'
                    break
                solver.budget = TrainingBudget(max_nodes=114, max_infosets=DENOM,
                                               max_depth=8, seconds=left - 0.01)
                rec['attempted_sweeps'] += 1
                solver.last_attempt_nodes = 0
                try:
                    solver.iterate(game['sample_factory'])
                except TrainingBudgetExceeded as exc:
                    rec['status'] = 'STOP_BUDGET'
                    rec['termination_reason'] = str(exc)
                    break
                if solver.iterations in (RESUME_CHECKPOINT, TARGET):
                    label = str(solver.iterations)
                    cp = solver.checkpoint()
                    a_rows, b_rows = by_key(solver.acc_a), by_key(solver.acc_b)
                    write('s%s-native-%s.json' % (tag, label), cp)
                    write('s%s-equal-%s.json' % (tag, label),
                          dict(seed=seed, iterations=solver.iterations,
                               tap_commits=solver.tap_commits, equal=solver.equal,
                               native_checkpoint_sha256=cp['sha256']))
                    write('s%s-A-%s.json' % (tag, label),
                          dict(seed=seed, sweeps=solver.iterations, variant='A',
                               rows=a_rows))
                    write('s%s-B-%s.json' % (tag, label),
                          dict(seed=seed, sweeps=solver.iterations, variant='B',
                               rows=b_rows))
                    seed_left = deadline_abs - time.monotonic()
                    point = dict(sweeps=solver.iterations,
                                 wall_seconds=time.monotonic() - started,
                                 cpu_seconds=time.process_time() - cpu_start,
                                 total_nodes=solver.total_nodes,
                                 coverage=dict(A_linear_full_tree=classify(a_rows),
                                               B_uniform_full_tree=classify(b_rows),
                                               native_linear_average=classify(
                                                   solver.average),
                                               existing_equal_average=classify(
                                                   solver.equal)))
                    if solver.iterations == TARGET:
                        point['scores'] = [
                            score(a_rows, 'A_linear_full_tree', TARGET, seed_left),
                            score(b_rows, 'B_uniform_full_tree', TARGET, seed_left)]
                    rec['checkpoints'].append(point)
            else:
                rec['status'] = 'COMPLETE_TARGET'
                rec['termination_reason'] = 'max_complete_sweeps_16384'
        except BaseException as exc:
            rec['status'] = 'STOP_ERROR'
            rec['termination_reason'] = type(exc).__name__ + ':' + str(exc)
            rec['traceback'] = traceback.format_exc()

        save_started = time.monotonic()
        final_cp = solver.checkpoint()
        a_final, b_final = by_key(solver.acc_a), by_key(solver.acc_b)
        write('S11-A-FINAL-%s.json' % tag,
              dict(seed=seed, sweeps=solver.iterations, variant='A', rows=a_final))
        write('S11-B-FINAL-%s.json' % tag,
              dict(seed=seed, sweeps=solver.iterations, variant='B', rows=b_final))
        write('S11-native-final-%s.json' % tag, final_cp)
        write('S11-equal-final-%s.json' % tag,
              dict(seed=seed, iterations=solver.iterations,
                   tap_commits=solver.tap_commits, equal=solver.equal,
                   native_checkpoint_sha256=final_cp['sha256']))
        save_seconds = time.monotonic() - save_started
        rec.update(completed_sweeps=solver.iterations,
                   tap_commits=solver.tap_commits,
                   incomplete_started_sweeps=rec['attempted_sweeps']
                                             - (solver.iterations - base),
                   unstarted_sweeps=TARGET - solver.iterations,
                   total_nodes=solver.total_nodes,
                   pass_wall_seconds=time.monotonic() - started,
                   pass_cpu_seconds=time.process_time() - cpu_start,
                   save_seconds=save_seconds,
                   wall_seconds=used_before + (time.monotonic() - started),
                   cpu_seconds=time.process_time() - cpu_start, end_utc=utc(),
                   bypass=dict(solver.bypass),
                   final_coverage=dict(A_linear_full_tree=classify(a_final),
                                       B_uniform_full_tree=classify(b_final),
                                       native_linear_average=classify(solver.average),
                                       existing_equal_average=classify(solver.equal)),
                   reached_final_acceptance_point=(solver.iterations == TARGET))
        rec['_bundle'] = dict(native=final_cp,
                              equal=dict(seed=seed, iterations=solver.iterations,
                                         tap_commits=solver.tap_commits,
                                         equal=solver.equal,
                                         native_checkpoint_sha256=final_cp['sha256']),
                              a=dict(seed=seed, sweeps=solver.iterations, variant='A',
                                     rows=a_final),
                              b=dict(seed=seed, sweeps=solver.iterations, variant='B',
                                     rows=b_final))
        return rec

    return dict(game=game, root=root, plan=plan, menus=menus,
                denominator=DENOM, seeds=SEEDS, target=TARGET,
                run_seed=run_seed, restore_only=restore_only,
                classify=classify, score=score, snapshot=snapshot, by_key=by_key,
                historical_resume_available=False, runtime_execution='NOT_RUN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-source-only', action='store_true', required=True,
                        help='check frozen file hashes; never build, train or score')
    parser.parse_args()
    print(json.dumps(verify_source_files(), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
