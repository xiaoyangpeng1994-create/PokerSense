"""Opt-in offline arena state source for two pinned synthetic river fixtures.

State and own-memory come from actual reset/step events. No screen adapter,
platform authentication, training, V2 registration or quality promotion exists.
Host internals remain trusted Python state; only visible observations go to the
existing bridge. The second fixture retains its original stopped batch status.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import random
from threading import RLock
import time
import uuid

from poker_engine.desktop.aa_turn_runtime import (
    AATurnWindow, TurnEvidence, TurnIdentity,
)
from poker_engine.strategy.aa_frozen_policy import action_ids, canonical_hash
from poker_engine.strategy.aa_full_hand_arena import AAFullHandArena
from poker_engine.strategy.aa_rules_v2 import AARuleProfileV2
from tools.aa_research_equal_bridge import (
    ResearchArtifact, ResearchEqualBridge, _research_key,
)


SUPPORTED_FIXTURES = frozenset({
    'b0e5588e8402464339131644b4b814c435995e7e92f59bde343f1593aecf24f0',
    '691945d6cc90c596021c165e2d18515beba155a6eff0cbe38136da5ee6fe5067',
})
RULES = '47d4265a9afe6f1a1ba6971fd97d8897ac3cbb7cfd6c1dfa82f21653b7fb2934'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class ArenaSnapshot:
    source_id: str
    epoch: int
    revision: int
    identity: TurnIdentity
    source_at: float
    observation_json: bytes

    @property
    def observation(self):
        return json.loads(self.observation_json)


@dataclass(frozen=True)
class _Event:
    actor: int
    public_index: int
    visible_sha256: str
    action_id: str
    event_json: bytes


class ArenaContextSource:
    """Trusted simulation host; requests cannot substitute observation fields."""

    def __init__(self, artifact, *, enabled=False, clock=time.monotonic):
        if (not isinstance(artifact, ResearchArtifact) or type(enabled) is not bool
                or not callable(clock)):
            raise ValueError('LOADED_ARTIFACT_AND_EXPLICIT_OPT_IN_REQUIRED')
        self._bridge = ResearchEqualBridge(artifact, enabled=enabled)
        doc = artifact.document
        self._binding = deepcopy(doc['sources']['checkpoint']['binding'])
        self._fixture = deepcopy(doc['sources']['fixture'])
        if (canonical_hash(self._fixture) not in SUPPORTED_FIXTURES
                or self._binding['rules_fingerprint'] != RULES):
            raise ValueError('UNSUPPORTED_RESEARCH_FIXTURE')
        self._catalog = deepcopy(doc['sources']['catalog'])
        first = next(iter(self._catalog.values()))['visible_observation']
        self._rules = AARuleProfileV2.from_dict(first['rules'])
        self._artifact_sha256 = artifact.sha256
        self._enabled, self._clock = enabled, clock
        self._lock = RLock()
        self._source_id = 'offline-arena-' + uuid.uuid4().hex
        self._arena = None
        self._memory, self._journal = {}, ()
        self._epoch, self._revision = 0, 0
        self._source_at, self._receipt, self._window = None, None, None
        self._last_clock = None
        self._closed = False

    def _require_open(self):
        if self._closed:
            raise ValueError('CLOSED')
        if not self._enabled:
            raise ValueError('DISABLED')

    def _require_state(self):
        self._require_open()
        if self._arena is None:
            raise ValueError('NOT_RESET')

    def _now(self):
        with self._lock:
            value = self._clock()
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0):
                raise ValueError('INVALID_SOURCE_CLOCK')
            if self._last_clock is not None and value < self._last_clock:
                raise ValueError('CLOCK_REGRESSION')
            self._last_clock = float(value)
            return self._last_clock

    @property
    def epoch(self):
        with self._lock:
            return self._epoch

    @property
    def revision(self):
        with self._lock:
            return self._revision

    @property
    def source_at(self):
        with self._lock:
            return self._source_at

    @property
    def actor(self):
        with self._lock:
            self._require_state()
            return self._arena.actor

    @property
    def terminal(self):
        with self._lock:
            self._require_state()
            return self._arena.terminal

    def preload(self):
        with self._lock:
            self._require_open()
            self._bridge.preload()

    def close(self):
        with self._lock:
            self._closed = True
            if self._window is not None:
                self._window.invalidate('CLOSED')
        self._bridge.close()

    def _deck(self, deal, seed):
        from pokerkit import Deck
        if (not isinstance(deal, (tuple, list)) or
                list(deal) not in self._fixture['joint_deals']):
            raise ValueError('UNSUPPORTED_SYNTHETIC_JOINT_DEAL')
        holes = dict(self._fixture['folded_holes'])
        for seat, label in zip(self._fixture['active_seats'], deal):
            holes[str(seat)] = self._fixture['active_ranges'][str(seat)][label]
        deck = [card for seat in range(self._fixture['table_size'])
                for card in holes[str(seat)]]
        board, burns = self._fixture['board'], self._fixture['burns']
        deck += [burns[0], *board[:3], burns[1], board[3], burns[2], board[4]]
        if len(deck) != 20 or len(set(deck)) != 20:
            raise ValueError('SYNTHETIC_CARD_COLLISION')
        tail = [repr(card) for card in Deck.STANDARD if repr(card) not in deck]
        random.Random(seed).shuffle(tail)
        return deck + tail

    @staticmethod
    def _apply(arena, memory, journal, action_id):
        actor = arena.actor
        visible = arena.observe(actor)
        if action_id not in action_ids(visible):
            raise ValueError('OUTSIDE_NATIVE_MENU')
        index = len(visible['public_history'])
        visible_sha = canonical_hash(visible)
        try:
            arena.step(action_id)
        except RuntimeError as exc:
            raise ValueError('ARENA_TRANSACTION_FAILED') from exc
        event = arena.observe(actor)['public_history'][-1]
        memory[actor].append(dict(public_index=index,
                                  visible_observation_sha256=visible_sha,
                                  action_id=action_id))
        return journal + (_Event(actor, index, visible_sha, action_id, _json(event)),)

    @staticmethod
    def _check_memory(arena, memory, journal):
        history = arena.observe(arena.occupied_seats[0])['public_history']
        expected = {seat: [] for seat in arena.occupied_seats}
        if len(history) != len(journal):
            raise ValueError('MEMORY_INTEGRITY_MISMATCH')
        for index, entry in enumerate(journal):
            if (entry.public_index != index or _json(history[index]) != entry.event_json
                    or history[index]['actor'] != entry.actor
                    or history[index]['id'] != entry.action_id):
                raise ValueError('MEMORY_INTEGRITY_MISMATCH')
            expected[entry.actor].append(dict(
                public_index=index, visible_observation_sha256=entry.visible_sha256,
                action_id=entry.action_id))
        if _json(memory) != _json(expected):
            raise ValueError('MEMORY_INTEGRITY_MISMATCH')

    def _visible(self, arena, memory, journal, seat):
        self._check_memory(arena, memory, journal)
        if type(seat) is not int or seat not in arena.occupied_seats:
            raise ValueError('INVALID_OBSERVER')
        obs = arena.observe(seat)
        obs['own_memory'] = deepcopy(memory[seat])
        return obs

    def _scoped(self, arena, memory, journal):
        if arena.terminal:
            raise ValueError('TERMINAL')
        obs = self._visible(arena, memory, journal, arena.actor)
        try:
            key = _research_key(obs, self._binding)
            if (key not in self._catalog or canonical_hash(obs) !=
                    canonical_hash(self._catalog[key]['visible_observation'])):
                raise ValueError('OUTSIDE_RESEARCH_SCOPE')
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError('OUTSIDE_RESEARCH_SCOPE') from exc
        return obs

    def _publish(self, arena, memory, journal, epoch, revision):
        self._check_memory(arena, memory, journal)
        obs = None if arena.terminal else self._scoped(arena, memory, journal)
        now = self._now()
        receipt, window = None, None
        if obs is not None:
            identity = TurnIdentity(self._source_id, epoch, f'hand-{epoch}',
                                    f'turn-{revision}')
            receipt = ArenaSnapshot(self._source_id, epoch, revision, identity,
                                    now, _json(obs))
            evidence = TurnEvidence(f'simulation-{epoch}-{revision}',
                                    'verified_onset', now, 10)
            window = AATurnWindow(identity, evidence, now=now)
        self._arena, self._memory, self._journal = arena, memory, journal
        self._epoch, self._revision, self._source_at = epoch, revision, now
        self._receipt, self._window = receipt, window

    def reset(self, deal, *, seed=0):
        with self._lock:
            self._require_open()
            if type(seed) is not int:
                raise ValueError('INTEGER_SYNTHETIC_SEED_REQUIRED')
            arena = AAFullHandArena(
                self._rules, starting_stacks={s: self._fixture['starting_stacks']
                                              for s in range(6)},
                occupied_seats=range(6), dealer_seat=self._fixture['dealer'])
            try:
                arena.reset(seed, deck=self._deck(deal, seed))
            except RuntimeError as exc:
                raise ValueError('ARENA_TRANSACTION_FAILED') from exc
            memory, journal = {s: [] for s in arena.occupied_seats}, ()
            for seat, street, action, paid in self._fixture['prefix']:
                if arena.actor != seat or arena.street != street:
                    raise ValueError('ACTUAL_PREFIX_MISMATCH')
                journal = self._apply(arena, memory, journal, action)
                if json.loads(journal[-1].event_json)['paid'] != paid:
                    raise ValueError('ACTUAL_PREFIX_MISMATCH')
            self._publish(arena, memory, journal, self._epoch + 1, 0)
        return self

    def step(self, action_id, *, seat=None):
        with self._lock:
            self._require_state()
            self._scoped(self._arena, self._memory, self._journal)
            if seat is not None and (
                    type(seat) is not int or seat != self._arena.actor):
                raise ValueError('NON_ACTOR')
            arena, memory = self._arena.clone(), deepcopy(self._memory)
            journal = self._apply(arena, memory, self._journal, action_id)
            self._publish(arena, memory, journal, self._epoch, self._revision + 1)
        return self

    def observe(self, seat):
        with self._lock:
            self._require_state()
            return self._visible(self._arena, self._memory, self._journal, seat)

    def snapshot(self, seat=None):
        with self._lock:
            self._require_state()
            if seat is not None and (
                    type(seat) is not int or seat != self._arena.actor):
                raise ValueError('NON_ACTOR')
            obs = self._scoped(self._arena, self._memory, self._journal)
            if _json(obs) != self._receipt.observation_json:
                raise ValueError('STATE_CHANGED')
            return self._receipt

    def _context(self, receipt):
        with self._lock:
            self._require_state()
            if receipt is not self._receipt:
                raise ValueError('STATE_CHANGED')
            obs = self._scoped(self._arena, self._memory, self._journal)
            return dict(observation=obs, identity=self._receipt.identity,
                        artifact_sha256=self._artifact_sha256,
                        source_at=self._source_at)

    def lookup(self, receipt):
        try:
            with self._lock:
                self._require_state()
                if (not isinstance(receipt, ArenaSnapshot)
                        or receipt.source_id != self._source_id):
                    raise ValueError('INVALID_ARENA_SNAPSHOT')
                if receipt.epoch != self._epoch:
                    raise ValueError('STALE_ARENA_EPOCH')
                if receipt.revision != self._revision:
                    raise ValueError('STATE_CHANGED')
                if receipt is not self._receipt:
                    raise ValueError('INVALID_ARENA_SNAPSHOT')
                context = self._context(receipt)
                window = self._window
            result = self._bridge.lookup(
                context['observation'], artifact_sha256=self._artifact_sha256,
                simulation=True, identity=receipt.identity, window=window,
                source_at=receipt.source_at,
                current_context=lambda: self._context(receipt), clock=self._now)
            with self._lock:
                current = self._context(receipt)
                if _json(current['observation']) != receipt.observation_json:
                    raise ValueError('STATE_CHANGED')
                if result.get('status') == 'SHADOW_RESULT':
                    reason = window.check(now=self._now(), identity=receipt.identity,
                                          source_at=receipt.source_at,
                                          for_new_result=False)
                    if reason != 'WITHIN_BUDGET':
                        raise ValueError(reason)
                result.update(arena_epoch=receipt.epoch,
                              arena_revision=receipt.revision)
                return result
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return self._bridge._abstain(str(exc))
