"""AST-equivalent extraction of the historical synthetic observer.
Importing this module constructs no solver or arena and performs no iteration.
Full source/snippet hashes and transformation mapping are in SOURCE-MAP.json.
"""
from poker_engine.strategy.aa_mccfr import ExternalSamplingMCCFR

class PairedObserver(ExternalSamplingMCCFR):
    """Never feeds the second statistic into native strategy, traversal or RNG."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.equal = {}
        self._pending_equal = None
        self.tap_commits = 0

    def _merge(self, base, delta):
        if base is self.average:
            if self._pending_equal is not None:
                raise ValueError('duplicate_average_merge')
            t = self.iterations + 1
            increment = {key: {a: v / t for a, v in row.items()}
                         for key, row in delta.items()}
            self._pending_equal = ExternalSamplingMCCFR._merge(self.equal, increment)
        return ExternalSamplingMCCFR._merge(base, delta)

    def iterate(self, arena_factory):
        self._pending_equal = None
        try:
            result = super().iterate(arena_factory)
        except BaseException:
            self._pending_equal = None
            raise
        if self._pending_equal is None:
            raise ValueError('native_average_merge_not_observed')
        self.equal = self._pending_equal
        self._pending_equal = None
        self.tap_commits += 1
        return result

def mean_policy(table):
    result = {}
    for key, values in table.items():
        total = sum(values[a] for a in sorted(values))
        if total:
            result[key] = {a: v / total for a, v in values.items()}
    return result
