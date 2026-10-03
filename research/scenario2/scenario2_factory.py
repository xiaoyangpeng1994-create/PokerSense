"""Portable handoff shim: exact scenario constants/setup extracted from S11.
No game construction or training runs on import. Call load_game only in a
separately authorized cloud computation. Module keeps the original file path
for relative rule loading. This packaging pass never executes this module.
"""
import hashlib
from pathlib import Path
import sys

HEAD = '38a97da7f550047a12bfacd2e8174b1c225a0419'

SCOPE2_ID = 'aa-six-seat-two-active-full-chance-turn-river-scenario2-v1'

BOARD2 = ('2c', '5d', '9h', 'Js')

HOLDINGS2 = {1: {'AA': ('As', 'Ad'), 'TT': ('Tc', 'Td')},
             2: {'KK': ('Kh', 'Kd'), 'PAIR_8': ('8c', '8d')}}

DEALS2 = (('AA', 'KK'), ('AA', 'PAIR_8'), ('TT', 'KK'), ('TT', 'PAIR_8'))

ORIGINAL_DRIVER_SHA256 = '5ef1bf7e38f18e636570e569b6cad7c0385be0aeb201844752cc2e6257f21614'

def load_game(repo_root):
    REPO = Path(repo_root).resolve()
    ORIG_GAME = REPO / 'tools' / 'aa_turn_river_game.py'
    if hashlib.sha256(ORIG_GAME.read_bytes()).hexdigest() != '42066caa41afc3edc4270dbc106d499c53995cb53cd989a60a1c7a629833408e':
        raise ValueError('constructor source does not match frozen scenario')
    for path in (REPO, REPO / 'src'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    source = ORIG_GAME.read_text('utf-8')
    ns2 = {'__name__': 'aa_turn_river_game_scenario2', '__file__': str(ORIG_GAME)}
    exec(compile(source, str(ORIG_GAME), 'exec'), ns2)
    ns2['SCOPE_ID'] = SCOPE2_ID
    ns2['BOARD'] = BOARD2
    ns2['HOLDINGS'] = HOLDINGS2
    ns2['DEALS'] = DEALS2
    ns2['_strength'].cache_clear()
    game = ns2
    return game
