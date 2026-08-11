"""Replay a solver solution over a generated level, as a sequence of grid states.

The push solver returns a full player-move string (`R`/`L`/`D`/`U`, walking steps
included, not just the pushes). Applying it one move at a time turns a solved
level into an animation of the level actually being played.

Token ids follow diffusion/dataset.py; see render.py.
"""

import os
import sys

import numpy as np

SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SOURCE_DIR)
sys.path.insert(0, os.path.join(SOURCE_DIR, 'metrics'))

from test_solver import boxoban_to_matrix, solve_push

GRID = 10
DELTA = {'R': (0, 1), 'L': (0, -1), 'D': (1, 0), 'U': (-1, 0)}


def solution_moves(puzzle, max_states=200000):
    """Full player-move string for a solvable level, or None.

    Uses the same solver call as the corpus measurement, so a level captioned
    solvable here is solvable by the paper's definition.
    """
    try:
        matrix, pos = boxoban_to_matrix(puzzle)
        if pos is None:
            return None
        path, _n, _stats = solve_push(matrix, pos, max_states=max_states,
                                      return_stats=True)
        return path or None
    except Exception:
        return None


def decode(tokens):
    """(100,) token ids -> (walls, goals, boxes, player)."""
    walls, goals, boxes, player = set(), set(), set(), None
    for i, t in enumerate(tokens):
        rc = divmod(i, GRID)
        t = int(t)
        if t == 0:
            walls.add(rc)
        if t in (4, 5, 6):
            goals.add(rc)
        if t in (3, 5):
            boxes.add(rc)
        if t in (2, 6):
            player = rc
    return walls, goals, boxes, player


def encode(walls, goals, boxes, player):
    """(walls, goals, boxes, player) -> (100,) token ids."""
    out = np.ones(GRID * GRID, dtype=np.uint8)  # floor
    for rc in walls:
        out[rc[0] * GRID + rc[1]] = 0
    for rc in goals:
        out[rc[0] * GRID + rc[1]] = 4
    for rc in boxes:
        out[rc[0] * GRID + rc[1]] = 5 if rc in goals else 3
    if player is not None:
        out[player[0] * GRID + player[1]] = 6 if player in goals else 2
    return out


def replay(tokens, moves):
    """Grid state after each move: a list of (100,) arrays, starting with the
    initial state, so len(result) == len(moves) + 1.

    Stops early and returns what it has if a move is illegal, rather than
    producing a corrupt animation from a bad path.
    """
    walls, goals, boxes, player = decode(tokens)
    states = [encode(walls, goals, boxes, player)]
    if player is None:
        return states

    for m in moves:
        d = DELTA.get(m)
        if d is None:
            break
        tgt = (player[0] + d[0], player[1] + d[1])
        if not (0 <= tgt[0] < GRID and 0 <= tgt[1] < GRID) or tgt in walls:
            break
        if tgt in boxes:
            dest = (tgt[0] + d[0], tgt[1] + d[1])
            if not (0 <= dest[0] < GRID and 0 <= dest[1] < GRID) \
                    or dest in walls or dest in boxes:
                break
            boxes.discard(tgt)
            boxes.add(dest)
        player = tgt
        states.append(encode(walls, goals, boxes, player))

    return states


def solved(tokens):
    """True once every box sits on a goal."""
    _w, goals, boxes, _p = decode(tokens)
    return bool(boxes) and boxes <= goals
