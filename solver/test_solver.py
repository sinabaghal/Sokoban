"""
Test Sokoban solver on Boxoban dataset puzzles.
Standalone implementation without pygame dependency.

Boxoban format:          Solver format:
  # = wall                 + = wall
  (space) = floor          - = floor
  @ = player               * = player
  $ = box                  @ = box
  . = goal                 X = goal
  * = box on goal          $ = box on goal
  + = player on goal       % = player on goal
"""

import os
import time
from collections import deque

# Default data path (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'boxoban-levels', 'medium', 'valid', '000.txt'))


def parse_boxoban_file(filepath: str) -> list[str]:
    """Parse a Boxoban file and return list of puzzles."""
    puzzles = []
    current_puzzle = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith(';'):
                if current_puzzle:
                    puzzles.append('\n'.join(current_puzzle))
                    current_puzzle = []
            elif line:
                current_puzzle.append(line)

        if current_puzzle:
            puzzles.append('\n'.join(current_puzzle))

    return puzzles


def boxoban_to_matrix(puzzle: str) -> tuple[list[list[str]], tuple[int, int]]:
    """
    Convert Boxoban format to internal matrix format.
    Returns (matrix, player_pos).

    Internal format:
      + = wall
      - = floor
      * = player
      @ = box
      X = goal
      $ = box on goal
      % = player on goal
    """
    mapping = {
        '#': '+',   # wall
        ' ': '-',   # floor
        '@': '*',   # player
        '$': '@',   # box
        '.': 'X',   # goal
        '*': '$',   # box on goal
        '+': '%',   # player on goal
    }

    lines = puzzle.strip().split('\n')
    matrix = []
    player_pos = None

    for i, line in enumerate(lines):
        row = []
        for j, c in enumerate(line):
            converted = mapping.get(c, c)
            row.append(converted)
            if converted in ('*', '%'):
                player_pos = (i, j)
        matrix.append(row)

    return matrix, player_pos


def get_state(matrix: list[list[str]]) -> str:
    """Convert matrix to string state."""
    return ''.join(''.join(row) for row in matrix)


def state_to_matrix(state: str, height: int, width: int) -> list[list[str]]:
    """Convert string state back to matrix."""
    return [[state[i * width + j] for j in range(width)] for i in range(height)]


def is_solved(state: str) -> bool:
    """Check if puzzle is solved (no boxes not on goals)."""
    return '@' not in state


def find_boxes_and_goals(state: str, width: int) -> tuple[list, list]:
    """Find positions of boxes and goals."""
    boxes = []
    goals = []
    for pos, char in enumerate(state):
        if char == '@':  # box not on goal
            boxes.append((pos // width, pos % width))
        elif char in 'X%':  # goal or player on goal
            goals.append((pos // width, pos % width))
    return boxes, goals


def is_deadlock(state: str, height: int, width: int) -> bool:
    """Check for simple deadlock conditions."""
    boxes, goals = find_boxes_and_goals(state, width)

    # Corner deadlock: box in corner not on goal
    for bx, by in boxes:
        box_idx = bx * width + by

        # Check all four corners
        corners = [
            (box_idx - 1, box_idx - width),      # left + up
            (box_idx + 1, box_idx - width),      # right + up
            (box_idx - 1, box_idx + width),      # left + down
            (box_idx + 1, box_idx + width),      # right + down
        ]

        for adj1, adj2 in corners:
            if 0 <= adj1 < len(state) and 0 <= adj2 < len(state):
                if state[adj1] == '+' and state[adj2] == '+':
                    return True

    return False


def can_move(state: str, height: int, width: int, pos: tuple, move: tuple) -> tuple[str, bool]:
    """
    Try to move player in given direction.
    Returns (new_state, success).
    """
    x, y = pos
    dx, dy = move
    new_x, new_y = x + dx, y + dy
    push_x, push_y = x + 2*dx, y + 2*dy

    # Convert to 1D indices
    curr_idx = x * width + y
    target_idx = new_x * width + new_y
    push_idx = push_x * width + push_y

    # Bounds check
    if not (0 <= new_x < height and 0 <= new_y < width):
        return None, False

    target_cell = state[target_idx]

    # Can't move into wall
    if target_cell == '+':
        return None, False

    new_state = list(state)

    # Moving into empty floor or goal
    if target_cell in '-X':
        # Update current position
        new_state[curr_idx] = 'X' if state[curr_idx] == '%' else '-'
        # Update target position
        new_state[target_idx] = '%' if target_cell == 'X' else '*'
        return ''.join(new_state), True

    # Moving into box
    if target_cell in '@$':
        # Check if we can push
        if not (0 <= push_x < height and 0 <= push_y < width):
            return None, False

        push_cell = state[push_idx]

        # Can't push into wall or another box
        if push_cell in '+@$':
            return None, False

        # Push the box
        # Update push target
        new_state[push_idx] = '$' if push_cell == 'X' else '@'
        # Update box's old position (now player)
        new_state[target_idx] = '%' if target_cell == '$' else '*'
        # Update player's old position
        new_state[curr_idx] = 'X' if state[curr_idx] == '%' else '-'

        return ''.join(new_state), True

    return None, False


def solve_bfs(matrix: list[list[str]], player_pos: tuple, max_states: int = 100000) -> tuple[str, int]:
    """
    BFS solver.
    Returns (solution_path, depth) or (None, -1) if unsolvable.
    """
    height = len(matrix)
    width = len(matrix[0])

    initial_state = get_state(matrix)

    if is_solved(initial_state):
        return ('', 0)

    seen = {initial_state}
    q = deque([(initial_state, player_pos, '')])

    moves = [(0, 1, 'R'), (0, -1, 'L'), (1, 0, 'D'), (-1, 0, 'U')]
    states_explored = 0

    while q and states_explored < max_states:
        state, pos, path = q.popleft()
        states_explored += 1

        for dx, dy, direction in moves:
            new_state, success = can_move(state, height, width, pos, (dx, dy))

            if not success or new_state is None:
                continue

            if new_state in seen:
                continue

            if is_deadlock(new_state, height, width):
                continue

            seen.add(new_state)
            new_path = path + direction
            new_pos = (pos[0] + dx, pos[1] + dy)

            if is_solved(new_state):
                return (new_path, len(new_path))

            q.append((new_state, new_pos, new_path))

    return (None, -1)


# ---------------------------------------------------------------------------
# Push-based solver
#
# solve_bfs above branches on the 4 player *moves* every step, so it wastes its
# state budget shuffling the player around without pushing anything. This solver
# instead branches only on box *pushes*: at each node it flood-fills the squares
# the player can currently reach, and the only "moves" are (reach a box, push
# it). That collapses the branching factor from 4-per-step to ~(#boxes x 4) and
# lets a precomputed static deadlock table prune dead squares. Same interface as
# solve_bfs -- returns (move_path_string, num_moves) or (None, -1) -- so it's a
# drop-in replacement (the returned path is reconstructed as real U/D/L/R player
# moves, so GIF rendering still works).
# ---------------------------------------------------------------------------

_DIRS = [(0, 1, 'R'), (0, -1, 'L'), (1, 0, 'D'), (-1, 0, 'U')]


def _parse_matrix(matrix):
    """Extract walls/goals/boxes/player as coordinate sets from the internal matrix."""
    walls, goals, boxes = set(), set(), set()
    player = None
    for r, row in enumerate(matrix):
        for c, ch in enumerate(row):
            if ch == '+':
                walls.add((r, c))
            elif ch == 'X':          # empty goal
                goals.add((r, c))
            elif ch == '@':          # box, not on goal
                boxes.add((r, c))
            elif ch == '$':          # box on goal
                boxes.add((r, c))
                goals.add((r, c))
            elif ch == '*':          # player, not on goal
                player = (r, c)
            elif ch == '%':          # player on goal
                player = (r, c)
                goals.add((r, c))
    return walls, goals, boxes, player


def _compute_live_squares(walls, goals, height, width):
    """Squares from which a box can still reach some goal (via pushes).

    Computed by reverse-reachability: start at every goal and 'pull' a box
    outward one step at a time -- a box can be pulled from `s` to `frm = s + d`
    only if both `frm` and the square beyond it (where the puller stands) are
    non-wall. Any square NOT reached is a simple-deadlock square: a box there can
    never reach a goal, so any push landing there can be pruned immediately.
    """
    live = set(goals)
    q = deque(goals)
    while q:
        r, c = q.popleft()
        for dr, dc, _ in _DIRS:
            frm = (r + dr, c + dc)
            beyond = (r + 2 * dr, c + 2 * dc)
            if frm in live:
                continue
            if not (0 <= frm[0] < height and 0 <= frm[1] < width) or frm in walls:
                continue
            if not (0 <= beyond[0] < height and 0 <= beyond[1] < width) or beyond in walls:
                continue
            live.add(frm)
            q.append(frm)
    return live


def _player_reachable(player, walls, boxes, height, width):
    """Flood-fill the set of squares the player can reach (blocked by walls/boxes)."""
    seen = {player}
    q = deque([player])
    while q:
        r, c = q.popleft()
        for dr, dc, _ in _DIRS:
            nb = (r + dr, c + dc)
            if (0 <= nb[0] < height and 0 <= nb[1] < width
                    and nb not in walls and nb not in boxes and nb not in seen):
                seen.add(nb)
                q.append(nb)
    return seen


def _walk_path(start, goal, walls, boxes, height, width):
    """Shortest U/D/L/R move string for the player to walk start->goal (no pushing)."""
    if start == goal:
        return ''
    seen = {start}
    q = deque([(start, '')])
    while q:
        (r, c), path = q.popleft()
        for dr, dc, ch in _DIRS:
            nb = (r + dr, c + dc)
            if nb == goal:
                return path + ch
            if (0 <= nb[0] < height and 0 <= nb[1] < width
                    and nb not in walls and nb not in boxes and nb not in seen):
                seen.add(nb)
                q.append((nb, path + ch))
    return None


def _reconstruct_moves(player, boxes, walls, pushes, height, width):
    """Turn a push list [(box, (dr,dc,ch), dest), ...] into a full player-move string."""
    moves = []
    boxes = set(boxes)
    for box, (dr, dc, ch), dest in pushes:
        stand = (box[0] - dr, box[1] - dc)
        walk = _walk_path(player, stand, walls, boxes, height, width)
        if walk is None:
            return None  # shouldn't happen if pushes are legal
        moves.append(walk)
        moves.append(ch)          # the push step itself moves the player onto the box
        boxes.discard(box)
        boxes.add(dest)
        player = box
    return ''.join(moves)


def solve_push(matrix: list, player_pos: tuple, max_states: int = 200000,
               return_stats: bool = False) -> tuple:
    """Push-based BFS solver. Returns (move_path, num_moves) or (None, -1).

    If return_stats=True, returns a 3rd element: a dict with search-effort
    signals usable as difficulty proxies --
        'states_expanded': push-nodes popped from the queue before resolving
                           (~ A* closed-list length; the validated difficulty proxy)
        'pushes':          length of the solution in box pushes (-1 if unsolvable)
    """
    height = len(matrix)
    width = len(matrix[0])

    def _ret(path, n, states, pushes):
        if return_stats:
            return (path, n, {'states_expanded': states, 'pushes': pushes})
        return (path, n)

    walls, goals, boxes, player = _parse_matrix(matrix)
    if player is None:
        player = player_pos
    if player is None:
        return _ret(None, -1, 0, -1)

    boxes = frozenset(boxes)
    if all(b in goals for b in boxes):
        return _ret('', 0, 0, 0)

    live = _compute_live_squares(walls, goals, height, width)
    # If any box is already stuck on a dead square, it's unsolvable up front.
    if any(b not in live for b in boxes):
        return _ret(None, -1, 0, -1)

    # Captured for path reconstruction, since the loop below rebinds boxes/player.
    player_start = player
    boxes_start = boxes

    start_region = _player_reachable(player, walls, boxes, height, width)
    start_key = (boxes, min(start_region))
    seen = {start_key}
    q = deque([(boxes, player, [])])
    states = 0

    while q and states < max_states:
        boxes, player, pushes = q.popleft()
        states += 1
        region = _player_reachable(player, walls, boxes, height, width)

        for box in boxes:
            br, bc = box
            for dr, dc, ch in _DIRS:
                stand = (br - dr, bc - dc)          # player must stand opposite the push direction
                dest = (br + dr, bc + dc)            # box's destination
                if stand not in region:
                    continue
                if not (0 <= dest[0] < height and 0 <= dest[1] < width):
                    continue
                if dest in walls or dest in boxes:
                    continue
                if dest not in live:                 # static deadlock prune
                    continue

                new_boxes = frozenset(boxes - {box} | {dest})
                new_player = box                     # player ends where the box was
                new_region = _player_reachable(new_player, walls, new_boxes, height, width)
                key = (new_boxes, min(new_region))
                if key in seen:
                    continue
                seen.add(key)

                new_pushes = pushes + [(box, (dr, dc, ch), dest)]
                if all(b in goals for b in new_boxes):
                    path = _reconstruct_moves(player_start, boxes_start, walls,
                                              new_pushes, height, width)
                    if path is None:
                        # solvable; path reconstruction fell through
                        return _ret('', len(new_pushes), states, len(new_pushes))
                    return _ret(path, len(path), states, len(new_pushes))
                q.append((new_boxes, new_player, new_pushes))

    return _ret(None, -1, states, -1)


def test_puzzle(puzzle: str, puzzle_id: int, verbose: bool = False) -> dict:
    """Test a single puzzle and return results."""
    result = {
        'id': puzzle_id,
        'solvable': False,
        'solution': None,
        'depth': -1,
        'time': 0,
        'error': None
    }

    try:
        matrix, player_pos = boxoban_to_matrix(puzzle)

        if player_pos is None:
            result['error'] = 'No player found'
            return result

        start = time.time()
        solution, depth = solve_bfs(matrix, player_pos, max_states=200000)
        elapsed = time.time() - start

        result['time'] = elapsed
        result['solvable'] = solution is not None
        result['solution'] = solution
        result['depth'] = depth

        if verbose:
            status = "SOLVED" if solution else "UNSOLVED"
            print(f"Puzzle {puzzle_id:3d}: {status:8s} depth={depth:3d}  time={elapsed:.3f}s", end='')
            if solution and len(solution) <= 30:
                print(f"  path={solution}", end='')
            print()

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Puzzle {puzzle_id:3d}: ERROR - {e}")

    return result


def main():
    """Test solver on Boxoban medium puzzles."""

    # Path to Boxoban data
    data_file = DEFAULT_DATA_PATH

    print(f"Loading puzzles from {data_file}...")
    puzzles = parse_boxoban_file(data_file)
    print(f"Loaded {len(puzzles)} puzzles\n")

    # Test first N puzzles
    n_test = 10
    print(f"Testing first {n_test} puzzles...\n")

    results = []
    for i, puzzle in enumerate(puzzles[:n_test]):
        result = test_puzzle(puzzle, i, verbose=True)
        results.append(result)

    # Summary
    solved = sum(1 for r in results if r['solvable'])
    errors = sum(1 for r in results if r['error'])
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results)
    solved_depths = [r['depth'] for r in results if r['depth'] > 0]
    avg_depth = sum(solved_depths) / len(solved_depths) if solved_depths else 0

    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total tested:  {len(results)}")
    print(f"Solved:        {solved} ({100*solved/len(results):.1f}%)")
    print(f"Unsolved:      {len(results) - solved - errors}")
    print(f"Errors:        {errors}")
    print(f"Total time:    {total_time:.2f}s")
    print(f"Avg time:      {avg_time:.3f}s")
    print(f"Avg depth:     {avg_depth:.1f} moves")


if __name__ == '__main__':
    main()
