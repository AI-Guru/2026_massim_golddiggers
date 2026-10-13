"""Fair map generation: square maps of odd size, point-symmetric about the center.

Only the upper triangle of the map matrix is generated (cells with x > y, plus the half of the
diagonal before the center); every other cell is that half rotated by 180 degrees,
(x, y) -> (n-1-x, n-1-y). This is a rotation, not a mirror, so the two teams, which start in the
opposite corners (0, 0) and (n-1, n-1), see exactly the same world. The depot sits in the exact
center; the center and both corner blocks are kept free of obstacles, and any area cut off from
the depot is reopened (symmetrically).

- random:    obstacles scattered uniformly (MAPC 2006 Random1/2).
- labyrinth: concentric square walls around the depot, with holes (2006 Labyrinth).
- rows:      horizontal walls with alternating gates, bounded by the diagonal (2006 Labyrinth1,
             the map where random exploration fails).

Coordinates are (x, y) with (0, 0) top-left, as in the MAPC spec.
"""
from __future__ import annotations

import math
import random
from collections import deque

MAP_TYPES = ("random", "labyrinth", "rows")
CENTER_RADIUS = 2            # (2r+1)^2 free cells around the depot


def neighbours(x: int, y: int, w: int, h: int):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def reachable(start: tuple[int, int], obstacles: set, w: int, h: int) -> set:
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for nb in neighbours(*cell, w, h):
            if nb not in seen and nb not in obstacles:
                seen.add(nb)
                queue.append(nb)
    return seen


class Symmetry:
    def __init__(self, n: int):
        self.n, self.c = n, n // 2

    def rot(self, cell):
        return self.n - 1 - cell[0], self.n - 1 - cell[1]

    def in_half(self, cell) -> bool:
        """The generated half: strict upper triangle plus the diagonal before the center."""
        x, y = cell
        return x > y or (x == y and x < self.c)

    def half(self):
        return [(x, y) for y in range(self.n) for x in range(self.n) if self.in_half((x, y))]

    def complete(self, cells) -> set:
        """Upper-triangle cells -> the full symmetric set."""
        cells = {c for c in cells if self.in_half(c)}
        return cells | {self.rot(c) for c in cells}


def _protected(sym: Symmetry, team_size: int) -> set:
    n, c = sym.n, sym.c
    r = CENTER_RADIUS
    keep = {(x, y) for x in range(c - r, c + r + 1) for y in range(c - r, c + r + 1)}
    k = min(math.isqrt(team_size - 1) + 2, c - r)            # corner block side
    corner = {(x, y) for x in range(k) for y in range(k)}
    return keep | corner | {sym.rot(cell) for cell in corner}


def _random_half(rng, sym, n_obstacles, protected):
    n = sym.n
    depot = (sym.c, sym.c)
    candidates = [cell for cell in sym.half() if cell not in protected]
    rng.shuffle(candidates)
    obstacles: set = set()
    for cell in candidates:
        if len(obstacles) + 2 > n_obstacles:
            break
        pair = {cell, sym.rot(cell)}
        obstacles |= pair
        if len(reachable(depot, obstacles, n, n)) != n * n - len(obstacles):
            obstacles -= pair                                  # would cut something off
    return obstacles


def _labyrinth_half(rng, sym):
    n, c = sym.n, sym.c
    obstacles: set = set()
    for d in range(CENTER_RADIUS + 1, c, 3):
        lo, hi = c - d, c + d
        ring = {(x, y) for x in range(lo, hi + 1) for y in (lo, hi)}
        ring |= {(x, y) for y in range(lo, hi + 1) for x in (lo, hi)}
        half = sorted(cell for cell in ring if sym.in_half(cell))
        holes = set(rng.sample(half, min(len(half), 1 + rng.randrange(2))))
        obstacles |= {cell for cell in half if cell not in holes}
    return obstacles


def _rows_half(rng, sym):
    """Walls on rows 2, 5, 8, ... from the diagonal to the right edge, one gate per wall."""
    n = sym.n
    obstacles: set = set()
    gate_right = rng.random() < 0.5
    for y in range(2, sym.c - CENTER_RADIUS, 3):
        xs = list(range(y + 1, n))
        if len(xs) < 2:
            break
        gate = xs[-1] if gate_right else xs[0]
        obstacles |= {(x, y) for x in xs if x != gate}
        gate_right = not gate_right
    return obstacles


def _reconnect(sym: Symmetry, obstacles: set):
    """Open the shortest obstacle path (and its rotation) to every area not reachable from the depot."""
    n = sym.n
    depot = (sym.c, sym.c)
    while True:
        reach = reachable(depot, obstacles, n, n)
        cut_off = {(x, y) for x in range(n) for y in range(n)} - reach - obstacles
        if not cut_off:
            return
        prev = {cell: None for cell in reach}
        queue = deque(reach)
        target = None
        while queue and target is None:
            cell = queue.popleft()
            for nb in neighbours(*cell, n, n):
                if nb not in prev:
                    prev[nb] = cell
                    if nb in cut_off:
                        target = nb
                        break
                    queue.append(nb)
        cell = target
        while cell is not None:
            if cell in obstacles:
                obstacles.discard(cell)
                obstacles.discard(sym.rot(cell))
            cell = prev[cell]


def generate(map_type: str, rng: random.Random, size: int, n_obstacles: int,
             teams: list[str], team_size: int):
    """Returns (obstacles, depot, starts {team: [cells]}, sym)."""
    if size % 2 == 0:
        raise ValueError("map size must be odd")
    sym = Symmetry(size)
    protected = _protected(sym, team_size)
    if map_type == "random":
        half = _random_half(rng, sym, n_obstacles, protected)
    elif map_type == "labyrinth":
        half = _labyrinth_half(rng, sym)
    elif map_type == "rows":
        half = _rows_half(rng, sym)
    else:
        raise ValueError(f"unknown map type {map_type!r}, expected one of {MAP_TYPES}")
    obstacles = sym.complete(half) - protected
    _reconnect(sym, obstacles)
    depot = (sym.c, sym.c)

    # team 0 fills the cells nearest to (0, 0); team 1 gets exactly the rotated cells
    free = sorted(reachable(depot, obstacles, size, size) - {depot},
                  key=lambda cell: (cell[0] + cell[1], cell[1], cell[0]))
    first = free[:team_size]
    if len(first) < team_size or set(first) & {sym.rot(cell) for cell in first}:
        raise ValueError("map too small for the teams")
    starts = {teams[0]: first, teams[1]: [sym.rot(cell) for cell in first]}
    return obstacles, depot, starts, sym
