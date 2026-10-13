"""Pluggable strategies for agents/harness.py. Register new ones in REGISTRY."""
from __future__ import annotations

import random

from harness import MOVES, VIEW, Decision, Strategy, Team


class Greedy(Strategy):
    """Shared team map; nearest known unclaimed gold by BFS, deliver when full, explore unseen cells."""
    name = "greedy"

    def decide(self, p: dict, team: Team) -> Decision:
        with team.lock:
            return Decision(self._decide(p, team.memory, team.max_carry))

    @staticmethod
    def _decide(p, mem, max_carry):
        pos, items, name = tuple(p["pos"]), p["items"], p["agent"]
        cur = p["cells"].get("cur", [])
        near = {(pos[0] + VIEW[c][0], pos[1] + VIEW[c][1])
                for c, content in p["cells"].items() if "ally" in content or "enemy" in content}
        if pos == mem.depot:
            if items:
                return "drop"
            move, _ = mem.path_step(pos, lambda c: c != mem.depot, near)
            return move or "skip"
        if "gold" in cur and items < max_carry:
            mem.claims.pop(pos, None)
            return "pick"
        for cell, owner in list(mem.claims.items()):
            if owner == name:
                del mem.claims[cell]
        if items < max_carry:
            free_gold = {g for g in mem.gold if mem.claims.get(g) in (None, name)}
            if free_gold:
                move, target = mem.path_step(pos, lambda c: c in free_gold, near)
                if move:
                    # claim what we go for; with gold on board only take short detours
                    mem.claims[target] = name
                    if items == 0 or abs(target[0] - pos[0]) + abs(target[1] - pos[1]) <= 6:
                        return move
                    mem.claims.pop(target, None)
        if items:
            move, _ = mem.path_step(pos, lambda c: c == mem.depot, near, allow_depot=True)
            if move:
                return move
        move, _ = mem.path_step(pos, lambda c: c not in mem.seen, near)
        return move or random.choice(list(MOVES))


class RandomWalker(Strategy):
    """CLIMABot-like baseline: random walk, picks gold it steps on and delivers it."""
    name = "random"

    def decide(self, p: dict, team: Team) -> Decision:
        mem = team.memory
        pos, items = tuple(p["pos"]), p["items"]
        if pos == mem.depot:
            return Decision("drop" if items else random.choice(list(MOVES)))
        if "gold" in p["cells"].get("cur", []) and items < team.max_carry:
            return Decision("pick")
        if items:
            with team.lock:
                move, _ = mem.path_step(pos, lambda c: c == mem.depot, allow_depot=True)
            if move:
                return Decision(move)
        return Decision(random.choice(list(MOVES)))


from d3 import D3Strategy  # noqa: E402  (imports harness too)

REGISTRY: dict[str, type[Strategy]] = {"greedy": Greedy, "random": RandomWalker, "d3": D3Strategy}
