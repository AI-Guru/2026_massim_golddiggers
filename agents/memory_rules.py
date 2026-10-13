"""memory-rules: d3-memory without the model (ablation). Same memory, same facts, same goals;
the move is chosen by a rule instead of d3: the one that brings the goal distance down the most."""
from __future__ import annotations

import random
from collections import deque

from d3_memory import D3Memory
from harness import MOVES, Decision, Team


class MemoryRules(D3Memory):
    """Ablation: d3-memory's memory and facts, but a rule picks the move (no model)."""
    name = "memory-rules"

    def __init__(self, args):                    # no model server needed
        self.args = args
        self.recent: dict[str, deque] = {}
        self.targets: dict = {}
        self.batched = True
        self.version = "memory-rules (d3-memory facts, rule instead of d3)"

    @staticmethod
    def goal_keys(f: dict) -> list[str]:
        """The distance(s) the d3-memory prompt (v5) shows for the current goal, in order."""
        if f["items"] >= f["max_carry"]:
            return ["depot"]
        if f.get("my_gold_target") and f["now"].get("my_gold") is not None:
            return ["my_gold", "depot"] if f["items"] else ["my_gold"]
        if f["items"]:
            return ["depot"]
        if f.get("my_explore_target") and f["now"].get("my_explore") is not None:
            return ["my_explore"]
        return ["unknown"]

    def decide_batch(self, perceptions: list[dict], team: Team) -> dict[str, Decision]:
        with team.lock:
            facts = self._team_facts(perceptions, team)
        out = {}
        for p in perceptions:
            f = facts[p["agent"]]
            acts = f["actions"]
            if f["on_depot"] and f["items"] and "drop" in acts:
                action = "drop"
            elif "pick" in acts:
                action = "pick"
            else:
                key = self.goal_keys(f)[0]
                moves = {a: i.get(key) for a, i in acts.items() if a in MOVES}
                known = {a: d for a, d in moves.items() if d is not None}
                if known:
                    best = min(known.values())
                    action = random.choice(sorted(a for a, d in known.items() if d == best))
                else:
                    action = random.choice(sorted(moves) or ["skip"])
            self.recent.setdefault(p["agent"], deque(maxlen=4)).append(tuple(p["pos"]))
            out[p["agent"]] = Decision(action, 0, {"memory": {"target": self.targets.get(p["agent"])}})
        return out
