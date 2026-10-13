"""d3-memory: the d3 strategy plus per-miner memory of targets (the model still makes every move).

Differences to the d3 baseline:
- remembered gold target: a miner keeps its gold until it is picked up or gone (d3 re-matches
  every step, so targets flip when teammates move); only miners without a target are matched,
  closest pairs first, one miner per nugget;
- split exploration: an exploring miner gets its own unexplored target cell, at least
  EXPLORE_SPACING steps (Manhattan) from its teammates' targets, and keeps it until it has been seen;
- prompt v5: the goal sentence names the remembered target and moves refer to it;
- distances route around the miners next to an agent (queues of full miners jammed the lane to
  the depot), and a miner that already carries gold only targets gold within CARRY_DETOUR steps.
Always batched (it needs the whole team per step).
"""
from __future__ import annotations

from d3 import PROMPTS, D3Strategy
from harness import Team

EXPLORE_SPACING = 8
CARRY_DETOUR = 6          # a miner already carrying gold only targets gold this close (else: deliver)


class D3Memory(D3Strategy):
    """d3 plus memory: remembered gold targets, split exploration targets, prompt v5."""
    name = "d3-memory"

    @staticmethod
    def add_arguments(parser):
        pass                              # shares the d3 options (--d3-url, --d3-model, --d3-sample)

    def __init__(self, args):
        super().__init__(args)
        self.render = PROMPTS["v5"]
        self.version = f"d3-memory {self.model} prompt v5"
        self.batched = True
        self.targets: dict[str, tuple[str, tuple]] = {}      # agent -> ("gold"|"explore", cell)
        print("d3-memory: prompt v5, remembered gold and exploration targets", flush=True)

    def _decision(self, p, answer, tokens, facts, rendered, extra):
        d = super()._decision(p, answer, tokens, facts, rendered, extra)
        d.info["prompt_variant"] = "v5"
        d.info["memory"] = {"target": self.targets.get(p["agent"])}
        return d

    def _team_facts(self, perceptions: list[dict], team: Team) -> dict[str, dict]:
        mem = team.memory
        by_name = {p["agent"]: p for p in perceptions}
        dist_from = {}

        def dist(agent):
            if agent not in dist_from:
                pos = tuple(by_name[agent]["pos"])
                dist_from[agent] = mem.distances(lambda c, pos=pos: c == pos, allow_depot=False)
            return dist_from[agent]

        # 1. keep remembered targets that are still valid
        taken = set()
        for name, p in by_name.items():
            target = self.targets.get(name)
            room = p["items"] < team.max_carry
            if target is None:
                continue
            kind, cell = target
            pos = tuple(p["pos"])
            near_enough = p["items"] == 0 or dist(name).get(cell, 10**6) <= CARRY_DETOUR
            if kind == "gold" and room and near_enough and cell in mem.gold and cell != pos and cell not in taken:
                taken.add(cell)
            elif kind == "explore" and p["items"] == 0 and cell not in mem.seen and cell in dist(name):
                taken.add(cell)
            else:
                del self.targets[name]
        # a gold target replaces an exploration target as soon as gold is known
        free_gold = [g for g in mem.gold if g not in taken]
        seekers = [n for n, p in by_name.items() if p["items"] < team.max_carry
                   and self.targets.get(n, ("explore",))[0] != "gold"]
        # 2. match miners without a gold target to free gold, closest pairs first
        pairs = sorted((dist(n)[g], n, g) for n in seekers for g in free_gold
                       if g in dist(n) and g != tuple(by_name[n]["pos"])
                       and (by_name[n]["items"] == 0 or dist(n)[g] <= CARRY_DETOUR))
        for _, n, g in pairs:
            if self.targets.get(n, ("explore",))[0] != "gold" and g not in taken:
                if n in self.targets:
                    taken.discard(self.targets[n][1])
                self.targets[n] = ("gold", g)
                taken.add(g)
        # 3. empty miners without any target explore: their own unseen cell, spaced apart
        explore_cells = [cell for kind, cell in self.targets.values() if kind == "explore"]
        for n, p in by_name.items():
            if n in self.targets or p["items"] != 0:
                continue
            options = sorted((d, c) for c, d in dist(n).items() if c not in mem.seen)
            spaced = [(d, c) for d, c in options
                      if all(abs(c[0] - e[0]) + abs(c[1] - e[1]) >= EXPLORE_SPACING for e in explore_cells)]
            pick = (spaced or options or [None])[0]
            if pick:
                self.targets[n] = ("explore", pick[1])
                explore_cells.append(pick[1])

        facts = {}
        for n, p in by_name.items():
            kind, cell = self.targets.get(n, (None, None))
            facts[n] = self._facts(p, team, cell if kind == "gold" else None,
                                   cell if kind == "explore" else None, route_around=True)
        return facts
