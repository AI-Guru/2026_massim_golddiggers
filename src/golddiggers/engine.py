"""Gold Miners simulation (MAPC 2006/2007 "Gold Rush" rules), stepped explicitly.

Agents queue one action per step with act(); step() resolves all queued actions at once
(missing actions count as skip). Rules follow the 2007 spec (ac07-scenario-extended.txt):

- 3x3 local view, absolute position, number of carried items; any cell may be reported as
  "unknown" (information distortion), but never wrongly.
- Actions: skip up down left right pick drop mark unmark. A failed action is a skip.
- Fatigue: P = p_sim + (p_max - p_sim) / max_carry * items, for both action failure and
  information distortion.
- Pushing: moving into an agent that skips pushes it one cell, if the cell behind is free.
- Depot: only enterable while carrying gold; drop must be the next action (drops all items,
  scores one point each); then leave. depot_policy decides when an agent is teleported away,
  teleport_target where to (see _teleport_target); "original" + "random" is MAPC 2006/07.
- Gold generation as in the 2006 server: every `gold_frequency` steps, with probability
  `gold_probability`, `gold_number` items appear on random empty cells.

2006 rules are a parameter set: team_size=4, max_carry=1, pushing=False, p_max=p_sim.

Interpretations where the spec is silent (documented in README):
- Depot punishments look at the agent's requested action, so a random fatigue failure on the
  depot is not punished.
- Mutual pushes (two adjacent agents moving into each other) pick the pusher by coin flip.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from . import maps

ACTIONS = ("skip", "up", "down", "left", "right", "pick", "drop", "mark", "unmark")
MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
VIEW = {"nw": (-1, -1), "n": (0, -1), "ne": (1, -1), "w": (-1, 0), "cur": (0, 0),
        "e": (1, 0), "sw": (-1, 1), "s": (0, 1), "se": (1, 1)}
MARK_LEN = 5
DEPOT_POLICIES = ("original", "leave_or_teleport", "teleport_on_drop")
TELEPORT_TARGETS = ("random", "ring", "home")
TEAM_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _.-")


@dataclass
class GameConfig:
    map_type: str = "random"
    width: int = 31             # maps are square with an odd size: width == height
    height: int = 31
    obstacles: int = 90          # used by the random map only
    initial_gold: int = 40
    steps: int = 500
    teams: list[str] = field(default_factory=lambda: ["red", "blue"])
    team_size: int = 6
    max_carry: int = 3
    pushing: bool = True
    p_fail: float = 0.05         # P_sim for action failure (spec: <= 0.2)
    p_fail_max: float = 0.3      # at max_carry items (spec: <= 0.5)
    p_info: float = 0.05         # P_sim for information distortion
    p_info_max: float = 0.3
    gold_frequency: int = 10
    gold_probability: float = 0.8
    gold_number: int = 2
    depot_policy: str = "leave_or_teleport"   # original | leave_or_teleport | teleport_on_drop
    teleport_target: str = "ring"             # random | ring | home
    teleport_radius: int = 6                  # ring: walking distance from the depot
    seed: int | None = None

    def validate(self):
        if self.map_type not in maps.MAP_TYPES:
            raise ValueError(f"map_type must be one of {maps.MAP_TYPES}")
        if self.width != self.height or self.width % 2 == 0 or not 9 <= self.width <= 99:
            raise ValueError("maps are square with an odd size: width == height, odd, 9..99")
        if len(self.teams) != 2 or len(set(self.teams)) != 2:
            raise ValueError("exactly two distinct team names are required")
        if not all(t.isalnum() for t in self.teams):
            raise ValueError("team names must be alphanumeric")
        if not 1 <= self.team_size <= 20:
            raise ValueError("team_size must be within 1..20")
        if self.max_carry < 1 or self.steps < 1 or self.gold_frequency < 1:
            raise ValueError("max_carry, steps and gold_frequency must be >= 1")
        if self.depot_policy not in DEPOT_POLICIES:
            raise ValueError(f"depot_policy must be one of {DEPOT_POLICIES}")
        if self.teleport_target not in TELEPORT_TARGETS:
            raise ValueError(f"teleport_target must be one of {TELEPORT_TARGETS}")
        if self.teleport_radius < 1:
            raise ValueError("teleport_radius must be >= 1")
        for name in ("p_fail", "p_fail_max", "p_info", "p_info_max", "gold_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within 0..1")


@dataclass
class Agent:
    name: str
    team: str
    x: int
    y: int
    items: int = 0
    in_depot_must_drop: bool = False     # entered the depot, next action must be drop
    in_depot_must_leave: bool = False    # dropped in the depot, must leave asap
    last_action: str = "skip"
    last_result: str = "none"            # ok | failed | impossible | pushed | teleported | none


class Game:
    def __init__(self, config: GameConfig):
        config.validate()
        self.config = config
        self.rng = random.Random(config.seed)
        c = config
        self.obstacles, self.depot, starts, self.sym = maps.generate(
            c.map_type, self.rng, c.width, c.obstacles, c.teams, c.team_size)
        self.starts = starts
        self.depot_distance = self._walking_distances(self.depot)
        self.team_names = {t: t for t in c.teams}
        self.agents: dict[str, Agent] = {}
        for team in c.teams:
            for i, (x, y) in enumerate(starts[team], start=1):
                name = f"{team}{i}"
                self.agents[name] = Agent(name, team, x, y)
        self.gold: set = set()
        self.marks: dict = {}
        self.score = {t: 0 for t in c.teams}
        self.step_no = 0
        self.queued: dict[str, tuple[str, str | None]] = {}
        self.stats = {"generated_gold": 0, "actions": 0, "failed": 0, "impossible": 0,
                      "pushes": 0, "teleports": 0, "picks": 0, "deliveries": 0}
        self.history: list[dict] = []
        self._initial_gold(c.initial_gold)
        self._perceptions = self._compute_perceptions()
        self._record()

    # ------------------------------------------------------------------ helpers
    @property
    def finished(self) -> bool:
        return self.step_no >= self.config.steps

    def _occupied(self) -> dict:
        return {(a.x, a.y): a for a in self.agents.values()}

    def _inside(self, x, y) -> bool:
        return 0 <= x < self.config.width and 0 <= y < self.config.height

    def _empty_cells(self) -> list:
        occ = self._occupied()
        return [(x, y) for x in range(self.config.width) for y in range(self.config.height)
                if (x, y) not in self.obstacles and (x, y) not in self.gold
                and (x, y) not in occ and (x, y) != self.depot]

    def _initial_gold(self, n: int):
        """Initial gold is part of the map: placed in rotated pairs (an odd count rounds down)."""
        empty = [cell for cell in self._empty_cells() if self.sym.in_half(cell)]
        for cell in self.rng.sample(empty, min(n // 2, len(empty))):
            self.gold |= {cell, self.sym.rot(cell)}
            self.stats["generated_gold"] += 2

    def _spawn_gold(self, n: int):
        empty = self._empty_cells()
        for cell in self.rng.sample(empty, min(n, len(empty))):
            self.gold.add(cell)
            self.stats["generated_gold"] += 1

    def _prob(self, base: float, top: float, items: int) -> float:
        return base + (top - base) / self.config.max_carry * items

    def _can_enter(self, agent: Agent, cell) -> bool:
        """Static check: inside, no obstacle, depot only with gold."""
        if not self._inside(*cell) or cell in self.obstacles:
            return False
        return not (cell == self.depot and agent.items == 0)

    def _winner(self):
        a, b = self.config.teams
        if self.score[a] == self.score[b]:
            return None
        return a if self.score[a] > self.score[b] else b

    # ------------------------------------------------------------------ agent API
    def act(self, name: str, action: str, param: str | None = None):
        if name not in self.agents:
            raise KeyError(name)
        if self.finished:
            raise RuntimeError("game is finished")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}, expected one of {ACTIONS}")
        if action == "mark":
            param = str(param or "")[:MARK_LEN]
        self.queued[name] = (action, param if action == "mark" else None)

    def perceive(self, name: str) -> dict:
        if name not in self.agents:
            raise KeyError(name)
        return self._perceptions[name]

    def _compute_perceptions(self) -> dict:
        """Computed once per step so repeated perceive calls agree (distortion is sampled once)."""
        occ = self._occupied()
        out = {}
        for a in self.agents.values():
            p_info = self._prob(self.config.p_info, self.config.p_info_max, a.items)
            cells = {}
            for cid, (dx, dy) in VIEW.items():
                x, y = a.x + dx, a.y + dy
                if not self._inside(x, y):
                    continue
                if self.rng.random() < p_info:
                    cells[cid] = ["unknown"]
                    continue
                content = []
                if (x, y) in self.obstacles:
                    content.append("obstacle")
                other = occ.get((x, y))
                if other is not None and other is not a:
                    content.append("ally" if other.team == a.team else "enemy")
                if (x, y) in self.gold:
                    content.append("gold")
                if (x, y) == self.depot:
                    content.append("depot")
                if (x, y) in self.marks:
                    content.append({"mark": self.marks[(x, y)]})
                cells[cid] = content or ["empty"]
            out[a.name] = {
                "agent": a.name, "team": a.team, "step": self.step_no,
                "steps": self.config.steps, "finished": self.finished,
                "pos": [a.x, a.y], "items": a.items, "cells": cells,
                "last_action": a.last_action, "last_result": a.last_result,
            }
            if self.finished:
                w = self._winner()
                out[a.name]["result"] = "draw" if w is None else ("win" if w == a.team else "lose")
                out[a.name]["score"] = self.score[a.team]
        return out

    def sim_info(self, name: str) -> dict:
        """What SIM-START told the agent: grid size, depot, steps, opponent."""
        a = self.agents[name]
        opponent = next(t for t in self.config.teams if t != a.team)
        return {"agent": name, "team": a.team, "opponent": opponent,
                "teammates": [n for n, b in self.agents.items() if b.team == a.team and n != name],
                "width": self.config.width, "height": self.config.height,
                "depot": list(self.depot), "steps": self.config.steps,
                "max_carry": self.config.max_carry}

    # ------------------------------------------------------------------ the step
    def step(self) -> dict:
        if self.finished:
            raise RuntimeError("game is finished")
        c = self.config
        occ = self._occupied()
        requested = {n: self.queued.get(n, ("skip", None)) for n in self.agents}
        self.queued = {}
        action = {n: r[0] for n, r in requested.items()}
        result = {n: "ok" for n in self.agents}
        events: list[dict] = []

        # 1. impossible actions -> skip
        for n, a in self.agents.items():
            act = action[n]
            here = (a.x, a.y)
            impossible = False
            if act in MOVES:
                dx, dy = MOVES[act]
                impossible = not self._can_enter(a, (a.x + dx, a.y + dy))
            elif act == "pick":
                impossible = here not in self.gold or a.items >= c.max_carry
            elif act == "drop":
                impossible = a.items == 0 or (here != self.depot and here in self.gold)
            elif act == "mark":
                impossible = here == self.depot
            elif act == "unmark":
                impossible = here not in self.marks
            if impossible:
                action[n], result[n] = "skip", "impossible"

        # 2. fatigue failures -> skip
        for n, a in self.agents.items():
            if action[n] != "skip":
                if self.rng.random() < self._prob(c.p_fail, c.p_fail_max, a.items):
                    action[n], result[n] = "skip", "failed"

        # 3. moves into cells free at step start: one random winner per cell
        targets: dict = {}
        for n, a in self.agents.items():
            if action[n] in MOVES:
                dx, dy = MOVES[action[n]]
                t = (a.x + dx, a.y + dy)
                if t not in occ:
                    targets.setdefault(t, []).append(n)
        moved: set = set()
        for t, names in targets.items():
            winner = self.rng.choice(names)
            for n in names:
                if n != winner:
                    action[n], result[n] = "skip", "failed"
            self.agents[winner].x, self.agents[winner].y = t
            moved.add(winner)

        # 4. pushes
        if c.pushing:
            self._resolve_pushes(action, result, moved, events)

        # 5. remaining moves -> skip
        for n in self.agents:
            if action[n] in MOVES and n not in moved:
                action[n], result[n] = "skip", "failed"

        # 6. non-move actions
        for n, a in self.agents.items():
            here = (a.x, a.y)
            act = action[n]
            if act == "pick":
                self.gold.discard(here)
                a.items += 1
                self.stats["picks"] += 1
            elif act == "drop":
                if here == self.depot:
                    self.score[a.team] += a.items
                    self.stats["deliveries"] += a.items
                    events.append({"type": "delivery", "agent": n, "items": a.items})
                    a.items = 0
                else:
                    a.items -= 1
                    self.gold.add(here)
            elif act == "mark":
                self.marks[here] = requested[n][1] or ""
            elif act == "unmark":
                self.marks.pop(here, None)

        self._depot_rules(requested, action, result, occ, events)

        for n, a in self.agents.items():
            a.last_action, a.last_result = requested[n][0], result[n]
            if requested[n][0] != "skip":
                self.stats["actions"] += 1
                if result[n] == "failed":
                    self.stats["failed"] += 1
                elif result[n] == "impossible":
                    self.stats["impossible"] += 1

        self.step_no += 1
        if self.step_no % c.gold_frequency == 0 and self.rng.random() < c.gold_probability:
            self._spawn_gold(c.gold_number)
        self._perceptions = self._compute_perceptions()
        self._record()
        return {"step": self.step_no, "finished": self.finished, "score": dict(self.score),
                "actions": {n: {"requested": requested[n][0], "result": result[n]}
                            for n in self.agents},
                "events": events}

    def _resolve_pushes(self, action, result, moved, events):
        occ = self._occupied()     # after step 3
        # mutual pushes: two adjacent agents moving into each other -> coin flip
        for n, a in self.agents.items():
            if action[n] not in MOVES or n in moved:
                continue
            dx, dy = MOVES[action[n]]
            other = occ.get((a.x + dx, a.y + dy))
            if other is None or other.name in moved or action[other.name] not in MOVES:
                continue
            odx, ody = MOVES[action[other.name]]
            if (other.x + odx, other.y + ody) == (a.x, a.y):
                loser = self.rng.choice([n, other.name])
                action[loser], result[loser] = "skip", "failed"
        pushers: dict = {}
        for n, a in self.agents.items():
            if action[n] not in MOVES or n in moved:
                continue
            dx, dy = MOVES[action[n]]
            victim = occ.get((a.x + dx, a.y + dy))
            if victim is not None and action[victim.name] == "skip":
                pushers.setdefault(victim.name, []).append(n)
        order = list(pushers)
        self.rng.shuffle(order)
        for victim_name in order:
            victim = self.agents[victim_name]
            pusher = self.agents[self.rng.choice(pushers[victim_name])]
            dx, dy = MOVES[action[pusher.name]]
            dest = (victim.x + dx, victim.y + dy)
            if (dest in occ or not self._can_enter(victim, dest)
                    or not self._can_enter(pusher, (victim.x, victim.y))
                    or (pusher.x, pusher.y) != (victim.x - dx, victim.y - dy)):
                continue
            del occ[(victim.x, victim.y)], occ[(pusher.x, pusher.y)]
            pusher.x, pusher.y = victim.x, victim.y
            victim.x, victim.y = dest
            occ[(pusher.x, pusher.y)], occ[dest] = pusher, victim
            moved.add(pusher.name)
            result[victim_name] = "pushed"
            self.stats["pushes"] += 1
            events.append({"type": "push", "pusher": pusher.name, "victim": victim_name})

    def _depot_rules(self, requested, action, result, occ_before, events):
        for n, a in self.agents.items():
            req = requested[n][0]
            teleport = False
            if (a.x, a.y) != self.depot:           # left (or was pushed out of) the depot
                a.in_depot_must_drop = a.in_depot_must_leave = False
                continue
            policy = self.config.depot_policy
            if a.in_depot_must_drop:
                if action[n] == "drop":
                    a.in_depot_must_drop, a.in_depot_must_leave = False, True
                    teleport = policy == "teleport_on_drop"
                elif req != "drop":
                    teleport = True
            elif a.in_depot_must_leave and policy == "leave_or_teleport":
                teleport = True                    # still here one step after dropping
            elif a.in_depot_must_leave:
                could_move = any(
                    self._inside(*cell) and cell not in self.obstacles and cell not in occ_before
                    for cell in ((a.x + dx, a.y + dy) for dx, dy in MOVES.values()))
                if could_move and req not in MOVES:
                    teleport = True
            else:
                a.in_depot_must_drop = True        # entered this step (or was pushed in)
            if teleport:
                self._teleport(a)
                result[n] = "teleported"
                events.append({"type": "teleport", "agent": n})

    def _walking_distances(self, start) -> dict:
        n = self.config.width
        dist = {start: 0}
        queue = [start]
        for cell in queue:
            for nb in maps.neighbours(*cell, n, n):
                if nb not in dist and nb not in self.obstacles:
                    dist[nb] = dist[cell] + 1
                    queue.append(nb)
        return dist

    def _teleport_target(self, a: Agent):
        """random: anywhere (MAPC original). ring: random free cell at walking distance
        teleport_radius from the depot (or the nearest distance beyond that has a free cell), the
        same penalty for everyone. home: the team's free start cell nearest to its corner."""
        occ = self._occupied()
        free = [cell for cell in self.depot_distance
                if cell not in occ and cell != self.depot]
        target = self.config.teleport_target
        if target == "home":
            home = [cell for cell in self.starts[a.team] if cell not in occ]
            if home:
                return home[0]
            corner = self.starts[a.team][0]
            return min(free, key=lambda c: (abs(c[0] - corner[0]) + abs(c[1] - corner[1]), c))
        if target == "ring":
            far = [c for c in free if self.depot_distance[c] >= self.config.teleport_radius]
            if far:
                d = min(self.depot_distance[c] for c in far)
                return self.rng.choice(sorted(c for c in far if self.depot_distance[c] == d))
        return self.rng.choice(sorted(free))

    def set_team_name(self, team: str, name: str):
        if team not in self.team_names:
            raise KeyError(team)
        name = name.strip()
        if not 1 <= len(name) <= 24 or not set(name) <= TEAM_NAME_CHARS:
            raise ValueError("team name: 1-24 characters out of letters, digits, space, _ . -")
        self.team_names[team] = name

    def _teleport(self, a: Agent):
        a.x, a.y = self._teleport_target(a)
        a.in_depot_must_drop = a.in_depot_must_leave = False
        self.stats["teleports"] += 1

    # ------------------------------------------------------------------ views
    def _record(self):
        carried = {t: 0 for t in self.config.teams}
        for a in self.agents.values():
            carried[a.team] += a.items
        self.history.append({"step": self.step_no, "score": dict(self.score),
                             "carried": carried, "gold_on_map": len(self.gold)})

    def state(self) -> dict:
        """Full god's-eye state for the frontend."""
        c = self.config
        return {
            "config": asdict(c),
            "step": self.step_no, "finished": self.finished, "winner": self._winner(),
            "depot": list(self.depot),
            "obstacles": sorted(map(list, self.obstacles)),
            "gold": sorted(map(list, self.gold)),
            "marks": [{"pos": list(p), "value": v} for p, v in sorted(self.marks.items())],
            "agents": [{"name": a.name, "team": a.team, "pos": [a.x, a.y], "items": a.items,
                        "last_action": a.last_action, "last_result": a.last_result,
                        "queued": self.queued.get(a.name, (None,))[0]}
                       for a in self.agents.values()],
            "score": dict(self.score), "stats": dict(self.stats),
            "team_names": dict(self.team_names),
            "history": self.history,
        }
