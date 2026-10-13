#!/usr/bin/env python3
"""Common agent harness for Gold Diggers: one loop for every strategy, only the strategy is pluggable.

Each step, every agent runs perceive -> decide -> act in its own thread (so decision-model requests
overlap), then the harness calls /api/stepgame. Per decision it records the decision time (wall
time of Strategy.decide, including any model request) and the prompt length in tokens (0 for
strategies without a model). Every step it records steps/NNNNNN.png (the field) and
steps/NNNNNN.json (server numbers, structured field, client numbers) unless --no-record; render
frames or a video from them with tools/render_run.py. When the run ends (game over or Ctrl+C) it
prints mean/std per agent and per team and writes runs/<time>_<red>-vs-<blue>/ with meta.json,
decisions.jsonl, summary.json and report.png (decision time and prompt length distributions per
agent, score over time).

Usage (from golddiggers/):
  uv run --extra agents python agents/harness.py --start '{"preset": "mapc2007"}' --red greedy --blue random
  uv run --extra agents python agents/harness.py --start '{"preset": "mapc2007"}' --red d3 --blue greedy
  uv run --extra agents python agents/harness.py --list-strategies
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
VIEW = {"nw": (-1, -1), "n": (0, -1), "ne": (1, -1), "w": (-1, 0), "cur": (0, 0),
        "e": (1, 0), "sw": (-1, 1), "s": (0, 1), "se": (1, 1)}
RUNS = Path(__file__).resolve().parent.parent / "runs"


def call(base: str, method: str, path: str, body=None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: HTTP {e.code} {e.read().decode()}") from None


class TeamMemory:
    """What a team has seen. Shared by its agents (MAPC allows local team communication)."""

    def __init__(self, width, height, depot):
        self.w, self.h, self.depot = width, height, tuple(depot)
        self.obstacles: set = set()
        self.gold: set = set()
        self.seen: set = set()
        self.claims: dict = {}      # gold cell -> agent

    def update(self, perception):
        x, y = perception["pos"]
        for cid, content in perception["cells"].items():
            if content == ["unknown"]:
                continue
            dx, dy = VIEW[cid]
            cell = (x + dx, y + dy)
            self.seen.add(cell)
            if "obstacle" in content:
                self.obstacles.add(cell)
            if "gold" in content:
                self.gold.add(cell)
            else:
                self.gold.discard(cell)
                self.claims.pop(cell, None)

    def path_step(self, start, goal_test, blocked=frozenset(), allow_depot=False):
        """BFS over known-free/unknown cells; returns (first move, target) or (None, None)."""
        start = tuple(start)
        prev = {start: None}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell != start and goal_test(cell):
                target = cell
                while prev[cell] != start:
                    cell = prev[cell]
                return next(m for m, (dx, dy) in MOVES.items()
                            if (start[0] + dx, start[1] + dy) == cell), target
            for dx, dy in MOVES.values():
                n = (cell[0] + dx, cell[1] + dy)
                if not (0 <= n[0] < self.w and 0 <= n[1] < self.h) or n in prev:
                    continue
                if n in self.obstacles or (n in blocked and not goal_test(n)):
                    continue
                if n == self.depot and not allow_depot and not goal_test(n):
                    continue
                prev[n] = cell
                queue.append(n)
        return None, None

    def distances(self, goal_test, allow_depot: bool, blocked=frozenset()) -> dict:
        """Multi-source BFS from all goal cells: cell -> steps to the nearest goal.
        blocked: extra cells to route around (e.g. cells with miners next to an agent)."""
        goals = [(x, y) for x in range(self.w) for y in range(self.h)
                 if goal_test((x, y)) and (x, y) not in self.obstacles]
        dist = {g: 0 for g in goals}
        queue = deque(goals)
        while queue:
            cell = queue.popleft()
            for dx, dy in MOVES.values():
                n = (cell[0] + dx, cell[1] + dy)
                if (0 <= n[0] < self.w and 0 <= n[1] < self.h and n not in dist and n not in blocked
                        and n not in self.obstacles and (allow_depot or n != self.depot)):
                    dist[n] = dist[cell] + 1
                    queue.append(n)
        return dist


@dataclass
class Team:
    """Handed to Strategy.decide: the team's shared memory, its lock and game facts."""
    name: str
    memory: TeamMemory
    lock: threading.Lock
    max_carry: int


@dataclass
class Decision:
    action: str
    prompt_tokens: int = 0
    info: dict = field(default_factory=dict)     # extra data for decisions.jsonl


class Strategy:
    """Subclass and register in strategies.REGISTRY. decide() is called from worker threads;
    lock team.lock while touching team.memory, but not while waiting for a model.
    A strategy with batched = True gets one decide_batch() call per team and step instead."""
    name = "base"
    batched = False

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        pass

    def __init__(self, args):
        self.args = args

    def decide(self, p: dict, team: Team) -> Decision:
        raise NotImplementedError

    def decide_batch(self, perceptions: list[dict], team: Team) -> dict[str, Decision]:
        return {p["agent"]: self.decide(p, team) for p in perceptions}


def mean_std(values):
    if not values:
        return 0.0, 0.0
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


class Recorder:
    """Per step: steps/NNNNNN.png (field rendered from /api/field) and steps/NNNNNN.json with the
    server's numbers (/api/stats), the structured field and this step's client-side numbers.
    A step's files show the state at that step plus the decisions the agents took from it."""

    def __init__(self, out: Path, url: str, fog: bool):
        from field_render import FieldRenderer
        self.dir = out / "steps"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.url, self.fog = url, fog
        self.renderer = FieldRenderer(url)

    def record(self, step: int, decisions: list[dict], wall_ms: float | None):
        stats = call(self.url, "GET", "/api/stats")
        field = call(self.url, "GET", "/api/field")
        teams = {}
        for t in stats["teams"]:
            mine = [d for d in decisions if d["team"] == t]
            teams[t] = {"decisions": len(mine),
                        "ms_mean": round(mean_std([d["ms"] for d in mine])[0], 2),
                        "ms_max": round(max((d["ms"] for d in mine), default=0), 2),
                        "tokens_mean": round(mean_std([d["prompt_tokens"] for d in mine])[0], 1),
                        "tokens_sum": sum(d["prompt_tokens"] for d in mine),
                        "errors": sum(1 for d in mine if "error" in d)}
            requests = []                        # the exact request bodies (one per team when batched)
            for d in mine:
                if "request" in d and d["request"] not in requests:
                    requests.append(d["request"])
            if requests:
                teams[t]["requests"] = requests
        client = {"step_wall_ms": round(wall_ms, 1) if wall_ms is not None else None, "teams": teams,
                  "agents": {d["agent"]: {k: d[k] for k in ("action", "ms", "prompt_tokens", "strategy")}
                             | {k: d[k] for k in ("probabilities", "prompt", "prompt_variant", "model", "error")
                                if k in d}
                             for d in decisions}}
        self.renderer.render(field, fog=self.fog).save(self.dir / f"{step:06d}.png")
        (self.dir / f"{step:06d}.json").write_text(json.dumps(
            {"step": step, "server": stats, "field": field, "client": client}))


class Run:
    def __init__(self, args, strategies: dict[str, Strategy]):
        self.args = args
        self.strategies = strategies                       # team -> Strategy
        names = "-vs-".join(s.name for s in strategies.values())
        self.out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_{names}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.recorder = None if args.no_record else Recorder(self.out, args.url, not args.no_fog)
        self.records: list[dict] = []
        self.records_lock = threading.Lock()
        self.history: list[dict] = []
        self.state: dict = {}

    def agent_step(self, name: str, step: int, teams: dict[str, Team]):
        p = call(self.args.url, "GET", f"/api/perceive/{name}")
        if p["finished"] or p["step"] != step:
            return
        team = teams[p["team"]]
        with team.lock:
            team.memory.update(p)
        t0 = time.perf_counter()
        decision = self.strategies[p["team"]].decide(p, team)
        ms = (time.perf_counter() - t0) * 1000
        call(self.args.url, "POST", f"/api/act/{name}", {"action": decision.action})
        with self.records_lock:
            self.records.append({"step": step, "agent": name, "team": p["team"],
                                 "strategy": self.strategies[p["team"]].name,
                                 "action": decision.action, "ms": round(ms, 2),
                                 "prompt_tokens": decision.prompt_tokens, **decision.info})

    def team_step(self, team_name: str, names: list[str], step: int, teams: dict[str, Team]):
        """Batched strategies: perceive all agents of one team, one decide_batch, act all.
        Every agent's decision time is the batch's wall time; its prompt tokens are its share
        (request tokens / batch size; the full count is kept as batch_tokens)."""
        team = teams[team_name]
        ps = [call(self.args.url, "GET", f"/api/perceive/{n}") for n in names]
        ps = [p for p in ps if not p["finished"] and p["step"] == step]
        if not ps:
            return
        with team.lock:
            for p in ps:
                team.memory.update(p)
        strategy = self.strategies[team_name]
        t0 = time.perf_counter()
        decisions = strategy.decide_batch(ps, team)
        ms = (time.perf_counter() - t0) * 1000
        for p in ps:
            d = decisions[p["agent"]]
            call(self.args.url, "POST", f"/api/act/{p['agent']}", {"action": d.action})
            with self.records_lock:
                self.records.append({"step": step, "agent": p["agent"], "team": team_name,
                                     "strategy": strategy.name, "action": d.action, "ms": round(ms, 2),
                                     "prompt_tokens": d.prompt_tokens, **d.info})

    def play(self):
        a = self.args
        self.state = call(a.url, "GET", "/api/state")
        if not self.state.get("running"):
            raise SystemExit("no game running: pass --start '{}'")
        cfg = self.state["config"]
        for team, strategy in self.strategies.items():
            call(a.url, "POST", f"/api/setteamname/{team}", {"name": strategy.name})
        teams = {t: Team(t, TeamMemory(cfg["width"], cfg["height"], self.state["depot"]),
                         threading.Lock(), cfg["max_carry"]) for t in self.strategies}
        names = [ag["name"] for ag in self.state["agents"] if ag["team"] in self.strategies]
        self.history = list(self.state["history"])
        (self.out / "meta.json").write_text(json.dumps(
            {"config": cfg, "strategies": {t: s.name for t, s in self.strategies.items()},
             "team_names": self.state.get("team_names"), "argv": sys.argv[1:]}, indent=1))
        step = self.state["step"]
        recorded = len(self.records)          # records before this index belong to earlier steps
        team_of = {ag["name"]: ag["team"] for ag in self.state["agents"]}
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            while not self.state["finished"]:
                t0 = time.perf_counter()
                # one task per agent, or one per team for batched strategies (teams never mixed)
                tasks = [(lambda t=t: self.team_step(t, [n for n in names if team_of[n] == t], step, teams))
                         for t, s in self.strategies.items() if s.batched]
                tasks += [(lambda n=n: self.agent_step(n, step, teams))
                          for n in names if not self.strategies[team_of[n]].batched]
                list(pool.map(lambda task: task(), tasks))
                wall = (time.perf_counter() - t0) * 1000
                if self.recorder:
                    self.recorder.record(step, self.records[recorded:], wall)
                recorded = len(self.records)
                if a.no_drive:                      # someone else steps
                    while (st := call(a.url, "GET", "/api/state"))["step"] == step:
                        time.sleep(0.05)
                    self.state = st
                else:
                    call(a.url, "POST", "/api/stepgame")
                    self.state = call(a.url, "GET", "/api/state")
                self.history = self.state["history"]
                step = self.state["step"]
                if step % a.print_every == 0 or self.state["finished"]:
                    print(f"step {step}: {self.state['score']}  (step wall {wall:.0f} ms)", flush=True)
                if a.delay:
                    time.sleep(a.delay)

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        order = list((self.state.get("config") or {}).get("teams", []))
        by_agent: dict = {}
        for r in self.records:
            by_agent.setdefault(r["agent"], []).append(r)
        def stats(rows):
            ms_m, ms_s = mean_std([r["ms"] for r in rows])
            tk_m, tk_s = mean_std([r["prompt_tokens"] for r in rows])
            return {"decisions": len(rows), "ms_mean": round(ms_m, 2), "ms_std": round(ms_s, 2),
                    "tokens_mean": round(tk_m, 1), "tokens_std": round(tk_s, 1)}
        return {
            "score": self.state.get("score"), "step": self.state.get("step"),
            "finished": self.state.get("finished"), "winner": self.state.get("winner"),
            "strategies": {t: s.name for t, s in self.strategies.items()},
            "versions": {t: getattr(s, "version", s.name) for t, s in self.strategies.items()},
            "config": self.state.get("config"),
            "agents": {n: {"team": rows[0]["team"], "strategy": rows[0]["strategy"], **stats(rows)}
                       for n, rows in sorted(by_agent.items(), key=lambda kv: agent_key(kv[0], order))},
            "teams": {t: {"strategy": s.name, **stats([r for r in self.records if r["team"] == t])}
                      for t, s in self.strategies.items()},
        }

    def report(self):
        summary = self.summary()
        print(f"\nfinal score {summary['score']} after {summary['step']} steps"
              f"{'' if summary['finished'] else ' (interrupted)'}")
        errors = sum(1 for r in self.records if "error" in r)
        if errors:
            print(f"{errors} decisions failed (model request errors; those agents skipped)")
        print(f"{'agent':10} {'strategy':10} {'n':>5}  {'decision ms':>17}  {'prompt tokens':>17}")
        rows = list(summary["agents"].items()) + [(f"[{t}]", v) for t, v in summary["teams"].items()]
        for name, s in rows:
            print(f"{name:10} {s['strategy']:10} {s['decisions']:5}  "
                  f"{s['ms_mean']:8.1f} ± {s['ms_std']:<6.1f}  {s['tokens_mean']:8.1f} ± {s['tokens_std']:<6.1f}")
        out = self.out
        if self.recorder and self.state.get("finished"):
            self.recorder.record(self.state["step"], [], None)     # the final state
        with open(out / "decisions.jsonl", "w") as f:     # request bodies only in the step files
            for r in self.records:
                f.write(json.dumps({k: v for k, v in r.items() if k != "request"}) + "\n")
        summary["history"] = self.history
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        try:
            plot(summary, self.records, out / "report.png")
            print(f"report: {out / 'report.png'}")
        except ImportError:
            print("matplotlib missing (uv sync --extra agents): no report.png")
        print(f"data:   {out}")


def agent_key(name: str, team_order: list[str]):
    team = name.rstrip("0123456789")
    rank = team_order.index(team) if team in team_order else len(team_order)
    return rank, int(name[len(team):] or 0)


TEAM_COLORS = ["#e34948", "#2a78d6"]        # validated categorical pair (red, blue)


def plot(summary: dict, records: list[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teams = list(summary["config"]["teams"])
    agents = list(summary["agents"])
    color = {t: TEAM_COLORS[i % 2] for i, t in enumerate(teams)}
    label = {t: f"{summary['strategies'].get(t, t)} ({t})" for t in teams}
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e3da", linewidth=0.8)
        ax.set_axisbelow(True)

    for ax, key, title in ((axes[0], "ms", "Decision time per agent (ms, wall clock incl. model request)"),
                           (axes[1], "prompt_tokens", "Prompt length per agent (tokens)")):
        data = [[r[key] for r in records if r["agent"] == a] or [0] for a in agents]
        if not any(v for d in data for v in d):           # e.g. prompt length of rule-based strategies
            ax.text(0.5, 0.5, "all 0 (no model prompts)", transform=ax.transAxes, ha="center",
                    va="center", color="#77756f")
            ax.set_yticks([])
            ax.set_xticks(range(1, len(agents) + 1), agents, fontsize=8)
            ax.set_xlim(0.5, len(agents) + 0.5)
            ax.set_title(title, loc="left", fontsize=11)
            continue
        bp = ax.boxplot(data, patch_artist=True, showfliers=True, widths=0.6,
                        medianprops={"color": "#0b0b0b", "linewidth": 1.5},
                        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
        for patch, a in zip(bp["boxes"], agents):
            patch.set_facecolor(color[summary["agents"][a]["team"]])
            patch.set_alpha(0.75)
        for i, a in enumerate(agents, start=1):
            s = summary["agents"][a]
            m, sd = (s["ms_mean"], s["ms_std"]) if key == "ms" else (s["tokens_mean"], s["tokens_std"])
            fmt = ".2f" if m < 10 else ".0f"
            ax.annotate(f"{m:{fmt}}±{sd:{fmt}}", (i, 1), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7, color="#52514e")
        ax.set_xticks(range(1, len(agents) + 1), agents, rotation=0, fontsize=8)
        ax.set_title(title, loc="left", fontsize=11, pad=14)

    hist = summary["history"]
    for t in teams:
        xs = [h["step"] for h in hist]
        ys = [h["score"][t] for h in hist]
        axes[2].plot(xs, ys, color=color[t], linewidth=2, label=label[t])
        if xs:
            axes[2].annotate(str(ys[-1]), (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                             va="center", fontsize=9, color="#52514e")
    axes[2].set_title("Score over time (gold delivered)", loc="left", fontsize=11)
    axes[2].set_xlabel("step")
    axes[2].legend(frameon=False, loc="upper left")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=color[t], alpha=0.75, label=label[t]) for t in teams],
                   frameon=False, loc="upper right", fontsize=8)
    fig.suptitle(f"Gold Diggers: {' vs '.join(label[t] for t in teams)}  ·  final {summary['score']}",
                 fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from strategies import REGISTRY

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8642", help="golddiggers server")
    ap.add_argument("--start", help="JSON parameters: start a new game first")
    ap.add_argument("--red", default="greedy", help="strategy for the first team")
    ap.add_argument("--blue", default="random", help="strategy for the second team")
    ap.add_argument("--teams", help="only control these teams (comma-separated; default: both)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to sleep after each step")
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--no-drive", action="store_true", help="do not call /api/stepgame")
    ap.add_argument("--no-record", action="store_true", help="no per-step screenshots and JSON")
    ap.add_argument("--no-fog", action="store_true", help="recorded screenshots without fog of war")
    ap.add_argument("--list-strategies", action="store_true")
    for cls in REGISTRY.values():
        cls.add_arguments(ap)
    args = ap.parse_args()
    if args.list_strategies:
        for name, cls in REGISTRY.items():
            print(f"{name:8} {(cls.__doc__ or '').strip().splitlines()[0]}")
        return
    for choice in (args.red, args.blue):
        if choice not in REGISTRY:
            raise SystemExit(f"unknown strategy {choice!r}, available: {', '.join(REGISTRY)}")

    if args.start:
        call(args.url, "POST", "/api/startgame", json.loads(args.start))
    state = call(args.url, "GET", "/api/state")
    if not state.get("running"):
        raise SystemExit("no game running: pass --start '{}'")
    teams = state["config"]["teams"]
    controlled = args.teams.split(",") if args.teams else teams
    instances: dict[str, Strategy] = {}                  # one instance per strategy name
    strategies = {}
    for team, choice in zip(teams, (args.red, args.blue)):
        if team in controlled:
            instances.setdefault(choice, REGISTRY[choice](args))
            strategies[team] = instances[choice]

    run = Run(args, strategies)
    try:
        run.play()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        run.report()


if __name__ == "__main__":
    main()
