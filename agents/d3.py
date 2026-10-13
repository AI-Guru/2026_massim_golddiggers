"""d3 strategy: every action is chosen by the vllm-sr/d3 decision model over POST /v1/systemone.

At start-up the strategy asks the server which models it serves (GET /v1/models) and uses
--d3-model, or the first model listed. Per decision the model sees the agent's state as text and
one choice question whose options are only the actions possible right now (skip only if nothing
else is: offered next to moves, d3-edge chose it ~90% of the time); each move option says
what lies in that cell and how it changes the BFS distance (over the team's shared map) to the
depot, the nearest known gold and the nearest unexplored cell. The model decides; the strategy only
computes facts. Prompt length comes from the server's usage.input_tokens.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections import deque

from harness import MOVES, Decision, Strategy, Team

CELL_WORDS = {"nw": "north-west", "n": "north (up)", "ne": "north-east", "w": "west (left)",
              "cur": "your cell", "e": "east (right)", "sw": "south-west", "s": "south (down)",
              "se": "south-east"}
TASK = ("You control one gold miner in a grid world. Your team scores one point for every gold item "
        "delivered to the depot (drop while standing on it). You can carry at most {max_carry} gold "
        "items; carrying more makes your actions fail more often. You may only enter the depot while "
        "carrying gold, and must step out right after dropping. Collect gold quickly, deliver it, and "
        "explore unknown areas when no gold is known. Coordinates are (x, y); x grows to the right "
        "(east), y grows downwards (south).")


def http_json(url: str, body=None, timeout: float = 60.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{url}: HTTP {e.code} {e.read().decode()[:300]}") from None
        except (urllib.error.URLError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(1.0)


def describe(content: list) -> str:
    return ", ".join(f'mark "{c["mark"]}"' if isinstance(c, dict) else c for c in content)


class D3Strategy(Strategy):
    """Every action chosen by the d3 decision model (/v1/systemone); the model is picked from /v1/models."""
    name = "d3"

    @staticmethod
    def add_arguments(parser):
        g = parser.add_argument_group("d3 strategy")
        g.add_argument("--d3-url", default="http://192.168.0.219:11490",
                       help="decision server base URL (serves /v1/models and /v1/systemone)")
        g.add_argument("--d3-model", help="model id (default: the first one the server lists)")
        g.add_argument("--d3-sample", action="store_true",
                       help="sample the action from the probabilities instead of the argmax")
        g.add_argument("--d3-batch", action="store_true",
                       help="one request per team and step: the task text as state, each miner's "
                            "situation in its own question (teams are never mixed)")
        g.add_argument("--d3-log-prompts", action="store_true",
                       help="store state text and options in decisions.jsonl")

    def __init__(self, args):
        super().__init__(args)
        self.base = args.d3_url.rstrip("/")
        try:
            listed = http_json(f"{self.base}/v1/models", timeout=10)
        except (urllib.error.URLError, ConnectionError, RuntimeError) as e:
            raise SystemExit(f"d3: cannot list models at {self.base}/v1/models: {e}") from None
        # OpenAI style {"data": [{"id"}]} (llama.cpp) or {"models": [{"name"}]} (d3-edge server)
        ids = ([m["id"] for m in listed.get("data", [])]
               or [m.get("model") or m["name"] for m in listed.get("models", [])])
        if not ids:
            raise SystemExit(f"{self.base}/v1/models lists no models")
        if args.d3_model and args.d3_model not in ids:
            raise SystemExit(f"model {args.d3_model!r} not served; available: {', '.join(ids)}")
        self.model = args.d3_model or ids[0]
        ctx = next((m.get("meta", {}).get("n_ctx") for m in listed.get("data", []) if m["id"] == self.model), None)
        print(f"d3: using model {self.model} at {self.base} (available: {', '.join(ids)}"
              f"{f', n_ctx {ctx}' if ctx else ''})", flush=True)
        self.recent: dict[str, deque] = {}
        self.batched = args.d3_batch

    def decide(self, p: dict, team: Team) -> Decision:
        recent = self.recent.setdefault(p["agent"], deque(maxlen=4))
        with team.lock:
            situation, options = self.build(p, team, recent)
        state = TASK.format(max_carry=team.max_carry) + "\n" + situation
        body = {"model": self.model, "state": state, "questions": {
            "action": {"type": "choice", "instructions": "Which action should the miner take now?",
                       "criteria": options}}}
        try:
            resp = http_json(f"{self.base}/v1/systemone", body)
        except (urllib.error.URLError, ConnectionError, RuntimeError, TimeoutError) as e:
            return Decision("skip", 0, {"error": str(e)[:200]})   # server hiccup: skip, keep playing
        answer = resp["answers"]["action"]
        probs = answer["probabilities"]
        action = answer["choice"]
        if self.args.d3_sample:
            action = random.choices(list(probs), weights=list(probs.values()))[0]
        recent.append(tuple(p["pos"]))
        info = {"model": resp.get("model", self.model),
                "probabilities": {k: round(v, 4) for k, v in probs.items()}}
        if self.args.d3_log_prompts:
            info.update(state=state, options=options)
        return Decision(action, resp.get("usage", {}).get("input_tokens", 0), info)

    def decide_batch(self, perceptions: list[dict], team: Team) -> dict[str, Decision]:
        """One request for the whole team (one state, one question per miner). Only the task text
        is the shared state; each miner's situation goes into its own question's instructions, so
        no question sees another miner's situation (putting all situations into the state tripled
        the tokens and changed 60% of the answers, see tools/bench_d3_batching.py)."""
        built = {}
        with team.lock:
            for p in perceptions:
                recent = self.recent.setdefault(p["agent"], deque(maxlen=4))
                built[p["agent"]] = self.build(p, team, recent)
        body = {"model": self.model, "state": TASK.format(max_carry=team.max_carry), "questions": {
            name: {"type": "choice", "criteria": options,
                   "instructions": situation + "\nWhich action should this miner take now?"}
            for name, (situation, options) in built.items()}}
        try:
            resp = http_json(f"{self.base}/v1/systemone", body)
        except (urllib.error.URLError, ConnectionError, RuntimeError, TimeoutError) as e:
            return {n: Decision("skip", 0, {"error": str(e)[:200]}) for n in built}
        tokens = resp.get("usage", {}).get("input_tokens", 0)
        out = {}
        for p in perceptions:
            name = p["agent"]
            answer = resp["answers"][name]
            probs = answer["probabilities"]
            action = answer["choice"]
            if self.args.d3_sample:
                action = random.choices(list(probs), weights=list(probs.values()))[0]
            self.recent[name].append(tuple(p["pos"]))
            info = {"model": resp.get("model", self.model), "batch_size": len(built),
                    "batch_tokens": tokens, "probabilities": {k: round(v, 4) for k, v in probs.items()}}
            if self.args.d3_log_prompts:
                info.update(state=built[name][0], options=built[name][1])
            out[name] = Decision(action, round(tokens / len(built)), info)
        return out

    @staticmethod
    def build(p: dict, team: Team, recent: deque):
        """(situation text without the task, {action: description}), only possible actions."""
        mem, max_carry = team.memory, team.max_carry
        pos, items, cells = tuple(p["pos"]), p["items"], p["cells"]
        cur = cells.get("cur", [])
        to_depot = mem.distances(lambda c: c == mem.depot, allow_depot=True)
        to_gold = mem.distances(lambda c: c in mem.gold and c != pos, allow_depot=False)
        to_unknown = mem.distances(lambda c: c not in mem.seen, allow_depot=False)

        def dist(label, table, cell):
            return f"{label} {table[cell]}" if cell in table else f"{label} unreachable/unknown"

        view = "; ".join(f"{CELL_WORDS[cid]}: {describe(content)}" for cid, content in cells.items())
        known_gold = sorted(mem.gold - {pos}, key=lambda g: abs(g[0] - pos[0]) + abs(g[1] - pos[1]))[:5]
        situation = "\n".join([
            f"Step {p['step']} of {p['steps']}. You are {p['agent']} (team {p['team']}).",
            f"Position {list(pos)}. Carrying {items} of {max_carry} gold. Depot at {list(mem.depot)}.",
            f"Now: {dist('steps to depot', to_depot, pos)}, "
            f"{dist('steps to nearest known gold', to_gold, pos)}, "
            f"{dist('steps to unexplored area', to_unknown, pos)}.",
            f"View: {view}.",
            f"Known gold nearby: {', '.join(str(list(g)) for g in known_gold) or 'none'}.",
            f"Last action: {p['last_action']} ({p['last_result']}). "
            f"Recent positions: {', '.join(str(list(c)) for c in recent) or 'none'}.",
        ])

        options: dict[str, str] = {}
        neighbour = {"up": "n", "down": "s", "left": "w", "right": "e"}
        for move, (dx, dy) in MOVES.items():
            target = (pos[0] + dx, pos[1] + dy)
            content = cells.get(neighbour[move])
            if content is None or "obstacle" in content:          # outside the grid or a tree
                continue
            if target == mem.depot and items == 0:
                continue
            what = describe(content)
            if "ally" in content or "enemy" in content:
                what += " there (moving pushes it if it stands still)"
            options[move] = (f"move to {list(target)} ({what}); {dist('then depot', to_depot, target)}, "
                             f"{dist('gold', to_gold, target)}, {dist('unexplored', to_unknown, target)}")
        if "gold" in cur and items < max_carry:
            options["pick"] = "pick up the gold in your cell"
        if items and (pos == mem.depot or "gold" not in cur):
            options["drop"] = ("drop all carried gold into the depot and score" if pos == mem.depot
                               else "drop one gold item here (does not score)")
        if not options:          # skip only as a last resort: d3-edge picks "do nothing" far too often
            options["skip"] = "do nothing this step"
        return situation, options
