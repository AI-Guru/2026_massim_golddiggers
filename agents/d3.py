"""d3 strategy: every action is chosen by a vllm-sr/d3 decision model over POST /v1/systemone.

At start-up the strategy asks the server which models it serves (GET /v1/models) and uses
--d3-model, or the first model listed. Per decision it computes *facts* (position, carried gold,
view, and for every possible action the BFS distance over the team's shared map to the depot, the
nearest known gold and the nearest unexplored cell) and renders them into a state text and one
choice question with a prompt variant (--d3-prompt, see PROMPTS). The model decides; the strategy
only computes facts. The facts are logged with every decision, so tools/d3_prompt_lab.py can
re-render logged situations with other variants and score them offline.

By default one request per team and step (--d3-no-batch for one per agent): only the task text is
the shared state, each miner's situation goes into its own question's instructions.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections import deque

from harness import MOVES, VIEW as VIEW_D, Decision, Strategy, Team

DIRS = {"up": "north", "down": "south", "left": "west", "right": "east"}
CELL_WORDS = {"nw": "north-west", "n": "north (up)", "ne": "north-east", "w": "west (left)",
              "cur": "your cell", "e": "east (right)", "sw": "south-west", "s": "south (down)",
              "se": "south-east"}


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


def list_models(base: str) -> tuple[list[str], dict]:
    listed = http_json(f"{base}/v1/models", timeout=10)
    # OpenAI style {"data": [{"id"}]} (llama.cpp) or {"models": [{"name"}]} (d3 servers)
    ids = ([m["id"] for m in listed.get("data", [])]
           or [m.get("model") or m["name"] for m in listed.get("models", [])])
    return ids, listed


def describe(content: list) -> str:
    return ", ".join(f'mark "{c["mark"]}"' if isinstance(c, dict) else c for c in content)


# ------------------------------------------------------------------------------------ facts
def assign_gold(perceptions: list[dict], team: Team) -> dict[str, tuple]:
    """Match the team's miners (with room) to known gold, closest pairs first, one miner per
    nugget, so teammates don't chase the same gold. Returns {agent: gold cell}."""
    mem = team.memory
    pairs = []
    for p in perceptions:
        if p["items"] >= team.max_carry:
            continue
        pos = tuple(p["pos"])
        dist = mem.distances(lambda c, pos=pos: c == pos, allow_depot=False)   # BFS from the miner
        pairs += [(dist[g], p["agent"], g) for g in mem.gold if g in dist and g != pos]
    assigned, taken = {}, set()
    for d, agent, g in sorted(pairs):
        if agent not in assigned and g not in taken:
            assigned[agent] = g
            taken.add(g)
    return assigned


def compute_facts(p: dict, team: Team, recent: deque, my_gold: tuple | None = None,
                  my_explore: tuple | None = None, route_around: bool = False) -> dict:
    """Everything a prompt may say, as plain data (JSON-serialisable, logged per decision).
    my_gold: the gold this miner is assigned to (batched mode), distances in "my_gold"."""
    mem = team.memory
    pos, items, cells = tuple(p["pos"]), p["items"], p["cells"]
    to_depot = mem.distances(lambda c: c == mem.depot, allow_depot=True)
    to_gold = mem.distances(lambda c: c in mem.gold and c != pos, allow_depot=False)
    to_mine = (mem.distances(lambda c: c == my_gold, allow_depot=False) if my_gold else to_gold)
    to_unknown = mem.distances(lambda c: c not in mem.seen, allow_depot=False)
    to_explore = (mem.distances(lambda c: c == my_explore, allow_depot=False) if my_explore else to_unknown)
    plain = {"depot": to_depot, "my_gold": to_mine, "my_explore": to_explore}
    if route_around:
        # distances that route around the miners next to this one (d3-memory): a side step then
        # shows as the shorter way round instead of a straight move into a queue
        here = {(pos[0] + VIEW_D[c][0], pos[1] + VIEW_D[c][1]) for c, content in cells.items()
                if "ally" in content or "enemy" in content}
        if here:
            to_depot = mem.distances(lambda c: c == mem.depot, allow_depot=True, blocked=here)
            if my_gold:
                to_mine = mem.distances(lambda c: c == my_gold, allow_depot=False, blocked=here)
            if my_explore:
                to_explore = mem.distances(lambda c: c == my_explore, allow_depot=False, blocked=here)
    known_gold = sorted(mem.gold - {pos}, key=lambda g: abs(g[0] - pos[0]) + abs(g[1] - pos[1]))[:5]
    actions = {}
    neighbour = {"up": "n", "down": "s", "left": "w", "right": "e"}
    for move, (dx, dy) in MOVES.items():
        target = (pos[0] + dx, pos[1] + dy)
        content = cells.get(neighbour[move])
        if content is None or "obstacle" in content:          # outside the grid or a tree
            continue
        if target == mem.depot and items == 0:
            continue
        occupied = "ally" in content or "enemy" in content
        if occupied and route_around:      # the move itself fails unless that miner moves away
            dd, dg, de = (plain["depot"].get(target), plain["my_gold"].get(target),
                          plain["my_explore"].get(target))
        else:
            dd, dg, de = to_depot.get(target), to_mine.get(target), to_explore.get(target)
        actions[move] = {"target": list(target), "content": describe(content),
                         "agent_there": occupied,
                         "depot": to_depot.get(target), "gold": to_gold.get(target),
                         "my_gold": dg, "unknown": to_unknown.get(target), "my_explore": de}
        if route_around:
            actions[move]["depot"] = dd
    cur = cells.get("cur", [])
    if "gold" in cur and items < team.max_carry:
        actions["pick"] = {}
    if items and pos == tuple(mem.depot):
        actions["drop"] = {"depot": True}
    elif items and "gold" not in cur:
        actions["drop"] = {"depot": False}                     # drops one item, does not score
    return {
        "agent": p["agent"], "team": p["team"], "step": p["step"], "steps": p["steps"],
        "pos": list(pos), "items": items, "max_carry": team.max_carry, "depot": list(mem.depot),
        "on_gold": "gold" in cur, "on_depot": pos == tuple(mem.depot),
        "now": {"depot": to_depot.get(pos), "gold": to_gold.get(pos), "my_gold": to_mine.get(pos),
                "my_explore": to_explore.get(pos),
                "unknown": to_unknown.get(pos)},
        "my_gold_target": list(my_gold) if my_gold else None,
        "my_explore_target": list(my_explore) if my_explore else None,
        "view": {cid: describe(content) for cid, content in cells.items()},
        "known_gold": [list(g) for g in known_gold],
        "last_action": p["last_action"], "last_result": p["last_result"],
        "recent": [list(c) for c in recent],
        "actions": actions,
    }


# ------------------------------------------------------------------------------- prompts
# A prompt variant renders facts into (task, situation, options). The task is shared by all
# miners of a team (the state of a batched request); the situation is per miner.

TASK_V0 = ("You control one gold miner in a grid world. Your team scores one point for every gold item "
           "delivered to the depot (drop while standing on it). You can carry at most {max_carry} gold "
           "items; carrying more makes your actions fail more often. You may only enter the depot while "
           "carrying gold, and must step out right after dropping. Collect gold quickly, deliver it, and "
           "explore unknown areas when no gold is known. Coordinates are (x, y); x grows to the right "
           "(east), y grows downwards (south).")


def _dist(label, value):
    return f"{label} {value}" if value is not None else f"{label} unreachable/unknown"


def render_v0(f: dict):
    """The original prompt: raw distances per move, drop-anywhere offered."""
    situation = "\n".join([
        f"Step {f['step']} of {f['steps']}. You are {f['agent']} (team {f['team']}).",
        f"Position {f['pos']}. Carrying {f['items']} of {f['max_carry']} gold. Depot at {f['depot']}.",
        f"Now: {_dist('steps to depot', f['now']['depot'])}, "
        f"{_dist('steps to nearest known gold', f['now']['gold'])}, "
        f"{_dist('steps to unexplored area', f['now']['unknown'])}.",
        "View: " + "; ".join(f"{CELL_WORDS[c]}: {v}" for c, v in f["view"].items()) + ".",
        f"Known gold nearby: {', '.join(map(str, f['known_gold'])) or 'none'}.",
        f"Last action: {f['last_action']} ({f['last_result']}). "
        f"Recent positions: {', '.join(map(str, f['recent'])) or 'none'}.",
    ])
    options = {}
    for a, info in f["actions"].items():
        if a in MOVES:
            what = info["content"] + (" there (moving pushes it if it stands still)" if info["agent_there"] else "")
            options[a] = (f"move to {info['target']} ({what}); {_dist('then depot', info['depot'])}, "
                          f"{_dist('gold', info['gold'])}, {_dist('unexplored', info['unknown'])}")
        elif a == "pick":
            options[a] = "pick up the gold in your cell"
        elif a == "drop":
            options[a] = ("drop all carried gold into the depot and score" if info["depot"]
                          else "drop one gold item here (does not score)")
    if not options:
        options["skip"] = "do nothing this step"
    return TASK_V0.format(max_carry=f["max_carry"]), situation, options


def _change(now, then, noun):
    """'closer to the depot (9->8)' style wording."""
    if now is None or then is None:
        return f"{noun} distance unknown"
    if then < now:
        return f"closer to {noun} ({now}->{then})"
    if then > now:
        return f"farther from {noun} ({now}->{then})"
    return f"same distance to {noun} ({now})"


TASK_V1 = ("You control one gold miner on a grid. Your team scores one point for each gold item you "
           "deliver: walk onto the depot while carrying gold, then drop it there. You carry at most "
           "{max_carry} items. Good play: pick up gold you stand on, walk to known gold, deliver when you "
           "are full or no gold is near, and explore unknown areas when no gold is known.")


def render_v1(f: dict):
    """No drop outside the depot; moves described as closer/farther from depot, gold, unexplored."""
    full = f["items"] >= f["max_carry"]
    lines = [f"You carry {f['items']} of {f['max_carry']} gold" + (" (full)." if full else ".")]
    if f["on_depot"]:
        lines.append("You are standing on the depot.")
    elif f["on_gold"]:
        lines.append("You are standing on gold.")
    lines.append(f"Distances: depot {f['now']['depot']}, nearest known gold "
                 f"{f['now']['gold'] if f['now']['gold'] is not None else 'none known'}, "
                 f"unexplored area {f['now']['unknown']}.")
    options = {}
    for a, info in f["actions"].items():
        if a in MOVES:
            parts = [_change(f["now"]["depot"], info["depot"], "the depot")]
            if f["now"]["gold"] is not None:
                parts.append(_change(f["now"]["gold"], info["gold"], "gold"))
            parts.append(_change(f["now"]["unknown"], info["unknown"], "unexplored area"))
            extra = " (another miner is there)" if info["agent_there"] else ""
            options[a] = f"step {DIRS[a]}{extra}: " + ", ".join(parts)
        elif a == "pick":
            options[a] = "pick up the gold you are standing on"
        elif a == "drop" and info["depot"]:
            options[a] = "drop your gold into the depot (scores)"
    if not options:
        options["skip"] = "do nothing this step"
    return TASK_V1.format(max_carry=f["max_carry"]), "\n".join(lines), options


def render_v2(f: dict):
    """v1 plus the current goal stated as a fact (full -> deliver, empty -> collect or explore)."""
    task, situation, options = render_v1(f)
    if f["items"] >= f["max_carry"]:
        goal = "You are full: deliver to the depot."
    elif f["now"]["gold"] is not None and f["items"] == 0:
        goal = "You carry nothing: go to the nearest known gold."
    elif f["now"]["gold"] is None and f["items"] == 0:
        goal = "No gold is known: explore."
    else:
        goal = "You carry some gold: collect more if gold is close, otherwise deliver."
    return task, situation + "\n" + goal, options


def _goal(f: dict) -> tuple[str, list[str]]:
    """The current goal as a fact sentence, and the distances that matter for it (most important first)."""
    if f["items"] >= f["max_carry"]:
        return "You are full: deliver to the depot.", ["depot"]
    if f["items"] and f["now"]["gold"] is not None:
        return "You carry some gold: collect more if gold is close, otherwise deliver.", ["gold", "depot"]
    if f["items"]:
        return "You carry some gold and no other gold is known: deliver it.", ["depot"]
    if f["now"]["gold"] is not None:
        return "You carry nothing: go to the nearest known gold.", ["gold"]
    return "No gold is known: explore unknown areas to find some.", ["unknown"]


NOUNS = {"depot": "the depot", "gold": "gold", "unknown": "unexplored area"}


def render_v3(f: dict):
    """v2, but each move only mentions the distances that matter for the current goal."""
    goal, keys = _goal(f)
    lines = [f"You carry {f['items']} of {f['max_carry']} gold"
             + (" (full)." if f["items"] >= f["max_carry"] else ".")]
    if f["on_depot"]:
        lines.append("You are standing on the depot.")
    elif f["on_gold"]:
        lines.append("You are standing on gold.")
    lines.append(goal)
    options = {}
    for a, info in f["actions"].items():
        if a in MOVES:
            parts = [_change(f["now"][k], info[k], NOUNS[k]) for k in keys]
            extra = " (another miner is there)" if info["agent_there"] else ""
            options[a] = f"step {DIRS[a]}{extra}: " + ", ".join(parts)
        elif a == "pick":
            options[a] = "pick up the gold you are standing on"
        elif a == "drop" and info["depot"]:
            options[a] = "drop your gold into the depot (scores)"
    if not options:
        options["skip"] = "do nothing this step"
    return TASK_V1.format(max_carry=f["max_carry"]), "\n".join(lines), options


def render_v4(f: dict):
    """v3 with team coordination: gold distances refer to the gold assigned to this miner (one
    miner per nugget), falling back to the nearest gold when nothing is assigned."""
    g = dict(f)
    if f.get("my_gold_target") and f["now"].get("my_gold") is not None:
        g["now"] = {**f["now"], "gold": f["now"]["my_gold"]}
        g["actions"] = {a: ({**i, "gold": i["my_gold"]} if a in MOVES else i) for a, i in f["actions"].items()}
    task, situation, options = render_v3(g)
    situation = situation.replace("go to the nearest known gold.", "go to your gold (teammates take the others).")
    return task, situation, options


def render_v5(f: dict):
    """Memory prompt (d3-memory): the goal names the miner's remembered target (gold it keeps
    until picked up or gone, or its own unexplored target cell) and moves refer to that target."""
    full = f["items"] >= f["max_carry"]
    now = f["now"]
    lines = [f"You carry {f['items']} of {f['max_carry']} gold" + (" (full)." if full else ".")]
    if f["on_depot"] and f["items"]:
        lines.append("You are standing on the depot: drop your gold to score.")
        keys = ["depot"]
    elif "pick" in f["actions"]:
        lines.append("You are standing on gold: pick it up.")
        keys = []
    elif full:
        lines.append("You are full: deliver to the depot.")
        keys = ["depot"]
    elif f.get("my_gold_target") and now.get("my_gold") is not None:
        # no coordinates: without the miner's own position they misled d3 (half the moves went the
        # wrong way in oscillating situations; 60/60 right without them)
        lines.append(f"You are heading to your gold, {now['my_gold']} steps away.")
        keys = ["my_gold", "depot"] if f["items"] else ["my_gold"]
    elif f["items"]:
        lines.append("You carry some gold and no free gold is known: deliver it.")
        keys = ["depot"]
    elif f.get("my_explore_target") and now.get("my_explore") is not None:
        lines.append(f"No free gold is known. You are exploring towards your own unexplored target, "
                     f"{now['my_explore']} steps away (teammates explore elsewhere).")
        keys = ["my_explore"]
    else:
        lines.append("No gold is known: explore unknown areas to find some.")
        keys = ["unknown"]
    nouns = {"depot": "the depot", "my_gold": "your gold", "my_explore": "your exploration target",
             "unknown": "unexplored area"}
    options = {}
    for a, info in f["actions"].items():
        if a in MOVES:
            parts = [_change(now[k], info[k], nouns[k]) for k in keys] or ["leaves the gold here"]
            extra = " (another miner is there)" if info["agent_there"] else ""
            options[a] = f"step {DIRS[a]}{extra}: " + ", ".join(parts)
        elif a == "pick":
            options[a] = "pick up the gold you are standing on"
        elif a == "drop" and info["depot"]:
            options[a] = "drop your gold into the depot (scores)"
    if not options:
        options["skip"] = "do nothing this step"
    return TASK_V1.format(max_carry=f["max_carry"]), "\n".join(lines), options


PROMPTS = {"v0": render_v0, "v1": render_v1, "v2": render_v2, "v3": render_v3, "v4": render_v4,
           "v5": render_v5}
DEFAULT_PROMPT = "v4"


# ----------------------------------------------------------------------------- strategy
class D3Strategy(Strategy):
    """Every action chosen by the d3 decision model (/v1/systemone); the model is picked from /v1/models."""
    name = "d3"

    @staticmethod
    def add_arguments(parser):
        g = parser.add_argument_group("d3 strategy")
        g.add_argument("--d3-url", default="http://192.168.0.219:11490",
                       help="decision server base URL (serves /v1/models and /v1/systemone)")
        g.add_argument("--d3-model", help="model id (default: the first one the server lists)")
        g.add_argument("--d3-prompt", default=DEFAULT_PROMPT, choices=list(PROMPTS),
                       help=f"prompt variant (default {DEFAULT_PROMPT})")
        g.add_argument("--d3-sample", action="store_true",
                       help="sample the action from the probabilities instead of the argmax")
        g.add_argument("--d3-no-batch", action="store_true",
                       help="one request per agent instead of one per team and step")

    def __init__(self, args):
        super().__init__(args)
        self.base = args.d3_url.rstrip("/")
        try:
            ids, listed = list_models(self.base)
        except (urllib.error.URLError, ConnectionError, RuntimeError) as e:
            raise SystemExit(f"d3: cannot list models at {self.base}/v1/models: {e}") from None
        if not ids:
            raise SystemExit(f"{self.base}/v1/models lists no models")
        if args.d3_model and args.d3_model not in ids:
            raise SystemExit(f"model {args.d3_model!r} not served; available: {', '.join(ids)}")
        self.model = args.d3_model or ids[0]
        self.render = PROMPTS[args.d3_prompt]
        self.version = f"d3 {self.model} prompt {args.d3_prompt}"
        self.batched = not args.d3_no_batch
        self.recent: dict[str, deque] = {}
        print(f"d3: model {self.model} at {self.base} (available: {', '.join(ids)}), prompt "
              f"{args.d3_prompt}, {'batched per team' if self.batched else 'one request per agent'}",
              flush=True)

    def _facts(self, p, team, my_gold=None, my_explore=None, route_around=False):
        return compute_facts(p, team, self.recent.setdefault(p["agent"], deque(maxlen=4)), my_gold,
                             my_explore, route_around)

    def _decision(self, p, answer, tokens, facts, rendered, extra):
        """extra carries the request body; the harness stores it once per team in the step files."""
        probs = answer["probabilities"]
        action = answer["choice"]
        if self.args.d3_sample:
            action = random.choices(list(probs), weights=list(probs.values()))[0]
        self.recent[p["agent"]].append(tuple(p["pos"]))
        info = {"model": self.model, "prompt_variant": self.args.d3_prompt, **extra,
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
                "prompt": {"task": rendered[0], "situation": rendered[1], "options": rendered[2]},
                "facts": facts}
        return Decision(action, tokens, info)

    def decide(self, p: dict, team: Team) -> Decision:
        with team.lock:
            facts = self._facts(p, team)
        task, situation, options = rendered = self.render(facts)
        body = {"model": self.model, "state": task + "\n" + situation, "questions": {
            "action": {"type": "choice", "instructions": "Which action should the miner take now?",
                       "criteria": options}}}
        try:
            resp = http_json(f"{self.base}/v1/systemone", body)
        except (urllib.error.URLError, ConnectionError, RuntimeError, TimeoutError) as e:
            return Decision("skip", 0, {"error": str(e)[:200]})   # server hiccup: skip, keep playing
        return self._decision(p, resp["answers"]["action"], resp.get("usage", {}).get("input_tokens", 0),
                              facts, rendered, {"request": body})

    def _team_facts(self, perceptions: list[dict], team: Team) -> dict[str, dict]:
        """Facts for every miner of a team (team.lock held). d3: fresh gold matching each step."""
        assigned = assign_gold(perceptions, team)
        return {p["agent"]: self._facts(p, team, assigned.get(p["agent"])) for p in perceptions}

    def decide_batch(self, perceptions: list[dict], team: Team) -> dict[str, Decision]:
        """One request for the whole team (one state, one question per miner). Only the task text
        is the shared state; each miner's situation goes into its own question's instructions, so
        no question sees another miner's situation (putting all situations into the state tripled
        the tokens and changed 60% of the answers, see tools/bench_d3_batching.py)."""
        with team.lock:
            facts = self._team_facts(perceptions, team)
        rendered = {n: self.render(f) for n, f in facts.items()}
        task = next(iter(rendered.values()))[0]
        body = {"model": self.model, "state": task, "questions": {
            n: {"type": "choice", "criteria": opts,
                "instructions": situation + "\nWhich action should this miner take now?"}
            for n, (_, situation, opts) in rendered.items()}}
        try:
            resp = http_json(f"{self.base}/v1/systemone", body)
        except (urllib.error.URLError, ConnectionError, RuntimeError, TimeoutError) as e:
            return {n: Decision("skip", 0, {"error": str(e)[:200]}) for n in facts}
        tokens = resp.get("usage", {}).get("input_tokens", 0)
        n = len(perceptions)
        return {p["agent"]: self._decision(p, resp["answers"][p["agent"]], round(tokens / n),
                                           facts[p["agent"]], rendered[p["agent"]],
                                           {"batch_size": n, "batch_tokens": tokens, "request": body})
                for p in perceptions}
