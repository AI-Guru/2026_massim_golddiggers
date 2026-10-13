# golddiggers

A server for the Multi-Agent Programming Contest **Gold Miners** scenario (MAPC 2006/2007,
[multiagentcontest.org/2007](https://multiagentcontest.org/2007/)). Agents talk to it over a JSON API,
and a web frontend shows the map and statistics live. Unlike the original MASSim server, steps do
not advance on a timer. Something has to call `stepgame`: a driver script, the frontend's step
buttons, or an experiment harness. This makes slow agents, such as LLMs, a non-issue.

![Gold Diggers frontend: a greedy team (red) against a random team (blue) on a labyrinth map, with fog of war, scores, stats and score over time](docs/screenshot.png)

```sh
cd golddiggers
uv sync --extra dev --extra agents
uv run golddiggers --port 8642                      # frontend: http://127.0.0.1:8642/
uv run python agents/harness.py --start '{"preset": "mapc2007", "seed": 1}' --red greedy --blue random
uv run pytest -q
```

With Docker (port 8642, restarts with Docker; rebuild after code changes):

```sh
docker compose up -d --build        # in golddiggers/
docker compose logs -f              # docker compose down to stop
```

## API

All bodies and responses are JSON. Agent names are `<team><n>`, e.g. `red1`…`red6`, `blue1`…`blue6`.

| endpoint | what it does |
|---|---|
| `POST /api/startgame` | Starts a new game, replacing the current one. Body: any `GameConfig` field (below), plus an optional `"preset"`. Returns the config and, per agent, what MAPC's SIM-START gave: team, opponent, teammates, grid size, depot, steps, max_carry. |
| `GET /api/perceive/{agent}` | The perception for the current step. Repeated calls return the same data. |
| `POST /api/act/{agent}` | `{"action": "up"}` or `{"action": "mark", "param": "abc"}`. Queues the action for the next step. Calling it again replaces the queued action. |
| `POST /api/setteamname/{team}` | `{"name": "greedy"}`: a display name for the team (1-24 letters, digits, space, `_.-`). The frontend shows it as "greedy (red)". The example agents set their strategy name. |
| `POST /api/stepgame` | Resolves all queued actions at once. Agents with nothing queued skip. Returns the step number, the score, each agent's requested action and result, and events (push, delivery, teleport). |
| `GET /api/state` | The full world state for the frontend: obstacles, gold, agents, marks, score, stats, per-step history. |
| `GET /api/config` | Defaults, presets and the list of actions. |

FastAPI's interactive docs are at `/docs`.

Perception (`GET /api/perceive/red1`):

```json
{"agent": "red1", "team": "red", "step": 12, "steps": 500, "finished": false,
 "pos": [4, 7], "items": 1,
 "cells": {"nw": ["obstacle"], "n": ["empty"], "ne": ["gold", {"mark": "x1"}], "w": ["ally"],
           "cur": ["empty"], "e": ["enemy"], "sw": ["unknown"], "s": ["depot"], "se": ["empty"]},
 "last_action": "pick", "last_result": "ok",
 "sim": {"opponent": "blue", "width": 30, "height": 30, "depot": [4, 8], "...": "..."},
 "queued": null}
```

- Cells outside the grid are left out.
- `unknown` means the server withheld that cell (information distortion). It never reports a cell wrongly.
- When the game is over, the perception also carries `result` (win/lose/draw) and `score`.
- `last_result` is one of `ok`, `failed` (random failure or lost conflict), `impossible`, `pushed`, `teleported`. The original protocol had no such feedback. Agents had to infer it from the next perception.

## Rules (2007 spec, with the 2006 rules as a parameter set)

- **Grid:** x grows to the right, y grows downwards, (0,0) is the top-left corner. There is one depot, shared by both teams.
- **Actions:** `skip up down left right pick drop mark unmark`. A failed action counts as skip.
- **Carrying:** an agent carries up to `max_carry` items. `drop` in the depot delivers all of them, one point each. `drop` elsewhere puts one item on the cell.
- **Fatigue:** failure probability is `p = p_sim + (p_max - p_sim) / max_carry * items`. It applies to both action failure (`p_fail` → `p_fail_max`) and missing cells in the perception (`p_info` → `p_info_max`).
- **Pushing:** moving into an agent whose action this step resolves to skip pushes it one cell, if the cell behind it is free. If two agents move into each other, a coin flip decides who pushes.
- **Move resolution:** the spec's six-step algorithm. Moves only succeed into cells that were free at the start of the step, so agents can't follow each other in a line within one step.
- **Depot:** an agent can enter only while carrying gold, and must `drop` on its next action.
  What happens afterwards is set by `depot_policy`:
  - `original` (MAPC 2006/07): leave at the first chance, meaning when a neighbouring cell is free. An agent boxed in by others is never removed. Without pushing, agents queueing around the depot then deadlock it for the rest of the game.
  - `leave_or_teleport` (default): an agent still on the depot one step after dropping is teleported, boxed in or not.
  - `teleport_on_drop`: delivering teleports the agent away immediately, so nobody ever stands on the depot.

  A depot violator is teleported to the cell chosen by `teleport_target`:
  - `random` (original): any free cell. Luck decides whether that helps or hurts.
  - `ring` (default): a random free cell at walking distance `teleport_radius` (6) from the depot. The cost is the same for everyone.
  - `home`: the team's start cells. This is fair because maps are symmetric.

  Not dropping right after entering is always a violation.
- **Gold generation:** works as in the 2006 server code. Every `gold_frequency` steps, with probability `gold_probability`, `gold_number` items appear on random empty cells.

`GameConfig` fields and defaults: `map_type` (`random`|`labyrinth`|`rows`), `width` 30, `height` 30,
`obstacles` 90 (random maps only), `initial_gold` 40, `steps` 500, `teams` ["red","blue"],
`team_size` 6, `max_carry` 3, `pushing` true, `p_fail` 0.05, `p_fail_max` 0.3, `p_info` 0.05,
`p_info_max` 0.3, `gold_frequency` 10, `gold_probability` 0.8, `gold_number` 2,
`depot_policy` leave_or_teleport, `teleport_target` ring, `teleport_radius` 6, `seed` null.

Presets:

- `mapc2006`: 4 agents, carry 1, no pushing, 25×25, using the parameters of the 2006 Random1 map.
- `mapc2007`: 6 agents, carry 3, pushing, 40×40, using the parameters of the 2007 Meadow map.
- `labyrinth`: concentric walls around a central depot, like 2006 Labyrinth.
- `rows`: walls with a single gate at alternating ends, like 2006 Labyrinth1. Random exploration fails on this map.

The maps are generated, not the original hand-made ones.

## Frontend

Served at `/`, polling `/api/state` four times a second. It shows:

- the pixel-art map; hover a cell to see what is on it;
- scores and a step progress bar;
- stat tiles (gold on the map, carried, delivered and generated; failure rate; pushes; teleports);
- a score-over-time chart with a crosshair;
- an agent table.

From the controls you can start games (preset, seed, steps, and any parameter as JSON) and step manually or with auto-step.

Sprites: the frontend uses `static/assets/<name>.png` (grass, tree, gold, depot, mark, miner0,
miner1) when the file exists, and otherwise falls back to built-in 16×16 pixel art.
`tools/gen_assets.sh` generates the sprites with Retro Diffusion (`general/skills/retro-diffusion-api`).
It is paid: about $1.26 per run.

## Agents (client side)

All agents run through one harness, [agents/harness.py](agents/harness.py). Only the strategy is pluggable.

- **The loop:** each step, every agent runs perceive → decide → act in its own thread, so model requests overlap. Then the harness calls `stepgame`, unless you pass `--no-drive`. Each team announces its strategy name via `setteamname`.
- **Per decision it records** the decision time (wall clock of `decide`, including any model request) and the prompt length in tokens (0 for rule-based strategies).
- **At the end of the run** (game over or Ctrl+C), it prints mean ± std per agent and per team, and writes `runs/<time>_<red>-vs-<blue>/` containing:
  - `decisions.jsonl`
  - `summary.json`
  - `report.png`: decision time and prompt length distributions per agent, plus score over time.

```sh
uv run python agents/harness.py --list-strategies
uv run python agents/harness.py --start '{"preset": "mapc2007"}' --red d3 --blue greedy
```

Strategies ([agents/strategies.py](agents/strategies.py), registry `REGISTRY`):

- `greedy`: keeps a shared team map, goes for the nearest known unclaimed gold by BFS, delivers when full, and otherwise explores the nearest unseen cell.
- `random`: random walk, like the organizers' 2006 CLIMABot baseline. It picks up any gold it steps on and delivers it.
- `d3` ([agents/d3.py](agents/d3.py)): every action is chosen by a d3 decision model over `/v1/systemone`.
  - It talks to `--d3-url` (default `http://192.168.0.219:11490`, the port for all d3 models). At start it asks `/v1/models` which models are served, and uses `--d3-model` or the first one listed.
  - The model gets the agent's state as text and one choice question. Only actions possible right now are offered, and each move option says what is in the target cell and how the move changes the distance to the depot, gold and unexplored area.
  - `--d3-sample` draws the action from the probabilities instead of taking the most likely one.
  - `--d3-log-prompts` stores the prompts in `decisions.jsonl`.
  - A failed model request makes that agent skip, and the error is recorded.

A new strategy subclasses `harness.Strategy`. It implements `decide(perception, team) -> Decision(action, prompt_tokens)` and is added to `REGISTRY`. While touching `team.memory`, hold `team.lock`, but not while waiting for a model.

## References

This is an independent reimplementation for experiments. It is not the official software.
The scenario, rules and protocol come from the Multi-Agent Programming Contest (MAPC) and its
MASSim simulation server, organized by Clausthal University of Technology and Utrecht University
(Mehdi Dastani, Jürgen Dix, Peter Novák).

- **Contest:** [multiagentcontest.org](https://multiagentcontest.org/), with the
  [2006 (CLIMA VII)](https://multiagentcontest.org/2006/) and [2007](https://multiagentcontest.org/2007/) editions.
- **Original specs:**
  - 2006: [scenario](https://multiagentcontest.org/2006/c7c-scenario-final.txt), [protocol](https://multiagentcontest.org/2006/c7c-protocol.txt)
  - 2007: [scenario](https://multiagentcontest.org/2007/ac07-scenario-extended.txt), [protocol](https://multiagentcontest.org/2007/ac07-protocol.txt)
- **Original MASSim 2006 server** (Java), monitor and demo agents: downloads on the [2006 page](https://multiagentcontest.org/2006/).
  The current MASSim is at [github.com/agentcontest](https://github.com/agentcontest).
- **Organizer reports:**
  - M. Dastani, J. Dix, P. Novák: *The Second Contest on Multi-Agent Systems Based on Computational Logic*,
    CLIMA VII, LNAI 4371, 2007 ([doi](https://doi.org/10.1007/978-3-540-69619-3_15),
    [tech report IfI-06-13](https://multiagentcontest.org/publications/ifi0613dastani.pdf))
  - M. Dastani, J. Dix, P. Novák: *Agent Contest Competition: 3rd Edition*, ProMAS 2007, LNAI 4908, 2008
    ([doi](https://doi.org/10.1007/978-3-540-79043-3_14),
    [tech report IfI-07-15](https://multiagentcontest.org/publications/ifi0715dastani.pdf))
  - T. Behrens, M. Dastani, J. Dix, M. Köster, P. Novák: *The Multi-Agent Programming Contest from 2005–2010:
    From Collecting Gold to Herding Cows*, Annals of Mathematics and Artificial Intelligence 59, 2010
    ([doi](https://doi.org/10.1007/s10472-010-9219-5),
    [tech report IfI-10-14](https://multiagentcontest.org/publications/ifi1014behrens.pdf))
  - T. Ahlbrecht, J. Dix, N. Fiekas, T. Krausburg: *The Multi-Agent Programming Contest: A résumé*
    ([arXiv:2006.02739](https://arxiv.org/abs/2006.02739))
- **The Jason gold-miners teams** (2006 winner and 2007 entry):
  [examples/gold-miners](https://github.com/jason-lang/jason/tree/master/examples) in the Jason distribution.

Differences from the original are described above:
- explicit stepping instead of timed steps;
- JSON over HTTP instead of XML over TCP;
- point-symmetric maps;
- configurable depot policies;
- extra perception feedback (`last_action`, `last_result`).

