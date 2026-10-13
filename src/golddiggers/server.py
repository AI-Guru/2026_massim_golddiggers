"""HTTP/JSON API for the Gold Miners simulation, plus the web frontend at /.

Agents:
  GET  /api/perceive/{agent}   perception of the current step (+ sim info)
  POST /api/act/{agent}        {"action": "up", "param": null}; replaces an earlier queued action
Game control:
  POST /api/startgame          GameConfig fields as JSON (all optional) -> agents and sim info
  POST /api/stepgame           resolve all queued actions (missing = skip) -> step summary
  POST /api/setteamname/{team} {"name": "greedy"}: display name of a team (e.g. its strategy)
  GET  /api/state              full state for the frontend (positions, history, stats)
  GET  /api/config             defaults, presets, action list

Run: uv run golddiggers --port 8642
"""
from __future__ import annotations

import argparse
import threading
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import ACTIONS, Game, GameConfig

STATIC = Path(__file__).resolve().parent / "static"

# Parameter sets close to the original contests (maps are generated, not the originals;
# sizes are rounded up to odd because maps are point-symmetric around a central depot).
PRESETS = {
    "mapc2006": {"map_type": "random", "width": 25, "height": 25, "obstacles": 20,
                 "initial_gold": 75, "steps": 500, "team_size": 4, "max_carry": 1,
                 "pushing": False, "p_fail": 0.02, "p_fail_max": 0.02, "p_info": 0.1,
                 "p_info_max": 0.1, "gold_frequency": 3, "gold_probability": 1.0,
                 "gold_number": 1},
    "mapc2007": {"map_type": "random", "width": 41, "height": 41, "obstacles": 95,
                 "initial_gold": 175, "steps": 800, "team_size": 6, "max_carry": 3,
                 "pushing": True, "p_fail": 0.05, "p_fail_max": 0.33, "p_info": 0.1,
                 "p_info_max": 0.3, "gold_frequency": 10, "gold_probability": 1.0,
                 "gold_number": 3},
    "labyrinth": {"map_type": "labyrinth", "width": 31, "height": 31, "initial_gold": 45,
                  "steps": 500, "p_info": 0.01, "p_info_max": 0.2},
    "rows": {"map_type": "rows", "width": 35, "height": 35, "initial_gold": 60, "steps": 500},
}

app = FastAPI(title="Gold Diggers", description="MAPC Gold Miners scenario server")
lock = threading.Lock()
game: Game | None = None


class TeamName(BaseModel):
    name: str


class Action(BaseModel):
    action: str
    param: str | None = None


def current() -> Game:
    if game is None:
        raise HTTPException(409, "no game running: POST /api/startgame first")
    return game


@app.post("/api/startgame")
def startgame(params: dict | None = None):
    global game
    params = dict(params or {})
    preset = params.pop("preset", None)
    if preset is not None:
        if preset not in PRESETS:
            raise HTTPException(422, f"unknown preset {preset!r}, expected one of {list(PRESETS)}")
        params = {**PRESETS[preset], **params}
    known = {f.name for f in fields(GameConfig)}
    unknown = set(params) - known
    if unknown:
        raise HTTPException(422, f"unknown parameters {sorted(unknown)}, known: {sorted(known)}")
    try:
        new = Game(GameConfig(**params))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from None
    with lock:
        game = new
        return {"status": "started", "config": asdict(new.config),
                "agents": {n: new.sim_info(n) for n in new.agents}}


@app.post("/api/stepgame")
def stepgame():
    with lock:
        g = current()
        if g.finished:
            raise HTTPException(409, "game is finished")
        return g.step()


@app.post("/api/setteamname/{team}")
def setteamname(team: str, body: TeamName):
    with lock:
        g = current()
        try:
            g.set_team_name(team, body.name)
        except KeyError:
            raise HTTPException(404, f"unknown team {team!r}, teams are {g.config.teams}") from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        return {"status": "ok", "team": team, "name": g.team_names[team]}


@app.get("/api/perceive/{agent}")
def perceive(agent: str):
    with lock:
        g = current()
        if agent not in g.agents:
            raise HTTPException(404, f"unknown agent {agent!r}")
        return {**g.perceive(agent), "sim": g.sim_info(agent),
                "queued": g.queued.get(agent, (None,))[0]}


@app.post("/api/act/{agent}")
def act(agent: str, body: Action):
    with lock:
        g = current()
        if agent not in g.agents:
            raise HTTPException(404, f"unknown agent {agent!r}")
        try:
            g.act(agent, body.action, body.param)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None
        return {"status": "queued", "agent": agent, "step": g.step_no, "action": body.action}


@app.get("/api/state")
def state():
    with lock:
        if game is None:
            return {"running": False}
        return {"running": True, **game.state()}


@app.get("/api/config")
def config():
    return {"defaults": asdict(GameConfig()), "presets": PRESETS, "actions": list(ACTIONS)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8642)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
