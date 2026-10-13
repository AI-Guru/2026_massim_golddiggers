"""Rule tests on small hand-placed worlds (no randomness: all probabilities 0)."""
import pytest
from fastapi.testclient import TestClient

from golddiggers import server
from golddiggers.engine import Game, GameConfig, maps


def world(**kw):
    cfg = dict(map_type="random", width=9, height=9, obstacles=0, initial_gold=0, steps=50,
               team_size=1, p_fail=0, p_fail_max=0, p_info=0, p_info_max=0,
               gold_probability=0, seed=1)
    cfg.update(kw)
    g = Game(GameConfig(**cfg))
    g.gold.clear()
    g.obstacles.clear()
    g.depot = (7, 7)
    return g


def place(g, name, x, y, items=0):
    a = g.agents[name]
    a.x, a.y, a.items = x, y, items
    g._perceptions = g._compute_perceptions()
    return a


def test_move_pick_deliver():
    g = world()
    r = place(g, "red1", 6, 7)
    place(g, "blue1", 0, 0)
    g.gold.add((6, 7))
    g.act("red1", "pick"); g.step()
    assert r.items == 1 and (6, 7) not in g.gold
    g.act("red1", "right"); g.step()
    assert (r.x, r.y) == (7, 7)
    g.act("red1", "drop"); g.step()
    assert g.score["red"] == 1 and r.items == 0
    g.act("red1", "left"); g.step()
    assert (r.x, r.y) == (6, 7) and r.last_result == "ok"


def test_depot_closed_without_gold():
    g = world()
    r = place(g, "red1", 6, 7)
    place(g, "blue1", 0, 0)
    g.act("red1", "right"); g.step()
    assert (r.x, r.y) == (6, 7) and r.last_result == "impossible"


def test_depot_teleport_if_no_drop():
    g = world()
    r = place(g, "red1", 6, 7, items=1)
    place(g, "blue1", 0, 0)
    g.act("red1", "right"); g.step()
    g.act("red1", "skip"); g.step()
    assert (r.x, r.y) != (7, 7) and r.last_result == "teleported" and r.items == 1


def test_depot_teleport_if_not_leaving():
    g = world()
    r = place(g, "red1", 6, 7, items=1)
    place(g, "blue1", 0, 0)
    g.act("red1", "right"); g.step()
    g.act("red1", "drop"); g.step()
    g.step()   # skip although a neighbour is free
    assert (r.x, r.y) != (7, 7) and g.stats["teleports"] == 1


def test_carry_limit_and_drop_all_in_depot():
    g = world(max_carry=3)
    r = place(g, "red1", 7, 6, items=3)
    place(g, "blue1", 0, 0)
    g.gold.add((7, 6))
    g.act("red1", "pick"); g.step()
    assert r.items == 3 and r.last_result == "impossible"
    g.act("red1", "down"); g.step()
    g.act("red1", "drop"); g.step()
    assert g.score["red"] == 3


def test_push():
    g = world()
    r = place(g, "red1", 2, 2)
    b = place(g, "blue1", 3, 2)
    g.act("red1", "right"); g.step()
    assert (r.x, r.y) == (3, 2) and (b.x, b.y) == (4, 2) and b.last_result == "pushed"


def test_no_push_against_obstacle_or_when_disabled():
    g = world()
    r = place(g, "red1", 2, 2)
    b = place(g, "blue1", 3, 2)
    g.obstacles.add((4, 2))
    g.act("red1", "right"); g.step()
    assert (r.x, r.y) == (2, 2) and (b.x, b.y) == (3, 2)
    g = world(pushing=False)
    r = place(g, "red1", 2, 2)
    b = place(g, "blue1", 3, 2)
    g.act("red1", "right"); g.step()
    assert (r.x, r.y) == (2, 2)


def test_acting_agent_cannot_be_pushed():
    g = world()
    r = place(g, "red1", 2, 2)
    b = place(g, "blue1", 3, 2)
    g.gold.add((3, 2))
    g.act("red1", "right"); g.act("blue1", "pick"); g.step()
    assert (b.x, b.y) == (3, 2) and b.items == 1 and (r.x, r.y) == (2, 2)


def test_mutual_push_coin_flip():
    g = world()
    r = place(g, "red1", 2, 2)
    b = place(g, "blue1", 3, 2)
    g.act("red1", "right"); g.act("blue1", "left"); g.step()
    assert {(r.x, r.y), (b.x, b.y)} in ({(3, 2), (4, 2)}, {(1, 2), (2, 2)})


def test_conflict_one_winner():
    g = world()
    r = place(g, "red1", 2, 2)
    b = place(g, "blue1", 4, 2)
    g.act("red1", "right"); g.act("blue1", "left"); g.step()
    assert sorted([(r.x, r.y), (b.x, b.y)]) in ([(2, 2), (3, 2)], [(3, 2), (4, 2)])


def test_fatigue_probability():
    g = world(p_fail=0.1, p_fail_max=0.5, max_carry=3)
    assert g._prob(0.1, 0.5, 0) == pytest.approx(0.1)
    assert g._prob(0.1, 0.5, 3) == pytest.approx(0.5)
    assert g._prob(0.1, 0.5, 1) == pytest.approx(0.2333, abs=1e-3)


def test_perception_and_marks():
    g = world()
    place(g, "red1", 0, 0)
    place(g, "blue1", 1, 0)
    g.act("red1", "mark", "abcdefg"); g.step()
    p = g.perceive("red1")
    assert set(p["cells"]) == {"cur", "e", "s", "se"}
    assert p["cells"]["e"] == ["enemy"]
    assert {"mark": "abcde"} in p["cells"]["cur"]


def test_info_distortion_is_stable_within_a_step():
    g = world(p_info=0.5, p_info_max=0.5)
    place(g, "red1", 3, 3)
    assert g.perceive("red1") == g.perceive("red1")


@pytest.mark.parametrize("map_type", maps.MAP_TYPES)
@pytest.mark.parametrize("seed", range(5))
def test_maps_are_fair(map_type, seed):
    n = 41
    g = Game(GameConfig(map_type=map_type, width=n, height=n, obstacles=200, initial_gold=60,
                        seed=seed))
    rot = g.sym.rot
    assert g.obstacles and g.obstacles == {rot(c) for c in g.obstacles}
    assert g.gold == {rot(c) for c in g.gold} and len(g.gold) == 60
    assert g.depot == (n // 2, n // 2)
    c = n // 2
    assert not any((x, y) in g.obstacles for x in range(c - 2, c + 3) for y in range(c - 2, c + 3))
    red = sorted((a.x, a.y) for a in g.agents.values() if a.team == "red")
    blue = sorted((a.x, a.y) for a in g.agents.values() if a.team == "blue")
    assert sorted(rot(p) for p in red) == blue
    assert all(x + y < 6 for x, y in red)                 # red in the (0, 0) corner
    assert not g.obstacles & {(x, y) for x in range(4) for y in range(4)}


def test_even_or_non_square_size_rejected():
    for w, h in ((30, 30), (31, 33)):
        with pytest.raises(ValueError):
            GameConfig(width=w, height=h).validate()


@pytest.mark.parametrize("map_type", maps.MAP_TYPES)
def test_maps_connected(map_type):
    g = Game(GameConfig(map_type=map_type, width=35, height=35, seed=3))
    free = maps.reachable(g.depot, g.obstacles, 35, 35)
    for a in g.agents.values():
        assert (a.x, a.y) in free
    assert all(cell in free for cell in g.gold)


def test_game_ends_with_result():
    g = world(steps=2)
    g.step(); g.step()
    assert g.finished and g.perceive("red1")["result"] == "draw"
    with pytest.raises(RuntimeError):
        g.step()


def test_api_roundtrip():
    c = TestClient(server.app)
    r = c.post("/api/startgame", json={"preset": "mapc2006", "seed": 5, "steps": 3})
    assert r.status_code == 200 and len(r.json()["agents"]) == 8
    p = c.get("/api/perceive/red1").json()
    assert p["step"] == 0 and "cells" in p and p["sim"]["opponent"] == "blue"
    assert c.post("/api/act/red1", json={"action": "fly"}).status_code == 422
    assert c.post("/api/act/nobody", json={"action": "up"}).status_code == 404
    assert c.post("/api/act/red1", json={"action": "up"}).json()["status"] == "queued"
    s = c.post("/api/stepgame").json()
    assert s["step"] == 1 and s["actions"]["red1"]["requested"] == "up"
    c.post("/api/stepgame"); c.post("/api/stepgame")
    assert c.post("/api/stepgame").status_code == 409
    st = c.get("/api/state").json()
    assert st["finished"] and len(st["history"]) == 4
    assert c.post("/api/startgame", json={"bogus": 1}).status_code == 422


def boxed_in(policy, target="ring", **kw):
    """red1 delivers on the depot while four blue agents (with gold, waiting) surround it."""
    g = world(team_size=4, pushing=False, depot_policy=policy, teleport_target=target, **kw)
    g.depot = (4, 4)
    g.depot_distance = g._walking_distances(g.depot)
    for a in g.agents.values():
        a.x, a.y = 0, 0
    place(g, "red1", 4, 4, items=1)
    g.agents["red1"].in_depot_must_drop = True
    for name, cell in zip(("blue1", "blue2", "blue3", "blue4"), ((3, 4), (5, 4), (4, 3), (4, 5))):
        place(g, name, *cell, items=1)
    for i, name in enumerate(("red2", "red3", "red4")):
        place(g, name, 8, 6 + i)
    return g


def test_original_policy_deadlocks_when_boxed_in():
    g = boxed_in("original")
    g.act("red1", "drop"); g.step()
    for _ in range(3):
        g.act("red1", "up"); g.step()
    assert (g.agents["red1"].x, g.agents["red1"].y) == (4, 4) and g.stats["teleports"] == 0


def test_leave_or_teleport_breaks_the_deadlock():
    g = boxed_in("leave_or_teleport")
    g.act("red1", "drop"); g.step()
    g.act("red1", "up"); g.step()
    r = g.agents["red1"]
    assert (r.x, r.y) != (4, 4) and r.last_result == "teleported"
    assert g.depot_distance[(r.x, r.y)] == g.config.teleport_radius


def test_teleport_on_drop_and_home_target():
    g = boxed_in("teleport_on_drop", target="home")
    g.act("red1", "drop"); g.step()
    r = g.agents["red1"]
    assert g.score["red"] == 1 and (r.x, r.y) in g.starts["red"]


def test_team_names_api():
    c = TestClient(server.app)
    c.post("/api/startgame", json={"seed": 1})
    assert c.post("/api/setteamname/red", json={"name": "greedy"}).json()["name"] == "greedy"
    assert c.post("/api/setteamname/green", json={"name": "x"}).status_code == 404
    assert c.post("/api/setteamname/blue", json={"name": "<script>"}).status_code == 422
    assert c.get("/api/state").json()["team_names"] == {"red": "greedy", "blue": "blue"}


def test_stats_and_field_api():
    c = TestClient(server.app)
    c.post("/api/startgame", json={"preset": "mapc2007", "seed": 3, "steps": 20})
    c.post("/api/act/red1", json={"action": "down"})
    c.post("/api/stepgame")
    s = c.get("/api/stats").json()
    assert s["step"] == 1 and "history" not in s and set(s["teams"]) == {"red", "blue"}
    assert s["agents"]["red1"]["actions"] == 1 and s["gold_on_map"] >= 0
    assert len(c.get("/api/stats?history=true").json()["history"]) == 2
    f = c.get("/api/field").json()
    assert len(f["rows"]) == 41 and all(len(r) == 41 for r in f["rows"])
    assert f["rows"][20][20] == "D" and sum(r.count("A") for r in f["rows"]) == 6
