#!/usr/bin/env python3
"""Round-robin tournament and comparison matrix (strategy x versus strategy y).

  play    play every pairing of the given strategies on the given seeds, each with both sides
          (red/blue), through agents/harness.py; games that already exist in runs/ (same red,
          blue, seed, steps and map size) are skipped
  matrix  build the matrix from all finished runs that match the settings: cell (row A, col B)
          = A's results against B over all games, whichever side A played

Writes runs/tournament/matrix.md, matrix.json and matrix.png (win rate heatmap with the scores).

Usage (from golddiggers/):
  uv run python tools/tournament.py play   --strategies random,greedy,d3,d3-memory --seeds 1,2
  uv run python tools/tournament.py matrix --strategies random,greedy,d3,d3-memory
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = RUNS / "tournament"


def finished_games(steps: int, size: int, preset_seeds: set | None = None, since: str = "") -> list[dict]:
    games = []
    for summary in RUNS.glob("*/summary.json"):
        if summary.parent.name < since:                   # run dirs start with YYYYMMDD-HHMMSS
            continue
        try:
            s = json.loads(summary.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cfg = s.get("config") or {}
        if not s.get("finished") or cfg.get("steps") != steps or cfg.get("width") != size:
            continue
        red, blue = cfg["teams"]
        if red not in s["strategies"] or blue not in s["strategies"]:
            continue                                      # only part of the teams was controlled
        if preset_seeds is not None and cfg.get("seed") not in preset_seeds:
            continue
        games.append({"red": s["strategies"][red], "blue": s["strategies"][blue], "seed": cfg.get("seed"),
                      "score_red": s["score"][red], "score_blue": s["score"][blue], "run": summary.parent.name})
    return games


def play(args):
    strategies = args.strategies.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    have = {(g["red"], g["blue"], g["seed"]) for g in finished_games(args.steps, args.size, since=args.since)}
    pairs = list(itertools.combinations(strategies, 2)) + ([(s, s) for s in strategies] if args.mirror else [])
    todo = [(r, b, seed) for a, c in pairs for seed in seeds
            for r, b in ({(a, c), (c, a)}) if (r, b, seed) not in have]
    print(f"{len(todo)} games to play ({len(have)} matching games exist)", flush=True)
    for i, (red, blue, seed) in enumerate(sorted(todo), start=1):
        start = {"preset": args.preset, "seed": seed, "steps": args.steps}
        cmd = [sys.executable, str(ROOT / "agents/harness.py"), "--start", json.dumps(start),
               "--red", red, "--blue", blue, "--print-every", "100000"]
        if args.no_record:
            cmd.append("--no-record")
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
        line = next((ln for ln in out.splitlines() if ln.startswith("final score")), "FAILED")
        print(f"[{i}/{len(todo)}] {red} vs {blue} seed {seed}: {line}", flush=True)


def matrix(args):
    strategies = args.strategies.split(",")
    seeds = {int(x) for x in args.seeds.split(",")} if args.seeds else None
    games = [g for g in finished_games(args.steps, args.size, seeds, args.since)
             if g["red"] in strategies and g["blue"] in strategies]
    cells = {(a, b): {"games": 0, "wins": 0, "draws": 0, "losses": 0, "score": 0, "opponent": 0}
             for a in strategies for b in strategies}
    for g in games:
        for me, other, mine, theirs in ((g["red"], g["blue"], g["score_red"], g["score_blue"]),
                                        (g["blue"], g["red"], g["score_blue"], g["score_red"])):
            if me == other and (me, other) == (g["blue"], g["red"]):
                continue                                  # mirror game: count once
            c = cells[(me, other)]
            c["games"] += 1
            c["score"] += mine
            c["opponent"] += theirs
            c["wins" if mine > theirs else "losses" if mine < theirs else "draws"] += 1
    totals = {}
    for a in strategies:
        row = [cells[(a, b)] for b in strategies if b != a]
        n = sum(c["games"] for c in row)
        points = sum(c["wins"] + 0.5 * c["draws"] for c in row)
        totals[a] = {"games": n, "win_rate": points / n if n else None,
                     "mean_score": sum(c["score"] for c in row) / n if n else None}
    ranking = sorted(strategies, key=lambda a: -(totals[a]["win_rate"] or -1))

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"# Comparison matrix ({len(games)} games, {args.steps} steps, {args.size}x{args.size})", "",
             "Cell = row strategy against column strategy: wins-draws-losses, mean score row : column.", "",
             "| | " + " | ".join(ranking) + " | overall win rate |",
             "|---|" + "---|" * (len(ranking) + 1)]
    for a in ranking:
        cols = []
        for b in ranking:
            c = cells[(a, b)]
            if not c["games"]:
                cols.append("–")
            else:
                cols.append(f"{c['wins']}-{c['draws']}-{c['losses']}<br>{c['score'] / c['games']:.0f} : "
                            f"{c['opponent'] / c['games']:.0f}")
        wr = totals[a]["win_rate"]
        lines.append(f"| **{a}** | " + " | ".join(cols) + f" | {'–' if wr is None else f'{100 * wr:.0f}%'} |")
    (OUT / "matrix.md").write_text("\n".join(lines) + "\n")
    (OUT / "matrix.json").write_text(json.dumps(
        {"strategies": ranking, "totals": totals, "games": games,
         "cells": {f"{a} vs {b}": c for (a, b), c in cells.items()}}, indent=1))
    print("\n".join(lines))
    try:
        heatmap(ranking, cells, OUT / "matrix.png", len(games), args)
        print(f"\n{OUT / 'matrix.png'}")
    except ImportError:
        pass


def heatmap(ranking, cells, path, n_games, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # diverging: blue (row loses) - grey midpoint (even) - red (row wins); validated pair poles
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    k = len(ranking)
    fig, ax = plt.subplots(figsize=(2.2 + 1.9 * k, 1.2 + 1.5 * k), constrained_layout=True)
    data = [[None] * k for _ in range(k)]
    for i, a in enumerate(ranking):
        for j, b in enumerate(ranking):
            c = cells[(a, b)]
            if c["games"] and a != b:
                data[i][j] = (c["wins"] + 0.5 * c["draws"]) / c["games"]
    import numpy as np
    arr = np.array([[np.nan if v is None else v for v in row] for row in data], dtype=float)
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
    for i, a in enumerate(ranking):
        for j, b in enumerate(ranking):
            c = cells[(a, b)]
            if a == b or not c["games"]:
                ax.text(j, i, "–", ha="center", va="center", color="#77756f")
                continue
            ax.text(j, i, f"{100 * arr[i, j]:.0f}% wins\n{c['score'] / c['games']:.0f} : {c['opponent'] / c['games']:.0f}\n"
                    f"{c['wins']}-{c['draws']}-{c['losses']} of {c['games']}",
                    ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(arr[i, j] - 0.5) > 0.35 else "#0b0b0b")
    ax.set_xticks(range(k), ranking)
    ax.set_yticks(range(k), ranking)
    ax.set_xlabel("against (column)")
    ax.set_ylabel("strategy (row)")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_title(f"Row vs column: win rate, mean score, W-D-L  ({n_games} games, {args.steps} steps)",
                 fontsize=10, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["play", "matrix"])
    ap.add_argument("--strategies", default="random,greedy,d3,d3-memory")
    ap.add_argument("--seeds", default="1,2", help="play: seeds to play; matrix: only these seeds (default all)")
    ap.add_argument("--preset", default="mapc2007")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--size", type=int, default=41, help="map size the games must have (mapc2007: 41)")
    ap.add_argument("--mirror", action="store_true", help="also play each strategy against itself")
    ap.add_argument("--since", default="", help="only runs from this time on, e.g. 20261013-1242 "
                                                "(strategies changed before; older games do not count)")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    if args.mode == "matrix" and "--seeds" not in sys.argv:
        args.seeds = None
    (play if args.mode == "play" else matrix)(args)


if __name__ == "__main__":
    main()
