#!/usr/bin/env python3
"""Render a recorded harness run: one frame per step (field screenshot left, numbers right), or a video.

A frame for step n shows the recorded screenshot of the field and, on the right, the score and
key numbers at step n plus charts up to step n: score over time, gold on the map and carried,
decision time per step (team mean, log scale) and prompt length per step (team mean).

Usage (from golddiggers/):
  uv run --extra agents python tools/render_run.py runs/<run> --step 120          # -> runs/<run>/frame_000120.png
  uv run --extra agents python tools/render_run.py runs/<run> --video             # -> runs/<run>/video.mp4
  uv run --extra agents python tools/render_run.py runs/<run> --video --fps 20 --workers 8

10 fps is the default: one simulation step per frame keeps single moves readable, and an 800-step
game becomes an 80 s video. The last frame is held for 2 s.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

W, H = 1920, 1080
FIELD = 1032                                      # field image side in the frame
BG, PANEL, GRID = "#121211", "#1a1a19", "#2a2a28"
TEXT, TEXT2, MUTED = "#f0efec", "#c3c2b7", "#8f8d85"
TEAM = ["#e66767", "#3987e5"]                     # validated dark-mode categorical pair
GOLD = "#e8b930"


@lru_cache(maxsize=4)
def load_run(run: str):
    """All step JSONs of a run, as time series."""
    d = Path(run)
    meta = json.loads((d / "meta.json").read_text())
    steps = sorted((d / "steps").glob("*.json"))
    frames = [json.loads(p.read_text()) for p in steps]
    teams = meta["config"]["teams"]
    s = {"step": [f["step"] for f in frames], "gold_on_map": [f["server"]["gold_on_map"] for f in frames]}
    for t in teams:
        s[f"score_{t}"] = [f["server"]["score"][t] for f in frames]
        s[f"carried_{t}"] = [f["server"]["teams"][t]["carried"] for f in frames]
        s[f"ms_{t}"] = [f["client"]["teams"].get(t, {}).get("ms_mean") if f["client"]["teams"].get(t, {}).get("decisions") else None
                        for f in frames]
        s[f"tokens_{t}"] = [f["client"]["teams"].get(t, {}).get("tokens_mean") if f["client"]["teams"].get(t, {}).get("decisions") else None
                            for f in frames]
    return meta, frames, s, [str(p.with_suffix(".png")) for p in steps]


def render_frame(run: str, index: int):
    """PIL image of the frame for the index-th recorded step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    meta, frames, s, pngs = load_run(run)
    f = frames[index]
    teams = meta["config"]["teams"]
    names = {t: f["server"]["teams"][t]["name"] for t in teams}
    total_steps = meta["config"]["steps"]
    n = index + 1

    frame = Image.new("RGB", (W, H), BG)
    field = Image.open(pngs[index]).convert("RGB").resize((FIELD, FIELD), Image.LANCZOS)
    frame.paste(field, (24, (H - FIELD) // 2))

    pw = W - FIELD - 72
    fig = plt.figure(figsize=(pw / 100, (H - 48) / 100), dpi=100, facecolor=BG)
    plt.rcParams.update({"font.family": "DejaVu Sans Mono", "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    gs = fig.add_gridspec(6, 1, height_ratios=[1.6, 1, 1, 1, 1, 1], hspace=0.65,
                          left=0.1, right=0.92, top=0.98, bottom=0.05)

    head = fig.add_subplot(gs[0])
    head.axis("off")
    head.text(0, 0.95, f"step {f['step']} / {total_steps}", fontsize=15, color=MUTED, va="top")
    for i, t in enumerate(teams):
        x = 0 if i == 0 else 1
        ha = "left" if i == 0 else "right"
        head.text(x, 0.68, f"{names[t]} ({t})", fontsize=14, color=TEAM[i], ha=ha, va="top", weight="bold")
        head.text(x, 0.45, str(f["server"]["score"][t]), fontsize=40, color=TEXT, ha=ha, va="top", weight="bold")
    head.text(0.5, 0.36, ":", fontsize=34, color=MUTED, ha="center", va="top")
    c = f["server"]["counters"]
    fail = f"{100 * c['failed'] / c['actions']:.1f}%" if c["actions"] else "–"
    head.text(0, 0.0, f"gold on map {f['server']['gold_on_map']}   carried {f['server']['carried']}   "
              f"delivered {c['deliveries']}   generated {c['generated_gold']}",
              fontsize=10.5, color=TEXT2, va="top")
    head.text(0, -0.17, f"failed actions {fail}   impossible {c['impossible']}   pushes {c['pushes']}   "
              f"teleports {c['teleports']}", fontsize=10.5, color=TEXT2, va="top")

    from matplotlib.ticker import MaxNLocator

    def chart(pos, title, keys, log=False, extra=None, integer=False):
        ax = fig.add_subplot(gs[pos], facecolor=PANEL)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT2, pad=6)
        ax.set_xlim(0, total_steps)
        if integer:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        xs = s["step"][:n]
        drew = False
        series = [s[f"{keys}_{t}"][:n] for t in teams] if keys else []
        if keys and not any(y for ys in series for y in ys if y is not None):
            ax.text(0.5, 0.5, "all 0 (no model prompts)" if keys == "tokens" else "no data",
                    transform=ax.transAxes, ha="center", va="center", color=MUTED)
            ax.set_yticks([])
            return ax
        last = []
        for i, (t, ys) in enumerate(zip(teams, series)):
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None and (not log or y > 0)]
            if pts:
                ax.plot(*zip(*pts), color=TEAM[i], linewidth=2, label=f"{names[t]} ({t})")
                last.append((pts[-1], i))
                drew = True
        last.sort(key=lambda item: item[0][1])           # end labels: the higher value goes up
        for k, ((x, y), i) in enumerate(last):
            dy = 0 if len(last) == 1 else (-6 if k == 0 else 6)
            ax.annotate(f"{y:.3g}", (x, y), xytext=(5, dy), textcoords="offset points",
                        va="center", fontsize=9, color=TEAM[i])
        if extra:
            extra(ax, xs)
        if log and drew:
            ax.set_yscale("log")
        return ax

    def gold_on_map(ax, xs):
        ys = s["gold_on_map"][:n]
        ax.plot(xs, ys, color=GOLD, linewidth=2)
        ax.annotate(str(ys[-1]), (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=GOLD)

    ax = chart(1, "score (gold delivered)", "score", integer=True)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TEXT2)
    chart(2, "gold carried per team", "carried", integer=True)
    chart(3, "gold on the map", None, extra=gold_on_map, integer=True)
    chart(4, "decision time per step, team mean (ms, log scale)", "ms", log=True)
    ax = chart(5, "prompt length per decision, team mean (tokens)", "tokens")
    ax.set_xlabel("step", fontsize=9)

    fig.canvas.draw()
    panel = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    frame.paste(panel, (FIELD + 48, 24))
    return frame


def _write(job):
    run, index, path = job
    render_frame(run, index).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory (runs/<time>_<red>-vs-<blue>)")
    ap.add_argument("--step", type=int, help="render this step as a PNG")
    ap.add_argument("--video", action="store_true", help="render all steps as an MP4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    run = str(Path(args.run))
    meta, frames, s, pngs = load_run(run)
    if not frames:
        raise SystemExit(f"{run} has no recorded steps (steps/*.json)")

    if args.step is not None:
        index = s["step"].index(args.step) if args.step in s["step"] else None
        if index is None:
            raise SystemExit(f"step {args.step} not recorded (have {s['step'][0]}..{s['step'][-1]})")
        out = args.out or f"{run}/frame_{args.step:06d}.png"
        render_frame(run, index).save(out)
        print(out)
    if args.video:
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found")
        out = args.out if args.out and args.step is None else f"{run}/video.mp4"
        hold = int(args.hold * args.fps)
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [(run, i, f"{tmp}/{i:06d}.png") for i in range(len(frames))]
            with ProcessPoolExecutor(args.workers) as pool:
                for k, _ in enumerate(pool.map(_write, jobs, chunksize=4), start=1):
                    if k % 100 == 0 or k == len(jobs):
                        print(f"\rframes {k}/{len(jobs)}", end="", flush=True)
            print()
            for j in range(hold):
                shutil.copy(jobs[-1][2], f"{tmp}/{len(frames) + j:06d}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                            "-i", f"{tmp}/%06d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", "-movflags", "+faststart", out], check=True)
        print(out)
    if args.step is None and not args.video:
        sys.exit("pass --step N and/or --video")


if __name__ == "__main__":
    main()
