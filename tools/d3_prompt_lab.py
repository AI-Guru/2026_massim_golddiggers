#!/usr/bin/env python3
"""Offline prompt lab for the d3 strategy: score prompt variants on logged situations.

Reads decision facts from harness runs (d3 logs them always, greedy with --greedy-log-facts),
samples situations stratified by type, renders each with every prompt variant (agents/d3.py
PROMPTS), asks the d3 server (batched: 6 situations per request, like a team in the game) and
scores the answers against a reference: the set of sensible actions in that situation
  on the depot with gold    -> drop
  standing on gold, room    -> pick
  full                      -> the moves that get closest to the depot
  carrying some             -> moves closest to the depot or to the nearest known gold
  empty, gold known         -> moves closest to the gold
  empty, no gold known      -> moves closest to unexplored area
Prints accuracy per variant and situation type, the most common wrong answers, tokens and time.

Usage: uv run python tools/d3_prompt_lab.py runs/*/decisions.jsonl [--per-type 30] [--variants v0,v1,v2]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agents"))
from d3 import PROMPTS, http_json, list_models  # noqa: E402
from harness import MOVES  # noqa: E402


def situation_type(f: dict) -> str:
    if f["on_depot"] and f["items"]:
        return "on depot with gold"
    if "pick" in f["actions"]:
        return "standing on gold"
    if f["items"] >= f["max_carry"]:
        return "full"
    if f["items"]:
        return "carrying some"
    return "empty, gold known" if f["now"]["gold"] is not None else "empty, no gold known"


def best_moves(f: dict, key: str) -> set:
    moves = {a: i[key] for a, i in f["actions"].items() if a in MOVES and i.get(key) is not None}
    if not moves:
        return set()
    m = min(moves.values())
    return {a for a, v in moves.items() if v == m}


def reference(f: dict) -> set:
    t = situation_type(f)
    if t == "on depot with gold":
        return {"drop"}
    if t == "standing on gold":
        return {"pick"}
    if t == "full":
        return best_moves(f, "depot")
    if t == "carrying some":
        return best_moves(f, "depot") | best_moves(f, "gold")
    if t == "empty, gold known":
        return best_moves(f, "gold")
    return best_moves(f, "unknown")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", nargs="+")
    ap.add_argument("--url", default="http://192.168.0.219:11490")
    ap.add_argument("--variants", default=",".join(PROMPTS))
    ap.add_argument("--per-type", type=int, default=30, help="situations per type")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=0, help="print N rendered example prompts per variant")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    model = list_models(base)[0][0]

    facts = []
    seen = set()
    for path in args.decisions:
        for line in open(path):
            r = json.loads(line)
            f = r.get("facts")
            if not f or not f["actions"]:
                continue
            key = json.dumps({k: f[k] for k in ("pos", "items", "actions", "now")}, sort_keys=True)
            if key not in seen and reference(f):
                seen.add(key)
                facts.append(f)
    rng = random.Random(args.seed)
    by_type = collections.defaultdict(list)
    for f in facts:
        by_type[situation_type(f)].append(f)
    sample = []
    for t, fs in sorted(by_type.items()):
        rng.shuffle(fs)
        sample += fs[:args.per_type]
    print(f"model {model}; {len(facts)} distinct situations, sampled {len(sample)}: "
          + ", ".join(f"{t} {min(len(fs), args.per_type)}" for t, fs in sorted(by_type.items())))

    def ask(variant, chunk):
        render = PROMPTS[variant]
        rendered = [render(f) for f in chunk]
        body = {"model": model, "state": rendered[0][0], "questions": {
            f"m{i}": {"type": "choice", "criteria": opts,
                      "instructions": situation + "\nWhich action should this miner take now?"}
            for i, (_, situation, opts) in enumerate(rendered)}}
        t0 = time.perf_counter()
        resp = http_json(f"{base}/v1/systemone", body, timeout=120)
        ms = (time.perf_counter() - t0) * 1000
        return [(f, resp["answers"][f"m{i}"]["choice"], rendered[i]) for i, f in enumerate(chunk)], \
            resp.get("usage", {}).get("input_tokens", 0), ms

    for variant in args.variants.split(","):
        # a request shares one task text: group by max_carry (the only fact in the task)
        chunks = []
        for mc in sorted({f["max_carry"] for f in sample}):
            group = [f for f in sample if f["max_carry"] == mc]
            chunks += [group[i:i + args.batch] for i in range(0, len(group), args.batch)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda c: ask(variant, c), chunks))
        answers = [a for res, _, _ in results for a in res]
        tokens = sum(t for _, t, _ in results) / len(answers)
        ms = statistics.fmean(m for _, _, m in results)
        ok = collections.Counter()
        total = collections.Counter()
        wrong = collections.Counter()
        for f, choice, _ in answers:
            t = situation_type(f)
            total[t] += 1
            if choice in reference(f):
                ok[t] += 1
            else:
                wrong[(t, choice if choice not in MOVES else "move (worse)")] += 1
        acc = sum(ok.values()) / len(answers)
        print(f"\n=== {variant}: {100 * acc:.0f}% sensible  ({tokens:.0f} tokens/situation, "
              f"{ms:.0f} ms per request of {args.batch})")
        for t in sorted(total):
            print(f"  {t:22} {ok[t]:3}/{total[t]:<3} {100 * ok[t] / total[t]:4.0f}%")
        print("  most common mistakes: " + ", ".join(f"{t} -> {c} ({n})" for (t, c), n in wrong.most_common(5)))
        for f, choice, (task, situation, opts) in answers[:args.show]:
            print(f"\n  --- example ({situation_type(f)}, answered {choice}, sensible {sorted(reference(f))})")
            print("  " + situation.replace("\n", "\n  "))
            for k, v in opts.items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
