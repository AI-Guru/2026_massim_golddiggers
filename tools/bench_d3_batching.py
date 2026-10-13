#!/usr/bin/env python3
"""Standalone benchmark: one d3 request per agent vs one batched request per team.

Takes real per-agent prompts (state + options) from a d3 harness run (decisions.jsonl) and asks
for the decisions of one team (6 agents of the same step) in three ways, several rounds each:
  sequential  6 single-question requests, one after the other
  parallel    6 single-question requests at once (threads)
  batched     1 request: shared task text + one block per miner, one question per miner
  batched-q   1 request: only the task text as state; each miner's situation goes into its
              own question's instructions
Reports wall time per round (mean ± std) and how often batched and single answers agree.

Usage: python3 tools/bench_d3_batching.py runs/<run>/decisions.jsonl [--url ...] [--rounds 10]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def split_task(state: str):
    task, _, rest = state.partition("\n")
    return task, rest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions")
    ap.add_argument("--url", default="http://192.168.0.219:11490")
    ap.add_argument("--rounds", type=int, default=10)
    args = ap.parse_args()
    base = args.url.rstrip("/")
    listed = json.loads(urllib.request.urlopen(f"{base}/v1/models", timeout=10).read())
    model = ([m["id"] for m in listed.get("data", [])]
             or [m.get("model") or m["name"] for m in listed.get("models", [])])[0]
    url = f"{base}/v1/systemone"

    rows = [json.loads(line) for line in open(args.decisions)]
    for r in rows:                            # current logs: prompt = {task, situation, options}
        if "prompt" in r and isinstance(r["prompt"], dict):
            r["state"] = r["prompt"]["task"] + "\n" + r["prompt"]["situation"]
            r["options"] = r["prompt"]["options"]
    rows = [r for r in rows if r.get("strategy") == "d3" and "state" in r]
    by_step: dict = {}
    for r in rows:
        by_step.setdefault((r["step"], r["team"]), []).append(r)
    groups = [g for g in by_step.values() if len(g) >= 6][-args.rounds:]
    if not groups:
        raise SystemExit("no step with 6 logged d3 prompts (run the harness with a d3 team)")
    print(f"model {model}, {len(groups)} rounds of {len(groups[0])} agents")

    def single(r):
        body = {"model": model, "state": r["state"], "questions": {"action": {
            "type": "choice", "instructions": "Which action should the miner take now?",
            "criteria": r["options"]}}}
        resp = post(url, body)
        return resp["answers"]["action"]["choice"], resp.get("usage", {}).get("input_tokens", 0)

    def batched(group):
        task = split_task(group[0]["state"])[0]
        blocks = [f"Miner {r['agent']}:\n{split_task(r['state'])[1]}" for r in group]
        body = {"model": model, "state": task + "\n\n" + "\n\n".join(blocks), "questions": {
            r["agent"]: {"type": "choice", "instructions": f"Which action should miner {r['agent']} take now?",
                         "criteria": r["options"]} for r in group}}
        resp = post(url, body)
        return ({a: ans["choice"] for a, ans in resp["answers"].items()},
                resp.get("usage", {}).get("input_tokens", 0))

    def batched_q(group):
        task = split_task(group[0]["state"])[0]
        body = {"model": model, "state": task, "questions": {
            r["agent"]: {"type": "choice",
                         "instructions": split_task(r["state"])[1] + "\nWhich action should this miner take now?",
                         "criteria": r["options"]} for r in group}}
        resp = post(url, body)
        return ({a: ans["choice"] for a, ans in resp["answers"].items()},
                resp.get("usage", {}).get("input_tokens", 0))

    single(groups[0][0])                                    # warm-up
    times = {"sequential": [], "parallel": [], "batched": [], "batched-q": []}
    tokens = {"single": [], "batched": [], "batched-q": []}
    agree = total = agree_q = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for group in groups:
            t = time.perf_counter()
            seq = [single(r) for r in group]
            times["sequential"].append(time.perf_counter() - t)
            t = time.perf_counter()
            list(pool.map(single, group))
            times["parallel"].append(time.perf_counter() - t)
            t = time.perf_counter()
            choices, btok = batched(group)
            times["batched"].append(time.perf_counter() - t)
            t = time.perf_counter()
            choices_q, qtok = batched_q(group)
            times["batched-q"].append(time.perf_counter() - t)
            tokens["single"].append(sum(tok for _, tok in seq))
            tokens["batched"].append(btok)
            tokens["batched-q"].append(qtok)
            for r, (choice, _) in zip(group, seq):
                total += 1
                agree += choices.get(r["agent"]) == choice
                agree_q += choices_q.get(r["agent"]) == choice

    print(f"{'mode':11} {'ms per team decision':>22}")
    for mode, ts in times.items():
        ms = [t * 1000 for t in ts]
        print(f"{mode:11} {statistics.fmean(ms):10.0f} ± {statistics.stdev(ms) if len(ms) > 1 else 0:<6.0f}")
    print("prompt tokens per team decision: " + ", ".join(
        f"{k} {statistics.fmean(v):.0f}" for k, v in tokens.items()))
    print(f"same answer as single requests: batched {agree}/{total} ({100 * agree / total:.0f}%), "
          f"batched-q {agree_q}/{total} ({100 * agree_q / total:.0f}%)")


if __name__ == "__main__":
    main()
