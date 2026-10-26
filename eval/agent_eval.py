"""Evaluate the agent itself, not just the measurements.

    venv/bin/python -m eval.agent_eval            # Bedrock vs the rules engine, 8 scenarios x 6
    venv/bin/python -m eval.agent_eval --n 2      # quick run

Each case is a synthetic photo plus a patient history. The same case runs through
the rules engine (the clinic's policy, deterministic) and through the Bedrock agent,
from the same starting state. We report:
  - outcome agreement: same final result (recorded or not, retake, which nurse request);
  - guard refusals: tool calls the plain-code guards rejected (and whether the agent
    recovered to the right outcome anyway);
  - grounded numbers: every decimal number in the patient message appears in a tool output;
  - steps, latency and tokens per visit.
Needs AWS credentials with Bedrock access. Writes eval/agent_results.json and eval/agent_summary.md.
Uses a temporary local store, never the deployed bucket.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

from app import agent, patients, store, synth

OUT = Path(__file__).resolve().parent

# name, photo settings, history as [(days before today, area)], area today
SCENARIOS = [
    ("baseline", {}, [], 6.0),
    ("on_track", {}, [(28, 8.0), (14, 5.5)], 3.6),
    ("stalled", {}, [(28, 6.6), (14, 6.4)], 6.2),
    ("worsening", {}, [(21, 5.0), (7, 4.6)], 6.0),
    ("tilted", {"tilt": 56}, [(14, 6.0)], 5.5),
    ("too_far", {"ppm": 2.0}, [(14, 6.0)], 5.5),
    ("blurry", {"blur": 5.0}, [(14, 6.0)], 5.5),
    ("glare", {"glare": True}, [(14, 6.0)], 5.5),
]


def make_photo(rng, area, skin, tilt=12.0, ppm=6.0, blur=0.0, glare=False):
    plane, mm2 = synth.render_plane(rng, area, skin)
    shot = synth.Shot(tilt, ppm, blur, glare, skin, mm2 / 100)
    return synth.photograph(rng, plane, 8.0, shot), mm2 / 100


def reset(history, today_area_true=None):
    for k in store.keys("alerts/"):
        store.delete(k)
    start = date.today() - timedelta(days=max([d for d, _ in history], default=0))
    hist = [{"date": (date.today() - timedelta(days=d)).isoformat(),
             "day": (date.today() - timedelta(days=d) - start).days, "area_cm2": a,
             "area_sd_cm2": round(0.02 * a, 2)} for d, a in history]
    patients.save({"id": "e1", "name": "Eval", "wound": "test target", "demo": "evaluation",
                   "start_date": start.isoformat(), "history": hist})


def summary_key(run):
    o = run["outcome"] or {}
    alerts = sorted(store.get_json(k)["kind"] for k in store.keys("alerts/"))
    recorded = len(patients.get("e1")["history"])
    return {"kind": o.get("kind"), "alerts": alerts, "records": recorded}


def numbers(text):
    return [float(x) for x in re.findall(r"\d+\.\d+", text or "")]


def grounded(run):
    """Every decimal in the patient message must appear in some tool output (to 0.01)."""
    blob = json.dumps([s.get("output") for s in run["steps"]], default=str)
    seen = [float(x) for x in re.findall(r"-?\d+\.\d+", blob)]
    seen += [abs(v) * 100 for v in seen]  # fractions shown as percentages

    def matches(n, v):  # equal, or v rounded to the decimals the message used
        places = len(f"{n}".split(".")[1])
        return abs(n - v) < 0.011 or abs(n - round(v, places)) < 1e-9
    return all(any(matches(n, v) for v in seen) for n in numbers(run["answer"]))


def run_case(engine, history, img_bytes):
    reset(history)
    store.put("photos/eval.jpg", img_bytes, "image/jpeg")
    os.environ["AGENT_ENGINE"] = engine
    run = agent.run("e1", "eval")
    return run, summary_key(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="variants per scenario")
    a = ap.parse_args()
    store.LOCAL = Path(tempfile.mkdtemp(prefix="agent-eval-"))
    os.environ.pop("BUCKET", None)
    rows = []
    for name, settings, history, area in SCENARIOS:
        for i in range(a.n):
            rng = np.random.default_rng(1000 * len(rows) + i)
            img, true = make_photo(rng, area, skin=i % 6, **settings)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rules, want = run_case("rules", history, buf.tobytes())
            llm, got = run_case("bedrock", history, buf.tobytes())
            refusals = [s for s in llm["steps"] if "error" in s]
            row = {"scenario": name, "variant": i, "skin": i % 6, "true_cm2": true,
                   "expected": want, "got": got, "agree": want == got, "engine": llm["engine"],
                   "note": llm.get("note"), "steps": len([s for s in llm["steps"] if s["tool"] != "agent_note"]),
                   "rules_steps": len(rules["steps"]), "refusals": [s["error"] for s in refusals],
                   "grounded": grounded(llm), "ms": llm["ms"], "usage": llm.get("usage"),
                   "answer": llm["answer"], "model_answer": llm.get("model_answer"),
                   "message_replaced": llm.get("message_replaced"),
                   "tools": [s["tool"] for s in llm["steps"]]}
            rows.append(row)
            print(f"{name:10s} v{i} skin{i % 6} agree={row['agree']} steps={row['steps']} "
                  f"refusals={len(refusals)} grounded={row['grounded']} {row['ms']} ms")
    llm_rows = [r for r in rows if r["engine"] == "bedrock"]
    tok_in = sum((r["usage"] or {}).get("inputTokens", 0) for r in llm_rows)
    tok_out = sum((r["usage"] or {}).get("outputTokens", 0) for r in llm_rows)
    s = {
        "cases": len(rows), "ran_on_bedrock": len(llm_rows), "model": agent.MODEL,
        "agreement": sum(r["agree"] for r in llm_rows) / max(len(llm_rows), 1),
        "by_scenario": {n: f"{sum(r['agree'] for r in llm_rows if r['scenario'] == n)}/"
                           f"{sum(1 for r in llm_rows if r['scenario'] == n)}" for n, *_ in SCENARIOS},
        "cases_with_refusal": sum(bool(r["refusals"]) for r in llm_rows),
        "refusals_then_correct": sum(bool(r["refusals"]) and r["agree"] for r in llm_rows),
        "grounded_messages": sum(r["grounded"] for r in llm_rows) / max(len(llm_rows), 1),
        "messages_replaced": sum(bool(r["message_replaced"]) for r in llm_rows),
        "median_steps": float(np.median([r["steps"] for r in llm_rows])) if llm_rows else None,
        "median_ms": float(np.median([r["ms"] for r in llm_rows])) if llm_rows else None,
        "p95_ms": float(np.percentile([r["ms"] for r in llm_rows], 95)) if llm_rows else None,
        "input_tokens_per_visit": tok_in / max(len(llm_rows), 1),
        "output_tokens_per_visit": tok_out / max(len(llm_rows), 1),
    }
    (OUT / "agent_results.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1, default=str))
    if not llm_rows:
        print(f"No visit reached Bedrock ({rows[0]['note'] if rows else 'no cases'}); nothing to compare.")
        return
    L = ["# Agent evaluation: Bedrock agent vs the clinic's rules", "",
         f"{s['cases']} visits ({len(SCENARIOS)} scenarios × {a.n} variants, six skin tones), synthetic "
         f"photos. Model: `{s['model']}`. Each visit also ran through the deterministic rules engine "
         "from the same state; that result is the expected one.", "",
         "| Result | Value |", "|---|---|",
         f"| Same outcome as the rules (record / retake / nurse request) | {100 * s['agreement']:.0f}% |",
         f"| Visits where a guard refused a tool call | {s['cases_with_refusal']} "
         f"({s['refusals_then_correct']} still ended correctly) |",
         f"| Patient messages whose numbers all come from tool outputs | {100 * s['grounded_messages']:.0f}% |",
         f"| Model replies replaced by the message guard (leaked internals, opinion, length) | "
         f"{s['messages_replaced']} |",
         f"| Median tool calls per visit | {s['median_steps']} |",
         f"| Latency per visit, median / p95 | {s['median_ms']:.0f} / {s['p95_ms']:.0f} ms |",
         f"| Tokens per visit, input / output | {s['input_tokens_per_visit']:.0f} / {s['output_tokens_per_visit']:.0f} |",
         "", "| Scenario | Agreement |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["by_scenario"].items()]
    bad = [r for r in llm_rows if not r["agree"]]
    if bad:
        L += ["", "Disagreements:", ""]
        L += [f"- {r['scenario']} v{r['variant']}: expected {r['expected']}, got {r['got']}; tools {r['tools']}"
              for r in bad]
    if len(llm_rows) < len(rows):
        L += ["", f"{len(rows) - len(llm_rows)} visits fell back to the rules engine (Bedrock unavailable)."]
    (OUT / "agent_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
