"""Reproducible evaluation on synthetic photos with known ground truth.

    venv/bin/python -m eval.run_eval            # 600 hard + 150 clean cases, held-out seed 11
    venv/bin/python -m eval.run_eval --seed 2026 # tuning seed (gates, uncertainty fit)
    venv/bin/python -m eval.run_eval --seed 7    # earlier held-out seed, since used to diagnose failures
    venv/bin/python -m eval.run_eval --n 100    # quick run

Every case is measured twice: as the pipeline decides (gates on), and forced
(gates off), so we can show what the gates protect against. Writes
eval/results.json and eval/summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from app import policy, synth, vision

OUT = Path(__file__).resolve().parent


def _init():
    cv2.setNumThreads(1)  # one process per core; OpenCV's own threads would oversubscribe


def one(args):
    seed, hard = args
    rng = np.random.default_rng(seed)
    img, shot = synth.random_case(rng, hard=hard)
    t = time.perf_counter()
    check = vision.check_photo(img)
    first = vision.measure(img, "grabcut")
    second = None
    d = policy.decide(check, first)
    if d["action"] == "cross_check":
        second = vision.measure(img, "colour")
        d = policy.decide(check, first, second)
    ms = (time.perf_counter() - t) * 1000
    row = {"seed": seed, "hard": hard, **shot.dict(), "action": d["action"],
           "reasons": d.get("reasons", []), "ms": round(ms),
           "metrics": check["metrics"]}
    if first.get("found"):
        row.update(area=first["area_cm2"], err=first["area_cm2"] / shot.true_cm2 - 1,
                   ruler_err=first["ruler_area_cm2"] / shot.true_cm2 - 1,
                   glare_fraction=first["glare_fraction"], edge_contrast=first["edge_contrast"],
                   perimeter_cm=first["perimeter_cm"], length_cm=first["length_cm"])
    return row


def pct(x):
    return f"{100 * x:.1f}%"


def summarise(rows):
    hard = [r for r in rows if r["hard"]]
    clean = [r for r in rows if not r["hard"]]
    acc = [r for r in hard if r["action"] == "record"]
    rej = [r for r in hard if r["action"] == "retake"]
    nurse = [r for r in hard if r["action"] == "nurse_outline"]
    forced = [r for r in hard if "err" in r]
    ae = np.array([abs(r["err"]) for r in acc]) if acc else np.array([np.nan])
    ce = np.array([abs(r["err"]) for r in clean if "err" in r and r["action"] == "record"])
    bad_accept = sum(abs(r["err"]) > 0.10 for r in acc)
    rej_bad = [r for r in rej if "err" in r]
    caught = sum(abs(r["err"]) > 0.10 for r in rej_bad)
    needless = sum(abs(r["err"]) <= 0.05 for r in rej_bad)
    reasons = {}
    for r in rej:
        for c in r["reasons"]:
            reasons[c] = reasons.get(c, 0) + 1
    ruler = np.array([r["ruler_err"] for r in acc])
    s = {
        "cases_hard": len(hard), "cases_clean": len(clean),
        "recorded": len(acc), "retake": len(rej), "nurse_outline": len(nurse),
        "recorded_mape": float(np.mean(ae)), "recorded_p95_abs_err": float(np.percentile(ae, 95)),
        "recorded_within_5pct": float(np.mean(ae <= 0.05)),
        "recorded_within_10pct": float(np.mean(ae <= 0.10)),
        "bad_accepts_over_10pct": int(bad_accept),
        "forced_mape_all_hard": float(np.mean([abs(r["err"]) for r in forced])),
        "retakes_with_forced_err_over_10pct": int(caught),
        "retakes_with_forced_err_under_5pct": int(needless),
        "retakes_measurable": len(rej_bad),
        "retake_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "clean_recorded": int(len(ce)), "clean_mape": float(np.mean(ce)) if len(ce) else None,
        "ruler_mean_overestimate": float(np.mean(ruler)) if len(ruler) else None,
        "median_ms": float(np.median([r["ms"] for r in rows])),
        "by_skin_tone": {},
    }
    for tone in range(len(synth.SKIN_TONES)):
        t = [r for r in hard if r["skin"] == tone]
        ta = [r for r in t if r["action"] == "record"]
        s["by_skin_tone"][tone + 1] = {
            "cases": len(t), "recorded": len(ta),
            "mape": float(np.mean([abs(r["err"]) for r in ta])) if ta else None,
            "card_not_found": sum("card_not_found" in r["reasons"] for r in t),
            "wound_not_found": sum("wound_not_found" in r["reasons"] for r in t),
        }
    return s


def markdown(s):
    L = ["# Evaluation summary (synthetic test targets, not patients)", "",
         f"Hard set: {s['cases_hard']} simulated phone photos with random tilt (0-60°), distance, blur, "
         f"glare and six skin tones. Clean set: {s['cases_clean']} near-ideal photos.", "",
         "| Result | Value |", "|---|---|",
         f"| Recorded / retake / nurse outline | {s['recorded']} / {s['retake']} / {s['nurse_outline']} |",
         f"| Area error of recorded measurements (mean abs) | {pct(s['recorded_mape'])} |",
         f"| 95th percentile abs error, recorded | {pct(s['recorded_p95_abs_err'])} |",
         f"| Recorded within 5% / 10% of true area | {pct(s['recorded_within_5pct'])} / {pct(s['recorded_within_10pct'])} |",
         f"| Recorded with error over 10% (bad accepts) | {s['bad_accepts_over_10pct']} |",
         f"| Same pipeline with gates off (mean abs error) | {pct(s['forced_mape_all_hard'])} |",
         f"| Retakes that would have been >10% off | {s['retakes_with_forced_err_over_10pct']} of {s['retakes_measurable']} measurable |",
         f"| Retakes that would have been within 5% (needless) | {s['retakes_with_forced_err_under_5pct']} of {s['retakes_measurable']} |",
         f"| Clean set error (mean abs) | {pct(s['clean_mape']) if s['clean_mape'] is not None else 'n/a'} |",
         f"| Ruler length × width vs true area (mean) | +{pct(s['ruler_mean_overestimate'])} |",
         f"| Median processing time per photo | {s['median_ms']:.0f} ms |", "",
         "Retake reasons: " + ", ".join(f"{k} {v}" for k, v in s["retake_reasons"].items()), "",
         "| Skin tone (1 light → 6 dark) | Cases | Recorded | Error | Card not found | Wound not found |",
         "|---|---|---|---|---|---|"]
    for k, v in s["by_skin_tone"].items():
        L.append(f"| {k} | {v['cases']} | {v['recorded']} | {pct(v['mape']) if v['mape'] is not None else 'n/a'} "
                 f"| {v['card_not_found']} | {v['wound_not_found']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    jobs = [(a.seed * 10000 + i, True) for i in range(a.n)]
    jobs += [(a.seed * 10000 + 5000 + i, False) for i in range(a.n // 4)]
    with Pool(max(1, (os.cpu_count() or 2) - 2), initializer=_init) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    s = summarise(rows)
    (OUT / "results.json").write_text(json.dumps({"summary": s, "gates": vision.GATES,
                                                  "rows": rows}, indent=1, default=float))
    (OUT / "summary.md").write_text(markdown(s))
    print(markdown(s))


if __name__ == "__main__":
    main()
