"""Does the vision model's outline review catch bad outlines? Real photos, expert masks as the truth.

    AWS_DEFAULT_REGION=us-east-1 venv/bin/python -m eval.review_eval      # ~400 Bedrock calls, about $0.20

For each FUSeg validation photo, the outline from the learned segmenter and from the classical colour
method (mostly wrong on real photos, so they supply bad examples) is shown to the model (app/review.py).
An outline is "good" if its Dice against the experts' mask is >= 0.7 and "bad" if < 0.4.
Writes eval/review_summary.md and eval/review_results.json.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import cv2
import numpy as np

from app import review
from eval.fuseg_eval import DATA, outline, scores

OUT = Path(__file__).resolve().parent


def main():
    names = sorted(os.listdir(DATA / "validation" / "images"))
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    cases = []
    for n in names:
        img = cv2.imread(str(DATA / "validation" / "images" / n))
        truth = cv2.imread(str(DATA / "validation" / "labels" / n), cv2.IMREAD_GRAYSCALE)
        for method in ("dnn", "grabcut"):
            mask = outline(img, method)
            if mask.any():
                cases.append((n, method, img, mask, scores(mask, truth)["dice"]))

    def run(c):
        n, method, img, mask, dice = c
        return {"photo": n, "method": method, "dice": dice, **review.review(img, mask, client)}

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(run, cases))
    good = [r for r in rows if r["dice"] >= 0.7]
    bad = [r for r in rows if r["dice"] < 0.4]
    frac = lambda rs, v: sum(r["verdict"] == v for r in rs) / max(len(rs), 1)
    s = {"reviews": len(rows), "good": len(good), "bad": len(bad),
         "bad_flagged": frac(bad, "misses"), "bad_passed": frac(bad, "follows"),
         "good_passed": frac(good, "follows"), "good_flagged": frac(good, "misses"),
         "unsure": frac(rows, "unsure"), "errors": frac(rows, "error"),
         "input_tokens": sum((r.get("usage") or {}).get("inputTokens", 0) for r in rows),
         "median_ms": float(np.median([r["ms"] for r in rows]))}
    L = ["# Outline review on real wound photos", "",
         f"{s['reviews']} outlines on {len(names)} FUSeg validation photos (learned segmenter and classical "
         f"method), each shown to {review.MODEL} as photo + outline. Truth: Dice against the experts' mask.", "",
         "| Outlines | Count | Model said follows | Model said misses |", "|---|---|---|---|",
         f"| Good (Dice ≥ 0.7) | {s['good']} | {100 * s['good_passed']:.0f}% | {100 * s['good_flagged']:.0f}% |",
         f"| Bad (Dice < 0.4) | {s['bad']} | {100 * s['bad_passed']:.0f}% | {100 * s['bad_flagged']:.0f}% |", "",
         f"Unsure: {100 * s['unsure']:.0f}% · errors: {100 * s['errors']:.0f}% · median {s['median_ms']:.0f} ms per review · "
         f"{s['input_tokens'] / max(len(rows), 1):.0f} input tokens per review."]
    (OUT / "review_results.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1, default=str))
    (OUT / "review_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
