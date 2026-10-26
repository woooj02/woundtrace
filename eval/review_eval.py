"""Does the vision model's outline check catch bad outlines? Real photos, expert masks as the truth.

    AWS_DEFAULT_REGION=us-east-1 venv/bin/python -m eval.review_eval      # ~300 Bedrock calls, about $0.15

For each FUSeg validation photo, the outline from the learned segmenter and from the classical colour
method (mostly wrong on real photos, so they supply bad examples) is checked by app/review.py: the model
marks the wound on the plain photo and code compares its box with the outline's. An outline is "good" if
its Dice against the experts' mask is >= 0.7 and "bad" if < 0.4. The 40 outlines used to design the check
(eval/review_dev.json) are left out. Writes eval/review_summary.md and eval/review_results.json.
(The first design, showing the model the overlay and asking whether it follows the wound, passed every
outline, good or bad: eval/review_overlay_summary.md.)
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
    dev = {tuple(c) for c in json.loads((OUT / "review_dev.json").read_text())["cases"]}
    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    cases = []
    for n in names:
        img = cv2.imread(str(DATA / "validation" / "images" / n))
        truth = cv2.imread(str(DATA / "validation" / "labels" / n), cv2.IMREAD_GRAYSCALE)
        for method in ("dnn", "grabcut"):
            if (n, method) in dev:
                continue
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
         "errors": frac(rows, "error"), "min_box_iou": review.MIN_BOX_IOU,
         "input_tokens": sum((r.get("usage") or {}).get("inputTokens", 0) for r in rows),
         "median_ms": float(np.median([r["ms"] for r in rows]))}
    L = ["# Outline check on real wound photos", "",
         f"{s['reviews']} outlines on {len(names)} FUSeg validation photos (learned segmenter and classical "
         f"method). {review.MODEL} is shown the plain photo and marks the wound; the outline passes if the "
         f"boxes overlap (IoU ≥ {review.MIN_BOX_IOU}). Truth: Dice against the experts' mask. The 40 outlines "
         "used to design the check are left out.", "",
         "| Outlines | Count | Passed (agree) | Flagged for the nurse |", "|---|---|---|---|",
         f"| Good (Dice ≥ 0.7) | {s['good']} | {100 * s['good_passed']:.0f}% | {100 * s['good_flagged']:.0f}% |",
         f"| Bad (Dice < 0.4) | {s['bad']} | {100 * s['bad_passed']:.0f}% | {100 * s['bad_flagged']:.0f}% |", "",
         f"Errors: {100 * s['errors']:.0f}% · median {s['median_ms']:.0f} ms per check · "
         f"{s['input_tokens'] / max(len(rows), 1):.0f} input tokens per check.", "",
         "First design (the model shown the outline overlay and asked whether it follows the wound): it said "
         "\"follows\" for 100% of good and 100% of bad outlines (eval/review_overlay_summary.md), so it was replaced."]
    (OUT / "review_results.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1, default=str))
    (OUT / "review_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
