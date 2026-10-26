"""Three cited FUSeg examples for the README and the video: the experts' outline, ours, and the classical one.

    venv/bin/python -m scripts.fuseg_figure      # after eval.fuseg_eval; writes docs/fuseg_examples.png

Picks the validation photos at the 90th, 50th and 10th percentile of the learned outline's Dice, so the
figure shows a typical result and a poor one, not only the best. Photos: FUSeg 2021 (Wang et al.), AZH Wound
and Vascular Center, shared publicly by the authors with the clinic's permission.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from eval.fuseg_eval import DATA, outline

ROOT = Path(__file__).resolve().parent.parent
TILE = 420


def tile(img, masks, caption):
    t = img.copy()
    for mask, colour, th in masks:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, colour, th, cv2.LINE_AA)
    t = cv2.resize(t, (TILE, TILE), interpolation=cv2.INTER_AREA)
    bar = np.full((44, TILE, 3), 255, np.uint8)
    cv2.putText(bar, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([t, bar])


def main():
    rows = json.loads((ROOT / "eval" / "fuseg_results.json").read_text())["rows"]
    dnn = sorted((r for r in rows if r["method"] == "dnn"), key=lambda r: r["dice"])
    classical = {r["photo"]: r["dice"] for r in rows if r["method"] == "grabcut"}
    picks = [dnn[int(q * (len(dnn) - 1))] for q in (0.9, 0.5, 0.1)]
    cols = []
    for r in picks:
        img = cv2.imread(str(DATA / "validation" / "images" / r["photo"]))
        truth = (cv2.imread(str(DATA / "validation" / "labels" / r["photo"]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) * 255
        ours, old = outline(img, "dnn"), outline(img, "grabcut")
        cols.append(np.vstack([
            tile(img, [], f"FUSeg validation {r['photo']}"),
            tile(img, [(truth, (255, 255, 255), 3), (ours, (60, 230, 60), 2)],
                 f"experts (white) vs ours (green): Dice {r['dice']:.2f}"),
            tile(img, [(truth, (255, 255, 255), 3), (old, (40, 40, 240), 2)],
                 f"classical colour method (red): Dice {classical[r['photo']]:.2f}" if old.any()
                 else "classical colour method: no outline found")]))
    gap = np.full((cols[0].shape[0], 16, 3), 255, np.uint8)
    out = np.hstack([c for col in cols for c in (col, gap)][:-1])
    foot = np.full((40, out.shape[1], 3), 255, np.uint8)
    cv2.putText(foot, "Photos: FUSeg 2021 (Wang et al.), AZH Wound and Vascular Center. 90th / 50th / 10th "
                "percentile of our Dice on the 200 validation photos.", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (80, 80, 80), 1, cv2.LINE_AA)
    path = ROOT / "docs" / "fuseg_examples.png"
    cv2.imwrite(str(path), np.vstack([out, foot]))
    print(path, [(r["photo"], round(r["dice"], 2)) for r in picks])


if __name__ == "__main__":
    main()
