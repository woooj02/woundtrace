"""Real wound examples for the README and the video, from the CC BY 4.0 lower-limb dataset.

    venv/bin/python -m scripts.examples_figure     # after eval.external_eval; writes docs/real_examples.png

Picks the wound photos at the 90th, 50th and 10th percentile of our Dice, so the figure shows a typical
result and a poor one, not only the best, plus one held-out healthy foot. Only this dataset is shown as
pictures: its CC BY 4.0 licence allows reuse with credit; FUSeg and CO2Wounds (CC BY-NC) appear as numbers only.
Photos: Lower Limb and Feet Wound Image Dataset, Islam et al., Mendeley Data 10.17632/hsj38fwnvr.3.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app import vision
from eval.external_eval import LL

ROOT = Path(__file__).resolve().parent.parent
TILE = 420


def tile(img, masks, lines):
    t = img.copy()
    for mask, colour, th in masks:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, colour, th, cv2.LINE_AA)
    t = cv2.resize(t, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
    bar = np.full((70, TILE, 3), 255, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([t, bar])


def main():
    rows = json.loads((ROOT / "eval" / "external_results.json").read_text())["rows"]
    ll = sorted((r for r in rows if r["dataset"] == "lowerlimb"), key=lambda r: r["dice"])
    picks = [ll[int(q * (len(ll) - 1))] for q in (0.9, 0.5, 0.1)]
    cols = []
    for r, q in zip(picks, ("90th", "50th", "10th")):
        img = cv2.imread(str(LL / "wound_main" / r["photo"]))
        truth = (cv2.imread(str(LL / "wound_mask" / r["photo"].replace("main", "mask")), cv2.IMREAD_GRAYSCALE) > 127)
        truth = truth.astype(np.uint8) * 255
        ours = vision.dnn_mask(img)
        cols.append(tile(img, [(truth, (255, 255, 255), 5), (ours, (60, 230, 60), 2)],
                         [f"{q} percentile: Dice {r['dice']:.2f}", "dataset mask (white), ours (green)"]))
    healthy = next(r for r in rows if r["dataset"] == "healthy feet" and not r["found"])
    img = cv2.imread(str(LL / "Nomal" / healthy["photo"]))
    cols.append(tile(img, [(vision.dnn_mask(img), (60, 230, 60), 2)],
                     ["healthy foot (held out)", "no outline drawn: correct"]))
    gap = np.full((cols[0].shape[0], 16, 3), 255, np.uint8)
    out = np.hstack([c for col in cols for c in (col, gap)][:-1])
    foot = np.full((40, out.shape[1], 3), 255, np.uint8)
    cv2.putText(foot, "Photos: Lower Limb and Feet Wound Image Dataset (Islam et al., Mendeley Data, CC BY 4.0). "
                "Never used to train our model. Its masks include skin around the wound.", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    path = ROOT / "docs" / "real_examples.png"
    cv2.imwrite(str(path), np.vstack([out, foot]))
    print(path, [(r["photo"], round(r["dice"], 2)) for r in picks], healthy["photo"])


if __name__ == "__main__":
    main()
