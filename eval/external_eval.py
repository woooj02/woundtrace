"""Wound outlines on two public datasets the model never saw, from other clinics and other wound types.

    venv/bin/python -m eval.external_eval

- CO2Wounds-V2 (Sánchez-Jiménez et al., 2024; Mendeley Data 10.17632/s2w7rjwz49.2, CC BY-NC 3.0): chronic
  wounds of leprosy patients, photographed with smartphones by medical staff in a hospital in Colombia.
  Its own validation split (122 photos with masks). The current model also trained on its train split.
- Lower Limb and Feet Wound Image Dataset (Islam et al., 2025; Mendeley Data 10.17632/hsj38fwnvr.3, CC BY
  4.0): 2,686 lower-limb wound photos (several wound types) with masks, and 2,757 photos of healthy feet.
  No photo is a near-duplicate of a FUSeg photo (grey-thumbnail correlation, max 0.93, checked by eye).

  The lower-limb wound photos are never used in training: their masks include the skin around the wound,
  unlike FUSeg's and CO2Wounds' (and unlike the open-wound area a nurse records), so they are test-only.
Healthy feet have no wound, so the right answer is "no outline": this measures false alarms. Every other
healthy photo (sorted by name) is used in training as a no-wound example; only the other half is tested here.
Skin tone is estimated per photo from the skin around the wound (ITA°, Chardon 1991) in uncalibrated light,
so the bins are approximate. The datasets are not in this repository; download them into data/ (see README).
Writes eval/external_summary.md and eval/external_results.json.
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from app import vision
from eval.fuseg_eval import scores

ROOT = Path(__file__).resolve().parent.parent
CO2 = ROOT / "data" / "co2wounds" / "x" / "CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients"
LL = ROOT / "data" / "lowerlimb" / "x"
OUT = Path(__file__).resolve().parent

# ITA° bins (Del Bino et al. 2006): very light > 55 > light > 41 > intermediate > 28 > tan > 10 > brown > -30 > dark
ITA_BINS = [(55, "very light"), (41, "light"), (28, "intermediate"), (10, "tan"), (-30, "brown"), (-999, "dark")]


def jobs():
    for n in sorted(os.listdir(CO2 / "split" / "val")):
        yield "co2wounds", str(CO2 / "imgs" / n), str(CO2 / "masks" / (n.rsplit(".", 1)[0] + ".png"))
    for n in sorted(os.listdir(LL / "wound_main")):
        yield "lowerlimb", str(LL / "wound_main" / n), str(LL / "wound_mask" / n.replace("main", "mask"))
    for n in sorted(os.listdir(LL / "Nomal"))[1::2]:
        yield "healthy feet", str(LL / "Nomal" / n), None


def ita(img: np.ndarray, truth: np.ndarray) -> float | None:
    """Skin tone from a ring of skin around the wound (median L*, b* in CIELAB)."""
    k = max(3, int(0.04 * max(img.shape[:2])))
    ring = cv2.dilate(truth, np.ones((3 * k, 3 * k), np.uint8)) & ~cv2.dilate(truth, np.ones((k, k), np.uint8))
    if ring.sum() == 0:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, b = lab[..., 0][ring > 0] * 100 / 255, lab[..., 2][ring > 0] - 128
    return float(np.degrees(np.arctan2(np.median(L) - 50, np.median(b))))


def one(job):
    ds, ip, mp = job
    cv2.setNumThreads(1)
    img = cv2.imread(ip)
    pred = vision.dnn_mask(img)
    row = {"dataset": ds, "photo": os.path.basename(ip)}
    if mp is None:
        row["found"] = bool(pred.any())
        row["area_frac"] = float((pred > 0).mean())
        return row
    truth = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > (0 if ds == "co2wounds" else 127)).astype(np.uint8) * 255
    if truth.shape != pred.shape:
        truth = cv2.resize(truth, pred.shape[::-1], interpolation=cv2.INTER_NEAREST)
    p, t = pred > 0, truth > 0
    row.update(scores(pred, truth), ita=ita(img, truth),
               precision=float((p & t).sum() / p.sum()) if p.any() else None,
               recall=float((p & t).sum() / t.sum()) if t.any() else None)
    return row


def tone(a):
    return next(name for lo, name in ITA_BINS if a > lo)


def main():
    with Pool(int(os.environ.get("EVAL_PROCS", max(1, (os.cpu_count() or 2) - 2)))) as pool:
        rows = pool.map(one, list(jobs()), chunksize=8)
    L = ["# Real wound photos from two other datasets (never used in training)", "",
         "The app's segmenter (cv2.dnn + GrabCut snap) against each dataset's own masks. Outline only: these "
         "photos have no WoundTrace card.", "",
         "| Dataset | Photos | Mean Dice | Median Dice | Dice ≥ 0.7 | Missed (no outline) | Precision | Recall |",
         "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for ds in ("co2wounds", "lowerlimb"):
        r = [x for x in rows if x["dataset"] == ds]
        d = np.array([x["dice"] for x in r])
        prec = [x["precision"] for x in r if x["precision"] is not None]
        summary[ds] = {"n": len(r), "dice": float(d.mean()), "dice_median": float(np.median(d)),
                       "dice_ge_07": float((d >= 0.7).mean()), "missed": sum(not x["found"] for x in r),
                       "precision": float(np.mean(prec)), "recall": float(np.mean([x["recall"] for x in r]))}
        s = summary[ds]
        L.append(f"| {ds} | {s['n']} | {s['dice']:.3f} | {s['dice_median']:.3f} | {100 * s['dice_ge_07']:.0f}% | "
                 f"{s['missed']} | {s['precision']:.2f} | {s['recall']:.2f} |")
    h = [x for x in rows if x["dataset"] == "healthy feet"]
    fa = [x for x in h if x["found"]]
    summary["healthy feet"] = {"n": len(h), "false_outline": len(fa),
                               "false_outline_median_area_frac": float(np.median([x["area_frac"] for x in fa])) if fa else None}
    L += ["", f"Healthy feet (no wound): an outline was drawn on {len(fa)} of {len(h)} photos "
          f"({100 * len(fa) / len(h):.1f}%)" + (f"; median size {100 * summary['healthy feet']['false_outline_median_area_frac']:.2f}% of the photo." if fa else "."),
          "", "## By estimated skin tone (both wound datasets)", "",
          "| Skin tone (ITA°) | Photos | Mean Dice | Dice ≥ 0.7 | Missed |", "|---|---|---|---|---|"]
    by = {}
    for x in rows:
        if x.get("ita") is not None:
            by.setdefault(tone(x["ita"]), []).append(x)
    summary["by_tone"] = {}
    for _, name in ITA_BINS:
        r = by.get(name, [])
        if not r:
            continue
        d = np.array([x["dice"] for x in r])
        summary["by_tone"][name] = {"n": len(r), "dice": float(d.mean()), "dice_ge_07": float((d >= 0.7).mean()),
                                    "missed": sum(not x["found"] for x in r)}
        L.append(f"| {name} | {len(r)} | {d.mean():.3f} | {100 * (d >= 0.7).mean():.0f}% | {summary['by_tone'][name]['missed']} |")
    L += ["", "Skin tone is estimated from the skin around each wound (ITA°, Del Bino bins) in uncalibrated clinic "
          "light, so a photo can land one bin off. The two datasets mark wounds differently from FUSeg (the model's "
          "training data), so part of the lower Dice is a difference in where annotators draw the edge.",
          "", "Sources: CO2Wounds-V2 (Mendeley Data 10.17632/s2w7rjwz49.2, CC BY-NC 3.0); Lower Limb and Feet "
          "Wound Image Dataset (Mendeley Data 10.17632/hsj38fwnvr.3, CC BY 4.0)."]
    L += ["", "Precision: the share of the outlined pixels that are inside the dataset's mask. Recall: the share of "
          "the mask that was outlined. High precision with low recall means the outline sits on the wound but "
          "covers less than the annotators marked.", "", f"Model: {vision.MODEL_PATH.name}."]
    tag = os.environ.get("EVAL_TAG", "")
    (OUT / f"external_results{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    (OUT / f"external_summary{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
