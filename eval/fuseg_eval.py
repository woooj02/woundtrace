"""Wound outlines on real clinical photos: the FUSeg 2021 dataset (expert-annotated foot ulcers).

    venv/bin/python -m eval.fuseg_eval                 # validation split (200 photos), every method
    venv/bin/python -m eval.fuseg_eval --method dnn    # one method

Data: Wang et al., FUSeg: The Foot Ulcer Segmentation Challenge (MICCAI 2021), de-identified photos from the
AZH Wound and Vascular Center, shared publicly by the authors for research. Download it into data/fuseg/
(see README); it is not part of this repository. These photos have no reference card, so this measures the
outline only (Dice and IoU against the experts' masks); scale is tested separately.
Writes eval/fuseg_summary.md and eval/fuseg_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from app import vision

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fuseg" / "repo" / "data" / "Foot Ulcer Segmentation Challenge"
OUT = Path(__file__).resolve().parent


def outline(img: np.ndarray, method: str) -> np.ndarray:
    """The wound mask our pipeline gives for a photo without a card (the photo is the plane)."""
    if method in ("dnn", "dnn-raw"):  # dnn-raw: the model's outline without the GrabCut snap
        return vision.dnn_mask(img, refine=method == "dnn")
    valid = (img.max(axis=2) > 8).astype(np.uint8) * 255  # FUSeg pads with black
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    seg = vision.segment(img, valid, method)
    return seg["mask"] if seg.get("found") else np.zeros(img.shape[:2], np.uint8)


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    p, t = pred > 0, truth > 0
    inter, union = np.logical_and(p, t).sum(), np.logical_or(p, t).sum()
    dice = 2 * inter / (p.sum() + t.sum()) if p.sum() + t.sum() else 1.0
    return {"dice": float(dice), "iou": float(inter / union) if union else 1.0,
            "found": bool(p.any()), "area_ratio": float(p.sum() / t.sum()) if t.sum() else None}


def one(args):
    name, method, split = args
    cv2.setNumThreads(1)
    img = cv2.imread(str(DATA / split / "images" / name))
    truth = cv2.imread(str(DATA / split / "labels" / name), cv2.IMREAD_GRAYSCALE)
    return {"photo": name, "method": method, **scores(outline(img, method), truth)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", help="colour, grabcut, dnn (default: all available)")
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()
    methods = a.method or (["colour", "grabcut"] + (["dnn-raw", "dnn"] if vision.DNN_AVAILABLE else []))
    names = sorted(os.listdir(DATA / a.split / "images"))
    jobs = [(n, m, a.split) for m in methods for n in names]
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        rows = pool.map(one, jobs, chunksize=4)
    L = ["# Real wound photos: outline accuracy on FUSeg", "",
         f"{len(names)} de-identified clinical foot-ulcer photos ({a.split} split of FUSeg 2021, Wang et al.), "
         "compared with the wound-care experts' masks. No reference card in these photos: outline only.", "",
         "| Method | Mean Dice | Mean IoU | Dice ≥ 0.7 | Missed (no outline) | Area ratio, median |",
         "|---|---|---|---|---|---|"]
    notes = ["", "colour, grabcut: the classical pipeline (Lab colour distance from the skin, then GrabCut), tuned on "
             "synthetic targets. dnn-raw: the learned segmenter alone (LR-ASPP MobileNetV3, ONNX, run by cv2.dnn). "
             "dnn: the app's default, the same model with its outline snapped to the colour edge by GrabCut. "
             "The model was trained on the FUSeg train split plus synthetic photos (eval/train_segmenter.py); "
             "these validation photos were not used for training or for choosing the model."]
    summary = {}
    for m in methods:
        r = [x for x in rows if x["method"] == m]
        d = np.array([x["dice"] for x in r])
        ratios = [x["area_ratio"] for x in r if x["found"] and x["area_ratio"]]
        summary[m] = {"n": len(r), "dice": float(d.mean()), "iou": float(np.mean([x["iou"] for x in r])),
                      "dice_ge_07": float((d >= 0.7).mean()), "missed": sum(not x["found"] for x in r),
                      "area_ratio_median": float(np.median(ratios)) if ratios else None}
        s = summary[m]
        L.append(f"| {m} | {s['dice']:.3f} | {s['iou']:.3f} | {100 * s['dice_ge_07']:.0f}% | {s['missed']} | "
                 f"{s['area_ratio_median']:.2f} |" if s["area_ratio_median"] else f"| {m} | {s['dice']:.3f} | "
                 f"{s['iou']:.3f} | {100 * s['dice_ge_07']:.0f}% | {s['missed']} | – |")
    L += notes
    (OUT / "fuseg_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    (OUT / "fuseg_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
