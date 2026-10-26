"""Render each step of the OpenCV pipeline for one photo, for the report and the video.

    venv/bin/python -m scripts.pipeline_frames PHOTO [OUT_DIR] [--scale S]

Writes 1-photo.png ... 7-calibration.png (1920x1080, dark background, a caption on each): what the
code actually computes on that photo, step by step.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from app import card, vision

W, H = 1920, 1080


def frame(img, title, sub="", side=()):
    """Fit an image into a 1920x1080 frame with a title bar, and optional readings in a side panel."""
    canvas = np.full((H, W, 3), (20, 21, 11), np.uint8)
    top, panel = 120, (600 if side else 0)
    k = min((W - 80 - panel) / img.shape[1], (H - top - 40) / img.shape[0])
    small = cv2.resize(img, (round(img.shape[1] * k), round(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    y0 = top + (H - top - 40 - small.shape[0]) // 2
    x0 = 40 if side else (W - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    for i, line in enumerate(side):
        cv2.putText(canvas, line, (W - panel + 20, top + 80 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    (220, 230, 225), 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (40, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (191, 212, 45), 3, cv2.LINE_AA)
    cv2.putText(canvas, sub, (40, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 230, 225), 2, cv2.LINE_AA)
    return canvas


def crop_to(img, pts, margin=0.25):
    """Crop to the bounding box of the given points (card corners, wound outline) plus a margin."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    m = margin * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    x1, y1 = int(min(img.shape[1], x1 + m)), int(min(img.shape[0], y1 + m))
    return img[y0:y1, x0:x1]


def label(img, text, org, scale=1.0):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("out", nargs="?", default="submission/video/assets/pipeline")
    ap.add_argument("--scale", type=float, default=1.0, help="card print scale, if known")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(a.photo)
    t = max(2, round(max(img.shape[:2]) / 500))
    found = vision.find_card(img)
    m = vision.measure(img, card_scale=a.scale)
    cal = vision.find_id1_card(img)
    poi = [q for q in found.get("corners", {}).values()]
    if m.get("found"):
        poi.append(cv2.perspectiveTransform(cv2.findNonZero(m["_mask"]).reshape(-1, 1, 2).astype(np.float32),
                                            np.linalg.inv(m["_M"])).reshape(-1, 2))
    if cal.get("found"):
        poi.append(np.array(cal["corners_px"]))
    poi = np.vstack([np.asarray(p, np.float32).reshape(-1, 2) for p in poi])
    cut = lambda im: crop_to(im, poi)
    cv2.imwrite(str(out / "1-photo.png"), frame(cut(img), "1  The photo", "Phone camera, the printed card beside the wound"))

    # 2. Markers, pose axes, quality readings
    chk = vision.check_photo(img)
    m2 = img.copy()
    K = vision._intrinsics(img.shape)
    for mid, q in found.get("corners", {}).items():
        cv2.polylines(m2, [q.astype(np.int32)], True, (255, 160, 0), t + 1)
        obj = np.c_[card.marker_corners_mm(mid), np.zeros(4)].astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, q, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            cv2.drawFrameAxes(m2, K, None, rvec, tvec, 25, t + 3)
    mt = chk["metrics"]
    readings = [f"tilt  {mt.get('tilt_deg', 0):.0f} deg", "   solvePnP (IPPE)",
                f"scale  {mt.get('px_per_mm', 0):.1f} px/mm",
                f"blur  {mt.get('edge_mm', 0):.2f} mm", "   marker edge rise width",
                f"flatness  {mt.get('card_rms_mm', 0):.2f} mm", "   homography residual",
                "check: " + ("PASSED" if chk["ok"] else ", ".join(p['code'] for p in chk['problems']))]
    cv2.imwrite(str(out / "2-markers.png"), frame(cut(m2), "2  Find the card, check the photo",
                                                  "ArUco markers (sub-pixel corners) and their 3D pose axes",
                                                  side=readings))

    # 3. Rectified plane with a mm grid
    flat, valid, M = vision.rectify(img, found["H"])
    ppoi = [cv2.findNonZero(vision._card_mask(flat.shape, pad=0)).reshape(-1, 2)]
    if m.get("found"):
        ppoi.append(cv2.findNonZero(m["_mask"]).reshape(-1, 2))
    ppoi = np.vstack(ppoi)
    pcut = lambda im: crop_to(im, ppoi, 0.15)
    g = flat.copy()
    s = vision.RECT_PX_PER_MM
    for x in range(0, g.shape[1], round(10 * s)):
        cv2.line(g, (x, 0), (x, g.shape[0]), (255, 255, 255), 1)
    for y in range(0, g.shape[0], round(10 * s)):
        cv2.line(g, (0, y), (g.shape[1], y), (255, 255, 255), 1)
    cv2.imwrite(str(out / "3-flattened.png"), frame(pcut(g), "3  Flatten to true millimetres",
                                                    "Homography from the 8 marker corners -> warpPerspective at 6 px/mm (grid: 10 mm)"))

    # 4. Wound probability (learned segmenter) or colour distance (classical)
    if vision.DNN_AVAILABLE:
        prob = cv2.warpPerspective(vision.dnn_prob(img).astype(np.float32), M, (flat.shape[1], flat.shape[0]))
        heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        sub = "Learned segmenter (MobileNetV3 LR-ASPP, ONNX) run by OpenCV 5's DNN module, warped onto the plane"
    else:
        lab = cv2.cvtColor(flat, cv2.COLOR_BGR2Lab).astype(np.float32)
        d = np.linalg.norm(lab - np.median(lab[valid > 0], axis=0), axis=2)
        heat = cv2.applyColorMap(np.clip(d * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        sub = "Colour distance from the skin (Lab)"
    cv2.imwrite(str(out / "4-probability.png"), frame(pcut(cv2.addWeighted(flat, 0.35, heat, 0.65, 0)), "4  Where is the wound?", sub))
    if vision.DNN_AVAILABLE and m.get("found"):
        sn = flat.copy()
        raw = ((prob > 0.5) * 255).astype(np.uint8)
        cs, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sn, cs, -1, (0, 200, 255), 2)
        cs, _ = cv2.findContours(m["_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(sn, cs, -1, (60, 255, 60), 2)
        cv2.imwrite(str(out / "4b-snap.png"), frame(crop_to(sn, cv2.findNonZero(m["_mask"]).reshape(-1, 2), 0.6),
                                                    "4b  Snap to the edge",
                                                    "Model at 512 px (orange) -> GrabCut seeded by its probabilities, at 6 px/mm (green)"))

    # 5. Outline on the plane with metrics
    o = flat.copy()
    if m.get("found"):
        cs, _ = cv2.findContours(m["_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, (60, 255, 60), 3)
        x, y, w, h = cv2.boundingRect(m["_mask"])
        label(o, f"{m['area_cm2']:.2f} cm2   {m['length_cm']:.1f} x {m['width_cm']:.1f} cm",
              (max(10, x - 40), max(40, y - 20)), 1.2)
    cv2.imwrite(str(out / "5-outline.png"), frame(pcut(o), "5  Outline and measure",
                                                  f"Largest confident region, holes filled; area, Feret length, width, perimeter ({m.get('method', '')})"))

    # 6. Back on the photo
    if m.get("found"):
        cv2.imwrite(str(out / "6-result.png"), frame(cut(vision.annotate(img, m)), "6  Back on the photo",
                                                     "The outline projected through the inverse homography"))

    # 7. Plastic card, if present
    if cal.get("found"):
        c7 = img.copy()
        cv2.polylines(c7, [np.round(np.array(cal["corners_px"])).astype(np.int32)], True, (255, 0, 255), t + 2)
        side = ["plastic card (ID-1)", f"  measured {cal['measured_mm'][0]:.1f} x {cal['measured_mm'][1]:.1f}",
                "  in printed-card mm", "  true 85.6 x 54.0 mm", "", f"card printed at {cal['scale']:.1%}",
                f"areas x {cal['scale'] ** 2:.3f}"]
        cv2.imwrite(str(out / "7-calibration.png"), frame(cut(c7), "7  Check the printed card's size",
                                                          "Canny + contours -> a rounded rectangle with the ID-1 aspect ratio",
                                                          side=side))
    print(f"wrote {out}/ (area {m.get('area_cm2')}, method {m.get('method')})")


if __name__ == "__main__":
    main()
