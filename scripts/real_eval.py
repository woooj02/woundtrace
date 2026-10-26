"""Run the pipeline on real phone photos of the printed test sheets.

    venv/bin/python -m scripts.real_eval

real/a, real/b: colour test sheets A (7.07 cm²) and B (2.66 cm²), faulty printer.
real/bw-medium: black-ink geometry sheet (4.11 cm²).
real/coin: an Australian $1 coin (25.00 mm, 4.91 cm²) next to a separately printed card.
That card printed at 92.3% (a ruler read one 30 mm square as 27.7 mm), so it is measured
with that manual calibration. On a test sheet the card and the target print together,
so the print scale cancels (scale 1). The coin's face is 3 mm above the paper: at
13-16 cm that alone reads about 4-5% large.
Writes real/out/<set>-<name>.jpg (outline drawn) and real/summary.md.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

from app import policy, vision

ROOT = Path(__file__).resolve().parent.parent / "real"
SETS = {"a": (7.07, 1.0), "b": (2.66, 1.0), "bw-medium": (4.11, 1.0),
        "coin": (round(math.pi * 12.5 ** 2 / 100, 3), 27.7 / 30.0)}


def main():
    rows = []
    (ROOT / "out").mkdir(exist_ok=True)
    for sheet, (true, scale) in SETS.items():
        for f in sorted(glob.glob(str(ROOT / sheet / "*.jpg"))):
            img = cv2.imread(f)
            c = vision.check_photo(img)
            m = vision.measure(img, "grabcut", card_scale=scale)
            d = policy.decide(c, m)
            if d["action"] == "cross_check":
                d = policy.decide(c, m, vision.measure(img, "colour", card_scale=scale))
            name = f"{sheet}-{os.path.basename(f)}"
            if m.get("found"):
                cv2.imwrite(str(ROOT / "out" / name), vision.annotate(img, m))
            mm = c["metrics"]
            rows.append({"photo": name, "true": true, "area": m.get("area_cm2"),
                         "sd": vision.area_sd_cm2(m, c["metrics"]) if m.get("found") else None,
                         "err": m["area_cm2"] / true - 1 if m.get("found") else None,
                         "action": d["action"], "reasons": d.get("reasons") or [p["code"] for p in c["problems"]],
                         "tilt": mm.get("tilt_deg"), "px_per_mm": mm.get("px_per_mm"),
                         "contrast": m.get("edge_contrast")})
    rec = [r for r in rows if r["action"] == "record"]
    L = ["# Real phone photos with known true areas", "",
         f"{len(rows)} iPhone photos in ordinary room light. Colour sheets A and B: the printer "
         "had a colour fault (banding), so the printed skin is striped, a harder background than "
         "real skin. bw-medium: black-ink sheet. coin: a real object (Australian $1, 25.00 mm), "
         "card calibrated by ruler (92.3% print); its 3 mm thickness alone reads ~4-5% large. "
         "Not real wounds.", "",
         f"Card found: {sum(r['px_per_mm'] is not None for r in rows)}/{len(rows)} · "
         f"recorded: {len(rec)} · bad accepts (recorded, error > 10%): "
         f"{sum(abs(r['err']) > 0.10 for r in rec)}", ""]
    if rec:
        e = np.array([abs(r["err"]) for r in rec])
        L += [f"Recorded area error: mean {100 * e.mean():.1f}%, max {100 * e.max():.1f}%", ""]
    inside = [abs(r["area"] - r["true"]) <= 2 * r["sd"] for r in rec if r["sd"]]
    if inside:
        L += [f"Recorded measurements inside their stated ±2 sd: {sum(inside)}/{len(inside)}", ""]
    L += ["| Photo | True cm² | Measured cm² | Error | Decision | Reasons | Tilt ° | px/mm | Edge contrast |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        err = f"{100 * r['err']:+.1f}%" if r["err"] is not None else "–"
        L.append(f"| {r['photo']} | {r['true']} | {r['area'] if r['area'] is not None else '–'} | {err} | "
                 f"{r['action']} | {', '.join(r['reasons'])} | {r['tilt']} | {r['px_per_mm']} | {r['contrast']} |")
    (ROOT / "summary.md").write_text("\n".join(L) + "\n")
    errs = [abs(r["err"]) for r in rec]
    summary = {"photos": len(rows), "card_found": sum(r["px_per_mm"] is not None for r in rows),
               "recorded": len(rec), "bad_accepts": sum(e > 0.10 for e in errs),
               "mape": float(np.mean(errs)) if errs else None,
               "inside_2sd": [sum(inside), len(inside)]}
    (ROOT / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
