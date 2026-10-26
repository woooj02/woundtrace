"""Run the pipeline on real phone photos of the printed test sheets.

    venv/bin/python -m scripts.real_eval

real/a/*.jpg show sheet A (true area 7.07 cm²), real/b/*.jpg sheet B (2.66 cm²).
Writes real/out/<sheet>-<name>.jpg (outline drawn) and real/summary.md.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np

from app import policy, vision

ROOT = Path(__file__).resolve().parent.parent / "real"
TRUE = {"a": 7.07, "b": 2.66}


def main():
    rows = []
    (ROOT / "out").mkdir(exist_ok=True)
    for sheet, true in TRUE.items():
        for f in sorted(glob.glob(str(ROOT / sheet / "*.jpg"))):
            img = cv2.imread(f)
            c = vision.check_photo(img)
            m = vision.measure(img, "grabcut")
            d = policy.decide(c, m)
            if d["action"] == "cross_check":
                d = policy.decide(c, m, vision.measure(img, "colour"))
            name = f"{sheet}-{os.path.basename(f)}"
            if m.get("found"):
                cv2.imwrite(str(ROOT / "out" / name), vision.annotate(img, m))
            mm = c["metrics"]
            rows.append({"photo": name, "true": true, "area": m.get("area_cm2"),
                         "err": m["area_cm2"] / true - 1 if m.get("found") else None,
                         "action": d["action"], "reasons": d.get("reasons") or [p["code"] for p in c["problems"]],
                         "tilt": mm.get("tilt_deg"), "px_per_mm": mm.get("px_per_mm"),
                         "contrast": m.get("edge_contrast")})
    rec = [r for r in rows if r["action"] == "record"]
    L = ["# Real photos of printed test sheets", "",
         f"{len(rows)} phone photos (iPhone, ordinary room light) of two printed sheets. "
         "The printer had a colour fault (banding), so the printed skin is striped: a harder "
         "background than real skin. True areas are printed on the sheets.", "",
         f"Card found: {sum(r['px_per_mm'] is not None for r in rows)}/{len(rows)} · "
         f"recorded: {len(rec)} · bad accepts (recorded, error > 10%): "
         f"{sum(abs(r['err']) > 0.10 for r in rec)}", ""]
    if rec:
        e = np.array([abs(r["err"]) for r in rec])
        L += [f"Recorded area error: mean {100 * e.mean():.1f}%, max {100 * e.max():.1f}%", ""]
    L += ["| Photo | True cm² | Measured cm² | Error | Decision | Reasons | Tilt ° | px/mm | Edge contrast |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        err = f"{100 * r['err']:+.1f}%" if r["err"] is not None else "–"
        L.append(f"| {r['photo']} | {r['true']} | {r['area'] if r['area'] is not None else '–'} | {err} | "
                 f"{r['action']} | {', '.join(r['reasons'])} | {r['tilt']} | {r['px_per_mm']} | {r['contrast']} |")
    (ROOT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
