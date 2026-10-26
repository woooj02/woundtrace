"""Card calibration on synthetic photos: a card printed at the wrong size, a plastic ID-1 card beside it.

    venv/bin/python -m eval.calibration_eval

For each print scale and tilt: the scale the plastic card gives, and the wound area before and
after correction, against the true area.
"""

from __future__ import annotations

import numpy as np

from app import synth, vision


def main():
    print("# Card calibration on synthetic photos (`venv/bin/python -m eval.calibration_eval`)\n")
    print("| Card printed at | Tilt | Measured scale | True cm² | Uncorrected | Corrected |")
    print("|---|---|---|---|---|---|")
    for ps in (0.925, 0.96, 1.0, 1.04):
        for tilt in (5, 20, 35):
            rng = np.random.default_rng(int(ps * 1000) + tilt)
            plane, mm2 = synth.render_plane(rng, 6.0, 1, size_mm=(230.0, 150.0), print_scale=ps,
                                            id1_at=(128.0, 58.0))
            true = mm2 / 100
            img = synth.photograph(rng, plane, 8.0, synth.Shot(tilt, 4.5, 0.0, False, 1, true))
            cal = vision.find_id1_card(img)
            raw = vision.measure(img)
            fixed = vision.measure(img, card_scale=cal.get("scale", 1.0),
                                   exclude=[cal["corners_px"]] if cal.get("found") else None)
            err = lambda m: f"{m['area_cm2']} ({100 * (m['area_cm2'] / true - 1):+.1f}%)" if m.get("found") else "–"
            print(f"| {ps:.1%} | {tilt}° | {cal.get('scale', cal.get('reason'))} | {true:.2f} | {err(raw)} | {err(fixed)} |")


if __name__ == "__main__":
    main()
