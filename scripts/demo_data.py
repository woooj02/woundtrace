"""Demo patients and printable test sheets. Test targets only: no real people or wounds.

    venv/bin/python -m scripts.demo_data seed     # create the two demo patients
    venv/bin/python -m scripts.demo_data sheets   # print/*.png, 300 dpi, Letter

Each demo patient is one wound outline (fixed shape seed) at shrinking sizes.
Earlier visits are created by photographing the synthetic target in software
and measuring it with the real pipeline, so every stored number is a real
measurement of a known target. Today's visit is the printed sheet, photographed
live on a phone.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

from app import patients, store, synth, vision

ROOT = Path(__file__).resolve().parent.parent
DEMO = [
    {"id": "demo-a", "name": "Demo patient A", "wound": "Diabetic foot ulcer, left heel (test target)",
     "skin": 1, "shape_seed": 11, "areas": [8.0, 7.7, 7.4, 7.2], "today": 7.0},
    {"id": "demo-b", "name": "Demo patient B", "wound": "Pressure injury, sacrum (test target)",
     "skin": 4, "shape_seed": 23, "areas": [6.0, 4.9, 3.9, 3.2], "today": 2.6},
]


def seed() -> None:
    """Reset the demo: two patients with four weekly visits each, and an empty nurse queue."""
    for key in store.keys("alerts/"):
        store.delete(key)
    start = date.today() - timedelta(days=28)
    for d in DEMO:
        p = {"id": d["id"], "name": d["name"], "wound": d["wound"], "start_date": start.isoformat(),
             "demo": "Test target, not a person. Earlier visits are software photos of the same "
                     "printed target design, measured by the real pipeline.",
             "card": {"scale": 1.0, "method": "test sheet",
                      "note": "The sheet's card and wound are printed together, so print scale cancels.",
                      "date": start.isoformat()},
             "history": []}
        patients.save(p)
        rng = np.random.default_rng(d["shape_seed"])
        for week, area in enumerate(d["areas"]):
            plane, true_mm2 = synth.render_plane(rng, area, d["skin"], shape_seed=d["shape_seed"])
            shot = synth.Shot(tilt_deg=float(rng.uniform(5, 20)), px_per_mm=float(rng.uniform(5, 7)),
                              blur_sigma=0.0, glare=False, skin=d["skin"], true_cm2=true_mm2 / 100)
            img = synth.photograph(rng, plane, 8.0, shot)
            chk = vision.check_photo(img)
            assert chk["ok"], "seed photo failed its own quality check"
            m = vision.measure(img, vision.PRIMARY)
            m["area_sd_cm2"] = vision.area_sd_cm2(m, chk["metrics"])
            pid = f"seed-{d['id']}-w{week}"
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            store.put(f"photos/{pid}.jpg", buf.tobytes(), "image/jpeg")
            ok, buf = cv2.imencode(".jpg", vision.annotate(img, m), [cv2.IMWRITE_JPEG_QUALITY, 85])
            store.put(f"annotated/{pid}-grabcut.jpg", buf.tobytes(), "image/jpeg")
            clean = {k: v for k, v in m.items() if not k.startswith("_")}
            e = patients.add_measurement(d["id"], pid, clean, on=start + timedelta(days=7 * week))
            print(f"{d['id']} day {e['day']:2d}: true {true_mm2 / 100:.2f} cm², measured {e['area_cm2']:.2f}")
    patients.save({**NEW_PATIENT, "start_date": date.today().isoformat(), "history": []})
    print("demo-c: new patient, card not checked")


NEW_PATIENT = {"id": "demo-c", "name": "Demo patient C",
               "wound": "New patient: printed card not size-checked yet (test object)",
               "demo": "Test object, not a person. Shows the one-time card check: the agent will not "
                       "record until a plastic card (or a ruler reading) confirms the card's print size."}


def sheets() -> None:
    out = ROOT / "print"
    out.mkdir(exist_ok=True)
    dpi = 300
    ppm = dpi / 25.4
    W, H = round(215.9 * ppm), round(279.4 * ppm)
    for d in DEMO:
        page = np.full((H, W, 3), 255, np.uint8)
        rng = np.random.default_rng(d["shape_seed"] + 1000)
        plane, true_mm2 = synth.render_plane(rng, d["today"], d["skin"], px_per_mm=ppm,
                                             shape_seed=d["shape_seed"])
        ox, oy = round(8 * ppm), round(20 * ppm)
        page[oy:oy + plane.shape[0], ox:ox + plane.shape[1]] = plane
        y = oy + plane.shape[0] + round(14 * ppm)
        lines = [f"WoundTrace test target - {d['name']}, today's visit",
                 f"True wound area: {true_mm2 / 100:.2f} cm2  (test target, not a real wound)",
                 "Print at 100% / Actual size. Check: the bar below must measure 100 mm."]
        for i, t in enumerate(lines):
            cv2.putText(page, t, (ox, y + i * round(8 * ppm)), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (40, 40, 40), 3, cv2.LINE_AA)
        by = y + len(lines) * round(8 * ppm)
        cv2.rectangle(page, (ox, by), (ox + round(100 * ppm), by + round(3 * ppm)), (0, 0, 0), -1)
        path = out / f"{d['id']}-today.png"
        cv2.imwrite(str(path), page)
        # Embed the DPI so "Actual size" prints 1:1.
        _set_png_dpi(path, dpi)
        print(path, f"true {true_mm2 / 100:.2f} cm²")
    # A spare card on its own, for photographing anything else.
    card_page = np.full((H, W, 3), 255, np.uint8)
    from app import card
    card.draw(card_page, ppm)
    path = out / "card-only.png"
    cv2.imwrite(str(path), card_page)
    _set_png_dpi(path, dpi)
    print(path)


BW = [("bw-small", 1.5, 41), ("bw-medium", 4.0, 42), ("bw-large", 9.0, 43)]


def bw_page(area_cm2: float, shape_seed: int, ppm: float, size_mm=(215.9, 279.4), card_at=(15.0, 15.0)):
    """A grayscale geometry target: textured light 'skin' over the whole page, a dark granular
    wound with mid-grey slough, and the card. Prints with black ink only. Returns (gray image,
    true area in mm²). Tests scale and outline geometry; colour is tested elsewhere."""
    rng = np.random.default_rng(shape_seed)
    W, H = round(size_mm[0] * ppm), round(size_mm[1] * ppm)
    page = 205 + synth._noise_field(rng, (H, W), round(6 * ppm), 7) + rng.normal(0, 3, (H, W))
    cx, cy = card_at[0] + synth.WOUND_CENTER[0], card_at[1] + synth.WOUND_CENTER[1]
    poly = synth.wound_polygon(rng, area_cm2, center=(cx, cy))
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(poly * ppm * 16).astype(np.int32)], 255, lineType=cv2.LINE_8, shift=4)
    true_mm2 = float(np.count_nonzero(mask)) / ppm ** 2
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ppm * 5),) * 2))
    page -= 14 * cv2.GaussianBlur((ring > 0).astype(np.float32), (0, 0), ppm * 1.5)
    tex = synth._noise_field(rng, (H, W), max(2, round(ppm * 1.2)), 22)
    wound = 80 + tex
    slough = synth._noise_field(rng, (H, W), round(ppm * 4), 1) > 0.8
    wound = np.where(slough, 145 + tex * 0.4, wound)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), ppm * 0.15)
    page = np.clip(page * (1 - soft) + wound * soft, 0, 255).astype(np.uint8)
    page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    ox, oy = round(card_at[0] * ppm), round(card_at[1] * ppm)
    sub = page[oy:, ox:].copy()
    from app import card
    card.draw(sub, ppm)
    page[oy:, ox:] = sub
    return cv2.cvtColor(page, cv2.COLOR_BGR2GRAY), true_mm2


def bw_sheets() -> None:
    """Black-ink-only geometry targets: print/bw-*.png (print in grayscale at 100%)."""
    out = ROOT / "print"
    out.mkdir(exist_ok=True)
    dpi = 300
    ppm = dpi / 25.4
    for name, area, seed_ in BW:
        page, true_mm2 = bw_page(area, seed_, ppm)
        # Caption and check bar at the bottom, outside the measurement window around the card.
        x, y = round(15 * ppm), round(238 * ppm)
        cv2.rectangle(page, (x - 20, y - 70), (round(200 * ppm), round(272 * ppm)), 255, -1)
        lines = [f"WoundTrace geometry target ({name}). True wound area: {true_mm2 / 100:.2f} cm2",
                 "Print in grayscale, 100% / Actual size. The bar must measure 100 mm."]
        for i, t in enumerate(lines):
            cv2.putText(page, t, (x, y + i * round(7 * ppm)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 30, 3, cv2.LINE_AA)
        by = y + len(lines) * round(7 * ppm)
        cv2.rectangle(page, (x, by), (x + round(100 * ppm), by + round(3 * ppm)), 0, -1)
        path = out / f"{name}.png"
        cv2.imwrite(str(path), page)
        _set_png_dpi(path, dpi)
        print(path, f"true {true_mm2 / 100:.2f} cm²")


def samples() -> None:
    """Software photos of today's targets for judges without a printer (web/samples)."""
    out = ROOT / "web" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    cases = [("a-good", 0, dict(tilt_deg=12, px_per_mm=6.0, blur_sigma=0.0, glare=False)),
             ("a-tilted", 0, dict(tilt_deg=56, px_per_mm=5.0, blur_sigma=0.0, glare=False)),
             ("a-glare", 0, dict(tilt_deg=8, px_per_mm=6.0, blur_sigma=0.0, glare=True)),
             ("b-good", 1, dict(tilt_deg=15, px_per_mm=6.5, blur_sigma=0.0, glare=False))]
    for name, i, kw in cases:
        d = DEMO[i]
        for attempt in range(40):  # pick a glare spot that actually lands on the wound
            rng = np.random.default_rng(d["shape_seed"] + 1000 + attempt)
            plane, mm2 = synth.render_plane(np.random.default_rng(d["shape_seed"] + 1000), d["today"],
                                            d["skin"], shape_seed=d["shape_seed"])
            img = synth.photograph(rng, plane, 8.0, synth.Shot(skin=d["skin"], true_cm2=mm2 / 100, **kw))
            m = vision.measure(img)
            if not kw["glare"] or (m.get("found") and m["glare_fraction"] > 0.1):
                break
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(name, f"true {mm2 / 100:.2f}", vision.check_photo(img)["ok"], m.get("area_cm2"),
              m.get("glare_fraction"))
    # New patient C: a card the printer shrank to 92.5%, without and with a plastic card beside it.
    for name, id1 in (("c-no-card", False), ("c-card-check", True)):
        rng = np.random.default_rng(77)
        plane, mm2 = synth.render_plane(rng, 5.0, 2, size_mm=(230.0, 150.0), print_scale=0.925,
                                        id1_at=(128.0, 58.0) if id1 else None)
        img = synth.photograph(rng, plane, 8.0, synth.Shot(tilt_deg=10, px_per_mm=4.5, blur_sigma=0.0,
                                                           glare=False, skin=2, true_cm2=mm2 / 100))
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cal = vision.find_id1_card(img)
        print(name, f"true {mm2 / 100:.2f}", "uncorrected", vision.measure(img).get("area_cm2"),
              "card scale", cal.get("scale"))


def _set_png_dpi(path: Path, dpi: int) -> None:
    import struct
    import zlib
    data = path.read_bytes()
    ppm = round(dpi / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    i = data.index(b"IDAT") - 4
    path.write_bytes(data[:i] + chunk + data[i:])


if __name__ == "__main__":
    {"seed": seed, "sheets": sheets, "bw": bw_sheets, "samples": samples}[sys.argv[1]]()
