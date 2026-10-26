"""OpenCV 5 measurement: find the card, check the photo, rectify to true scale, measure the wound.

Everything here is deterministic. The agent decides what to call and what to do
with the results; these functions decide nothing on their own beyond reporting
numbers and the plain-code quality gates in GATES.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import card

# Quality gates, tuned on synthetic seed 2026 (eval/run_eval.py) and set more
# conservatively than the optimum there, because synthetic skin is perfectly flat.
# Reported results use a different seed.
GATES = {
    "min_px_per_mm": 2.5,     # resolution at the card: below this the edge is a few pixels wide
    "max_tilt_deg": 45.0,     # camera tilt from straight above (estimated, phone intrinsics assumed)
    "max_edge_mm": 1.3,       # marker edge blur, 10-90% rise width converted to millimetres
    "max_card_rms_mm": 0.6,   # homography residual: card bent, curled or misdetected
    "min_card_brightness": 70,
    "max_glare_fraction": 0.04,
}
FIX = {
    "card_not_found": "Place the printed card flat next to the wound, fully in the picture.",
    "card_partly_hidden": "Both black squares on the card must be visible. Move the card or the phone.",
    "too_far": "Move the phone closer so the wound and card fill most of the picture.",
    "tilted": "Hold the phone flat, straight above the wound, not at an angle.",
    "blurry": "Hold still and tap the screen to focus before taking the photo.",
    "card_bent": "Lay the card flat on the skin; it looks bent or curled.",
    "too_dark": "Turn on a light or move near a window. The photo is too dark.",
    "glare": "There is shine on the wound. Turn off the flash or tilt the light away.",
    "wound_not_found": "No wound edge was found next to the card. Make sure the wound is in the photo.",
    "card_not_calibrated": "One-time check of your printed card: lay any bank, library or gift card "
                           "flat next to it (cover the card number) and take the photo again.",
}

ARUCO = cv2.aruco.ArucoDetector(card.DICT, (lambda p: (
    setattr(p, "cornerRefinementMethod", cv2.aruco.CORNER_REFINE_SUBPIX), p)[1])(
    cv2.aruco.DetectorParameters()))

# Learned wound segmenter (ONNX), run with OpenCV's DNN module. Trained by eval/train_segmenter.py.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "wound_seg.onnx"
DNN_AVAILABLE = MODEL_PATH.exists()
PRIMARY = "dnn" if DNN_AVAILABLE else "grabcut"  # what the app measures with first


DNN_SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
_NET = None


def dnn_prob(img: np.ndarray) -> np.ndarray:
    """Per-pixel wound probability for the photo, from the ONNX model via cv2.dnn."""
    global _NET
    if _NET is None:
        _NET = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    h, w = img.shape[:2]
    k = DNN_SIZE / max(h, w)
    sh, sw = round(h * k), round(w * k)
    box = np.zeros((DNN_SIZE, DNN_SIZE, 3), np.uint8)  # letterbox, as in training
    box[:sh, :sw] = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    blob = (cv2.dnn.blobFromImage(box, 1 / 255.0, swapRB=True) - _MEAN) / _STD
    _NET.setInput(blob)
    logit = _NET.forward()[0, 0, :sh, :sw]
    return cv2.resize(1 / (1 + np.exp(-logit)), (w, h), interpolation=cv2.INTER_LINEAR)


def dnn_mask(img: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """All wound regions in a photo without a card (used for the FUSeg evaluation)."""
    m = (dnn_prob(img) > thr).astype(np.uint8) * 255
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.0002 * m.size]
    return _fill_holes(np.isin(cc, keep).astype(np.uint8) * 255)


RECT_PX_PER_MM = 6.0
# Measurement window around the card, in mm. The wound must lie inside it.
WINDOW_MM = (-70.0, -70.0, 200.0, 190.0)


def decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    return img


def _intrinsics(shape) -> np.ndarray:
    h, w = shape[:2]
    f = 0.75 * max(w, h)  # typical phone main camera (~26 mm equivalent); not calibrated
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def find_card(img: np.ndarray) -> dict:
    """Detect the markers and fit image → mm homography from their corners."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO.detectMarkers(gray)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            if int(i) in card.MARKERS and int(i) not in found:
                found[int(i)] = c.reshape(4, 2).astype(np.float32)
    out = {"markers": sorted(found), "corners": found}
    if not found:
        return out
    img_pts = np.concatenate([found[i] for i in sorted(found)])
    mm_pts = np.concatenate([card.marker_corners_mm(i) for i in sorted(found)])
    H, _ = cv2.findHomography(img_pts, mm_pts, 0)
    proj = cv2.perspectiveTransform(img_pts[None], H)[0]
    out["H"] = H
    out["rms_mm"] = float(np.sqrt(np.mean(np.sum((proj - mm_pts) ** 2, 1))))
    sides = [np.linalg.norm(c - np.roll(c, -1, 0), axis=1).mean() for c in found.values()]
    out["px_per_mm"] = float(np.mean(sides) / card.MARKER_MM)
    obj = np.concatenate([mm_pts, np.zeros((len(mm_pts), 1), np.float32)], 1)
    ok, rvec, _ = cv2.solvePnP(obj, img_pts, _intrinsics(img.shape), None,
                               flags=cv2.SOLVEPNP_IPPE)
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        out["tilt_deg"] = float(math.degrees(math.acos(min(1.0, abs(R[2, 2])))))
    return out


def _edge_width_px(gray: np.ndarray, quad: np.ndarray) -> float | None:
    """Median 10-90% rise width of the marker's outer black-to-white edges, in pixels."""
    side = float(np.linalg.norm(quad - np.roll(quad, -1, 0), axis=1).mean())
    reach = max(4.0, min(side / 6.5, 25.0))  # stay inside the black border cell
    centre = quad.mean(0)
    widths = []
    s = np.linspace(-reach, reach, int(reach * 4) + 1)
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        t = b - a
        n = np.array([t[1], -t[0]]) / (np.linalg.norm(t) + 1e-9)
        if np.dot((a + b) / 2 - centre, n) < 0:
            n = -n  # point outward, from black to white
        for u in np.linspace(0.2, 0.8, 9):
            p = a + u * t
            xs = (p[0] + s * n[0]).astype(np.float32)[None]
            ys = (p[1] + s * n[1]).astype(np.float32)[None]
            prof = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
            prof = prof.astype(np.float32)
            lo, hi = np.median(prof[:4]), np.median(prof[-4:])
            if hi - lo < 40:
                continue
            q = (prof - lo) / (hi - lo)
            i10 = np.argmax(q > 0.1)
            i90 = np.argmax(q > 0.9)
            if i90 > i10:
                widths.append((s[i90] - s[i10]))
    return float(np.median(widths)) if widths else None


def check_photo(img: np.ndarray) -> dict:
    """Quality metrics and the gates that failed, each with a plain fix for the patient."""
    found = find_card(img)
    m: dict = {"markers": found["markers"], "image_px": [img.shape[1], img.shape[0]]}
    problems: list[str] = []
    warnings: list[str] = []
    if not found["markers"]:
        problems.append("card_not_found")
    else:
        if len(found["markers"]) < len(card.MARKERS):
            # One marker still gives scale, but card flatness can't be checked.
            warnings.append("card_partly_hidden")
        m["px_per_mm"] = round(found["px_per_mm"], 2)
        m["card_rms_mm"] = round(found["rms_mm"], 3)
        if "tilt_deg" in found:
            m["tilt_deg"] = round(found["tilt_deg"], 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        widths = [w for q in found["corners"].values() if (w := _edge_width_px(gray, q)) is not None]
        if widths:
            m["edge_px"] = round(float(np.median(widths)), 2)
            m["edge_mm"] = round(m["edge_px"] / found["px_per_mm"], 3)
        # Card brightness: the white strip between the markers, in the photo.
        Hinv = np.linalg.inv(found["H"])
        strip = cv2.perspectiveTransform(np.array([[[45, 12], [90, 12], [90, 38], [45, 38]]],
                                                  np.float32), Hinv)[0]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, strip.astype(np.int32), 255)
        if mask.any():
            m["card_brightness"] = round(float(np.percentile(gray[mask > 0], 90)), 1)

        g = GATES
        if m["px_per_mm"] < g["min_px_per_mm"]:
            problems.append("too_far")
        if m.get("tilt_deg", 0) > g["max_tilt_deg"]:
            problems.append("tilted")
        if m.get("edge_mm") is None or m["edge_mm"] > g["max_edge_mm"]:
            problems.append("blurry")
        if len(found["markers"]) > 1 and m["card_rms_mm"] > g["max_card_rms_mm"]:
            problems.append("card_bent")
        if m.get("card_brightness", 255) < g["min_card_brightness"]:
            problems.append("too_dark")
    return {"ok": not problems, "metrics": m, "warnings": warnings,
            "problems": [{"code": p, "fix": FIX[p]} for p in problems]}


# Area uncertainty (relative sd) from this photo's own evidence. Edge placement
# dominates: a boundary off by d mm changes the area by about perimeter x d.
# Constants fitted by eval/fit_uncertainty.py on the tuning seed; coverage of the
# +/- 2 sd interval is checked on a held-out seed (see eval/uncertainty.md).
AREA_SD_K = (0.0039, 0.2052, 0.0)  # floor, edge term, card-flatness term (0: synthetic cards are flat)


def area_sd_cm2(m: dict, photo: dict) -> float:
    a_mm2, p_mm = m["area_cm2"] * 100, m["perimeter_cm"] * 10
    soft = photo.get("edge_mm", 1.0) + 1 / photo["px_per_mm"]
    k0, k1, k2 = AREA_SD_K
    rel = math.sqrt(k0 ** 2 + (k1 * p_mm * soft / max(a_mm2, 1e-6)) ** 2
                    + (k2 * photo.get("card_rms_mm", 0.0)) ** 2)
    return round(m["area_cm2"] * rel, 2)


def rectify(img: np.ndarray, H: np.ndarray):
    """Warp the photo onto the skin plane at RECT_PX_PER_MM, plus a mask of valid pixels."""
    x0, y0, x1, y1 = WINDOW_MM
    s = RECT_PX_PER_MM
    T = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1]])
    size = (round((x1 - x0) * s), round((y1 - y0) * s))
    M = T @ H
    flat = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, size,
                                flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))
    return flat, valid, M


def _card_mask(shape, pad: float = 4) -> np.ndarray:
    x0, y0 = WINDOW_MM[:2]
    s = RECT_PX_PER_MM
    cx0, cy0, cx1, cy1 = card.CARD_RECT
    m = np.zeros(shape[:2], np.uint8)
    cv2.rectangle(m, (round((cx0 - pad - x0) * s), round((cy0 - pad - y0) * s)),
                  (round((cx1 + pad - x0) * s), round((cy1 + pad - y0) * s)), 255, -1)
    return m


def segment(flat: np.ndarray, valid: np.ndarray, method: str = "grabcut") -> dict:
    """Wound mask on the rectified plane.

    colour:  Lab distance from the median skin colour, Otsu threshold, largest blob.
    grabcut: the colour mask refined by GrabCut, which follows the real edge better
             when slough or periwound redness blur the colour boundary.
    """
    s = RECT_PX_PER_MM
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2Lab).astype(np.float32)
    region = (valid > 0) & (_card_mask(flat.shape) == 0)
    if region.sum() < 1000:
        return {"found": False}
    # The card lies on skin, so a band just around it is a clean skin sample even
    # when most of the frame is bed sheet or table.
    band = (_card_mask(flat.shape, 15) > 0) & region
    skin = np.median(lab[band if band.sum() > 500 else region], axis=0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(1.5 * s) | 1,) * 2)
    # A wound lies wholly inside the photo. Blobs touching the photo's edge are
    # background (table, clothing) or a wound cut off by the frame.
    edge = (valid == 0).astype(np.uint8)
    edge[[0, -1], :] = 1
    edge[:, [0, -1]] = 1
    edge = cv2.dilate(edge, np.ones((round(2 * s),) * 2, np.uint8))
    h, w = flat.shape[:2]

    d, dist, seed, background = _seed(lab, skin, region, edge, k)
    if seed is None:
        return {"found": False}
    # Second pass: the skin right around the wound is a better reference than the
    # skin around the card, which may be partly bed sheet or table.
    near = cv2.dilate(seed, np.ones((round(15 * s),) * 2, np.uint8))
    ring = (near > 0) & (cv2.dilate(seed, np.ones((round(6 * s),) * 2, np.uint8)) == 0)
    ring &= region & ~background
    if ring.sum() > 500:
        skin = np.median(lab[ring], axis=0)
        d2, dist2, seed2, _ = _seed(lab, skin, region, edge, k)
        if seed2 is not None:
            d, dist, seed = d2, dist2, seed2
    blob, sweep = _edge_seeking_outline(seed, d, dist, region, edge, k)

    # If the outline still swallowed far more than the seed, the edge is weak
    # (blur, odd light, strong redness around the wound): report it.
    flooded = np.count_nonzero(blob) > 2.0 * np.count_nonzero(_fill_holes(seed))

    if method == "grabcut":
        x, y, bw, bh = cv2.boundingRect(blob)
        pad = round(12 * s)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        roi = flat[y0:y1, x0:x1]
        b = blob[y0:y1, x0:x1]
        gc = np.full(b.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(b, np.ones((round(6 * s),) * 2, np.uint8)) > 0] = cv2.GC_PR_BGD
        gc[b > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(b, np.ones((round(2.5 * s),) * 2, np.uint8)) > 0] = cv2.GC_FGD
        gc[region[y0:y1, x0:x1] == 0] = cv2.GC_BGD
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(roi, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
            refined = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, k)
            out = np.zeros_like(blob)
            out[y0:y1, x0:x1] = _largest(refined)
            out = _fill_holes(out)
            if np.count_nonzero(out) >= 0.3 * np.count_nonzero(blob):
                blob = out
            else:
                method = "colour"  # GrabCut collapsed; keep the colour outline and say so
        except cv2.error:
            method = "colour"  # GrabCut needs both classes present; fall back honestly

    # Edge contrast: colour distance just inside vs just outside the boundary.
    ring_in = cv2.subtract(blob, cv2.erode(blob, np.ones((round(1.5 * s),) * 2, np.uint8)))
    ring_out = cv2.subtract(cv2.dilate(blob, np.ones((round(1.5 * s),) * 2, np.uint8)), blob)
    ring_out[~region] = 0
    contrast = float(dist[ring_in > 0].mean() - dist[ring_out > 0].mean()) if ring_out.any() else 0.0
    return {"found": True, "mask": blob, "method": method, "edge_contrast": round(contrast, 1),
            "flooded": bool(flooded), "threshold_sweep": sweep,
            "skin_lab": [round(float(v), 1) for v in skin]}


MAX_ELONGATION = 8.0  # long side / short side of the tightest rotated box
MIN_THICKNESS_MM = 3.0  # diameter of the largest circle that fits inside


def _elongation(mask: np.ndarray) -> float:
    pts = cv2.findNonZero(mask.astype(np.uint8))
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return max(w, h) / max(min(w, h), 1.0)


def _thickness_mm(mask: np.ndarray) -> float:
    m = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    return 2 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max()) / RECT_PX_PER_MM


def segment_dnn(img: np.ndarray, flat: np.ndarray, valid: np.ndarray, M: np.ndarray) -> dict:
    """Wound mask on the rectified plane from the learned segmenter.

    The model scores every pixel of the original photo; the probability map is warped onto the skin
    plane with the same homography as the photo, then the same plain-code rules apply: not on the
    card, not cut off by the frame, at least 0.3 cm², one wound (the most confident one)."""
    s = RECT_PX_PER_MM
    prob = cv2.warpPerspective(dnn_prob(img).astype(np.float32), M, (flat.shape[1], flat.shape[0]),
                               flags=cv2.INTER_LINEAR)
    region = (valid > 0) & (_card_mask(flat.shape) == 0)
    mask = ((prob > 0.5) & region).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (round(1.5 * s) | 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    edge = (valid == 0).astype(np.uint8)
    edge[[0, -1], :] = 1
    edge[:, [0, -1]] = 1
    edge = cv2.dilate(edge, np.ones((round(2 * s),) * 2, np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(mask)
    best, best_score = 0, 0.0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 30 * s * s:
            continue
        comp = cc == i
        if edge[comp].any():
            continue
        score = float(prob[comp].sum())
        if score > best_score:
            best, best_score = i, score
    if not best:
        return {"found": False}
    blob = _fill_holes((cc == best).astype(np.uint8) * 255)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2Lab).astype(np.float32)
    return {"found": True, "mask": blob, "method": "dnn",
            "edge_contrast": round(_edge_contrast(lab, blob, region), 1),
            "confidence": round(float(prob[blob > 0].mean()), 3), "flooded": False}


def _edge_contrast(lab: np.ndarray, blob: np.ndarray, region: np.ndarray) -> float:
    """Colour distance from the surrounding skin, just inside vs just outside the outline."""
    s = RECT_PX_PER_MM
    near = cv2.dilate(blob, np.ones((round(15 * s),) * 2, np.uint8))
    ring = (near > 0) & (cv2.dilate(blob, np.ones((round(6 * s),) * 2, np.uint8)) == 0) & region
    if ring.sum() < 200:
        return 0.0
    d = lab - np.median(lab[ring], axis=0)
    dist = cv2.GaussianBlur(np.sqrt(0.25 * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2), (0, 0), 0.6 * s)
    ring_in = cv2.subtract(blob, cv2.erode(blob, np.ones((round(1.5 * s),) * 2, np.uint8)))
    ring_out = cv2.subtract(cv2.dilate(blob, np.ones((round(1.5 * s),) * 2, np.uint8)), blob)
    ring_out[~region] = 0
    return float(dist[ring_in > 0].mean() - dist[ring_out > 0].mean()) if ring_out.any() else 0.0


def _seed(lab, skin, region, edge, k):
    """Colour distance from skin, and the most wound-like blob that doesn't touch the frame.

    Off-skin background is usually the most distinct colour in the photo, so a
    single Otsu split can put "table + slough" against "skin + wound bed". Blobs
    touching the frame are dropped and the split is redone without them.
    """
    s = RECT_PX_PER_MM
    d = lab - skin
    dist = np.sqrt(0.25 * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    dist = cv2.GaussianBlur(dist, (0, 0), 0.6 * s)
    background = np.zeros(region.shape, bool)
    for _ in range(2):
        pool = region & ~background
        if pool.sum() < 1000:
            return d, dist, None, background
        hi = np.percentile(dist[pool], 99.5)
        d8 = np.clip(dist / max(hi, 1e-6) * 255, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(d8[pool][:, None], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = ((d8 > t) & pool).astype(np.uint8) * 255
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask)
        best, best_score, touching = 0, 0.0, []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 30 * s * s:  # under 0.3 cm²
                continue
            comp = cc[y:y + bh, x:x + bw] == i
            if edge[y:y + bh, x:x + bw][comp].any():
                touching.append(i)
                continue
            if _elongation(comp) > MAX_ELONGATION or _thickness_mm(comp) < MIN_THICKNESS_MM:
                continue  # a line or thin edge (text, a bandage or paper edge, a seam), not a wound
            score = area * float(dist[y:y + bh, x:x + bw][comp].mean())
            if score > best_score:
                best, best_score = i, score
        if not touching:
            break
        background |= np.isin(cc, touching)
    if not best:
        return d, dist, None, background
    return d, dist, (cc == best).astype(np.uint8) * 255, background


def _edge_seeking_outline(seed, d, dist, region, edge, k, steps: int = 16):
    """Choose the colour threshold whose outline sits on the strongest edge.

    A wound's border is a sharp colour step; reddened skin around it fades out
    gradually. So among candidate thresholds we keep the one whose boundary has
    the highest mean gradient of the colour-distance map. Slough (yellower than
    skin, not redder) is always counted as wound, at every threshold.
    """
    s = RECT_PX_PER_MM
    x, y, bw, bh = cv2.boundingRect(seed)
    pad = round(15 * s)
    h, w = seed.shape
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
    sl = (slice(y0, y1), slice(x0, x1))
    dr, reg, ed, sd = dist[sl], region[sl], edge[sl], seed[sl]
    gx = cv2.Sobel(dr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dr, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    da = cv2.GaussianBlur(d[..., 1][sl], (0, 0), 0.6 * s)
    db = cv2.GaussianBlur(d[..., 2][sl], (0, 0), 0.6 * s)
    slough = (db > 12) & (da < 0.5 * db)
    core = cv2.erode(sd, np.ones((round(2 * s),) * 2, np.uint8))
    if not core.any():
        core = sd
    top = float(np.percentile(dr[sd > 0], 90))
    filled = _fill_holes(sd)
    seed_area = np.count_nonzero(filled)
    # Periwound redness sits at the rim of the seed, so trimming it removes pixels
    # near the border. An outline that leaves holes deep inside the seed is tracing
    # something within the wound (slough flecks), not the wound's edge.
    dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    deep = dt >= 0.6 * dt.max()
    best, best_score, tried = None, -1.0, []
    for T in np.linspace(0.25 * top, 0.85 * top, steps):
        m = (((dr > T) | slough) & reg).astype(np.uint8) * 255
        m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, cc, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros_like(m)
        for i in range(1, n):
            comp = cc == i
            near_core = (core[comp] > 0).any() or (
                st[i, cv2.CC_STAT_AREA] > 4 * s * s and
                cv2.dilate(core, np.ones((round(8 * s),) * 2, np.uint8))[comp].any())
            if near_core and not ed[comp].any():
                keep[comp] = 255
        keep = _fill_holes(keep)
        area = np.count_nonzero(keep)
        if area < 0.3 * seed_area or area > 4 * seed_area:
            continue
        if np.count_nonzero(keep[deep]) < 0.9 * np.count_nonzero(deep):
            continue
        boundary = cv2.subtract(keep, cv2.erode(keep, np.ones((3, 3), np.uint8))) > 0
        score = float(grad[boundary].mean()) if boundary.any() else 0.0
        tried.append([round(float(T), 1), round(area / s / s / 100, 2), round(score, 2)])
        if score > best_score:
            best, best_score = keep, score
    out = np.zeros_like(seed)
    out[sl] = best if best is not None else _fill_holes(sd)
    return out, tried


def _largest(mask: np.ndarray) -> np.ndarray:
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (cc == i).astype(np.uint8) * 255


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, -1)
    return out


def shape_metrics(mask: np.ndarray, card_scale: float = 1.0) -> dict:
    """card_scale: how large the printed card really is relative to its design (0.925 = the
    printer shrank it to 92.5%). Rectified pixels are design-mm, so true mm = design mm x scale."""
    s = RECT_PX_PER_MM / card_scale
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    area_mm2 = float(np.count_nonzero(mask)) / s ** 2
    hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float64)
    dists = np.linalg.norm(hull[:, None] - hull[None], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    length = dists[i, j] / s
    axis = (hull[j] - hull[i]) / (dists[i, j] + 1e-9)
    perp = hull @ np.array([-axis[1], axis[0]])
    width = (perp.max() - perp.min()) / s
    return {
        "area_cm2": round(float(area_mm2) / 100, 2),
        "length_cm": round(float(length) / 10, 2),
        "width_cm": round(float(width) / 10, 2),
        # What a paper ruler gives: length x width. It overestimates irregular wounds.
        "ruler_area_cm2": round(float(length * width) / 100, 2),
        "perimeter_cm": round(float(cv2.arcLength(c, True)) / s / 10, 2),
        "thickness_cm": round(_thickness_mm(mask) * card_scale / 10, 2),
        "solidity": round(float(cv2.contourArea(c) / max(cv2.contourArea(hull.astype(np.float32)), 1)), 3),
    }


def measure(img: np.ndarray, method: str | None = None, card_scale: float = 1.0,
            exclude: list | None = None) -> dict:
    """Rectify and measure. Returns metrics plus the mask and warp for drawing.

    method: dnn, grabcut or colour (default: PRIMARY). exclude: image-space quads to leave out
    (a calibration card lying in the photo)."""
    method = method or PRIMARY
    found = find_card(img)
    if "H" not in found:
        return {"found": False, "reason": "card_not_found"}
    flat, valid, M = rectify(img, found["H"])
    for quad in exclude or []:
        q = cv2.perspectiveTransform(np.array([quad], np.float32), M)[0]
        cv2.fillConvexPoly(valid, np.round(q).astype(np.int32), 0)
        valid = cv2.erode(valid, np.ones((round(3 * RECT_PX_PER_MM),) * 2, np.uint8))
    seg = segment_dnn(img, flat, valid, M) if method == "dnn" else segment(flat, valid, method)
    if not seg["found"]:
        return {"found": False, "reason": "wound_not_found"}
    mask = seg["mask"]
    if not mask.any():
        return {"found": False, "reason": "wound_not_found"}
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    zone = cv2.dilate(mask, np.ones((round(3 * RECT_PX_PER_MM),) * 2, np.uint8)) > 0
    shiny = (hsv[..., 2] > 235) & (hsv[..., 1] < 50)
    glare = float(shiny[zone].mean()) if zone.any() else 0.0
    out = {"found": True, "method": seg["method"], "edge_contrast": seg["edge_contrast"],
           **({"dnn_confidence": seg["confidence"]} if "confidence" in seg else {}),
           "slough_growth_refused": seg["flooded"],
           "glare_fraction": round(glare, 4), "card_scale": card_scale,
           **shape_metrics(mask, card_scale)}
    out["_mask"], out["_M"], out["_flat"] = mask, M, flat
    return out


ID1_MM = (85.60, 53.98)  # ISO/IEC 7810 ID-1: bank, library, gift and ID cards


def find_id1_card(img: np.ndarray) -> dict:
    """Calibrate the printed card's real size against a standard plastic card in the photo.

    Home printers often shrink "100%" prints to fit their margins, and a card printed at
    92.5% makes every area 17% too large, while still looking perfect. The fix: rectify the
    photo with the printed card (design mm), find a rounded rectangle with the ID-1 aspect
    ratio, and compare its measured size with its known 85.60 x 53.98 mm.
    Returns found, scale (true size / design size of the printed card), and evidence.
    """
    found = find_card(img)
    if "H" not in found:
        return {"found": False, "reason": "card_not_found"}
    s = 4.0  # px per design-mm, enough for a sub-mm side estimate
    x0, y0, x1, y1 = -150.0, -150.0, 300.0, 280.0  # the plastic card can lie anywhere nearby
    T = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1]])
    size = (round((x1 - x0) * s), round((y1 - y0) * s))
    M = T @ found["H"]
    flat = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), M, size, flags=cv2.INTER_NEAREST)
    gray = cv2.GaussianBlur(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    med = float(np.median(gray[valid > 0])) if valid.any() else 128.0
    edges = cv2.Canny(gray, 0.4 * med, 1.0 * med)
    edges[cv2.erode(valid, np.ones((9, 9), np.uint8)) == 0] = 0
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    want = ID1_MM[0] / ID1_MM[1]
    cands = []
    for c in contours:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        long_, short = max(w, h) / s, min(w, h) / s
        if not (55 < long_ < 115 and 35 < short < 75):  # a card printed at 75%-155%
            continue
        if abs(long_ / short / want - 1) > 0.04:
            continue
        fill = cv2.contourArea(cv2.convexHull(c)) / (w * h)
        if fill < 0.93:  # a closed, nearly rectangular outline (ID-1 corners are rounded)
            continue
        cands.append((long_, short, (cx, cy), (w, h), ang))
    if not cands:
        return {"found": False, "reason": "id_card_not_found"}
    # The thickened edge line yields an outer and an inner contour around the same card;
    # the true edge lies midway, so average the largest and smallest of that pair.
    best = max(cands, key=lambda k: k[0] * k[1])
    same = [k for k in cands if np.hypot(k[2][0] - best[2][0], k[2][1] - best[2][1]) < 3 * s]
    inner = min(same, key=lambda k: k[0] * k[1])
    long_, short = (best[0] + inner[0]) / 2, (best[1] + inner[1]) / 2
    _, _, centre, wh, ang = best
    scale_long, scale_short = ID1_MM[0] / long_, ID1_MM[1] / short
    if abs(scale_long - scale_short) > 0.025:
        return {"found": False, "reason": "id_card_inconsistent",
                "scale_long": round(scale_long, 4), "scale_short": round(scale_short, 4)}
    box = cv2.boxPoints((centre, wh, ang))
    corners = cv2.perspectiveTransform(box[None].astype(np.float32), np.linalg.inv(M))[0]
    return {"found": True, "scale": round((scale_long + scale_short) / 2, 4),
            "measured_mm": [round(long_, 1), round(short, 1)],
            "scale_long": round(scale_long, 4), "scale_short": round(scale_short, 4),
            "corners_px": corners.round(1).tolist()}


def annotate(img: np.ndarray, result: dict, card_found: dict | None = None,
             extra_quads: list | None = None) -> np.ndarray:
    """The photo with the measured outline projected back onto it (and any calibration card)."""
    out = img.copy()
    t = max(2, round(max(img.shape[:2]) / 600))
    for q in extra_quads or []:
        cv2.polylines(out, [np.round(np.array(q)).astype(np.int32)], True, (255, 0, 255), t)
    found = card_found or find_card(img)
    for q in found.get("corners", {}).values():
        cv2.polylines(out, [q.astype(np.int32)], True, (255, 160, 0), t)
    if result.get("found"):
        contours, _ = cv2.findContours(result["_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        Minv = np.linalg.inv(result["_M"])
        for c in contours:
            pts = cv2.perspectiveTransform(c.astype(np.float32), Minv)
            cv2.polylines(out, [pts.astype(np.int32)], True, (60, 255, 60), t)
        label = f"{result['area_cm2']:.2f} cm2"
        cv2.putText(out, label, (20, 30 + 20 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t,
                    (0, 0, 0), t * 3, cv2.LINE_AA)
        cv2.putText(out, label, (20, 30 + 20 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t,
                    (60, 255, 60), t, cv2.LINE_AA)
    return out
