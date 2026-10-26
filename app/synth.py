"""Synthetic wound targets with known area, and simulated phone photos of them.

These are test targets, not patients. They give a ground truth that real photos
cannot: the exact area in mm², and the exact tilt, blur and glare of each shot.
The same renderer makes the printable sheets used in the live demo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import card

PAGE_MM = (200.0, 150.0)       # the plane we render: card at the top, wound below
WOUND_CENTER = (100.0, 98.0)
# Six flat skin tones spanning very light to very dark (RGB). A rough proxy for
# the Fitzpatrick scale: real skin has texture, hair and shine these lack.
SKIN_TONES = [(241, 204, 180), (224, 172, 140), (198, 140, 105),
              (161, 102, 72), (114, 72, 50), (72, 46, 34)]


def _noise_field(rng, shape, scale_px, amp):
    small = rng.normal(0, 1, (max(2, shape[0] // scale_px), max(2, shape[1] // scale_px)))
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC) * amp


def wound_polygon(rng, area_cm2: float, center=WOUND_CENTER, n=360) -> np.ndarray:
    """An irregular closed outline in mm, roughly the requested area."""
    r0 = math.sqrt(area_cm2 * 100 / math.pi)
    theta = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r = np.ones(n)
    for k in range(2, 6):
        r += rng.uniform(0, 0.22 / k * 2) * np.cos(k * theta + rng.uniform(0, 2 * math.pi))
    aspect = rng.uniform(0.65, 1.0)
    rot = rng.uniform(0, math.pi)
    # Squash into an ellipse without changing the area much.
    x = r0 * r * np.cos(theta) / aspect ** 0.5
    y = r0 * r * np.sin(theta) * aspect ** 0.5
    xr = x * math.cos(rot) - y * math.sin(rot) + center[0]
    yr = x * math.sin(rot) + y * math.cos(rot) + center[1]
    return np.stack([xr, yr], 1)


def render_plane(rng, area_cm2: float, skin: int, px_per_mm: float = 8.0,
                 size_mm=PAGE_MM, slough: bool = True, shape_seed: int | None = None,
                 print_scale: float = 1.0, id1_at: tuple | None = None, return_mask: bool = False):
    """Skin, wound and card on the plane. Returns (BGR image, true area in mm²), plus the wound mask
    if return_mask.

    print_scale: the printed card's real size / its design size (a home printer that shrank it to
    fit the margins gives ~0.93). id1_at: top-left (mm) of a plastic ID-1 card lying on the skin,
    for calibration; needs a wider plane than the default."""
    w, h = round(size_mm[0] * px_per_mm), round(size_mm[1] * px_per_mm)
    tone = np.array(SKIN_TONES[skin][::-1], np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = tone
    img += _noise_field(rng, (h, w), round(6 * px_per_mm), 6)[..., None]
    img += rng.normal(0, 3, (h, w, 1))

    # A fixed shape_seed keeps the same outline at any size: one wound healing.
    poly = wound_polygon(rng if shape_seed is None else np.random.default_rng(shape_seed), area_cm2)
    mask = np.zeros((h, w), np.uint8)
    pts = np.round(poly * px_per_mm * 16).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_8, shift=4)
    true_mm2 = float(np.count_nonzero(mask)) / px_per_mm ** 2

    # Periwound redness just outside the edge: part of what makes real edges hard.
    ring = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(px_per_mm * 5),) * 2))
    ring = cv2.GaussianBlur((ring > 0).astype(np.float32), (0, 0), px_per_mm * 1.5)
    redden = np.array([-10, -18, 12], np.float32)
    img += ring[..., None] * redden * rng.uniform(0.3, 1.0)

    # Granulation tissue: dark-to-bright red with a grainy texture, plus slough.
    base = np.array([48, 44, 168], np.float32) * rng.uniform(0.75, 1.1)
    tex = _noise_field(rng, (h, w), max(2, round(px_per_mm * 1.2)), 22)[..., None]
    wound = base + tex * np.array([0.4, 0.5, 1.0], np.float32)
    if slough:
        blobs = (_noise_field(rng, (h, w), round(px_per_mm * 4), 1) > rng.uniform(0.6, 1.1))
        wound = np.where(blobs[..., None], np.array([95, 175, 205], np.float32) + tex * 0.4, wound)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), px_per_mm * 0.15)[..., None]
    img = img * (1 - soft) + wound * soft
    img = np.clip(img, 0, 255).astype(np.uint8)
    if id1_at is not None:
        draw_id1(img, px_per_mm, id1_at)
    card.draw(img, px_per_mm * print_scale)
    return (img, true_mm2, mask) if return_mask else (img, true_mm2)


ID1 = (85.60, 53.98, 3.18)  # width, height, corner radius in mm (ISO/IEC 7810 ID-1)


def draw_id1(img: np.ndarray, px_per_mm: float, at_mm: tuple, colour=(170, 90, 30)) -> None:
    """A plain plastic card with rounded corners, true size, top-left at at_mm."""
    w, h, r = (v * px_per_mm for v in ID1)
    x0, y0 = at_mm[0] * px_per_mm, at_mm[1] * px_per_mm
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(m, (round(x0 + r), round(y0)), (round(x0 + w - r), round(y0 + h)), 255, -1)
    cv2.rectangle(m, (round(x0), round(y0 + r)), (round(x0 + w), round(y0 + h - r)), 255, -1)
    for cx, cy in ((x0 + r, y0 + r), (x0 + w - r, y0 + r), (x0 + r, y0 + h - r), (x0 + w - r, y0 + h - r)):
        cv2.circle(m, (round(cx), round(cy)), round(r), 255, -1)
    img[m > 0] = colour


@dataclass
class Shot:
    tilt_deg: float
    px_per_mm: float
    blur_sigma: float
    glare: bool
    skin: int
    true_cm2: float

    def dict(self):
        return asdict(self)


def photograph(rng, plane: np.ndarray, plane_px_per_mm: float, shot: Shot,
               size=(1600, 1200), mask: np.ndarray | None = None):
    """Simulate a phone photo of the plane: perspective, blur, glare, noise, JPEG.
    With a plane mask, also returns that mask as the camera sees it (for training)."""
    W, H = size
    f = 0.75 * max(W, H)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    c = np.array([WOUND_CENTER[0] * 0.7 + 67.5 * 0.3, WOUND_CENTER[1] * 0.7 + 25 * 0.3])
    d = f / shot.px_per_mm
    axis_angle = rng.uniform(0, 2 * math.pi)
    axis = np.array([math.cos(axis_angle), math.sin(axis_angle), 0.0])
    R_tilt, _ = cv2.Rodrigues(axis * math.radians(shot.tilt_deg))
    R_spin, _ = cv2.Rodrigues(np.array([0, 0, rng.uniform(-math.pi, math.pi)]))
    R = R_spin @ R_tilt
    A = np.column_stack([R[:, 0], R[:, 1], -R[:, :2] @ c + np.array([0, 0, d])])
    Hm = K @ A @ np.diag([1 / plane_px_per_mm, 1 / plane_px_per_mm, 1])
    table = tuple(int(v) for v in rng.integers(60, 140, 3))
    img = cv2.warpPerspective(plane, Hm, (W, H), flags=cv2.INTER_AREA, borderValue=table)
    img = img.astype(np.float32)

    gx = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(1.0, 1.15), W)
    img *= gx[None, :, None]
    if shot.glare:
        cx, cy = cv2.perspectiveTransform(
            np.array([[[WOUND_CENTER[0] * plane_px_per_mm, WOUND_CENTER[1] * plane_px_per_mm]]]), Hm)[0, 0]
        cx += rng.uniform(-40, 40); cy += rng.uniform(-40, 40)
        yy, xx = np.mgrid[0:H, 0:W]
        rx, ry = rng.uniform(40, 120), rng.uniform(25, 70)
        g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 2.5)
        img = img + g[..., None] * 255 * 1.6
    if shot.blur_sigma > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), shot.blur_sigma)
    img += rng.normal(0, 3, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    photo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if mask is None:
        return photo
    return photo, cv2.warpPerspective(mask, Hm, (W, H), flags=cv2.INTER_NEAREST)


def random_case(rng, hard: bool = True):
    """One evaluation case: a plane, a photo of it, and the ground truth."""
    shot = Shot(
        tilt_deg=float(rng.uniform(0, 60 if hard else 25)),
        px_per_mm=float(rng.uniform(1.8 if hard else 4.0, 9.0)),
        blur_sigma=float(rng.choice([0, 0, 0.8, 1.5, 3, 5]) if hard else 0.0),
        glare=bool(hard and rng.random() < 0.2),
        skin=int(rng.integers(0, len(SKIN_TONES))),
        true_cm2=0.0,
    )
    plane, true_mm2 = render_plane(rng, float(rng.uniform(1.5, 25)), shot.skin)
    shot.true_cm2 = true_mm2 / 100
    return photograph(rng, plane, 8.0, shot), shot
