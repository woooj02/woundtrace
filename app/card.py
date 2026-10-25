"""The printed reference card, in millimetres on the plane of the skin.

Two 30 mm ArUco markers (DICT_4X4_50, ids 0 and 1) sit on a white card. Their
eight corners give the homography from the photo to true millimetres, and the
second marker lets us check the card is flat and fully in view.
"""

from __future__ import annotations

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_MM = 30.0
MARKERS = {0: (10.0, 10.0), 1: (95.0, 10.0)}  # top-left corner of each marker
CARD_RECT = (5.0, 5.0, 130.0, 45.0)            # x0, y0, x1, y1 of the white card


def marker_corners_mm(marker_id: int) -> np.ndarray:
    """Corners in ArUco order: top-left, top-right, bottom-right, bottom-left."""
    x, y = MARKERS[marker_id]
    s = MARKER_MM
    return np.array([(x, y), (x + s, y), (x + s, y + s), (x, y + s)], np.float32)


def draw(canvas: np.ndarray, px_per_mm: float, label: str = "WoundTrace card") -> None:
    """Draw the card onto a BGR canvas whose origin is the plane's (0, 0) mm."""
    x0, y0, x1, y1 = (round(v * px_per_mm) for v in CARD_RECT)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (170, 170, 170), max(1, round(px_per_mm * 0.3)))
    side = round(MARKER_MM * px_per_mm)
    for mid, (mx, my) in MARKERS.items():
        tile = cv2.aruco.generateImageMarker(DICT, mid, side, borderBits=1)
        x, y = round(mx * px_per_mm), round(my * px_per_mm)
        canvas[y:y + side, x:x + side] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    scale = px_per_mm / 12
    for i, text in enumerate([label, "30 mm markers", "print at 100%"]):
        cv2.putText(canvas, text, (round(45 * px_per_mm), round((19 + i * 8) * px_per_mm)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (60, 60, 60), max(1, round(2 * scale)),
                    cv2.LINE_AA)
