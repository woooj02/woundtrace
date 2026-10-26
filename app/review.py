"""A second pair of eyes on OpenCV's outline: the vision-language model checks the overlay.

OpenCV measures; it cannot tell when it outlined the wrong thing with a confident, sharp edge (a fold,
a shadow, printed text). The agent's model can look at the picture. This module shows it two crops side
by side, the photo and the same photo with OpenCV's outline, and asks one narrow question. The answer is
structured, and plain code decides what it means (policy.decide): an outline the model disputes is never
recorded; it goes to the nurse.
"""

from __future__ import annotations

import json
import os
import re
import time

import cv2
import numpy as np

MODEL = os.environ.get("BEDROCK_MODEL", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")

PROMPT = """You are checking a computer-vision measurement of a skin wound.
Left: a region of a photo. Right: the same region with a green outline that an algorithm drew around what
it thinks is the wound (the open or scabbed lesion).
Does the green outline follow the edge of the wound?
- "follows": it traces the wound's border reasonably closely.
- "misses": it traces something that is not the wound, leaves out a large part of the wound, or takes in a
  lot of healthy skin.
- "unsure": you cannot tell from the picture.
Reply with JSON only: {"verdict": "follows" | "misses" | "unsure", "issue": "at most 12 words"}"""


def overlay_pair(img: np.ndarray, mask: np.ndarray, size: int = 448) -> bytes:
    """The photo crop and the same crop with the outline, side by side, as JPEG bytes."""
    x, y, w, h = cv2.boundingRect(mask)
    pad = int(max(w, h) * 0.6) + 20
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    crop, m = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    k = size / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    drawn = crop.copy()
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(drawn, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    gap = np.full((crop.shape[0], 12, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".jpg", np.hstack([crop, gap, drawn]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def review(img: np.ndarray, mask: np.ndarray, client=None) -> dict:
    """Ask the model whether the outline follows the wound. Never raises: errors come back as 'error'."""
    t = time.perf_counter()
    try:
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=REGION)
        r = client.converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [
                {"image": {"format": "jpeg", "source": {"bytes": overlay_pair(img, mask)}}},
                {"text": PROMPT}]}],
            inferenceConfig={"maxTokens": 80, "temperature": 0})
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
        m = re.search(r"\{.*\}", text, re.S)
        ans = json.loads(m.group(0)) if m else {}
        verdict = ans.get("verdict") if ans.get("verdict") in ("follows", "misses", "unsure") else "unsure"
        return {"verdict": verdict, "issue": str(ans.get("issue", ""))[:120], "model": MODEL,
                "ms": round((time.perf_counter() - t) * 1000), "usage": r.get("usage")}
    except Exception as exc:  # the check is extra caution; its failure must not lose the visit
        return {"verdict": "error", "issue": f"{type(exc).__name__}: {str(exc)[:100]}", "model": MODEL,
                "ms": round((time.perf_counter() - t) * 1000)}
