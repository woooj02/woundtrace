"""A second pair of eyes on OpenCV's outline: the vision-language model finds the wound on its own.

OpenCV measures; it cannot tell when it outlined the wrong thing with a confident, sharp edge (a fold,
a shadow, printed text). The agent's model can look at the picture. It is NOT shown OpenCV's outline:
shown the overlay and asked "does this follow the wound?", it said yes to every outline, good or bad
(eval/review_summary.md, first run). Instead it is shown the plain photo and asked where the wound is.
Plain code compares its box with the outline's box; little or no overlap means the two disagree, and
policy.decide sends the outline to the nurse instead of recording it.
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
MIN_BOX_IOU = 0.1  # chosen on 40 FUSeg outlines (20 good, 20 bad); scored on the other 261 in eval/review_eval.py
MAX_SIDE = 1024

PROMPT = """This is a photo that may show a skin wound (an ulcer, open sore or scab), possibly next to a
printed card with black squares and a plastic card. Ignore the cards.
Find the wound. Reply with JSON only:
{"wound_visible": true or false, "box": [x1, y1, x2, y2]}
where box is the tight bounding box of the whole wound, in coordinates scaled 0-1000 across the image width and height."""


def box_iou(a, b) -> float:
    """Intersection over union of two boxes (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def review(img: np.ndarray, mask: np.ndarray, client=None) -> dict:
    """Ask the model where the wound is and compare with OpenCV's outline (a mask on the photo).
    Never raises: errors come back as verdict 'error'."""
    t = time.perf_counter()
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(mask)
    outline_box = [x, y, x + bw, y + bh]
    try:
        if client is None:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=REGION)
        k = min(1.0, MAX_SIDE / max(h, w))
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        r = client.converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [
                {"image": {"format": "jpeg", "source": {"bytes": buf.tobytes()}}},
                {"text": PROMPT}]}],
            inferenceConfig={"maxTokens": 120, "temperature": 0})
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
        m = re.search(r"\{.*\}", text, re.S)
        ans = json.loads(m.group(0)) if m else {}
        out = {"model": MODEL, "outline_box": outline_box, "usage": r.get("usage")}
        box = ans.get("box")
        if not ans.get("wound_visible") or not (isinstance(box, list) and len(box) == 4):
            out.update(verdict="misses", iou=0.0, model_box=None, issue="the vision model saw no wound")
        else:
            mb = [box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000]
            iou = box_iou(mb, outline_box)
            out.update(model_box=[round(v) for v in mb], iou=round(iou, 3),
                       verdict="follows" if iou >= MIN_BOX_IOU else "misses",
                       issue=("the vision model found the wound where OpenCV outlined it" if iou >= MIN_BOX_IOU
                              else "the vision model found the wound somewhere else"))
        out["ms"] = round((time.perf_counter() - t) * 1000)
        return out
    except Exception as exc:  # the check is extra caution; its failure must not lose the visit
        return {"verdict": "error", "issue": f"{type(exc).__name__}: {str(exc)[:100]}", "model": MODEL,
                "outline_box": outline_box, "ms": round((time.perf_counter() - t) * 1000)}
