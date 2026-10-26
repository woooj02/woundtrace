"""The plain-code rules the agent must obey. The LLM can reason about these
results and word the messages, but it cannot override a gate.

Healing rule (configurable): flag a wound for clinician review when its area has not
fallen by more than 50% after at least 4 weeks. The Society for Vascular Surgery
guideline (Hingorani et al., J Vasc Surg 2016) recommends considering adjunctive
therapy for diabetic foot ulcers that fail to improve by >50% in area after a minimum
of 4 weeks of standard care; Sheehan et al. (Diabetes Care 2003, n=203) found the
4-week percent area change predicts 12-week healing. It is a prompt for a clinician to
review the care plan: never a diagnosis, a prediction for this patient, or a treatment
recommendation.
"""

from __future__ import annotations

import os

MIN_EDGE_CONTRAST = 6.0      # below this the outline is unreliable: a nurse must trace or confirm it
CROSS_CHECK_CONTRAST = 9.0   # below this, cross-check with a second segmentation method
MAX_METHOD_DISAGREEMENT = 0.15
MAX_GLARE = 0.04
MAX_ELONGATION = 8.0         # length / width: longer and thinner than this is not a plausible outline
MIN_THICKNESS_CM = 0.3       # widest point inside the outline; thinner is a line or an edge, not a wound
STALL_DAYS = int(os.environ.get("STALL_DAYS", "28"))
STALL_REDUCTION = float(os.environ.get("STALL_REDUCTION", "0.50"))
WORSE_INCREASE = 0.10        # 10% larger than the last measurement


def measurement_problems(m: dict) -> list[str]:
    """Reasons a measurement should not be recorded as it is."""
    if not m.get("found"):
        return [m.get("reason", "wound_not_found")]
    out = []
    if m["glare_fraction"] > MAX_GLARE:
        out.append("glare")
    return out


def methods_disagree(a: dict, b: dict) -> float:
    return abs(a["area_cm2"] - b["area_cm2"]) / max(a["area_cm2"], b["area_cm2"], 1e-6)


def assess_healing(history: list[dict], area_cm2: float, day: int,
                   area_sd_cm2: float | None = None) -> dict:
    """Compare a new area with the patient's baseline and last measurement.

    A rise counts as worsening only if it is over 10% and also larger than the
    measurement noise of the two photos (2 sd of the difference)."""
    if not history:
        return {"status": "baseline", "detail": "First measurement: this becomes the baseline."}
    base, last = history[0], history[-1]
    days = day - base["day"]
    reduction = 1 - area_cm2 / base["area_cm2"]
    change_last = area_cm2 / last["area_cm2"] - 1
    out = {"baseline_cm2": base["area_cm2"], "days_since_baseline": days,
           "reduction_from_baseline": round(reduction, 3),
           "change_since_last": round(change_last, 3), "last_cm2": last["area_cm2"]}
    noise = 2 * ((area_sd_cm2 or 0) ** 2 + (last.get("area_sd_cm2") or 0) ** 2) ** 0.5
    out["change_noise_cm2"] = round(noise, 2)
    if change_last > WORSE_INCREASE and area_cm2 - last["area_cm2"] > noise:
        out.update(status="worsening",
                   detail=f"Area grew {change_last:.0%} since the last photo.")
    elif days >= STALL_DAYS and reduction < STALL_REDUCTION:
        out.update(status="stalled",
                   detail=f"{reduction:.0%} smaller after {days} days, below the >{STALL_REDUCTION:.0%} "
                          f"area reduction by week 4 used to prompt a care-plan review (SVS 2016).")
    elif days < STALL_DAYS:
        out.update(status="too_early",
                   detail=f"{reduction:.0%} smaller after {days} days; benchmark is checked at 4 weeks.")
    else:
        out.update(status="on_track", detail=f"{reduction:.0%} smaller after {days} days.")
    out["needs_nurse"] = out["status"] in ("worsening", "stalled")
    return out


def decide(photo_check: dict, first: dict | None, second: dict | None = None,
           review: dict | None = None) -> dict:
    """The deterministic pipeline used by the rules engine and the evaluation:
    gate the photo, measure, cross-check a weak outline, and settle on an action.
    review: the vision model's check of the outline overlay (review.py); 'misses' means a nurse
    confirms the outline before anything is recorded."""
    if not photo_check["ok"]:
        return {"action": "retake", "reasons": [p["code"] for p in photo_check["problems"]]}
    probs = measurement_problems(first)
    if probs:
        return {"action": "retake", "reasons": probs}
    if (first["length_cm"] > MAX_ELONGATION * max(first["width_cm"], 0.01)
            or first["thickness_cm"] < MIN_THICKNESS_CM):
        return {"action": "nurse_outline", "reasons": ["implausible_shape"]}
    if first["edge_contrast"] < MIN_EDGE_CONTRAST:
        return {"action": "nurse_outline", "reasons": ["weak_edge"]}
    if first["edge_contrast"] < CROSS_CHECK_CONTRAST:
        if second is None:
            return {"action": "cross_check"}
        if not second.get("found") or methods_disagree(first, second) > MAX_METHOD_DISAGREEMENT:
            return {"action": "nurse_outline", "reasons": ["uncertain_edge"]}
    if review and review.get("verdict") == "misses":
        return {"action": "nurse_outline", "reasons": ["outline_disputed"]}
    return {"action": "record", "reasons": []}
