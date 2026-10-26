"""The plain-code rules the agent must obey. The LLM can reason about these
results and word the messages, but it cannot override a gate.

Healing rule: a diabetic foot ulcer whose area has not fallen by at least half
after four weeks of standard care is unlikely to heal by twelve weeks
(Sheehan et al., Diabetes Care 2003). We use it as a prompt for a nurse to
review the care plan, never as a diagnosis.
"""

from __future__ import annotations

MIN_EDGE_CONTRAST = 6.0      # below this the outline is unreliable: a nurse must trace or confirm it
CROSS_CHECK_CONTRAST = 9.0   # below this, cross-check with a second segmentation method
MAX_METHOD_DISAGREEMENT = 0.15
MAX_GLARE = 0.04
MAX_ELONGATION = 8.0         # length / width: longer and thinner than this is not a plausible outline
MIN_THICKNESS_CM = 0.3       # widest point inside the outline; thinner is a line or an edge, not a wound
STALL_DAYS = 28
STALL_REDUCTION = 0.50
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
                   detail=f"Only {reduction:.0%} smaller after {days} days; "
                          f"the 4-week benchmark is {STALL_REDUCTION:.0%}.")
    elif days < STALL_DAYS:
        out.update(status="too_early",
                   detail=f"{reduction:.0%} smaller after {days} days; benchmark is checked at 4 weeks.")
    else:
        out.update(status="on_track", detail=f"{reduction:.0%} smaller after {days} days.")
    out["needs_nurse"] = out["status"] in ("worsening", "stalled")
    return out


def decide(photo_check: dict, first: dict | None, second: dict | None = None) -> dict:
    """The deterministic pipeline used by the rules engine and the evaluation:
    gate the photo, measure, cross-check a weak outline, and settle on an action."""
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
    return {"action": "record", "reasons": []}
