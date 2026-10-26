"""The tools the agent can call during one visit.

The LLM chooses which tool to call next and words the messages. Plain code
here enforces the invariants it cannot talk its way around:
  - nothing is measured from a photo that failed the quality check;
  - healing is assessed from the number the vision code measured, never from a
    number the model typed;
  - nothing is recorded unless policy.decide says "record";
  - every message to a nurse waits for a human to approve it.
"""

from __future__ import annotations

import time
import uuid
from datetime import date

import cv2

from . import patients, policy, store, vision

TOOLS: dict[str, dict] = {}


def tool(**properties: dict):
    """Register a function as a tool. Keyword args are its JSON-schema properties."""
    def register(fn):
        TOOLS[fn.__name__] = {
            "fn": fn,
            "spec": {
                "name": fn.__name__,
                "description": " ".join((fn.__doc__ or "").split()),
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        }
        return fn
    return register


def specs() -> list[dict]:
    return [t["spec"] for t in TOOLS.values()]


class Visit:
    """State for one run: the patient, the photo and what the tools found."""

    def __init__(self, patient_id: str, photo_id: str):
        self.patient_id = patient_id
        self.photo_id = photo_id
        self.check: dict | None = None
        self.measures: dict[str, dict] = {}
        self.decision: dict | None = None
        self.healing: dict | None = None
        self.calibration: dict | None = None
        self.outcome: dict | None = None  # set by the tool that ends the visit
        self._img = None
        self._card = None

    def image(self):
        if self._img is None:
            raw = store.get(f"photos/{self.photo_id}.jpg")
            if raw is None:
                raise ValueError(f"no photo {self.photo_id}")
            self._img = vision.decode(raw)
        return self._img


def call(visit: Visit, name: str, args: dict):
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}")
    if args.get("photo_id", visit.photo_id) != visit.photo_id:
        raise ValueError("this visit can only use its own photo_id")
    if args.get("patient_id", visit.patient_id) != visit.patient_id:
        raise ValueError("this visit can only use its own patient_id")
    return TOOLS[name]["fn"](visit, **args)


PHOTO = {"type": "string", "description": "The photo_id given in the task."}
PATIENT = {"type": "string", "description": "The patient_id given in the task."}


@tool(photo_id=PHOTO)
def check_photo(v: Visit, photo_id: str) -> dict:
    """Run OpenCV quality checks on the photo: finds the printed ArUco card, then
    estimates resolution (px/mm), camera tilt, blur (marker edge width in mm), card
    flatness and brightness. Returns ok, metrics, problems each with a plain fix for the
    patient, and card_calibrated (whether this patient's printed card has been size-checked).
    Always call this first."""
    v.check = vision.check_photo(v.image())
    return {**v.check, "card_calibrated": patients.card_scale(patients.get(v.patient_id)) is not None}


@tool(patient_id=PATIENT, photo_id=PHOTO)
def calibrate_card(v: Visit, patient_id: str, photo_id: str) -> dict:
    """Check how large the patient's printed card really came out, using a standard plastic
    card (bank, library or gift card, 85.60 x 53.98 mm) lying in the same photo. Home
    printers often shrink the card, which would make every area too large. Needed once per
    printed card, before any measurement can be recorded. Returns the scale or why it failed."""
    if v.check is None or not v.check["ok"]:
        raise PermissionError("check_photo has not passed for this photo")
    r = vision.find_id1_card(v.image())
    v.calibration = r
    out = {k: val for k, val in r.items() if k != "corners_px"}
    if r["found"]:
        patients.set_card(patient_id, {"scale": r["scale"], "method": "plastic card (ID-1)",
                                       "photo_id": photo_id, "measured_mm": r["measured_mm"]})
        out["calibrated"] = True
        out["meaning"] = f"The printed card is {r['scale']:.1%} of its intended size; measurements are corrected."
    else:
        out["calibrated"] = False
    return out


@tool(photo_id=PHOTO, method={"type": "string", "enum": ["grabcut", "colour"],
                              "description": "grabcut (default) or colour, for a second opinion"})
def measure_wound(v: Visit, photo_id: str, method: str) -> dict:
    """Rectify the photo to true millimetres using the card, outline the wound and
    measure area, length, width and perimeter. Also reports glare on the wound and
    edge_contrast (how clear the outline is). Only allowed after check_photo passed.
    Returns a 'decision' from the clinic's rules: record, retake, cross_check (measure
    again with the other method) or nurse_outline."""
    if v.check is None or not v.check["ok"]:
        raise PermissionError("check_photo has not passed for this photo; ask for a retake instead")
    scale = patients.card_scale(patients.get(v.patient_id))
    exclude = [v.calibration["corners_px"]] if v.calibration and v.calibration.get("found") else None
    result = vision.measure(v.image(), method, card_scale=scale or 1.0, exclude=exclude)
    result["card_calibrated"] = scale is not None
    if result.get("found"):
        ann = vision.annotate(v.image(), result, extra_quads=exclude)
        ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
        store.put(f"annotated/{photo_id}-{method}.jpg", buf.tobytes(), "image/jpeg")
        result["annotated"] = f"annotated/{photo_id}-{method}.jpg"
        result["area_sd_cm2"] = vision.area_sd_cm2(result, v.check["metrics"])
    clean = {k: val for k, val in result.items() if not k.startswith("_")}
    v.measures[method] = clean
    others = [m for k, m in v.measures.items() if k != method]
    first = v.measures.get("grabcut") or clean
    second = clean if first is not clean else (others[0] if others else None)
    v.decision = policy.decide(v.check, first, second)
    clean["decision"] = v.decision
    return clean


@tool(patient_id=PATIENT)
def get_history(v: Visit, patient_id: str) -> dict:
    """The patient's wound profile and earlier measurements (date, day number, area)."""
    p = patients.get(patient_id)
    return {"patient": {k: p.get(k) for k in ("id", "name", "wound", "start_date", "demo", "card")},
            "history": p["history"]}


@tool(patient_id=PATIENT, photo_id=PHOTO)
def assess_healing(v: Visit, patient_id: str, photo_id: str) -> dict:
    """Compare this photo's measured area with the baseline and the last visit using the
    4-week rule (an ulcer that has not shrunk by half in 4 weeks needs its care plan
    reviewed). Uses the measured number from measure_wound and its uncertainty: a change
    smaller than the measurement noise is not called worsening."""
    m = v.measures.get("grabcut") or next(iter(v.measures.values()), None)
    if not m or not m.get("found"):
        raise PermissionError("measure_wound has not produced an area for this photo")
    p = patients.get(patient_id)
    v.healing = policy.assess_healing(p["history"], m["area_cm2"], patients.day_of(p),
                                      m.get("area_sd_cm2"))
    return v.healing


@tool(patient_id=PATIENT, photo_id=PHOTO)
def record_measurement(v: Visit, patient_id: str, photo_id: str) -> dict:
    """Save this photo's measurement to the patient's record. Allowed only when the
    clinic's rules decided 'record'."""
    if not v.decision or v.decision["action"] != "record":
        raise PermissionError(f"rules decided {v.decision and v.decision['action']!r}, not 'record'")
    if patients.card_scale(patients.get(patient_id)) is None:
        raise PermissionError("the printed card has not been size-checked: call calibrate_card on a "
                              "photo with a plastic card, or ask the patient for one")
    m = v.measures.get("grabcut") or next(iter(v.measures.values()))
    entry = patients.add_measurement(patient_id, photo_id, m)
    v.outcome = v.outcome or {"kind": "recorded", "entry": entry}
    return {"saved": entry}


@tool(photo_id=PHOTO, message={"type": "string",
                               "description": "Short, kind instruction to the patient: what to change and why."})
def request_retake(v: Visit, photo_id: str, message: str) -> dict:
    """Ask the patient for a new photo. Use when check_photo failed, calibrate_card failed,
    or the measurement decision is 'retake'. Ends the visit."""
    if v.decision and v.decision["action"] == "record":
        raise PermissionError("this photo is measurable; record it instead of asking for a retake")
    if v.check and not v.check["ok"]:
        reasons = [p["code"] for p in v.check["problems"]]
    elif v.calibration and not v.calibration.get("found"):
        reasons = ["card_not_calibrated"]
    else:
        reasons = (v.decision or {}).get("reasons", [])
    v.outcome = {"kind": "retake", "message": message, "reasons": reasons,
                 "fixes": [vision.FIX[r] for r in reasons if r in vision.FIX]}
    return {"sent_to_patient": True, **v.outcome}


@tool(patient_id=PATIENT, photo_id=PHOTO,
      kind={"type": "string", "enum": ["care_plan_review", "confirm_outline"],
            "description": "care_plan_review when healing stalled or worsened; confirm_outline when the edge is uncertain"},
      note={"type": "string", "description": "Two or three factual sentences for the nurse, with the numbers."})
def request_nurse_review(v: Visit, patient_id: str, photo_id: str, kind: str, note: str) -> dict:
    """Draft a message to the patient's nurse. It is NOT sent: it waits in the nurse
    queue until a human approves it. Use for stalled or worsening healing, or when the
    decision is nurse_outline."""
    alert = {
        "id": uuid.uuid4().hex[:10], "patient_id": patient_id, "photo_id": photo_id,
        "kind": kind, "note": note, "status": "awaiting_approval",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence": {"measure": v.measures.get("grabcut"), "healing": v.healing,
                     "decision": v.decision},
    }
    store.put_json(f"alerts/{alert['id']}.json", alert)
    v.outcome = {"kind": "nurse_review", "alert_id": alert["id"],
                 "recorded": bool(v.outcome and v.outcome.get("kind") == "recorded")}
    return {"queued_for_approval": alert["id"], "status": alert["status"]}


def today() -> str:
    return date.today().isoformat()
