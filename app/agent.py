"""The visit agent: perceive (OpenCV tools) → decide → act, with a recorded trace.

Two engines, same tools, same guards:
  bedrock: an LLM on Amazon Bedrock (Converse API) chooses the next tool and
           writes the messages. Used when Bedrock is reachable and today's
           LLM budget is not spent.
  rules:   no LLM. A fixed sequence driven by policy.decide, with template
           messages. Used offline, in tests, and as the fallback. The UI
           always shows which engine ran.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date

from . import patients, policy, store, tools, vision

MODEL = os.environ.get("BEDROCK_MODEL", "us.amazon.nova-2-lite-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "10"))
DAILY_LLM_RUNS = int(os.environ.get("DAILY_LLM_RUNS", "150"))

SYSTEM = """You are WoundTrace, the photo assistant for a home wound-care service.
A patient has sent a photo of their wound next to a printed reference card. Your job
is one visit: get a trustworthy measurement or a better photo, and tell the nurse
when the numbers say they should look.

How to work:
1. Call check_photo first. If it fails, call request_retake with a short, kind
   message built from the fixes it returned. Do not measure a failed photo.
   If it passes but card_calibrated is false, call calibrate_card. If that fails, call
   request_retake asking for the one-time card check (a plastic card next to the printed
   card); do not measure.
2. Otherwise call measure_wound with method "grabcut". Follow its decision:
   - retake: request_retake, explaining the problem (for example glare).
   - cross_check: call measure_wound again with method "colour"; follow the new decision.
   - nurse_outline: request_nurse_review with kind "confirm_outline".
   - record: call get_history and assess_healing, then record_measurement.
     If assess_healing says needs_nurse, also call request_nurse_review with kind
     "care_plan_review" and a note giving baseline, current area ± its uncertainty, days
     and change, and saying it is a photo measurement, not a diagnosis.
3. Your final reply is sent to the patient word for word: speak to them as "you", in
   two or three plain sentences with no markdown. Say what happened and what happens
   next, and mention the area if one was recorded. Never describe your steps, tools or
   ids in it.

Rules: you measure, you never diagnose. Don't say how the wound looks or whether it is
healthy, and don't comment on infection or treatment.
Nurse messages are drafts that a human approves. Use only numbers the tools returned."""


def engine() -> str:
    if os.environ.get("AGENT_ENGINE") in ("rules", "bedrock"):
        return os.environ["AGENT_ENGINE"]
    return "bedrock" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("USE_BEDROCK") else "rules"


def _llm_budget_left() -> bool:
    key = f"usage/{date.today().isoformat()}.json"
    u = store.get_json(key, {"runs": 0})
    return u["runs"] < DAILY_LLM_RUNS


def _count_llm_run(usage: dict) -> None:
    key = f"usage/{date.today().isoformat()}.json"
    u = store.get_json(key, {"runs": 0, "input_tokens": 0, "output_tokens": 0})
    u["runs"] += 1
    u["input_tokens"] += usage.get("inputTokens", 0)
    u["output_tokens"] += usage.get("outputTokens", 0)
    store.put_json(key, u)


# Outgoing-message guard: the model's reply goes to a patient, so plain code checks it.
# A reply that leaks internals (tool names, ids, "the user"), offers a clinical opinion,
# or runs long is replaced by the template message for the same outcome, and the trace
# says so.
LEAK = re.compile(r"check_photo|calibrate_card|measure_wound|get_history|assess_healing|record_measurement|"
                  r"request_retake|request_nurse_review|photo_id|patient_id|\bthe user\b|\btool",
                  re.I)
OPINION = re.compile(r"looks? (good|great|fine|healthy|bad|worse|infected)|\bhealthy\b|"
                     r"\binfect|\bdiagnos|antibiotic", re.I)
MAX_MESSAGE = 500


def message_problem(text: str, visit: tools.Visit) -> str | None:
    if not text.strip():
        return "empty"
    if len(text) > MAX_MESSAGE:
        return "too long"
    if LEAK.search(text) or visit.patient_id in text or visit.photo_id in text:
        return "mentions internal steps or ids"
    if OPINION.search(text):
        return "clinical opinion"
    return None


def template_message(visit: tools.Visit) -> str:
    o = visit.outcome or {}
    m = visit.measures.get("grabcut") or next(iter(visit.measures.values()), {})
    if o.get("kind") == "retake":
        return f"I couldn't measure this photo. {' '.join(o.get('fixes') or [o.get('message', '')])}".strip()
    if o.get("kind") == "recorded":
        return f"Measured {m['area_cm2']:.2f} cm² and saved it to your record."
    if o.get("kind") == "nurse_review" and o.get("recorded"):
        return (f"Measured {m['area_cm2']:.2f} cm² and saved it to your record. "
                "I've asked your nurse to review how it's healing.")
    if o.get("kind") == "nurse_review":
        return ("Your photo is saved. The edge of the wound is hard to see, so your nurse "
                "will check the outline before it is recorded.")
    return "Your photo was received. Your nurse will follow up if anything is needed."


def run(patient_id: str, photo_id: str) -> dict:
    started = time.monotonic()
    visit = tools.Visit(patient_id, photo_id)
    eng = engine()
    note = None
    if eng == "bedrock" and not _llm_budget_left():
        eng, note = "rules", "Daily LLM budget reached; the rules engine ran instead."
    if eng == "bedrock":
        try:
            result = _run_bedrock(visit)
        except Exception as exc:  # Bedrock down or denied: say so and fall back
            note = f"Bedrock unavailable ({type(exc).__name__}: {str(exc)[:160]}); rules engine ran instead."
            visit = tools.Visit(patient_id, photo_id)
            eng, result = "rules", None
    if eng == "rules":
        result = _run_rules(visit)
    if eng == "bedrock":
        result["answer"] = re.sub(r"\*\*|__|`|^#+\s*", "", result["answer"], flags=re.M)  # plain text only
        # A retake ends the visit with a message the model already wrote for the patient
        # (request_retake's message); use it, not whatever text the model adds afterwards.
        o = visit.outcome or {}
        if o.get("kind") == "retake" and o.get("message"):
            result["model_answer"], result["answer"] = result["answer"], o["message"]
        if why := message_problem(result["answer"], visit):
            result.setdefault("model_answer", result["answer"])
            result["answer"] = template_message(visit)
            result["message_replaced"] = why
    result.update(engine=eng, model=MODEL if eng == "bedrock" else None, note=note,
                  ms=int((time.monotonic() - started) * 1000),
                  patient_id=patient_id, photo_id=photo_id, outcome=visit.outcome,
                  decision=visit.decision, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    result["id"] = f"{photo_id}"
    store.put_json(f"runs/{photo_id}.json", result)
    # One structured line per visit: CloudWatch Logs Insights can query these.
    print(json.dumps({"event": "visit", "engine": eng, "patient": patient_id, "photo": photo_id,
                      "outcome": (visit.outcome or {}).get("kind"),
                      "decision": (visit.decision or {}).get("action"),
                      "steps": [s["tool"] for s in result["steps"]], "ms": result["ms"],
                      "message_replaced": result.get("message_replaced"),
                      "usage": result.get("usage")}))
    return result


def _step(visit, name, args, steps):
    t = time.perf_counter()
    step = {"tool": name, "input": args}
    try:
        step["output"] = tools.call(visit, name, args)
    except Exception as exc:
        step["error"] = f"{type(exc).__name__}: {exc}"
    step["ms"] = round((time.perf_counter() - t) * 1000)
    steps.append(step)
    return step


def _run_rules(visit: tools.Visit) -> dict:
    """The deterministic path. Same tools, same guards, template wording."""
    steps: list[dict] = []
    pid, ph = visit.patient_id, visit.photo_id
    chk = _step(visit, "check_photo", {"photo_id": ph}, steps)["output"]
    if not chk["ok"]:
        fixes = " ".join(p["fix"] for p in chk["problems"])
        _step(visit, "request_retake", {"photo_id": ph, "message": fixes}, steps)
        return {"answer": f"I couldn't measure this photo. {fixes}", "steps": steps}
    if not chk["card_calibrated"]:
        cal = _step(visit, "calibrate_card", {"patient_id": pid, "photo_id": ph}, steps)["output"]
        if not cal["calibrated"]:
            fix = vision.FIX["card_not_calibrated"]
            _step(visit, "request_retake", {"photo_id": ph, "message": fix}, steps)
            return {"answer": f"Before I can measure: {fix}", "steps": steps}
    m = _step(visit, "measure_wound", {"photo_id": ph, "method": "grabcut"}, steps)["output"]
    if m["decision"]["action"] == "cross_check":
        m = _step(visit, "measure_wound", {"photo_id": ph, "method": "colour"}, steps)["output"]
    action = m["decision"]["action"]
    if action == "retake":
        fixes = " ".join(vision.FIX.get(r, r) for r in m["decision"]["reasons"])
        _step(visit, "request_retake", {"photo_id": ph, "message": fixes}, steps)
        return {"answer": f"I couldn't measure this photo reliably. {fixes}", "steps": steps}
    if action == "nurse_outline":
        _step(visit, "request_nurse_review", {
            "patient_id": pid, "photo_id": ph, "kind": "confirm_outline",
            "note": f"The wound edge is unclear in today's photo (edge contrast "
                    f"{m.get('edge_contrast')}). Please confirm or trace the outline."}, steps)
        return {"answer": "Your photo is saved. The edge of the wound is hard to see, so your "
                          "nurse will check the outline before it is recorded.", "steps": steps}
    _step(visit, "get_history", {"patient_id": pid}, steps)
    h = _step(visit, "assess_healing", {"patient_id": pid, "photo_id": ph}, steps)["output"]
    _step(visit, "record_measurement", {"patient_id": pid, "photo_id": ph}, steps)
    answer = f"Measured {m['area_cm2']:.2f} cm² and saved it to your record."
    if h.get("needs_nurse"):
        sd = m.get("area_sd_cm2")
        _step(visit, "request_nurse_review", {
            "patient_id": pid, "photo_id": ph, "kind": "care_plan_review",
            "note": f"Area {m['area_cm2']:.2f}{f' ± {2 * sd:.2f}' if sd else ''} cm² on day "
                    f"{h['days_since_baseline']}, baseline {h['baseline_cm2']:.2f} cm² "
                    f"({h['reduction_from_baseline']:.0%} smaller). {h['detail']} "
                    "Photo measurement from a printed reference card; not a diagnosis."}, steps)
        answer += " I've asked your nurse to review how it's healing."
    return {"answer": answer, "steps": steps}


def _run_bedrock(visit: tools.Visit) -> dict:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=REGION)
    tool_config = {"tools": [{"toolSpec": {"name": s["name"], "description": s["description"],
                                           "inputSchema": {"json": s["input_schema"]}}}
                             for s in tools.specs()]}
    task = (f"New photo from patient_id={visit.patient_id}, photo_id={visit.photo_id}. "
            f"Today is {date.today().isoformat()}.")
    messages = [{"role": "user", "content": [{"text": task}]}]
    steps: list[dict] = []
    usage = {"inputTokens": 0, "outputTokens": 0}
    for _ in range(MAX_STEPS):
        r = client.converse(modelId=MODEL, system=[{"text": SYSTEM}], messages=messages,
                            toolConfig=tool_config, inferenceConfig={"maxTokens": 800})
        for k in usage:
            usage[k] += r.get("usage", {}).get(k, 0)
        msg = r["output"]["message"]
        messages.append(msg)
        texts = [b["text"] for b in msg["content"] if "text" in b]
        if r["stopReason"] != "tool_use":
            _count_llm_run(usage)
            return {"answer": "\n".join(texts).strip(), "steps": steps, "usage": usage}
        if texts:
            steps.append({"tool": "agent_note", "output": " ".join(texts).strip()})
        results = []
        for b in msg["content"]:
            if "toolUse" not in b:
                continue
            tu = b["toolUse"]
            step = _step(visit, tu["name"], tu.get("input") or {}, steps)
            if "error" in step:
                results.append({"toolResult": {"toolUseId": tu["toolUseId"], "status": "error",
                                               "content": [{"text": step["error"]}]}})
            else:
                results.append({"toolResult": {"toolUseId": tu["toolUseId"],
                                               "content": [{"json": json.loads(json.dumps(step["output"], default=str))}]}})
        messages.append({"role": "user", "content": results})
    _count_llm_run(usage)
    return {"answer": "Stopped after the step limit.", "steps": steps, "usage": usage}
