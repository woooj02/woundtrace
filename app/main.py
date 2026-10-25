"""HTTP layer: the web app, photo upload, the visit agent and the nurse queue."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import agent, patients, store, tools, vision

WEB = Path(__file__).resolve().parent.parent / "web"
MAX_UPLOAD = 6 * 1024 * 1024

app = FastAPI(title="woundtrace")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "engine": agent.engine(), "model": agent.MODEL,
            "opencv": cv2.__version__, "tools": sorted(tools.TOOLS),
            "storage": "s3" if store._bucket() else "local", "gates": vision.GATES}


@app.get("/api/patients")
def list_patients():
    out = []
    for p in patients.all_patients():
        alerts = [a for a in _alerts() if a["patient_id"] == p["id"]]
        out.append({**p, "pending": sum(a["status"] == "awaiting_approval" for a in alerts),
                    "day": patients.day_of(p)})
    return out


@app.get("/api/patients/{pid}")
def patient(pid: str):
    _safe(pid)
    try:
        p = patients.get(pid)
    except KeyError:
        raise HTTPException(404, "no such patient")
    return {**p, "day": patients.day_of(p),
            "alerts": [a for a in _alerts() if a["patient_id"] == pid]}


@app.post("/api/photos")
async def upload(file: UploadFile = File(...), patient_id: str = Form(...)):
    _safe(patient_id)
    data = await file.read()
    if not data:
        raise HTTPException(422, "empty file")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "photo too large (max 6 MB)")
    try:
        img = vision.decode(data)
    except ValueError:
        raise HTTPException(422, "that file is not an image")
    h, w = img.shape[:2]
    if max(h, w) > 2400:  # keep processing time bounded on Lambda
        f = 2400 / max(h, w)
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    photo_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    store.put(f"photos/{photo_id}.jpg", buf.tobytes(), "image/jpeg")
    return {"photo_id": photo_id, "width": img.shape[1], "height": img.shape[0]}


class VisitRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=40)
    photo_id: str = Field(min_length=1, max_length=40)


@app.post("/api/visits")
def visit(req: VisitRequest):
    _safe(req.patient_id), _safe(req.photo_id)
    try:
        patients.get(req.patient_id)
    except KeyError:
        raise HTTPException(404, "no such patient")
    if store.get(f"photos/{req.photo_id}.jpg") is None:
        raise HTTPException(404, "no such photo")
    try:
        return agent.run(req.patient_id, req.photo_id)
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/runs/{photo_id}")
def get_run(photo_id: str):
    _safe(photo_id)
    r = store.get_json(f"runs/{photo_id}.json")
    if r is None:
        raise HTTPException(404, "no such run")
    return r


@app.get("/api/img/{kind}/{name}")
def image(kind: str, name: str):
    if kind not in ("photos", "annotated"):
        raise HTTPException(404)
    _safe(name.removesuffix(".jpg"))
    data = store.get(f"{kind}/{name}")
    if data is None:
        raise HTTPException(404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


class Decision(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    note: str = Field(default="", max_length=500)


@app.post("/api/alerts/{alert_id}/decide")
def decide(alert_id: str, d: Decision):
    _safe(alert_id)
    a = store.get_json(f"alerts/{alert_id}.json")
    if a is None:
        raise HTTPException(404, "no such alert")
    if a["status"] != "awaiting_approval":
        raise HTTPException(409, f"already {a['status']}")
    a["status"] = "approved" if d.decision == "approve" else "rejected"
    a["decided"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    a["decision_note"] = d.note
    store.put_json(f"alerts/{alert_id}.json", a)
    print(f'{{"event": "alert_decided", "alert": "{alert_id}", "status": "{a["status"]}"}}')
    return a


def _alerts() -> list[dict]:
    out = [store.get_json(k) for k in store.keys("alerts/")]
    return sorted([a for a in out if a], key=lambda a: a["created"], reverse=True)


def _safe(s: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", s):
        raise HTTPException(422, "bad id")
