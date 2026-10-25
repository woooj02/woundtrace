"""Offline tests: rules engine, local storage in a temp folder, synthetic photos."""

import io
from datetime import date, timedelta

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import agent, patients, policy, store, synth, tools, vision
from app.main import app


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ENGINE", "rules")
    monkeypatch.delenv("BUCKET", raising=False)
    monkeypatch.setattr(store, "LOCAL", tmp_path)
    patients.save({"id": "t1", "name": "Test", "wound": "test target", "demo": "test",
                   "start_date": (date.today() - timedelta(days=28)).isoformat(), "history": []})


client = TestClient(app)


def photo(area=6.0, skin=1, tilt=10.0, ppm=6.0, blur=0.0, glare=False, seed=3):
    rng = np.random.default_rng(seed)
    plane, mm2 = synth.render_plane(rng, area, skin)
    shot = synth.Shot(tilt, ppm, blur, glare, skin, mm2 / 100)
    return synth.photograph(rng, plane, 8.0, shot), mm2 / 100


def upload(img, pid="t1"):
    ok, buf = cv2.imencode(".jpg", img)
    r = client.post("/api/photos", data={"patient_id": pid},
                    files={"file": ("p.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")})
    assert r.status_code == 200, r.text
    return r.json()["photo_id"]


def test_health_reports_opencv5():
    h = client.get("/api/health").json()
    assert h["opencv"].startswith("5.") and "measure_wound" in h["tools"]


def test_measurement_accuracy_on_good_photo():
    img, true = photo()
    assert vision.check_photo(img)["ok"]
    m = vision.measure(img)
    assert abs(m["area_cm2"] / true - 1) < 0.05
    assert m["ruler_area_cm2"] > m["area_cm2"]  # length x width overestimates


def test_tilted_photo_gets_retake():
    img, _ = photo(tilt=55)
    pid = upload(img)
    r = client.post("/api/visits", json={"patient_id": "t1", "photo_id": pid}).json()
    assert r["outcome"]["kind"] == "retake" and "tilted" in r["outcome"]["reasons"]
    assert [s["tool"] for s in r["steps"]] == ["check_photo", "request_retake"]
    assert patients.get("t1")["history"] == []


def test_good_photo_is_recorded_as_baseline():
    img, true = photo()
    pid = upload(img)
    r = client.post("/api/visits", json={"patient_id": "t1", "photo_id": pid}).json()
    assert r["outcome"]["kind"] == "recorded"
    h = patients.get("t1")["history"]
    assert len(h) == 1 and abs(h[0]["area_cm2"] / true - 1) < 0.05


def test_stalled_healing_queues_nurse_alert_for_approval():
    p = patients.get("t1")
    p["history"] = [{"date": p["start_date"], "day": 0, "area_cm2": 6.3}]
    patients.save(p)
    img, _ = photo(area=6.0)
    pid = upload(img)
    r = client.post("/api/visits", json={"patient_id": "t1", "photo_id": pid}).json()
    assert r["outcome"]["kind"] == "nurse_review"
    a = client.get("/api/patients/t1").json()["alerts"][0]
    assert a["status"] == "awaiting_approval" and a["kind"] == "care_plan_review"
    d = client.post(f"/api/alerts/{a['id']}/decide", json={"decision": "approve"}).json()
    assert d["status"] == "approved"
    assert client.post(f"/api/alerts/{a['id']}/decide", json={"decision": "approve"}).status_code == 409


def test_tools_refuse_to_skip_the_rules():
    img, _ = photo(tilt=55)
    pid = upload(img)
    v = tools.Visit("t1", pid)
    with pytest.raises(PermissionError):
        tools.call(v, "measure_wound", {"photo_id": pid, "method": "grabcut"})
    tools.call(v, "check_photo", {"photo_id": pid})
    with pytest.raises(PermissionError):
        tools.call(v, "record_measurement", {"patient_id": "t1", "photo_id": pid})
    with pytest.raises(ValueError):
        tools.call(v, "check_photo", {"photo_id": "someone-elses"})


def test_healing_rule():
    hist = [{"day": 0, "area_cm2": 8.0}, {"day": 21, "area_cm2": 7.4}]
    assert policy.assess_healing(hist, 7.1, 28)["status"] == "stalled"
    assert policy.assess_healing(hist, 3.9, 28)["status"] == "on_track"
    assert policy.assess_healing(hist, 8.5, 28)["status"] == "worsening"
    assert policy.assess_healing(hist, 7.1, 14)["status"] == "too_early"
    assert policy.assess_healing([], 7.1, 0)["status"] == "baseline"


def test_bad_inputs():
    r = client.post("/api/photos", data={"patient_id": "t1"},
                    files={"file": ("x.jpg", io.BytesIO(b"not an image"), "image/jpeg")})
    assert r.status_code == 422
    assert client.post("/api/visits", json={"patient_id": "../x", "photo_id": "a"}).status_code == 422
    assert client.post("/api/visits", json={"patient_id": "t1", "photo_id": "nope"}).status_code == 404


def test_tool_specs_are_closed():
    for spec in tools.specs():
        s = spec["input_schema"]
        assert spec["description"] and s["additionalProperties"] is False
        assert set(s["required"]) == set(s["properties"])
