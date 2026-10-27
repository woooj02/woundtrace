"""Offline tests: rules engine, local storage in a temp folder, synthetic photos."""

import io
import json
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
                   "start_date": (date.today() - timedelta(days=28)).isoformat(), "history": [],
                   "card": {"scale": 1.0, "method": "test"}})


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


def test_worsening_must_exceed_measurement_noise():
    hist = [{"day": 0, "area_cm2": 5.0, "area_sd_cm2": 0.05}, {"day": 14, "area_cm2": 4.0, "area_sd_cm2": 0.3}]
    # 12% larger, but within 2 sd of the difference: not called worsening
    assert policy.assess_healing(hist, 4.48, 21, area_sd_cm2=0.3)["status"] != "worsening"
    precise = [dict(h, area_sd_cm2=0.05) for h in hist]
    assert policy.assess_healing(precise, 4.48, 21, area_sd_cm2=0.05)["status"] == "worsening"


def test_measurement_reports_uncertainty():
    img, true = photo()
    pid = upload(img)
    r = client.post("/api/visits", json={"patient_id": "t1", "photo_id": pid}).json()
    m = [s for s in r["steps"] if s["tool"] == "measure_wound"][0]["output"]
    assert 0 < m["area_sd_cm2"] < 0.1 * m["area_cm2"]
    assert abs(m["area_cm2"] - true) < 4 * m["area_sd_cm2"] + 0.05


def card_photo(print_scale, with_id1, area=6.0, seed=5):
    rng = np.random.default_rng(seed)
    plane, mm2 = synth.render_plane(rng, area, 1, size_mm=(230.0, 150.0), print_scale=print_scale,
                                    id1_at=(128.0, 58.0) if with_id1 else None)
    return synth.photograph(rng, plane, 8.0, synth.Shot(12.0, 4.5, 0.0, False, 1, mm2 / 100)), mm2 / 100


def uncalibrated():
    p = patients.get("t1")
    p.pop("card")
    patients.save(p)


def test_uncalibrated_card_without_plastic_card_asks_for_card_check():
    uncalibrated()
    img, _ = card_photo(0.925, with_id1=False)
    r = client.post("/api/visits", json={"patient_id": "t1", "photo_id": upload(img)}).json()
    assert r["outcome"]["kind"] == "retake" and r["outcome"]["reasons"] == ["card_not_calibrated"]
    assert patients.get("t1")["history"] == []


def test_plastic_card_corrects_a_shrunken_print():
    uncalibrated()
    img, true = card_photo(0.925, with_id1=True)
    raw = vision.measure(img)["area_cm2"]
    assert raw / true - 1 > 0.12  # uncorrected: the shrunken card inflates the area
    r = client.post("/api/visits", json={"patient_id": "t1", "photo_id": upload(img)}).json()
    assert r["outcome"]["kind"] == "recorded"
    assert abs(patients.get("t1")["card"]["scale"] - 0.925) < 0.01
    assert abs(patients.get("t1")["history"][0]["area_cm2"] / true - 1) < 0.03


def test_manual_card_check_and_record_guard():
    uncalibrated()
    img, _ = photo()
    pid = upload(img)
    v = tools.Visit("t1", pid)
    tools.call(v, "check_photo", {"photo_id": pid})
    tools.call(v, "measure_wound", {"photo_id": pid, "method": "grabcut"})
    with pytest.raises(PermissionError):
        tools.call(v, "record_measurement", {"patient_id": "t1", "photo_id": pid})
    assert client.post("/api/patients/t1/card", json={"marker_mm": 27.7}).json()["scale"] == 0.9233
    assert client.post("/api/patients/t1/card", json={"marker_mm": 3}).status_code == 422


def test_disputed_outline_goes_to_the_nurse():
    ok = {"ok": True, "problems": []}
    m = {"found": True, "glare_fraction": 0.0, "edge_contrast": 20.0, "length_cm": 3.0, "width_cm": 2.5,
         "thickness_cm": 2.0, "area_cm2": 6.0}
    assert policy.decide(ok, m, review={"verdict": "follows"})["action"] == "record"
    assert policy.decide(ok, m, review={"verdict": "misses"}) == {"action": "nurse_outline",
                                                                  "reasons": ["outline_disputed"]}


def test_bedrock_visits_cannot_record_without_an_outline_review():
    img, _ = photo()
    pid = upload(img)
    v = tools.Visit("t1", pid)
    v.needs_review = True
    tools.call(v, "check_photo", {"photo_id": pid})
    tools.call(v, "measure_wound", {"photo_id": pid, "method": "dnn"})
    with pytest.raises(PermissionError):
        tools.call(v, "record_measurement", {"patient_id": "t1", "photo_id": pid})


def test_visits_are_rate_limited_per_visitor(monkeypatch):
    from app import main
    monkeypatch.setattr(main, "VISITS_PER_HOUR", 1)
    pid = upload(photo()[0])
    assert client.post("/api/visits", json={"patient_id": "t1", "photo_id": pid}).status_code == 200
    assert client.post("/api/visits", json={"patient_id": "t1", "photo_id": pid}).status_code == 429


class FakeBedrock:
    """Answers the outline check with a fixed wound box (0-1000 coordinates)."""

    def __init__(self, box, visible=True):
        self.reply = json.dumps({"wound_visible": visible, "box": box})

    def converse(self, **kw):
        assert "image" in kw["messages"][0]["content"][0]  # the plain photo, no outline drawn
        return {"output": {"message": {"content": [{"text": self.reply}]}}, "usage": {}}


def test_outline_check_compares_the_models_box_with_the_outline():
    from app import review
    img = np.zeros((1000, 1000, 3), np.uint8)
    mask = np.zeros((1000, 1000), np.uint8)
    mask[400:600, 400:600] = 255
    assert review.review(img, mask, FakeBedrock([390, 410, 610, 590]))["verdict"] == "follows"
    far = review.review(img, mask, FakeBedrock([50, 50, 150, 150]))
    assert far["verdict"] == "misses" and far["iou"] == 0.0
    assert review.review(img, mask, FakeBedrock(None, visible=False))["verdict"] == "misses"
    assert review.review(img, mask, client=object())["verdict"] == "error"  # never raises


def test_photo_without_card_shows_the_wound_but_never_a_size():
    pid = upload(cv2.imread("web/samples/real-no-card.jpg"))  # a real wound photo, no card (CC BY 4.0 dataset)
    r = client.post("/api/visits", json={"patient_id": "t1", "photo_id": pid}).json()
    chk = r["steps"][0]["output"]
    assert r["outcome"]["kind"] == "retake" and "card_not_found" in r["outcome"]["reasons"]
    assert chk["wound_preview"]["wound_seen"] and not chk["wound_preview"]["measured"]
    assert "area_cm2" not in json.dumps(chk) and r["answer"].startswith("I can see the wound")
    assert patients.get("t1")["history"] == []
