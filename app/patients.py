"""Patient records. The demo patients are test targets, not real people."""

from __future__ import annotations

import time
from datetime import date

from . import store


def get(pid: str) -> dict:
    p = store.get_json(f"patients/{pid}.json")
    if p is None:
        raise KeyError(f"no patient {pid}")
    return p


def all_patients() -> list[dict]:
    out = []
    for key in store.keys("patients/"):
        p = store.get_json(key)
        if p:
            out.append(p)
    return sorted(out, key=lambda p: p["id"])


def save(p: dict) -> None:
    store.put_json(f"patients/{p['id']}.json", p)


def day_of(p: dict, on: date | None = None) -> int:
    return ((on or date.today()) - date.fromisoformat(p["start_date"])).days


def add_measurement(pid: str, photo_id: str, m: dict, on: date | None = None) -> dict:
    p = get(pid)
    entry = {"date": (on or date.today()).isoformat(), "day": day_of(p, on),
             "area_cm2": m["area_cm2"], "area_sd_cm2": m.get("area_sd_cm2"), "length_cm": m["length_cm"], "width_cm": m["width_cm"],
             "ruler_area_cm2": m["ruler_area_cm2"], "photo_id": photo_id,
             "method": m["method"], "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p["history"].append(entry)
    save(p)
    return entry


def card_scale(p: dict) -> float | None:
    """How large this patient's printed card really is vs its design (None: not checked yet)."""
    return (p.get("card") or {}).get("scale")


def set_card(pid: str, card: dict) -> dict:
    p = get(pid)
    p["card"] = {**card, "date": date.today().isoformat()}
    save(p)
    return p["card"]
