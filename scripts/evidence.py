"""Collect the evaluation results into web/evidence.json for the app's Evidence section.

    venv/bin/python -m scripts.evidence

Every number shown in the app comes from these files, written by the evaluation scripts:
eval/results.json (synthetic, held-out seed), eval/uncertainty.json, real/results.json,
eval/agent_results.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rel):
    return json.loads((ROOT / rel).read_text())


def main():
    syn = load("eval/results.json")["summary"]
    unc = load("eval/uncertainty.json")
    real = load("real/results.json")["summary"]
    agent = load("eval/agent_results.json")["summary"]
    inside, n = real["inside_2sd"]
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "synthetic": {"cases": syn["cases_hard"] + syn["cases_clean"], "recorded_mape": syn["recorded_mape"],
                      "bad_accepts": syn["bad_accepts_over_10pct"], "forced_mape": syn["forced_mape_all_hard"],
                      "ruler": syn["ruler_mean_overestimate"]},
        "uncertainty": {"within_2sd": unc["check"]["within_2sd"], "median_sd": unc["check"]["median_sd"]},
        "real": real,
        "agent": {"cases": agent["ran_on_bedrock"], "model": agent["model"], "agreement": agent["agreement"],
                  "refusals": agent["cases_with_refusal"], "replaced": agent["messages_replaced"],
                  "median_ms": agent["median_ms"]},
        "limits": [
            "Not tested on real wounds or patients: accuracy on real wounds is unknown.",
            "Synthetic skin is flat, smooth and evenly lit; curved limbs, hair and real tissue are not modelled.",
            f"On real photos the stated uncertainty is too narrow: {inside} of {n} recorded measurements "
            "fell inside their ± 2 sd.",
            "Real test targets: colour sheets from a faulty printer, a black-ink sheet and a coin "
            "(3 mm thick, so it reads about 4-5% large).",
            "A card printed at the wrong size biases every area: the app requires a one-time card check "
            "(plastic card or ruler) before it records anything.",
            "The healing flag follows a guideline threshold for diabetic foot ulcers and only prompts a "
            "clinician review: it does not diagnose, predict an outcome or recommend treatment.",
        ],
    }
    (ROOT / "web" / "evidence.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
