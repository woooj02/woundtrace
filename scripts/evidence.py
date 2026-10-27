"""Collect the evaluation results into web/evidence.json for the app's Evidence section.

    venv/bin/python -m scripts.evidence

Every number shown in the app comes from these files, written by the evaluation scripts:
eval/fuseg_results.json and eval/external_results.json (real wound photos), eval/review_results.json (the vision model's outline check),
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
    fus = load("eval/fuseg_results.json")["summary"]
    rev = load("eval/review_results.json")["summary"]
    ext = load("eval/external_results.json")["summary"]
    inside, n = real["inside_2sd"]
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "fuseg": {"photos": fus["dnn"]["n"], "dice": fus["dnn"]["dice"], "dice_ge_07": fus["dnn"]["dice_ge_07"],
                  "missed": fus["dnn"]["missed"], "classical_dice": fus["grabcut"]["dice"],
                  "classical_missed": fus["grabcut"]["missed"]},
        "clinics": {"co2_photos": ext["co2wounds"]["n"], "co2_dice": ext["co2wounds"]["dice"],
                    "co2_dice_ge_07": ext["co2wounds"]["dice_ge_07"],
                    "ll_photos": ext["lowerlimb"]["n"], "ll_dice": ext["lowerlimb"]["dice"],
                    "ll_precision": ext["lowerlimb"]["precision"], "ll_recall": ext["lowerlimb"]["recall"],
                    "healthy_photos": ext["healthy feet"]["n"], "healthy_false": ext["healthy feet"]["false_outline"]},
        "review": {"outlines": rev["reviews"], "bad": rev["bad"], "good": rev["good"],
                   "bad_flagged": rev["bad_flagged"], "good_flagged": rev["good_flagged"],
                   "median_ms": rev["median_ms"]},
        "synthetic": {"cases": syn["cases_hard"] + syn["cases_clean"], "recorded_mape": syn["recorded_mape"],
                      "bad_accepts": syn["bad_accepts_over_10pct"], "forced_mape": syn["forced_mape_all_hard"],
                      "ruler": syn["ruler_mean_overestimate"]},
        "uncertainty": {"within_2sd": unc["check"]["within_2sd"], "median_sd": unc["check"]["median_sd"]},
        "real": real,
        "agent": {"cases": agent["ran_on_bedrock"], "model": agent["model"], "agreement": agent["agreement"],
                  "refusals": agent["cases_with_refusal"], "replaced": agent["messages_replaced"],
                  "median_ms": agent["median_ms"]},
        "limits": [
            "Not tested on patients' own photos: the real-wound tests (three clinics' datasets) have outlines "
            "only, without a reference card, so area accuracy on real wounds is untested.",
            f"On a clinic the model never trained on (lower-limb set), mean Dice is {ext['lowerlimb']['dice']:.2f}: "
            f"its outlines sit on the wound (precision {ext['lowerlimb']['precision']:.2f}) but cover less than "
            f"that dataset marks (recall {ext['lowerlimb']['recall']:.2f}), and it misses "
            f"{ext['lowerlimb']['missed']} of {ext['lowerlimb']['n']} wounds.",
            "The vision model's outline check also sends 28% of good outlines to the nurse; its first design "
            "(judging the overlay) passed every bad outline and was replaced.",
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
