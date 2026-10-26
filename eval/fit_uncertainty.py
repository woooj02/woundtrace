"""Fit and check the per-measurement area uncertainty.

    venv/bin/python -m eval.fit_uncertainty FIT.json CHECK.json [eval/uncertainty.json]

The model gives each recorded area a standard deviation (relative):

    sd² = k0² + (k1 · P · (blur + pixel) / A)² + (k2 · card_rms)²

  P, A        perimeter (mm) and area (mm²) of the outline
  blur        edge width of the card markers (mm): how soft edges are in this photo
  pixel       size of one photo pixel on the skin (mm)
  card_rms    how far the card corners are from a flat-plane fit (mm)

The first term is edge placement: a boundary shift of d mm changes the area by
about P·d. k0..k2 are fitted by maximum likelihood on FIT (the tuning seed) and the
coverage of the ±2 sd interval is checked on CHECK (a held-out seed).
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np


def features(rows):
    r = [x for x in rows if x["action"] == "record" and "err" in x]
    A = np.array([x["area"] * 100 for x in r])
    P = np.array([x["perimeter_cm"] * 10 for x in r])
    m = [x["metrics"] for x in r]
    blur = np.array([v.get("edge_mm", 1.0) for v in m])
    pix = np.array([1 / v["px_per_mm"] for v in m])
    rms = np.array([v.get("card_rms_mm", 0.0) for v in m])
    err = np.array([x["err"] for x in r])  # measured / true - 1
    return np.stack([P * (blur + pix) / A, rms]), err


def sd(k, X):
    return np.sqrt(k[0] ** 2 + (k[1] * X[0]) ** 2 + (k[2] * X[1]) ** 2)


def nll(k, X, err):
    s = sd(k, X)
    return float(np.sum(np.log(s) + 0.5 * (err / s) ** 2))


def fit(X, err):
    grid0 = np.geomspace(0.002, 0.05, 25)
    grid1 = np.concatenate([[0], np.geomspace(0.05, 3, 30)])
    grid2 = np.concatenate([[0], np.geomspace(0.005, 0.5, 20)])
    return min(itertools.product(grid0, grid1, grid2), key=lambda k: nll(k, X, err))


def coverage(k, X, err):
    z = np.abs(err) / sd(k, X)
    return {"n": int(len(err)), "within_1sd": float(np.mean(z <= 1)),
            "within_2sd": float(np.mean(z <= 2)), "median_sd": float(np.median(sd(k, X)))}


def main():
    fit_rows = json.load(open(sys.argv[1]))["rows"]
    chk_rows = json.load(open(sys.argv[2]))["rows"]
    X, err = features(fit_rows)
    k = fit(X, err)
    Xc, errc = features(chk_rows)
    out = {"k": [round(float(v), 4) for v in k], "fit": coverage(k, X, err),
           "check": coverage(k, Xc, errc), "bias_check": float(np.mean(errc))}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
