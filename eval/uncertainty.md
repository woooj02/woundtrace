# Per-measurement uncertainty

Each recorded area comes with a standard deviation computed from that photo's own evidence
(`vision.area_sd_cm2`):

    sd / area = sqrt( k0² + (k1 · perimeter · (edge blur + pixel size) / area)² + (k2 · card rms)² )

Edge placement dominates: a boundary off by d mm changes the area by about perimeter × d, and
d grows with blur and coarser pixels. Fitted by maximum likelihood on the tuning seed
(`eval/fit_uncertainty.py`): k0 = 0.0039, k1 = 0.205, k2 = 0.

| Synthetic set | Recorded | Inside ±1 sd | Inside ±2 sd | Median sd |
|---|---|---|---|---|
| Tuning seed 2026 (fit) | 372 | 63.7% | 97.0% | 1.3% |
| Held-out seed 11 (check) | 389 | 66.3% | 96.6% | 1.2% |

A calibrated model would put 68% inside ±1 sd and 95% inside ±2 sd: the held-out check matches.

**Limits (honest):**
- k2 (card not flat) fitted to 0 because synthetic cards are perfectly flat; curved skin is not modelled yet.
- On real phone photos of the colour sheets (faulty printer) the model is overconfident: 2 of 4
  recorded measurements fall inside ±2 sd. Real-world error includes printing, lens and light effects
  the simulator lacks. Next: fit a real-world floor on black-and-white printed targets and check it on
  a separate real set (coins on skin).

How it is used: the app shows area ± 2 sd, and a rise only counts as "worsening" when it is over 10%
and larger than 2 sd of the difference between the two photos.
