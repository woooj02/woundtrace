# Evaluation summary (synthetic test targets, not patients)

Hard set: 600 simulated phone photos with random tilt (0-60°), distance, blur, glare and six skin tones. Clean set: 150 near-ideal photos.

| Result | Value |
|---|---|
| Recorded / retake / nurse outline | 245 / 355 / 0 |
| Area error of recorded measurements (mean abs) | 1.5% |
| 95th percentile abs error, recorded | 5.5% |
| Recorded within 5% / 10% of true area | 93.9% / 98.8% |
| Recorded with error over 10% (bad accepts) | 3 |
| Same pipeline with gates off (mean abs error) | 5.8% |
| Retakes that would have been >10% off | 59 of 323 measurable |
| Retakes that would have been within 5% (needless) | 208 of 323 |
| Clean set error (mean abs) | 0.7% |
| Ruler length × width vs true area (mean) | +43.3% |
| Median processing time per photo | 1540 ms |

Retake reasons: blurry 148, tilted 147, too_far 79, glare 39, card_not_found 32, card_bent 4

| Skin tone (1 light → 6 dark) | Cases | Recorded | Error | Card not found | Wound not found |
|---|---|---|---|---|---|
| 1 | 100 | 38 | 2.4% | 7 | 0 |
| 2 | 85 | 34 | 1.1% | 4 | 0 |
| 3 | 113 | 51 | 1.2% | 4 | 0 |
| 4 | 106 | 39 | 1.2% | 6 | 0 |
| 5 | 89 | 35 | 1.3% | 6 | 0 |
| 6 | 107 | 48 | 1.6% | 5 | 0 |
