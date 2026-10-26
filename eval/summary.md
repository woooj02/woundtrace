# Evaluation summary (synthetic test targets, not patients)

Hard set: 600 simulated phone photos with random tilt (0-60°), distance, blur, glare and six skin tones. Clean set: 150 near-ideal photos.

| Result | Value |
|---|---|
| Recorded / retake / nurse outline | 245 / 355 / 0 |
| Area error of recorded measurements (mean abs) | 1.4% |
| 95th percentile abs error, recorded | 4.4% |
| Recorded within 5% / 10% of true area | 96.7% / 100.0% |
| Recorded with error over 10% (bad accepts) | 0 |
| Same pipeline with gates off (mean abs error) | 14.4% |
| Retakes that would have been >10% off | 80 of 322 measurable |
| Retakes that would have been within 5% (needless) | 221 of 322 |
| Clean set error (mean abs) | 0.8% |
| Ruler length × width vs true area (mean) | +43.9% |
| Median processing time per photo | 1746 ms |

Retake reasons: blurry 148, tilted 147, too_far 79, glare 39, card_not_found 32, card_bent 4

| Skin tone (1 light → 6 dark) | Cases | Recorded | Error | Card not found | Wound not found |
|---|---|---|---|---|---|
| 1 | 100 | 38 | 2.0% | 7 | 0 |
| 2 | 85 | 34 | 1.7% | 4 | 0 |
| 3 | 113 | 51 | 1.6% | 4 | 0 |
| 4 | 106 | 39 | 0.8% | 6 | 0 |
| 5 | 89 | 35 | 0.9% | 6 | 0 |
| 6 | 107 | 48 | 1.4% | 5 | 0 |
