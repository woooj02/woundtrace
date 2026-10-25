# Evaluation summary (synthetic test targets, not patients)

Hard set: 600 simulated phone photos with random tilt (0-60°), distance, blur, glare and six skin tones. Clean set: 150 near-ideal photos.

| Result | Value |
|---|---|
| Recorded / retake / nurse outline | 232 / 366 / 2 |
| Area error of recorded measurements (mean abs) | 2.1% |
| 95th percentile abs error, recorded | 3.5% |
| Recorded within 5% / 10% of true area | 97.0% / 98.7% |
| Recorded with error over 10% (bad accepts) | 3 |
| Same pipeline with gates off (mean abs error) | 13.4% |
| Retakes that would have been >10% off | 92 of 331 measurable |
| Retakes that would have been within 5% (needless) | 209 of 331 |
| Clean set error (mean abs) | 3.4% |
| Ruler length × width vs true area (mean) | +43.0% |
| Median processing time per photo | 1235 ms |

Retake reasons: blurry 181, tilted 132, too_far 79, glare 44, card_not_found 29, card_bent 1

| Skin tone (1 light → 6 dark) | Cases | Recorded | Error | Card not found | Wound not found |
|---|---|---|---|---|---|
| 1 | 105 | 40 | 1.7% | 3 | 0 |
| 2 | 81 | 27 | 1.3% | 6 | 0 |
| 3 | 105 | 38 | 1.2% | 6 | 0 |
| 4 | 93 | 34 | 0.8% | 4 | 0 |
| 5 | 111 | 50 | 2.6% | 4 | 0 |
| 6 | 105 | 43 | 4.0% | 6 | 0 |
