# Evaluation summary (synthetic test targets, not patients)

Hard set: 600 simulated phone photos with random tilt (0-60°), distance, blur, glare and six skin tones. Clean set: 150 near-ideal photos.

| Result | Value |
|---|---|
| Recorded / retake / nurse outline | 245 / 355 / 0 |
| Area error of recorded measurements (mean abs) | 1.2% |
| 95th percentile abs error, recorded | 4.4% |
| Recorded within 5% / 10% of true area | 96.3% / 99.6% |
| Recorded with error over 10% (bad accepts) | 1 |
| Same pipeline with gates off (mean abs error) | 8.0% |
| Retakes that would have been >10% off | 64 of 322 measurable |
| Retakes that would have been within 5% (needless) | 216 of 322 |
| Clean set error (mean abs) | 0.5% |
| Ruler length × width vs true area (mean) | +42.9% |
| Median processing time per photo | 1557 ms |

Retake reasons: blurry 148, tilted 147, too_far 79, glare 39, card_not_found 32, card_bent 4

| Skin tone (1 light → 6 dark) | Cases | Recorded | Error | Card not found | Wound not found |
|---|---|---|---|---|---|
| 1 | 100 | 38 | 1.5% | 7 | 0 |
| 2 | 85 | 34 | 1.4% | 4 | 0 |
| 3 | 113 | 51 | 1.1% | 4 | 0 |
| 4 | 106 | 39 | 0.8% | 6 | 0 |
| 5 | 89 | 35 | 1.0% | 6 | 0 |
| 6 | 107 | 48 | 1.2% | 5 | 0 |
