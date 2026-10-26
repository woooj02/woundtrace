# Agent evaluation: Bedrock agent vs the clinic's rules

60 visits (10 scenarios × 6 variants, six skin tones), synthetic photos. Model: `us.amazon.nova-2-lite-v1:0`. Each visit also ran through the deterministic rules engine from the same state; that result is the expected one.

| Result | Value |
|---|---|
| Same outcome as the rules (record / retake / nurse request) | 100% |
| Visits where a guard refused a tool call | 0 (0 still ended correctly) |
| Patient messages whose numbers all come from tool outputs | 100% |
| Model replies replaced by the message guard (leaked internals, opinion, length) | 1 |
| Median tool calls per visit | 3.0 |
| Latency per visit, median / p95 | 4781 / 7736 ms |
| Tokens per visit, input / output | 16169 / 236 |

Calibration (card printed at 92.5%): measured scale 0.921, recorded area error 0.8% mean abs (uncorrected it would be about +17%).

| Scenario | Agreement |
|---|---|
| baseline | 6/6 |
| on_track | 6/6 |
| stalled | 6/6 |
| worsening | 6/6 |
| tilted | 6/6 |
| too_far | 6/6 |
| blurry | 6/6 |
| glare | 6/6 |
| uncalibrated | 6/6 |
| calibrate | 6/6 |
