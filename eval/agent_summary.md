# Agent evaluation: Bedrock agent vs the clinic's rules

48 visits (8 scenarios × 6 variants, six skin tones), synthetic photos. Model: `us.amazon.nova-2-lite-v1:0`. Each visit also ran through the deterministic rules engine from the same state; that result is the expected one.

| Result | Value |
|---|---|
| Same outcome as the rules (record / retake / nurse request) | 100% |
| Visits where a guard refused a tool call | 0 (0 still ended correctly) |
| Patient messages whose numbers all come from tool outputs | 96% |
| Median tool calls per visit | 4.0 |
| Latency per visit, median / p95 | 4702 / 6888 ms |
| Tokens per visit, input / output | 11635 / 259 |

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
