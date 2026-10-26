# Outline review on real wound photos

301 outlines on 200 FUSeg validation photos (learned segmenter and classical method), each shown to us.amazon.nova-2-lite-v1:0 as photo + outline. Truth: Dice against the experts' mask.

| Outlines | Count | Model said follows | Model said misses |
|---|---|---|---|
| Good (Dice ≥ 0.7) | 180 | 100% | 0% |
| Bad (Dice < 0.4) | 106 | 100% | 0% |

Unsure: 0% · errors: 0% · median 2139 ms per review · 432 input tokens per review.
