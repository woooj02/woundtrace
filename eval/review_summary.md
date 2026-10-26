# Outline check on real wound photos

261 outlines on 200 FUSeg validation photos (learned segmenter and classical method). us.amazon.nova-2-lite-v1:0 is shown the plain photo and marks the wound; the outline passes if the boxes overlap (IoU ≥ 0.1). Truth: Dice against the experts' mask. The 40 outlines used to design the check are left out.

| Outlines | Count | Passed (agree) | Flagged for the nurse |
|---|---|---|---|
| Good (Dice ≥ 0.7) | 159 | 72% | 28% |
| Bad (Dice < 0.4) | 86 | 21% | 79% |

Errors: 0% · median 1036 ms per check · 377 input tokens per check.

First design (the model shown the outline overlay and asked whether it follows the wound): it said "follows" for 100% of good and 100% of bad outlines (eval/review_overlay_summary.md), so it was replaced.
