# Real wound photos: outline accuracy on FUSeg

200 de-identified clinical foot-ulcer photos (validation split of FUSeg 2021, Wang et al.), compared with the wound-care experts' masks. No reference card in these photos: outline only.

| Method | Mean Dice | Mean IoU | Dice ≥ 0.7 | Missed (no outline) | Area ratio, median |
|---|---|---|---|---|---|
| colour | 0.031 | 0.024 | 2% | 97 | 1.56 |
| grabcut | 0.033 | 0.026 | 2% | 97 | 1.83 |
