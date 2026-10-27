# Real wound photos from two other datasets (never used in training)

The app's segmenter (cv2.dnn + GrabCut snap) against each dataset's own masks. Outline only: these photos have no WoundTrace card.

| Dataset | Photos | Mean Dice | Median Dice | Dice ≥ 0.7 | Missed (no outline) | Precision | Recall |
|---|---|---|---|---|---|---|---|
| co2wounds | 122 | 0.767 | 0.857 | 79% | 2 | 0.82 | 0.79 |
| lowerlimb | 2686 | 0.389 | 0.361 | 22% | 428 | 0.83 | 0.37 |

Healthy feet (no wound): an outline was drawn on 0 of 1378 photos (0.0%).

## By estimated skin tone (both wound datasets)

| Skin tone (ITA°) | Photos | Mean Dice | Dice ≥ 0.7 | Missed |
|---|---|---|---|---|
| very light | 340 | 0.410 | 26% | 40 |
| light | 221 | 0.460 | 28% | 15 |
| intermediate | 250 | 0.412 | 24% | 23 |
| tan | 301 | 0.387 | 23% | 48 |
| brown | 718 | 0.391 | 23% | 126 |
| dark | 978 | 0.407 | 26% | 178 |

Skin tone is estimated from the skin around each wound (ITA°, Del Bino bins) in uncalibrated clinic light, so a photo can land one bin off. The two datasets mark wounds differently from FUSeg (the model's training data), so part of the lower Dice is a difference in where annotators draw the edge.

Sources: CO2Wounds-V2 (Mendeley Data 10.17632/s2w7rjwz49.2, CC BY-NC 3.0); Lower Limb and Feet Wound Image Dataset (Mendeley Data 10.17632/hsj38fwnvr.3, CC BY 4.0).

Precision: the share of the outlined pixels that are inside the dataset's mask. Recall: the share of the mask that was outlined. High precision with low recall means the outline sits on the wound but covers less than the annotators marked.

Model: wound_seg.onnx.
