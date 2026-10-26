# Real wound photos: outline accuracy on FUSeg

200 de-identified clinical foot-ulcer photos (validation split of FUSeg 2021, Wang et al.), compared with the wound-care experts' masks. No reference card in these photos: outline only.

| Method | Mean Dice | Mean IoU | Dice ≥ 0.7 | Missed (no outline) | Area ratio, median |
|---|---|---|---|---|---|
| colour | 0.031 | 0.024 | 2% | 97 | 1.56 |
| grabcut | 0.033 | 0.026 | 2% | 97 | 1.83 |
| dnn-raw | 0.824 | 0.740 | 86% | 1 | 1.04 |
| dnn | 0.843 | 0.765 | 88% | 2 | 1.06 |

colour, grabcut: the classical pipeline (Lab colour distance from the skin, then GrabCut), tuned on synthetic targets. dnn-raw: the learned segmenter alone (LR-ASPP MobileNetV3, ONNX, run by cv2.dnn). dnn: the app's default, the same model with its outline snapped to the colour edge by GrabCut. The model was trained on the FUSeg train split plus synthetic photos (eval/train_segmenter.py); these validation photos were not used for training or for choosing the model.
