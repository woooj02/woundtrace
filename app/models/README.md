# wound_seg.onnx

The learned wound segmenter, run by OpenCV 5's DNN module (`cv2.dnn.readNetFromONNX`, see `app/vision.py`).

| | |
|---|---|
| Architecture | LR-ASPP, MobileNetV3-Large backbone (ImageNet-pretrained backbone from torchvision), 1 output channel |
| Input | 512 × 512 RGB, letterboxed, ImageNet mean/std; output: one logit per pixel |
| Training | `eval/train_segmenter.py --more-data`: 30 epochs, AdamW + one-cycle, BCE + Dice, on a GTX 1660 Ti |
| Data | FUSeg 2021 train split (729), CO2Wounds-V2 train split (437), 1,242 healthy-feet photos as no-wound examples, 500 synthetic photos from `app/synth.py`; 10% of each real set held back to pick the epoch |
| Held-back Dice | 0.905 |
| Size | 12.9 MB |

## Results on photos never used in training (`eval/fuseg_summary.md`, `eval/external_summary.md`)

| Test set | Photos | Mean Dice | Note |
|---|---|---|---|
| FUSeg 2021 validation (US clinic, foot ulcers) | 200 | 0.82 (with the GrabCut snap) | the previous model, FUSeg-only: 0.84 |
| CO2Wounds-V2 validation (Colombia, leprosy wounds, smartphones) | 122 | 0.77 | previous model: 0.48 |
| Lower-limb set (another clinic, never trained on) | 2,686 | 0.39 | previous: 0.33; its masks include surrounding skin |
| Healthy feet, held-out half | 1,378 | 0 given an outline | previous: 23% |

## Training data and licences

- Wang C. et al., *FUSeg: The Foot Ulcer Segmentation Challenge* (MICCAI 2021) and *Fully automatic wound
  segmentation with deep convolutional neural networks*, Scientific Reports 10, 21897 (2020). Photos from the
  AZH Wound and Vascular Center, shared publicly by the authors with the clinic's permission
  (github.com/uwm-bigdata/wound-segmentation).
- Sánchez-Jiménez D. et al., *CO2Wounds-V2: Extended Chronic Wounds Dataset From Leprosy Patients* (2024),
  Mendeley Data, doi 10.17632/s2w7rjwz49.2, **CC BY-NC 3.0**.
- Islam M. M. et al., *Lower Limb and Feet Wound Image Dataset for Medical Analysis* (2025), Mendeley Data,
  doi 10.17632/hsj38fwnvr.3, CC BY 4.0 (healthy-feet photos used for training; wound photos test-only).

None of the datasets is in this repository. Because part of the training data is CC BY-NC, **these weights
are for non-commercial research and demonstration only.**

## Limits

Trained on clinic photos of foot and leg wounds, plus synthetic targets. On a clinic it never saw it misses
16% of wounds and outlines less than that dataset marks. Not tested on patients' own phone photos. Skin
tone in the tests is estimated from the photos, not recorded. Research prototype, not a medical device.
