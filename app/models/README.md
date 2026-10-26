# wound_seg.onnx

The learned wound segmenter, run by OpenCV 5's DNN module (`cv2.dnn.readNetFromONNX`, see `app/vision.py`).

| | |
|---|---|
| Architecture | LR-ASPP, MobileNetV3-Large backbone (ImageNet-pretrained backbone from torchvision), 1 output channel |
| Input | 512 × 512 RGB, letterboxed, ImageNet mean/std; output: one logit per pixel |
| Training | `eval/train_segmenter.py`: 40 epochs, AdamW + one-cycle, BCE + Dice, on a GTX 1660 Ti |
| Data | FUSeg 2021 train split (810 photos, 10% held back to pick the epoch) + 500 synthetic photos from `app/synth.py` |
| Held-back Dice (train split) | 0.852 |
| FUSeg validation (200 photos, never used in training) | Dice 0.824 alone, 0.843 with the GrabCut snap: `eval/fuseg_summary.md` |
| Size | 12.9 MB |

## Training data

Wang C. et al., *FUSeg: The Foot Ulcer Segmentation Challenge*, 2021 (MICCAI 2021 challenge), and
Wang C. et al., *Fully automatic wound segmentation with deep convolutional neural networks*, Scientific
Reports 10, 21897 (2020). Photos from the AZH Wound and Vascular Center, Milwaukee, shared publicly by the
authors with the clinic's permission (github.com/uwm-bigdata/wound-segmentation). The dataset itself is not
in this repository.

## Limits

Trained on foot ulcers photographed in a clinic, plus synthetic targets. Not tested on other wound types,
on patients' own phone photos, or across skin tones on real skin (FUSeg does not report skin tone). Research
prototype, not a medical device.
