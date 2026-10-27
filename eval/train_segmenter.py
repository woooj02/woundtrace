"""Train the wound segmenter and export it to ONNX for OpenCV 5's DNN module.

    data/train-venv/bin/python -m eval.train_segmenter            # train, export app/models/wound_seg.onnx
    data/train-venv/bin/python -m eval.train_segmenter --check    # export an untrained model, test cv2.dnn

Training needs PyTorch (a separate venv, never deployed); the app only needs OpenCV.
Data: FUSeg 2021 train split (real, expert-annotated foot ulcers; 10% held back to pick the epoch) plus
synthetic photos from app/synth.py with exact masks, so the model also handles the printed and synthetic
test targets. The FUSeg validation split is never seen in training: eval/fuseg_eval.py reports on it.
Model: LR-ASPP with a MobileNetV3-Large backbone (ImageNet-pretrained backbone from torchvision).
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from app import synth

ROOT = Path(__file__).resolve().parent.parent
FUSEG = ROOT / "data" / "fuseg" / "repo" / "data" / "Foot Ulcer Segmentation Challenge"
SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
OUT = ROOT / "app" / "models" / "wound_seg.onnx"


def letterbox(img, size=SIZE, interp=cv2.INTER_AREA):
    """Fit the longer side to size and pad with black, as the app does at inference."""
    h, w = img.shape[:2]
    k = size / max(h, w)
    small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=interp)
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    out[:small.shape[0], :small.shape[1]] = small
    return out


def synthetic_set(n, seed=99):
    """Simulated phone photos of synthetic targets, with the wound mask as the camera sees it."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        img_plane, _, mask = synth.render_plane(rng, float(rng.uniform(1.5, 25)), int(rng.integers(0, 6)),
                                                return_mask=True)
        shot = synth.Shot(float(rng.uniform(0, 40)), float(rng.uniform(3.0, 9.0)),
                          float(rng.choice([0, 0, 0.8, 1.5])), bool(rng.random() < 0.15),
                          0, 0.0)
        photo, m = synth.photograph(rng, img_plane, 8.0, shot, mask=mask)
        out.append((letterbox(photo), letterbox(m, interp=cv2.INTER_NEAREST)))
    return out


CO2 = ROOT / "data" / "co2wounds" / "x" / "CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients"
HEALTHY = ROOT / "data" / "lowerlimb" / "x" / "Nomal"


def healthy_split():
    """Healthy feet: every other photo (sorted by name) for training, the rest held out for the false-alarm test."""
    names = sorted(os.listdir(HEALTHY))
    return names[::2], names[1::2]


def co2(split):
    """CO2Wounds-V2's own train or val split (val is never used for training)."""
    out = []
    for n in sorted(os.listdir(CO2 / "split" / split)):
        img = cv2.imread(str(CO2 / "imgs" / n))
        m = cv2.imread(str(CO2 / "masks" / (n.rsplit(".", 1)[0] + ".png")), cv2.IMREAD_GRAYSCALE)
        if img is not None and m is not None:
            out.append((letterbox(img), letterbox((m > 0).astype(np.uint8) * 255, interp=cv2.INTER_NEAREST)))
    return out


def healthy(names):
    return [(letterbox(img), np.zeros((SIZE, SIZE), np.uint8)) for img in (cv2.imread(str(HEALTHY / n)) for n in names)]


def fuseg(split, names=None):
    names = names or sorted(os.listdir(FUSEG / split / "images"))
    return [(cv2.imread(str(FUSEG / split / "images" / n)),
             cv2.imread(str(FUSEG / split / "labels" / n), cv2.IMREAD_GRAYSCALE)) for n in names]


class Aug(Dataset):
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, m = self.items[i]
        img, m = img.copy(), (m > 127).astype(np.uint8)
        if self.train:
            ang, sc = random.uniform(-180, 180), random.uniform(0.7, 1.4)
            A = cv2.getRotationMatrix2D((SIZE / 2 + random.uniform(-40, 40), SIZE / 2 + random.uniform(-40, 40)), ang, sc)
            img = cv2.warpAffine(img, A, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            m = cv2.warpAffine(m, A, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
            if random.random() < 0.5:
                img, m = img[:, ::-1], m[:, ::-1]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-6, 6)) % 180
            hsv[..., 1] *= random.uniform(0.7, 1.3)
            hsv[..., 2] *= random.uniform(0.6, 1.3)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
            if random.random() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 2.0))
        x = (img[..., ::-1].astype(np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), torch.from_numpy(m.astype(np.float32)[None].copy())


class Net(torch.nn.Module):
    """LR-ASPP MobileNetV3 with one output channel (wound logit), returning a plain tensor for ONNX."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.m = lraspp_mobilenet_v3_large(
            weights=None, num_classes=1,
            weights_backbone=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)

    def forward(self, x):
        return self.m(x)["out"]


def dice_loss(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


def export(model, path):
    model.eval().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, SIZE, SIZE), str(path), input_names=["image"],
                      output_names=["logit"], opset_version=17, dynamo=False)
    net = cv2.dnn.readNetFromONNX(str(path))
    net.setInput(np.zeros((1, 3, SIZE, SIZE), np.float32))
    out = net.forward()
    with torch.no_grad():
        ref = model(torch.zeros(1, 3, SIZE, SIZE)).numpy()
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB); cv2.dnn output {out.shape}, "
          f"max diff vs torch {np.abs(out - ref).max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--synthetic", type=int, default=500)
    ap.add_argument("--more-data", action="store_true",
                    help="also CO2Wounds-V2 train split and half the healthy-feet photos (as no-wound examples)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.check:
        export(Net(pretrained=False), ROOT / "data" / "check.onnx")
        return
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = sorted(os.listdir(FUSEG / "train" / "images"))
    random.shuffle(names)
    hold = names[:len(names) // 10]
    train_items = fuseg("train", names[len(hold):]) + synthetic_set(a.synthetic)
    hold_items = fuseg("train", hold)
    if a.more_data:
        c, h = co2("train"), healthy(healthy_split()[0])
        random.shuffle(c); random.shuffle(h)
        train_items += c[len(c) // 10:] + h[len(h) // 10:]
        hold_items += c[:len(c) // 10] + h[:len(h) // 10]
    print(f"train {len(train_items)} (FUSeg {len(names) - len(hold)} + synthetic {a.synthetic}"
          + (f" + CO2Wounds and healthy feet" if a.more_data else "") + f"), held back {len(hold_items)}, device {dev}")
    dl = DataLoader(Aug(train_items), batch_size=8, shuffle=True, num_workers=6, drop_last=True)
    hl = DataLoader(Aug(hold_items, train=False), batch_size=8, num_workers=4)
    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * len(dl))
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train(); t = time.time(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(3.0, device=dev)) \
                + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        model.eval(); dices = []
        with torch.no_grad():
            for x, y in hl:
                p = (torch.sigmoid(model(x.to(dev))) > 0.5).float().cpu()
                inter = (p * y).sum((1, 2, 3)); s = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
                dices += torch.where(s > 0, 2 * inter / s.clamp(min=1), torch.ones_like(s)).tolist()
        d = float(np.mean(dices))
        print(f"epoch {ep + 1:2d}  loss {tot / len(dl):.3f}  held-back Dice {d:.3f}  {time.time() - t:.0f}s", flush=True)
        if d > best:
            best, best_state = d, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"best held-back Dice {best:.3f}")
    export(model, a.out)


if __name__ == "__main__":
    main()
