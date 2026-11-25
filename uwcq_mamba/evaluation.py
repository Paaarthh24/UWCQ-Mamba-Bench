"""Inference, segmentation metrics (the MixerCSeg evaluation protocol) and crack-measurement errors."""
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import RENDER, source_of
from .geometry.measure import measure_mask

QUANTITIES = ["length_main_mm", "length_total_mm", "width_min_mm", "width_mean_mm", "width_max_mm", "area_mm2"]
NAMES = ["L_m", "L_t", "W_min", "W_avg", "W_max", "A"]
T = np.arange(0.0, 1.0, 0.01)


def read_image(path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)[None]


@torch.no_grad()
def predict_prob(model, x):
    """Crack probability (H, W) of one image tensor (1, 3, H, W) in [0, 1]."""
    dev = next(model.parameters()).device
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        lo = model(x.to(dev))
    return torch.softmax(lo.float(), 1)[0, 1].cpu().numpy()


def predict_mask(model, x):
    return predict_prob(model, x) >= 0.5


def hist(prob, gt):
    """Per-image counts of the saved uint8 value 255 p / max(p) over crack and background pixels (author protocol)."""
    m = prob.max()
    v = np.clip(np.rint(255.0 * prob / m), 0, 255).astype(np.uint8) if m > 0 else np.zeros(prob.shape, np.uint8)
    return np.stack([np.bincount(v[gt], minlength=256), np.bincount(v[~gt], minlength=256)])


def author_metrics(H):
    """mIoU and ODS: best threshold over the dataset; OIS: mean of each image's best F1; F1: pooled at threshold 0."""
    H = np.asarray(H, np.float64)
    above = np.cumsum(H[..., ::-1], -1)[..., ::-1]
    k = np.minimum(np.floor(255.0 * T).astype(int) + 1, 255)
    tp, fp = above[:, 0, k], above[:, 1, k]
    fn, tn = H[:, 0].sum(-1, keepdims=True) - tp, H[:, 1].sum(-1, keepdims=True) - fp
    P = np.where(tp + fp == 0, 1.0, tp / np.maximum(tp + fp, 1))
    R = np.where(tp + fn == 0, 0.0, tp / np.maximum(tp + fn, 1))
    F = np.where(P + R == 0, 0.0, 2 * P * R / np.maximum(P + R, 1e-12))
    iou = np.where(tp + fp + fn <= 0, 0.0, (tp / np.maximum(tp + fp + fn, 1) + tn / np.maximum(fn + fp + tn, 1)) / 2)
    TP, FP, FN = tp[:, 0].sum(), fp[:, 0].sum(), fn[:, 0].sum()
    Pp = 1.0 if TP == 0 and FP == 0 else TP / (TP + FP)
    Rp = TP / max(TP + FN, 1)
    return dict(mIoU=float(iou.mean(0).max()), ODS=float(F.mean(0).max()), OIS=float(F.max(1).mean()),
                F1=float(2 * Pp * Rp / max(Pp + Rp, 1e-12)))


def pooled_miou(counts):
    """Mean of crack and background IoU from pooled (tp, fp, fn, n_pixels) counts."""
    tp, fp, fn, n = np.sum(counts, 0)
    tn = n - tp - fp - fn
    return (tp / max(tp + fp + fn, 1) + tn / max(tn + fp + fn, 1)) / 2


def measure(mask, cfg):
    """Six quantities (mm) of a binary mask with the benchmark procedure."""
    try:
        r, _ = measure_mask(mask, cfg, binary=True, components=False, stray_filter=False, clean_mask=False)
        return [float(r.get(q, np.nan)) for q in QUANTITIES]
    except Exception:
        return [np.nan] * 6


def reference(dataset_root):
    """image id ("<folder>_img/<file>") -> the six reference quantities from measurement_gt.csv."""
    f = lambda x: float(x) if x not in (None, "") else np.nan
    return {r["image_id"]: [f(r[q + "_cons"]) for q in QUANTITIES]
            for r in csv.DictReader(open(Path(dataset_root) / "measurement_gt.csv"))}


def calibration_subset(items, k, cap=400):
    """k renders per training source photo (at most `cap` sources), chosen by a fixed hash: the train images on which each
    model's per-quantity measurement scale is fitted."""
    g = {}
    for it in items:
        g.setdefault(source_of(it[0]), []).append(it)
    srcs = sorted(g) if len(g) <= cap else sorted(g, key=lambda s: hashlib.md5(s.encode()).hexdigest())[:cap]
    return [it for s in srcs for it in sorted(g[s], key=lambda it: hashlib.md5(it[0].name.encode()).hexdigest())[:k]]


def water_type(path):
    m = RENDER.search(Path(path).stem)
    return m.group(1) if m else "air"
