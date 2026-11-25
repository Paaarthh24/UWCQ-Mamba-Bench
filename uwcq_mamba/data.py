"""Underwater crack renders: split listing, training crops with augmentation, and the dense geometry targets."""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

from .geometry.skeleton import edt_width, prune_spurs

JERLOV = ["I", "IA", "IB", "II", "III", "1C", "3C", "5C", "7C", "9C"]
RENDER = re.compile(r"_(I|IA|IB|II|III|1C|3C|5C|7C|9C)_(shallow|medium|deep)$")
READ = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def source_of(path):
    return RENDER.sub("", Path(path).stem)


def split_items(root, split_file, key):
    """(image, mask) pairs of every render whose source photo is listed under `key` ("train", "val" or "test") in the
    dataset's split file; keys there are "<folder>/<source stem>"."""
    root = Path(root)
    keys = set(json.loads((root / split_file).read_text())[key])
    out = []
    for folder in ("train", "val", "test"):
        if (root / f"{folder}_img").is_dir():
            out += [(q, root / f"{folder}_lab" / (q.stem + ".png")) for q in sorted((root / f"{folder}_img").glob("*.jpg"))
                    if f"{folder}/{source_of(q)}" in keys]
    return out


def read_pair(img_p, mask_p):
    img = cv2.cvtColor(cv2.imread(str(img_p), READ), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if m.shape != img.shape[:2]:
        raise ValueError(f"{img_p}: mask {m.shape} != image {img.shape[:2]}")
    return img, (m > 127).astype(np.uint8)


def make_targets(m, sigma=0.7, win=7, sdf_clip=20.0, spur_px=12.0):
    """Mask (0 / 1, 255 = ignore) -> skeleton, centreline heat map, log width, orientation (cos 2t, sin 2t), coherence and
    signed distance (+ outside, - inside, clipped at 20 px)."""
    crack = m == 1
    H, W = m.shape
    out = dict(skel=np.zeros((H, W), np.uint8), cl=np.zeros((H, W), np.float32), logw=np.zeros((H, W), np.float32),
               ori=np.zeros((2, H, W), np.float32), coh=np.zeros((H, W), np.float32))
    if crack.any():
        d_in, d_out = ndimage.distance_transform_edt(crack), ndimage.distance_transform_edt(~crack)
        out["sdf"] = np.clip(d_out - d_in, -sdf_clip, sdf_clip).astype(np.float32)
    else:
        out["sdf"] = np.full((H, W), sdf_clip, np.float32)
    if crack.sum() < 3:
        return out
    try:
        skel = prune_spurs(crack, spur_px, width_px=edt_width(crack))
    except ValueError:
        skel = skeletonize(crack)
    if not skel.any():
        return out
    d, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    out["skel"] = skel.astype(np.uint8)
    out["cl"] = np.exp(-(d ** 2) / (2 * sigma ** 2)).astype(np.float32)
    f = cv2.GaussianBlur(crack.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 3.0) for a in (gx * gx, gy * gy, gx * gy))
    c2, s2 = jxx - jyy, 2 * jxy
    mag = np.sqrt(c2 ** 2 + s2 ** 2)
    nz = mag > 1e-8
    cos2, sin2 = np.where(nz, -c2 / np.maximum(mag, 1e-8), 0.0), np.where(nz, -s2 / np.maximum(mag, 1e-8), 0.0)
    out["ori"] = np.stack([cos2 * crack, sin2 * crack]).astype(np.float32)
    out["coh"] = np.where(crack, mag / np.maximum(jxx + jyy, 1e-8), 0.0).clip(0, 1).astype(np.float32)
    mass = np.zeros((H, W), np.float32)
    np.add.at(mass, (iy[crack], ix[crack]), 1.0)
    ac, as_ = np.sqrt(np.clip((1 + cos2) / 2, 0, 1)), np.sqrt(np.clip((1 - cos2) / 2, 0, 1))
    step = np.where(skel, 1.0 / np.maximum(np.maximum(ac, as_), 1 / np.sqrt(2)), 0.0).astype(np.float32)
    k = np.ones((win, win), np.float32)
    num = cv2.filter2D(mass, -1, k, borderType=cv2.BORDER_CONSTANT)
    den = cv2.filter2D(step, -1, k, borderType=cv2.BORDER_CONSTANT)
    w_sk = np.where(skel, num / np.maximum(den, 1e-6), 0.0)
    out["logw"] = np.where(crack, np.log(np.maximum(w_sk[iy, ix], 1.0)), 0.0).astype(np.float32)
    return out


class CropDataset(Dataset):
    """Native-scale random crops: 70 % centred near a crack pixel, random rotation (+-30 deg, half the samples; fill =
    ignore), flips and 90-degree turns. Images in [0, 1]; mask 0 / 1 / 255 (ignore) plus the dense geometry targets."""

    def __init__(self, items, size=352, rot_deg=30.0, crack_centred_p=0.7):
        self.items, self.size, self.rot, self.ccp = items, size, rot_deg, crack_centred_p

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rng = np.random.default_rng([torch.initial_seed() % (2 ** 63), i, int(torch.randint(0, 2 ** 31, ()).item())])
        img, m = read_pair(*self.items[i][:2])
        s = self.size
        if rng.random() < 0.5 and self.rot:
            ang = float(rng.uniform(-self.rot, self.rot))
            H, W = m.shape
            M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=255)
        H, W = m.shape
        if H < s or W < s:
            ph, pw = max(0, s - H), max(0, s - W)
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
            m = np.pad(m, ((0, ph), (0, pw)), constant_values=255)
            H, W = m.shape
        ys, xs = np.nonzero(m == 1)
        if len(ys) and rng.random() < self.ccp:
            j = rng.integers(len(ys))
            y0 = int(np.clip(ys[j] - s // 2 + rng.integers(-s // 4, s // 4 + 1), 0, H - s))
            x0 = int(np.clip(xs[j] - s // 2 + rng.integers(-s // 4, s // 4 + 1), 0, W - s))
        else:
            y0, x0 = int(rng.integers(0, H - s + 1)), int(rng.integers(0, W - s + 1))
        img, m = img[y0:y0 + s, x0:x0 + s], m[y0:y0 + s, x0:x0 + s]
        if rng.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        if rng.random() < 0.5:
            img, m = img[::-1], m[::-1]
        k = int(rng.integers(4))
        img, m = np.rot90(img, k), np.rot90(m, k)
        m = np.ascontiguousarray(m)
        out = dict(image=torch.from_numpy(np.ascontiguousarray(img).astype(np.float32).transpose(2, 0, 1) / 255.0),
                   mask=torch.from_numpy(m.astype(np.int64)))
        out.update({k: torch.from_numpy(v) for k, v in make_targets(m).items()})
        return out
