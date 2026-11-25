"""Skeleton helpers: neighbour counts, end points, piece filtering and arc-length steps."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)


def _nbrs(skel):
    return cv2.filter2D(skel.astype(np.float32), -1, _K8, borderType=cv2.BORDER_CONSTANT)


def endpoints(skel):
    return skel & (_nbrs(skel) == 1)


def prune(skel, n=12):
    """Remove spurs shorter than about n px; closed loops and long branches survive."""
    if n <= 0 or not skel.any():
        return skel
    x1 = skel.copy()
    for _ in range(n):
        e = endpoints(x1)
        if not e.any():
            break
        x1 &= ~e
    grow = endpoints(x1)
    for _ in range(n):
        grow = cv2.dilate(grow.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & skel & ~x1
        if not grow.any():
            break
        x1 |= grow
        grow = grow & endpoints(x1)
    return x1


def skeleton_steps(skel):
    """Arc length per skeleton pixel, 1 / max(|cos t|, |sin t|), with the local skeleton direction t."""
    f = cv2.GaussianBlur(skel.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 2.0) for a in (gx * gx, gy * gy, gx * gy))
    out = np.zeros(skel.shape, np.float32)
    ys, xs = np.nonzero(skel)
    phi = 0.5 * np.arctan2(2 * jxy[ys, xs], jxx[ys, xs] - jyy[ys, xs])
    out[ys, xs] = 1.0 / np.maximum(np.maximum(np.abs(np.cos(phi)), np.abs(np.sin(phi))), 1 / np.sqrt(2))
    return out


_DISK1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


_NB = ((0, 1), (1, -1), (1, 0), (1, 1))


def keep_pieces(m, skel, step, min_len=0.0, min_elong=0.0):
    """Keep the longest skeleton piece and every sufficiently long and elongated piece."""
    lab, n = ndimage.label(skel, np.ones((3, 3), int))
    if n <= 1:
        return m, skel, step
    idx = range(1, n + 1)
    L = np.asarray(ndimage.sum(step, lab, idx), float)
    keep = L >= min_len
    if min_elong > 0:
        _, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
        A = np.asarray(ndimage.sum(m, lab[iy, ix] * m, idx), float)
        keep &= L * L / np.maximum(A, 1.0) >= min_elong
    keep[np.argmax(L)] = True
    if keep.all():
        return m, skel, step
    sk = np.isin(lab, np.nonzero(keep)[0] + 1)
    mlab, _ = ndimage.label(m, np.ones((3, 3), int))
    mk = np.isin(mlab, np.unique(mlab[sk]))
    return m & mk, sk, step * sk
