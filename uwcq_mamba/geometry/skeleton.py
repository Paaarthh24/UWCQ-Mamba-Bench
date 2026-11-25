"""Skeleton graph utilities shared by the width and length measurements."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

try:
    from skan import Skeleton
except ImportError as e:
    raise ImportError("skan is required for crack geometry (pip install skan)") from e


@dataclass
class Branch:
    rc: np.ndarray
    ends_degree: tuple[int, int]
    ds: np.ndarray = field(default=None)
    s: np.ndarray = field(default=None)

    @property
    def length(self) -> float:
        return float(self.ds.sum())


KULPA_AXIAL, KULPA_DIAG = 0.948, 1.343


def _arc_weights(rc: np.ndarray, smoothing: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel arc-length weights ds (sum = branch length) and cumulative s."""
    n = len(rc)
    if n == 1:
        return np.zeros(1), np.zeros(1)
    if smoothing == "savgol" and n >= 7:
        w = min(n if n % 2 else n - 1, 9)
        p = savgol_filter(rc.astype(float), w, 2, axis=0, mode="interp")
        p[0], p[-1] = rc[0], rc[-1]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    else:
        d = np.abs(np.diff(rc, axis=0)).sum(1)
        step = np.where(d == 2, KULPA_DIAG, KULPA_AXIAL) if smoothing == "kulpa" else np.where(d == 2, np.sqrt(2), 1.0)
    ds = np.zeros(n)
    ds[:-1] += step / 2
    ds[1:] += step / 2
    s = np.concatenate([[0.0], np.cumsum(step)])
    return ds, s


def prune_spurs(binary: np.ndarray, min_spur_px: float, width_px: np.ndarray | None = None,
                width_factor: float = 2.0, max_rounds: int = 3, keep_longest: bool = False) -> np.ndarray:
    """Remove tip-to-junction branches shorter than max(width_factor x local width, min_spur_px)."""
    skel = skeletonize(binary.astype(bool))
    for _ in range(max_rounds):
        if skel.sum() < 2:
            break
        sk = Skeleton(skel)
        removed = False
        cand = []
        n_at = {}
        for i in range(sk.n_paths):
            idx = sk.path(i)
            for e in (idx[0], idx[-1]):
                n_at[tuple(sk.coordinates[e].astype(int))] = n_at.get(tuple(sk.coordinates[e].astype(int)), 0) + 1
        for i in range(sk.n_paths):
            idx = sk.path(i)
            d0, d1 = sk.degrees[idx[0]], sk.degrees[idx[-1]]
            if not ((d0 == 1 and d1 >= 3) or (d1 == 1 and d0 >= 3)):
                continue
            rc = sk.coordinates[idx].astype(int)
            junction = rc[0] if d0 >= 3 else rc[-1]
            lw = width_px[junction[0], junction[1]] if width_px is not None else 0.0
            ds, _ = _arc_weights(rc, "plain")
            if ds.sum() < max(width_factor * lw, min_spur_px):
                tip_side = rc[1:] if d0 >= 3 else rc[:-1]
                cand.append((tuple(junction), ds.sum(), tip_side))
        if keep_longest:
            by_j = {}
            for c in cand:
                by_j.setdefault(c[0], []).append(c)
            cand = []
            for j, cs in by_j.items():
                cs.sort(key=lambda c: c[1])
                cand += cs[:-1] if n_at.get(j, 0) - len(cs) < 2 else cs
        for _, _, tip_side in cand:
            skel[tip_side[:, 0], tip_side[:, 1]] = False
            removed = True
        if not removed:
            break
        skel = skeletonize(skel)
    return skel


def scaled_length(rc: np.ndarray, s_row: float, s_col: float, smoothing: str = "savgol") -> float:
    """Length of an ordered pixel chain with different pixel sizes along rows and columns (mm per px)."""
    n = len(rc)
    if n < 2:
        return 0.0
    if smoothing == "savgol" and n >= 7:
        w = min(n if n % 2 else n - 1, 9)
        p = savgol_filter(rc.astype(float), w, 2, axis=0, mode="interp")
        p[0], p[-1] = rc[0], rc[-1]
    else:
        p = rc.astype(float)
    d = np.diff(p, axis=0) * np.array([s_row, s_col])
    return float(np.linalg.norm(d, axis=1).sum())


def branches_from_skeleton(skel: np.ndarray, smoothing: str = "savgol") -> list[Branch]:
    if skel.sum() < 2:
        return []
    sk = Skeleton(skel)
    out = []
    for i in range(sk.n_paths):
        idx = sk.path(i)
        rc = sk.coordinates[idx].astype(int)
        br = Branch(rc=rc, ends_degree=(int(sk.degrees[idx[0]]), int(sk.degrees[idx[-1]])))
        br.ds, br.s = _arc_weights(rc, smoothing)
        out.append(br)
    return out


def edt_width(binary: np.ndarray) -> np.ndarray:
    """Inscribed-circle width 2 x EDT."""
    return 2.0 * ndimage.distance_transform_edt(binary)
