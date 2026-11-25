"""Mask binarisation and cleaning."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import disk



def binarize(arr: np.ndarray, rule: dict) -> np.ndarray:
    """Binarise a raw label with a rule such as {"kind": "gt", "thr": 127}."""
    a = arr if arr.ndim == 2 else arr[..., 0]
    k = rule.get("kind", "nonzero")
    if k == "gt":
        return a > rule["thr"]
    if k == "eq":
        return a == rule["value"]
    if k == "nonzero":
        return a != 0
    raise ValueError(f"unknown binarisation rule {rule}")


def clean(mask: np.ndarray, open_close_px: int = 1, gap_close_px: int = 0,
          min_area_px: int = 0, max_hole_px: int = 0) -> tuple[np.ndarray, dict]:
    """Light cleaning with a log of pixel counts changed per operation."""
    log = {}
    m = mask.astype(bool)
    n0 = int(m.sum())
    if open_close_px > 0:
        se = disk(open_close_px)
        m2 = ndimage.binary_closing(ndimage.binary_opening(m, se), se)
        log["open_close_changed_px"] = int((m2 != m).sum()); m = m2
    if gap_close_px > 0:
        m2 = ndimage.binary_closing(m, disk(gap_close_px))
        log["gap_close_added_px"] = int((m2 & ~m).sum()); m = m2
    if max_hole_px > 0:
        holes = ndimage.binary_fill_holes(m) & ~m
        lab, nh = ndimage.label(holes)
        sizes = ndimage.sum(holes, lab, range(1, nh + 1))
        small = np.isin(lab, np.nonzero(sizes <= max_hole_px)[0] + 1)
        m = m | small
        log["holes_filled"] = int((sizes <= max_hole_px).sum()); log["holes_flagged"] = int((sizes > max_hole_px).sum())
    if min_area_px > 0:
        lab, nc = ndimage.label(m, structure=np.ones((3, 3)))
        sizes = ndimage.sum(m, lab, range(1, nc + 1))
        drop = np.isin(lab, np.nonzero(sizes < min_area_px)[0] + 1)
        log["tiny_removed_components"] = int((sizes < min_area_px).sum())
        m = m & ~drop
    log["pixels_before"], log["pixels_after"] = n0, int(m.sum())
    return m, log
