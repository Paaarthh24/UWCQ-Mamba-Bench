"""Camera-to-surface distance from a monocular disparity map (used by the v4/v5 pilots).

quad_fit: robust quadratic surface fit to the disparity, excluding crack pixels (mask
dilated by 10 px), so crack-shaped depth steps cannot leak the label into the degradation.
distance_from_disparity: far/near ratio = fitted disparity's 99th/1st percentile, clipped
to [1.05, 3]; nearest point d_near; 1/d linear in disparity.
"""
import numpy as np
from scipy import ndimage


def _design(h, w):
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    return np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1).reshape(-1, 6)


def quad_fit_coef(disp, crack):
    """Coefficients of the robust quadratic fit (see module docstring)."""
    A = _design(*disp.shape)
    keep = ~ndimage.binary_dilation(crack, iterations=10).ravel()
    idx = np.flatnonzero(keep)[:: max(1, keep.sum() // 20000)]
    wts = np.ones(len(idx))
    for _ in range(3):                                   # iteratively reweighted (Huber-like)
        coef = np.linalg.lstsq(A[idx] * wts[:, None], disp.ravel()[idx] * wts, rcond=None)[0]
        r = disp.ravel()[idx] - A[idx] @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-9
        wts = np.clip(1.5 * s / (np.abs(r) + 1e-9), None, 1.0)
    return coef


def quad_eval(coef, h, w):
    return (_design(h, w) @ np.asarray(coef)).reshape(h, w)


def quad_fit(disp, crack):
    return quad_eval(quad_fit_coef(disp, crack), *disp.shape)


def distance_from_disparity(fit, d_near):
    lo, hi = np.percentile(fit, [1, 99])
    lo = max(lo, 1e-3)
    ratio = float(np.clip(hi / lo, 1.05, 3.0))
    dn = np.clip((fit - lo) / (hi - lo + 1e-9), 0, 1)
    inv = 1.0 / (d_near * ratio) + dn * (1.0 / d_near - 1.0 / (d_near * ratio))
    return 1.0 / inv, ratio
