"""Depth-free underwater image synthesis following UW-GAN (Hambarde et al.,
IEEE TIM 2021, Sec. III-C, eq. 17 and Table I).

    I_uw(x) = Avg_Omega[ I(y) ] * N^D * N^d(x) + B * (1 - N^d(x))

N (normalised residual energy ratio, per RGB channel), B (background light)
and D (vertical depth) come from Table I, one row per water type.  UW-GAN takes
the horizontal distance d(x) from NYU depth maps; our crack photos are
close-range shots of a near-planar surface, so d(x) is a tilted plane
d0 * (1 + tilt * ramp(x)) instead of an estimated depth map.  Avg_Omega is a
self-guided filter (He et al.); N^d is treated as locally constant, so only
the scene radiance is filtered, as stated in the paper.  The geometry is never
warped, so the source crack mask stays valid unchanged.
"""
import numpy as np

# Table I of UW-GAN: N_rer (R, G, B), background B (R, G, B), vertical depth D (m)
UWGAN_TYPES = {
    1: dict(N=(0.62, 0.61, 0.50), B=(0.41, 0.40, 0.23), D=0.1),
    2: dict(N=(0.80, 0.96, 0.98), B=(0.10, 0.90, 0.10), D=0.1),
    3: dict(N=(0.01, 0.89, 0.49), B=(0.28, 0.73, 0.61), D=0.1),
    4: dict(N=(0.75, 0.88, 0.87), B=(0.35, 0.70, 0.60), D=0.5),
    5: dict(N=(0.67, 0.73, 0.67), B=(0.47, 0.58, 0.44), D=0.1),
    6: dict(N=(0.12, 0.89, 0.29), B=(0.10, 0.82, 0.23), D=0.5),
    7: dict(N=(0.01, 0.19, 0.89), B=(0.02, 0.08, 0.89), D=10.0),
}


def _box(a, r):
    """Mean over a (2r+1)^2 window with edge replication, via integral image."""
    p = np.pad(a, ((r + 1, r), (r + 1, r)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return s / (k * k)


def guided_filter_self(img, r=2, eps=1.2e-2):
    """Per-channel self-guided filter (He et al., TPAMI 2013) on a [0,1] image."""
    mean_i = _box(img, r)
    var_i = _box(img * img, r) - mean_i ** 2
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    return _box(a, r) * img + _box(b, r)


def distance_plane(h, w, d0, tilt, angle):
    """Depth-free distance map: a plane at d0 metres, tilted by +/- tilt/2 along
    direction `angle` (radians)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u = (np.cos(angle) * (xx / max(w - 1, 1) - 0.5)
         + np.sin(angle) * (yy / max(h - 1, 1) - 0.5))
    u = u / (np.abs(u).max() + 1e-12) * 0.5          # in [-0.5, 0.5]
    return d0 * (1.0 + tilt * u)


def synthesize(img, water_type, d0, tilt=0.0, angle=0.0, gf_radius=2, gf_eps=1.2e-2):
    """img: HxWx3 float in [0,1] (RGB).  Returns (I_uw, transmission HxWx3)."""
    p = UWGAN_TYPES[water_type]
    N = np.asarray(p["N"], dtype=np.float64)
    B = np.asarray(p["B"], dtype=np.float64)
    d = distance_plane(img.shape[0], img.shape[1], d0, tilt, angle)[..., None]
    t = N ** d                                        # N^d(x), per channel
    J = guided_filter_self(img.astype(np.float64), gf_radius, gf_eps)
    out = J * (N ** p["D"]) * t + B * (1.0 - t)
    return np.clip(out, 0.0, 1.0), t
