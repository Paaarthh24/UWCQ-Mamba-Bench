"""Underwater synthesis v5: RSUIGM image formation (Desai et al., ACM TOMM 2024, eq. 19)
with Jerlov water types, plus the lighting and camera effects of v3/v4.

    I = J * E0 * exp(-(Kd * d + c * z))  +  V_inf * (1 - exp(-c * z)),   V_inf = b * E(d) / c
    E(d) = E0 * exp(-Kd * d)

d: depth below the surface (category), z: camera-to-surface distance (estimated depth or plane).
Kd, a, b per RGB channel from Solonenko & Mobley (2015) (jerlov_rgb.json); c = a + b.
Additions kept from v3/v4 (not part of RSUIGM): patchy turbidity, caustics, an ROV lamp
(round trip exp(-2 c z), glow in the beam), forward-scatter blur driven by the scattering
optical depth b*z, and a camera pipeline (vignetting, raw noise, capped partial white
balance, auto-exposure, axial chromatic aberration, JPEG).  No step moves pixels.
"""
import io, json, os
import numpy as np
from PIL import Image
from scipy import ndimage

from uwcq_bench.uw_synthesis import distance_plane, guided_filter_self
from uwcq_bench.uw_synthesis_v2 import fbm, caustics, lamp_pattern, blur_rgb
from uwcq_bench.uw_synthesis_v3 import DEPTH_CATEGORIES, LAMP_RGB, sample_effects  # noqa: F401

JERLOV = json.load(open(os.path.join(os.path.dirname(__file__), "jerlov_rgb.json")))["types"]
JERLOV_TYPES = list(JERLOV)


SMOOTH_MAX = 800   # smooth fields (haze patches, caustics, lamp, beam glow) are computed at <= this size


def _smooth(h, w, make):
    """make(hh, ww) -> HxW[xC] smooth field; computed at reduced size for large images and
    bilinearly upsampled (identical to direct computation when max(h, w) <= SMOOTH_MAX)."""
    if max(h, w) <= SMOOTH_MAX:
        return make(h, w)
    f = SMOOTH_MAX / max(h, w)
    a = make(max(1, round(h * f)), max(1, round(w * f)))
    chans = a[..., None] if a.ndim == 2 else a
    up = np.stack([np.asarray(Image.fromarray(chans[..., i].astype(np.float32), "F").resize((w, h), Image.BILINEAR))
                   for i in range(chans.shape[-1])], -1).astype(np.float64)
    return up[..., 0] if a.ndim == 2 else up


def camera(img, fx, rng):
    """Linear RGB -> uint8 sRGB through the v3 camera pipeline."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    img = img * (1.0 - fx["vignette"] * r2)[..., None]
    m = np.clip(img, 0, None).reshape(-1, 3).mean(0) + 1e-6
    wb = (m.mean() / m) ** fx["awb_strength"]
    wb = np.clip(wb / wb[np.argmax(m)], 1.0, 2.5)
    gain = float(np.clip(0.18 / (m * wb).max(), 1.0, 48.0))
    sig = np.sqrt(np.clip(img, 0, None) * fx["shot_gain"] / gain + fx["read_noise"] ** 2 / gain)
    img = (img + sig * rng.standard_normal(img.shape)) * wb * gain
    img = np.clip(img, 0, 1) ** (1 / 2.2)
    cb = fx["chroma_blur"]
    img = np.stack([ndimage.gaussian_filter(img[..., 0], cb), img[..., 1],
                    ndimage.gaussian_filter(img[..., 2], 0.6 * cb)], -1)
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, "JPEG", quality=fx["jpeg_quality"])
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")), gain, wb, buf.getvalue()


def synthesize_rsuigm(img_srgb, jerlov_type, fx, rng, dist=None, return_jpeg=False):
    """img_srgb HxWx3 in [0,1]; fx from v3 sample_effects (uses D_m as depth d);
    dist: HxW camera distance z in metres (None: tilted plane).
    return_jpeg: also return the simulated camera's JPEG bytes (what the full dataset stores)."""
    k = JERLOV[jerlov_type]
    Kd, b, c = (np.asarray(k[q], dtype=np.float64) for q in ("Kd", "b", "c"))
    h, w = img_srgb.shape[:2]
    s = max(h, w)
    J = guided_filter_self(np.clip(img_srgb, 0, 1) ** 2.2, 2, 1.2e-2)

    z = distance_plane(h, w, fx["d0_m"], fx["tilt"], fx["angle"]) if dist is None else dist
    z = np.maximum(z * (1.0 + fx["turbidity_patchiness"] * _smooth(h, w, lambda a, b_: fbm(a, b_, rng, 4, 0.3))), 0.05)[..., None]
    T = np.exp(-c * z)                                                   # LOS transmission

    E_d = fx["ambient"] * np.exp(-Kd * fx["D_m"])                        # E(d), E0 = ambient
    E_obj = E_d * np.ones((h, w, 1))
    if fx["caustics"]:
        cz = _smooth(h, w, lambda a, b_: caustics(a, b_, rng, cell=0.12, sharp=0.03))
        E_obj = E_obj * np.clip(1.0 + fx["caustic_strength"] * (cz[..., None] - 1.0), 0.2, None)
    L = 0.0
    if fx["lamp"]:
        L = fx["lamp_power"] * _smooth(h, w, lambda a, b_: lamp_pattern(a, b_, fx["lamp_cx"], fx["lamp_cy"], fx["lamp_width"]))[..., None] * LAMP_RGB

    direct = J * (E_obj * T + L * T * T)                                 # RSUIGM D_lambda (+ lamp)
    tau_s = (b.mean() * z)                                               # scattering optical depth
    wgt = 1.0 - np.exp(-fx["forward_scatter_k"] * tau_s)
    direct = (1 - wgt) * direct + wgt * blur_rgb(direct, 0.004 * s * (1.0 + float(np.median(tau_s))))

    beam = _smooth(h, w, lambda a, b_: blur_rgb(fx["lamp_power"] * lamp_pattern(a, b_, fx["lamp_cx"], fx["lamp_cy"], fx["lamp_width"])[..., None]
                                              * LAMP_RGB * np.ones((a, b_, 3)), 0.15 * max(a, b_))) if fx["lamp"] else 0.0
    back = (b / c) * (E_d + 0.5 * beam) * (1.0 - T)                     # RSUIGM B_lambda, V_inf = b E(d) / c
    u8, gain, wb, jpeg = camera(direct + back, fx, rng)
    info = dict(exposure_gain=round(gain, 3), awb_gains=[round(float(v), 3) for v in wb],
                    irradiance_at_depth_rgb=[round(float(v), 5) for v in E_d],
                    mean_transmission_rgb=[round(float(v), 4) for v in T.reshape(-1, 3).mean(0)])
    return (u8, info, jpeg) if return_jpeg else (u8, info)
