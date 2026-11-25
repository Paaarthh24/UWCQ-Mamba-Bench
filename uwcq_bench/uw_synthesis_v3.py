"""Underwater synthesis v3: UW-GAN water types (Table I: N_rer and B) with
water-depth categories instead of Table I's fixed vertical depth D.

  shallow  D in [1, 5] m    sunlit; caustics likely; no lamp
  medium   D in [5, 15] m   dimmer, colour-shifted sunlight; lamp in half the images
  deep     D in [15, 30] m  almost no sunlight; lamp always

The camera stays close to the crack surface (d in [0.5, 2] m, tilted plane).
Chain (linear RGB): patchy turbidity -> ambient N^D (+ caustics) and lamp
(round trip, N^2d) -> distance-dependent forward-scatter blur -> backscatter
(ambient veil + lamp-beam glow) -> sensor (vignetting, auto-exposure, noise,
partial auto white balance,
axial chromatic aberration, JPEG).  No marine-snow particles.  No step moves
pixels, so the source crack mask stays valid for every output.
"""
import io
import numpy as np
from PIL import Image
from scipy import ndimage

from uwcq_bench.uw_synthesis import UWGAN_TYPES, distance_plane, guided_filter_self
from uwcq_bench.uw_synthesis_v2 import fbm, caustics, lamp_pattern, blur_rgb

DEPTH_CATEGORIES = {
    "shallow": dict(D=(1.0, 5.0), p_lamp=0.0, p_caustics=0.4, ambient=(0.8, 1.0), lamp_width=(0.35, 0.7)),
    "medium": dict(D=(5.0, 15.0), p_lamp=0.5, p_caustics=0.0, ambient=(0.6, 0.9), lamp_width=(0.35, 0.6)),
    "deep": dict(D=(15.0, 30.0), p_lamp=1.0, p_caustics=0.0, ambient=(0.4, 0.7), lamp_width=(0.25, 0.45)),
}
LAMP_RGB = np.array([1.0, 0.93, 0.82])          # warm-white LED/halogen


def sample_effects(category, rng):
    c = DEPTH_CATEGORIES[category]
    D = float(rng.uniform(*c["D"]))
    return dict(
        category=category, D_m=round(D, 2),
        d0_m=round(float(rng.uniform(0.5, 2.0)), 3),
        tilt=float(rng.uniform(0.0, 0.4)), angle=float(rng.uniform(0, 2 * np.pi)),
        turbidity_patchiness=float(rng.uniform(0.15, 0.4)),
        caustics=bool(rng.uniform() < c["p_caustics"]),
        # caustics fade with depth (focusing by surface waves weakens)
        caustic_strength=float(rng.uniform(0.2, 0.45) * np.clip(1.2 - D / 5.0, 0.2, 1.0)),
        ambient=float(rng.uniform(*c["ambient"])),
        lamp=bool(rng.uniform() < c["p_lamp"]),
        lamp_cx=float(rng.uniform(0.25, 0.75)), lamp_cy=float(rng.uniform(0.25, 0.75)),
        lamp_width=float(rng.uniform(*c["lamp_width"])),
        lamp_power=float(rng.uniform(0.6, 1.2)),
        forward_scatter_k=float(rng.uniform(0.15, 0.35)),
        vignette=float(rng.uniform(0.15, 0.35)),
        read_noise=float(rng.uniform(0.001, 0.004)),
        shot_gain=float(rng.uniform(0.001, 0.006)),
        chroma_blur=float(rng.uniform(0.3, 0.8)),
        awb_strength=float(rng.uniform(0.3, 0.7)),
        jpeg_quality=int(rng.integers(60, 91)),
    )


def synthesize_v3(img_srgb, water_type, fx, rng, dist=None):
    """dist: optional HxW camera-to-surface distance map in metres (v4: from estimated depth);
    None uses the tilted plane d0 * (1 + tilt * ramp) (v3)."""
    p = UWGAN_TYPES[water_type]
    N = np.asarray(p["N"], dtype=np.float64)
    B = np.asarray(p["B"], dtype=np.float64)
    h, w = img_srgb.shape[:2]
    s = max(h, w)
    J = guided_filter_self(np.clip(img_srgb, 0, 1) ** 2.2, 2, 1.2e-2)

    d = distance_plane(h, w, fx["d0_m"], fx["tilt"], fx["angle"]) if dist is None else dist
    d = np.maximum(d * (1.0 + fx["turbidity_patchiness"] * fbm(h, w, rng, 4, 0.3)), 0.05)[..., None]
    t = N ** d

    sun = fx["ambient"] * N ** fx["D_m"]                                 # sunlight left at depth D
    E_amb = sun * np.ones((h, w, 1))
    if fx["caustics"]:
        c = caustics(h, w, rng, cell=0.12, sharp=0.03)
        E_amb = E_amb * np.clip(1.0 + fx["caustic_strength"] * (c[..., None] - 1.0), 0.2, None)
    L = 0.0
    if fx["lamp"]:
        L = fx["lamp_power"] * lamp_pattern(h, w, fx["lamp_cx"], fx["lamp_cy"], fx["lamp_width"])[..., None] * LAMP_RGB

    direct = J * E_amb * t + J * L * t * t
    tau = -np.log(np.clip(t.mean(-1, keepdims=True), 1e-6, 1))
    wgt = 1.0 - np.exp(-fx["forward_scatter_k"] * tau)
    direct = (1 - wgt) * direct + wgt * blur_rgb(direct, 0.004 * s * (1.0 + float(np.median(tau))))

    beam = blur_rgb(np.broadcast_to(L, (h, w, 3)).copy(), 0.15 * s) if fx["lamp"] else 0.0
    img = direct + B * (1.0 - t) * (sun + 0.5 * beam)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    img = img * (1.0 - fx["vignette"] * r2)[..., None]
    # camera pipeline: partial grey-world white balance (dominant channel fixed, weak ones lifted),
    # then auto-exposure metered on the brightest balanced channel, noise added on the raw signal
    m = np.clip(img, 0, None).reshape(-1, 3).mean(0) + 1e-6
    wb = (m.mean() / m) ** fx["awb_strength"]
    wb = np.clip(wb / wb[np.argmax(m)], 1.0, 2.5)   # real cameras cap WB gains; empty channels stay empty
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
    u8 = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return u8, dict(exposure_gain=round(gain, 3), awb_gains=[round(float(v), 3) for v in wb],
                    sunlight_rgb=[round(float(v), 5) for v in np.ravel(sun)],
                    mean_transmission_rgb=[round(float(v), 4) for v in t.reshape(-1, 3).mean(0)])
