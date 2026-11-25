"""Depth-free underwater synthesis v2: UW-GAN water types (Table I) plus the
optical and sensor effects that a flat, close-range scene otherwise loses.

Order of the image-formation chain (all in linear RGB):
  1. patchy turbidity     d_eff(x) = d(x) * (1 + a * fbm(x))
  2. illumination         ambient sunlight N^D, optionally with caustics,
                          and/or an artificial lamp cone near the camera
  3. direct signal        GF(J) * E_amb(x) * N^d_eff  +  GF(J) * E_lamp(x) * N^(2 d_eff)
                          (lamp light travels camera -> surface -> camera)
  4. forward scatter      distance-dependent PSF blur of the direct signal
  5. backscatter          B * (1 - N^d_eff) * (ambient + widely blurred lamp beam)
  6. marine snow          lit, partly defocused particles in front of the surface
  7. sensor               vignetting, auto-exposure, Poisson-Gaussian noise,
                          axial chromatic aberration (per-channel blur), JPEG

No step moves pixels, so the source crack mask remains the label for every
output.  Marine snow can occlude a few crack pixels, as in real footage.
"""
import io
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from uwcq_bench.uw_synthesis import UWGAN_TYPES, distance_plane, guided_filter_self


def fbm(h, w, rng, octaves=4, base=0.25):
    """Fractal smooth noise in roughly [-1, 1]; `base` = largest feature size / image size."""
    s = max(h, w)
    out = np.zeros((h, w))
    amp = 1.0
    for o in range(octaves):
        sig = base * s / (2 ** o)
        n = ndimage.gaussian_filter(rng.standard_normal((h, w)), sig, mode="wrap")
        out += amp * n / (n.std() + 1e-12)
        amp *= 0.5
    return out / (np.abs(out).max() + 1e-12)


def caustics(h, w, rng, cell=0.08, sharp=0.015):
    """Worley-noise (F2 - F1) caustic network, domain-warped, mean ~1."""
    s = max(h, w)
    n_pts = int(1.0 / cell ** 2 * (h * w) / s ** 2) + 8
    pts = rng.uniform(0, 1, (n_pts, 2)) * [h, w]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    warp = 0.03 * s
    yy = yy + warp * fbm(h, w, rng, 3, 0.15)
    xx = xx + warp * fbm(h, w, rng, 3, 0.15)
    dist, _ = cKDTree(pts).query(np.stack([yy.ravel(), xx.ravel()], 1), k=2)
    f = ((dist[:, 1] - dist[:, 0]) / s).reshape(h, w)
    c = np.exp(-f / sharp)
    c = ndimage.gaussian_filter(c, 0.004 * s)
    return c / (c.mean() + 1e-12)


def lamp_pattern(h, w, cx, cy, width):
    """Gaussian lamp cone centred at (cx, cy) in normalised coords; peak 1."""
    s = max(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - cx * w) ** 2 + (yy - cy * h) ** 2) / (width * s) ** 2
    return np.exp(-0.5 * r2)


def blur_rgb(x, sigma):
    return np.stack([ndimage.gaussian_filter(x[..., c], sigma) for c in range(3)], -1) if sigma > 0 else x


def sample_effects(water_type, rng):
    """Per-image effect settings.  Caustics only in shallow water (D <= 0.5 m);
    type 7 (D = 10 m, no usable sunlight) always gets a lamp."""
    D = UWGAN_TYPES[water_type]["D"]
    lamp = True if D >= 10 else bool(rng.uniform() < 0.5)
    return dict(
        turbidity_patchiness=float(rng.uniform(0.2, 0.5)),
        caustics=bool(D <= 0.5 and rng.uniform() < 0.5),
        caustic_strength=float(rng.uniform(0.7, 1.0)),
        lamp=lamp,
        lamp_cx=float(rng.uniform(0.3, 0.7)), lamp_cy=float(rng.uniform(0.3, 0.7)),
        lamp_width=float(rng.uniform(0.25, 0.5)),
        lamp_power=float(rng.uniform(1.5, 3.0)),
        ambient=float(0.15 if D >= 10 else rng.uniform(0.6, 1.0)),
        forward_scatter_k=float(rng.uniform(0.15, 0.35)),
        snow_density=float(rng.uniform(0.3, 1.0)),
        vignette=float(rng.uniform(0.2, 0.45)),
        read_noise=float(rng.uniform(0.002, 0.006)),
        shot_gain=float(rng.uniform(0.002, 0.01)),
        chroma_blur=float(rng.uniform(0.4, 1.0)),
        jpeg_quality=int(rng.integers(45, 86)),
    )


def synthesize_v2(img_srgb, water_type, d0, tilt, angle, fx, rng):
    """img_srgb: HxWx3 float [0,1].  Returns (uint8 image, info dict)."""
    p = UWGAN_TYPES[water_type]
    N = np.asarray(p["N"], dtype=np.float64)
    B = np.asarray(p["B"], dtype=np.float64)
    h, w = img_srgb.shape[:2]
    s = max(h, w)
    J = np.clip(img_srgb, 0, 1) ** 2.2                                    # linearise
    J = guided_filter_self(J, 2, 1.2e-2)                                  # paper's Avg_Omega

    # 1. patchy turbidity
    d = distance_plane(h, w, d0, tilt, angle)
    d_eff = d * (1.0 + fx["turbidity_patchiness"] * fbm(h, w, rng, 4, 0.3))
    d_eff = np.maximum(d_eff, 0.05)[..., None]
    t = N ** d_eff                                                        # one-way transmission

    # 2. illumination
    E_amb = fx["ambient"] * (N ** p["D"]) * np.ones((h, w, 1))
    if fx["caustics"]:
        c = caustics(h, w, rng)
        E_amb = E_amb * np.clip(1.0 + fx["caustic_strength"] * (c[..., None] - 1.0), 0.2, None)
    L = np.zeros((h, w, 1))
    if fx["lamp"]:
        # with caustics (daylit shallow water) a lamp is a weaker fill light
        L = fx["lamp_power"] * (0.35 if fx["caustics"] else 1.0) * lamp_pattern(h, w, fx["lamp_cx"], fx["lamp_cy"], fx["lamp_width"])[..., None]

    # 3. direct signal (lamp: round trip, so transmission squared)
    direct = J * E_amb * t + J * L * t * t

    # 4. forward scatter: blend towards a wider blur as optical distance grows
    tau = -np.log(np.clip(t.mean(-1, keepdims=True), 1e-6, 1))           # optical depth
    wgt = 1.0 - np.exp(-fx["forward_scatter_k"] * tau)
    sig = 0.004 * s * (1.0 + float(np.median(tau)))
    direct = (1 - wgt) * direct + wgt * blur_rgb(direct, sig)

    # 5. backscatter: ambient veil + glow of the lamp beam in the water
    beam = blur_rgb(np.repeat(L, 3, -1), 0.15 * s) if fx["lamp"] else 0.0
    back = B * (1.0 - t) * (fx["ambient"] * N ** p["D"] + 0.6 * beam)

    img = direct + back

    # 6. marine snow (in front of the surface: little attenuation, lit by ambient + lamp)
    n_snow = int(fx["snow_density"] * 400 * h * w / 640 / 360)
    snow = np.zeros((h, w))
    if n_snow:
        ys, xs = rng.integers(0, h, n_snow), rng.integers(0, w, n_snow)
        size = rng.pareto(2.5, n_snow) * 0.6 + 0.4                        # mostly tiny, few large
        bright = rng.uniform(0.3, 1.0, n_snow)
        focus = rng.uniform(0, 1, n_snow)
        layers = [np.zeros((h, w)) for _ in range(3)]
        for y, x, sz, b, f in zip(ys, xs, size, bright, focus):
            layers[min(int(f * 3), 2)][y, x] += b * sz ** 2
        for k, lay in enumerate(layers):                                  # 3 defocus levels
            snow += ndimage.gaussian_filter(lay, (0.6 + 2.2 * k) * s / 640) * (1.0 + 2 * k)
        snow = np.clip(snow, 0, None)
        lit = fx["ambient"] * (N ** p["D"]).mean() + (L[..., 0] if fx["lamp"] else 0.0)
        img = img + (snow * lit)[..., None] * (0.6 + 0.4 * B)

    # 7. sensor
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
    img = img * (1.0 - fx["vignette"] * r2)[..., None]
    gain = 0.18 / max(img.mean(), 1e-4)                                   # auto-exposure: mid-grey metering
    gain = float(np.clip(gain, 1.0, 8.0))                                 # limited sensor gain
    sig_raw = np.sqrt(np.clip(img, 0, None) * fx["shot_gain"] / gain + fx["read_noise"] ** 2 / gain)
    img = (img + sig_raw * rng.standard_normal(img.shape)) * gain         # noise amplified by gain
    img = np.clip(img, 0, 1) ** (1 / 2.2)
    cb = fx["chroma_blur"]                                                # axial CA: R/B defocused
    img = np.stack([ndimage.gaussian_filter(img[..., 0], cb), img[..., 1],
                    ndimage.gaussian_filter(img[..., 2], 0.6 * cb)], -1)
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, "JPEG", quality=fx["jpeg_quality"])
    u8 = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return u8, dict(exposure_gain=round(float(gain), 3), n_snow=n_snow,
                    mean_transmission_rgb=[round(float(v), 4) for v in t.reshape(-1, 3).mean(0)])
