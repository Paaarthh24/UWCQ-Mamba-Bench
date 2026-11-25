"""Six crack measurements from a mask: main and total length, minimum, mean and maximum width, and area."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .hybrid_measure import endpoints, keep_pieces, prune, skeleton_steps
from .length import length_metrics
from .mask_qc import binarize, clean
from .width import aggregate, measure_width

PAD = 8
WIDTH_KEYS = ("width_min", "width_p10", "width_mean", "width_mean_area", "width_p50", "width_max", "width_max_raw")
QUANTITIES = ("length_main", "length_total", *WIDTH_KEYS, "area")
COLUMNS = ["image_id", "split", "W", "H", "n_components", "truncated", "length_main_px", "length_main_tipcorr_px",
           "length_total_px",
           "length_crosscheck_ok", "width_min_px", "width_p10_px", "width_mean_px", "width_mean_area_px", "width_p50_px",
           "width_max_px", "width_max_raw_px", "width_frac_below_res", "area_px",
           "length_main_kulpa_px", "length_total_kulpa_px", "length_crosscheck_rel", "width_edt_max_px", "n_stations",
           "width_min_upper_bound", "width_p10_upper_bound", "width_mean_area_rel_diff", "width_mean_area_ok",
           "n_tiny_components", "stray_removed_components", "stray_removed_px", "clean_changed_px", "holes_filled",
           "holes_flagged", "empty"] + [f"{p}_{q}" for q in QUANTITIES for p in ("valid", "reason")]
COMPONENT_COLUMNS = ["image_id", "component", "area_px", "length_main_px", "length_total_px", "truncated",
                     "width_min_px", "width_mean_px", "width_p50_px", "width_max_px", "width_frac_below_res", "tiny"]
NAN = float("nan")


def _crop(m):
    ys, xs = np.nonzero(m)
    y0, x0 = max(0, ys.min() - PAD), max(0, xs.min() - PAD)
    return m[y0:ys.max() + PAD + 1, x0:xs.max() + PAD + 1], (y0, x0)


def _border(shape, off, crop_shape, border_px):
    """Pixels of the crop that lie within border_px of the FULL image border (truncation test inside a crop)."""
    H, W = shape
    rr = np.arange(crop_shape[0])[:, None] + off[0]
    cc = np.arange(crop_shape[1])[None, :] + off[1]
    return (rr < border_px) | (cc < border_px) | (rr >= H - border_px) | (cc >= W - border_px)


def prepare(mask_raw, cfg, binary=False, stray_filter=True, clean_mask=True):
    """Steps 1-3 -> (bool mask at the original size, log dict). binary=True: mask_raw is already a bool prediction."""
    m = mask_raw.astype(bool) if binary else binarize(mask_raw, cfg["binarize"])
    log = dict(clean_changed_px=0, holes_filled=0, holes_flagged=0, n_tiny_components=0,
               stray_removed_components=0, stray_removed_px=0)
    if m.sum() < 3:
        return np.zeros_like(m), log
    full = np.zeros_like(m)
    c, (y0, x0) = _crop(m)
    cl = cfg["clean"]
    c, clog = clean(c, cl["open_close_px"], cl["gap_close_px"], cl["min_area_px"], cl["max_hole_px"]) if clean_mask else (c, {})
    log.update(clean_changed_px=clog.get("open_close_changed_px", 0), holes_filled=clog.get("holes_filled", 0),
               holes_flagged=clog.get("holes_flagged", 0))
    if c.sum() >= 3:
        skel = prune(skeletonize(c), int(cfg["width"]["spur_px"])) if stray_filter else None
        if skel is not None and skel.any():
            st = skeleton_steps(skel)
            lab0, n0 = ndimage.label(c, np.ones((3, 3), int))
            c2, _, _ = keep_pieces(c, skel, st, 0.0, 8.0)
            n1 = ndimage.label(c2, np.ones((3, 3), int))[1]
            log.update(stray_removed_components=int(n0 - n1), stray_removed_px=int(c.sum() - c2.sum()))
            c = c2
        ft = cfg["flag_tiny"]
        lab, n = ndimage.label(c, np.ones((3, 3), int))
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            comp = lab[sl] == i
            if comp.sum() < ft["min_area_px"] or skeletonize(comp).sum() < ft["min_skeleton_px"]:
                log["n_tiny_components"] += 1
    full[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = c
    return full, log


def _width_mm(st, s_row, s_col):
    """mm width of each valid station: local area / local length with per-axis pixel sizes."""
    out, g = [], math.sqrt(s_row * s_col)
    for s in st:
        if s.excluded is None and not s.below_res:
            if s.tangent is None:
                out.append(s.width * g)
            else:
                out.append(s.width * s_row * s_col / float(np.hypot(s_row * s.tangent[0], s_col * s.tangent[1])))
    return np.asarray(out, float)


def _mm_stats(w):
    if not len(w):
        return dict(p05=NAN, p10=NAN, p50=NAN, p95=NAN, mean=NAN, max=NAN)
    return dict(p05=float(np.percentile(w, 5)), p10=float(np.percentile(w, 10)), p50=float(np.percentile(w, 50)),
                p95=float(np.percentile(w, 95)), mean=float(w.mean()), max=float(w.max()))


def _core(m, shape, off, cfg, scale=None):
    """Steps 4-5 on a cropped mask -> dict of raw values (px; + mm when scale = (s_row, s_col)) and the station list."""
    w, L = cfg["width"], cfg["length"]
    border = _border(shape, off, m.shape, L["border_px"])
    H, W = shape
    mir = (off[0] == 0, off[0] + m.shape[0] == H, off[1] == 0, off[1] + m.shape[1] == W)
    try:
        st, skel = measure_width(m.astype(float), station_step=w["station_step_px"], spur_px=w["spur_px"],
                                 below_res=w["below_res_px"], keep_longest=True)
        lm = length_metrics(m, spur_px=L["spur_px"], smoothing=L["smoothing"], border_px=L["border_px"], image_border=border,
                          main_mode="nodes", keep_longest=True, mirror=mir, axis_scale=scale)
        lk = length_metrics(m, spur_px=L["spur_px"], smoothing=L["crosscheck_smoothing"], border_px=L["border_px"],
                            image_border=border, main_mode="nodes", keep_longest=True, mirror=mir)
    except ValueError:
        return None, []
    agg = aggregate(st, low_percentiles=True)
    ok = [s for s in st if s.excluded is None]
    tr = bool(lm["truncated"])
    ty, tx = np.nonzero(endpoints(skel))
    if len(ty):
        H, W = shape
        gy, gx = ty + off[0], tx + off[1]
        dist = np.minimum(np.minimum(gy, H - 1 - gy), np.minimum(gx, W - 1 - gx))
        edt = ndimage.distance_transform_edt(np.pad(m, 1))[ty + 1, tx + 1]
        tr |= bool((dist <= L["border_px"] + edt).any())
    v = dict(length_main=lm["path_length_main"], length_total=lm["path_length_total"], truncated=tr,
             main_border_ends=lm["main_border_ends"], main_free_end_radius=lm["main_free_end_radius"],
             length_main_kulpa=lk["path_length_main"], length_total_kulpa=lk["path_length_total"],
             width_p05=agg["width_p05"], width_p10=agg["width_p10"], width_mean=agg["width_mean"],
             width_p50=agg["width_p50"], width_p95=agg["width_p95"], width_max_raw=agg["width_max_diag"],
             width_frac_below_res=agg["width_frac_below_res"], n_stations=agg["n_stations"],
             width_edt_max=max((s.width_edt for s in ok), default=NAN), area=float(m.sum()))
    if scale is not None:
        sr, sc = scale
        v.update(length_main_mm=lm.get("path_length_main_mm", NAN), length_total_mm=lm.get("path_length_total_mm", NAN),
                 width_mm=_mm_stats(_width_mm(st, sr, sc)), area_mm2=float(m.sum()) * sr * sc)
    return v, st


def _val(x):
    return NAN if x is None or not np.isfinite(x) else float(x)


MM_COLS = ("length_main_mm", "length_main_tipcorr_mm", "length_total_mm", "width_min_mm", "width_p10_mm", "width_mean_mm",
           "width_mean_area_mm", "width_p50_mm", "width_max_mm", "width_max_raw_mm", "area_mm2")


def measure_mask(mask_raw, cfg, image_id="", split="", binary=False, components=True, stray_filter=True,
                 clean_mask=True, scale="cfg"):
    """Full record for one mask -> (row dict with COLUMNS, list of per-component dicts)."""
    H, W = mask_raw.shape[:2]
    sc = axis_scales(cfg.get("scale"), W, H) if isinstance(scale, str) else scale
    m, log = prepare(mask_raw, cfg, binary=binary, stray_filter=stray_filter, clean_mask=clean_mask)
    row = {k: NAN for k in COLUMNS}
    row.update(image_id=image_id, split=split, W=W, H=H, **log)
    comps = []
    if m.sum() < 3:
        row.update(empty=True, n_components=0, truncated=False, area_px=float(m.sum()))
        if sc is not None:
            row.update({k: NAN for k in MM_COLS}, scale_row_mm_per_px=sc[0], scale_col_mm_per_px=sc[1],
                       area_mm2=float(m.sum()) * sc[0] * sc[1])
        for q in QUANTITIES:
            row[f"valid_{q}"], row[f"reason_{q}"] = q == "area", (None if q == "area" else "EMPTY_MASK")
        return row, comps
    c, off = _crop(m)
    v, _ = _core(c, (H, W), off, cfg, sc)
    lab, ncomp = ndimage.label(c, np.ones((3, 3), int))
    row.update(empty=False, n_components=int(ncomp), area_px=float(c.sum()))
    if sc is not None:
        row.update({k: NAN for k in MM_COLS}, scale_row_mm_per_px=sc[0], scale_col_mm_per_px=sc[1],
                   area_mm2=float(c.sum()) * sc[0] * sc[1])

    def _components():
        """Per-component rows."""
        out = []
        if not components:
            return out
        ft = cfg["flag_tiny"]
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            comp = lab == i
            cc, coff = _crop(comp)
            cv_, _ = _core(cc, (H, W), (off[0] + coff[0], off[1] + coff[1]), cfg, sc)
            cv_ = cv_ or {}
            mmc = {}
            if sc is not None:
                wm = cv_.get("width_mm") or _mm_stats([])
                mmc = dict(length_main_mm=cv_.get("length_main_mm", NAN), length_total_mm=cv_.get("length_total_mm", NAN),
                           width_min_mm=wm["p05"], width_mean_mm=wm["mean"], width_p50_mm=wm["p50"], width_max_mm=wm["p95"],
                           area_mm2=float(comp.sum()) * sc[0] * sc[1])
            out.append(dict(image_id=image_id, component=i, area_px=float(comp.sum()),
                            length_main_px=cv_.get("length_main", NAN), length_total_px=cv_.get("length_total", NAN),
                            truncated=bool(cv_.get("truncated", False)), width_min_px=_val(cv_.get("width_p05")),
                            width_mean_px=_val(cv_.get("width_mean")), width_p50_px=_val(cv_.get("width_p50")),
                            width_max_px=_val(cv_.get("width_p95")),
                            width_frac_below_res=cv_.get("width_frac_below_res", NAN),
                            tiny=bool(comp.sum() < ft["min_area_px"] or skeletonize(cc).sum() < ft["min_skeleton_px"]), **mmc))
        return out
    if v is None:
        row["truncated"] = False
        for q in QUANTITIES:
            row[f"valid_{q}"], row[f"reason_{q}"] = q == "area", (None if q == "area" else "NO_SKELETON_GRAPH")
        return row, _components()
    if v["length_total"] <= 0:
        row["truncated"] = v["truncated"]
        for q in QUANTITIES:
            row[f"valid_{q}"], row[f"reason_{q}"] = q == "area", (None if q == "area" else "SKELETON_COLLAPSED")
        return row, _components()
    br = v["width_frac_below_res"]
    min_ub = bool(np.isfinite(br) and br > 0.05)
    p10_ub = bool(np.isfinite(br) and br > 0.10)
    floor = cfg["width"]["below_res_px"]
    w_area = v["area"] / v["length_total"] if v["length_total"] > 0 else NAN
    rel = abs(w_area / v["width_mean"] - 1) if np.isfinite(w_area) and np.isfinite(_val(v["width_mean"])) else NAN
    rel_len = abs(v["length_total_kulpa"] - v["length_total"]) / max(v["length_total"], 1e-9)
    tipcorr = v["length_main"] + v["main_free_end_radius"]
    if sc is not None:
        g, wm = math.sqrt(sc[0] * sc[1]), v["width_mm"]
        row.update(length_main_mm=v["length_main_mm"], length_main_tipcorr_mm=v["length_main_mm"] + v["main_free_end_radius"] * g,
                   length_total_mm=v["length_total_mm"],
                   width_min_mm=cfg["width"]["below_res_px"] * g if (np.isfinite(v["width_frac_below_res"]) and v["width_frac_below_res"] > 0.05) else wm["p05"],
                   width_p10_mm=cfg["width"]["below_res_px"] * g if (np.isfinite(v["width_frac_below_res"]) and v["width_frac_below_res"] > 0.10) else wm["p10"],
                   width_mean_mm=wm["mean"], width_p50_mm=wm["p50"], width_max_mm=wm["p95"], width_max_raw_mm=wm["max"],
                   width_mean_area_mm=v["area_mm2"] / v["length_total_mm"] if v["length_total_mm"] > 0 else NAN)
    row.update(truncated=v["truncated"], length_main_px=v["length_main"], length_main_tipcorr_px=tipcorr,
               length_total_px=v["length_total"],
               length_main_kulpa_px=v["length_main_kulpa"], length_total_kulpa_px=v["length_total_kulpa"],
               length_crosscheck_rel=rel_len, length_crosscheck_ok=bool(rel_len <= cfg["q2_gate"]["length_crosscheck_rel"]),
               width_min_px=floor if min_ub else _val(v["width_p05"]), width_min_upper_bound=min_ub,
               width_p10_px=floor if p10_ub else _val(v["width_p10"]), width_p10_upper_bound=p10_ub,
               width_mean_px=_val(v["width_mean"]), width_mean_area_px=w_area, width_mean_area_rel_diff=rel,
               width_mean_area_ok=bool(np.isfinite(rel) and rel <= 0.10), width_p50_px=_val(v["width_p50"]),
               width_max_px=_val(v["width_p95"]), width_max_raw_px=_val(v["width_max_raw"]),
               width_edt_max_px=_val(v["width_edt_max"]), width_frac_below_res=br, n_stations=v["n_stations"])
    hard = cfg["qc"]["hard_limits"]["width_p95_px_max"]
    for q in QUANTITIES:
        val = row[f"{q}_px"]
        valid, reason = bool(np.isfinite(val)), None
        if not valid:
            reason = "BELOW_RES" if (q.startswith("width") and v["n_stations"]) else "NO_STATIONS" if q.startswith("width") else "NOT_MEASURED"
        elif q.startswith("length") and v["truncated"]:
            reason = "TRUNCATED_LOWER_BOUND"
        elif q.startswith("width") and np.isfinite(row["width_max_px"]) and row["width_max_px"] > hard:
            valid, reason = False, "HARD_LIMIT"
        elif (q == "width_min" and min_ub) or (q == "width_p10" and p10_ub):
            reason = "UPPER_BOUND_LT_2PX"
        elif q == "length_total" and not row["length_crosscheck_ok"]:
            reason = "KULPA_CROSSCHECK_GT_1.5PCT"
        elif q == "width_mean_area" and not row["width_mean_area_ok"]:
            reason = "DIFFERS_FROM_STATION_MEAN_GT_10PCT"
        row[f"valid_{q}"], row[f"reason_{q}"] = valid, reason
    return row, _components()


def hfov_water_deg(hfov_air_deg, port="flat", n=1.33):
    """Horizontal field of view under water. Flat port: refraction narrows it; dome port: unchanged."""
    if port == "dome":
        return float(hfov_air_deg)
    if port != "flat":
        raise ValueError(f"port must be flat or dome, not {port!r}")
    return float(2 * math.degrees(math.asin(math.sin(math.radians(hfov_air_deg) / 2) / n)))


def mm_per_px(z_m, hfov_water_deg_, width_px):
    """Pinhole footprint of one pixel at distance z_m (fronto-parallel): 2 Z tan(HFOV / 2) / W, in mm."""
    return 1000.0 * 2 * z_m * math.tan(math.radians(hfov_water_deg_) / 2) / width_px


def axis_scales(cfg_scale, W, H):
    """(s_row, s_col) mm per px for a W x H image, or None (fixed_axes: separate scales for the long and short side)."""
    s = cfg_scale or {}
    if s.get("mode") == "fixed_axes":
        lo, hi = float(s["mm_per_px_short"]), float(s["mm_per_px_long"])
        return (lo, hi) if W >= H else (hi, lo)
    one = scale_for(s, W)
    return None if one is None else (one["scale_mm_per_px"], one["scale_mm_per_px"])


def scale_for(cfg_scale, width_px):
    """-> dict(scale_mm_per_px, scale_source, scale_rel_unc) or None while the scale is not configured."""
    s = cfg_scale or {}
    if s.get("mode") == "fixed_axes":
        ru = s.get("rel_unc")
        return dict(scale_mm_per_px=math.sqrt(float(s["mm_per_px_short"]) * float(s["mm_per_px_long"])),
                    scale_source=s.get("source", "fixed_axes"), scale_rel_unc=float("nan") if ru is None else float(ru))
    if s.get("mode") == "fixed":
        if not s.get("mm_per_px"):
            return None
        ru = s.get("rel_unc")
        return dict(scale_mm_per_px=float(s["mm_per_px"]), scale_source=s.get("source", "fixed"),
                    scale_rel_unc=float("nan") if ru is None else float(ru))
    if s.get("mode") == "calibrated":
        if not s.get("fx_px"):
            return None
        z = math.sqrt(s["z_min_m"] * s["z_max_m"])
        return dict(scale_mm_per_px=1000.0 * z / s["fx_px"], scale_source="checkerboard_fx_assumed_standoff",
                    scale_rel_unc=math.sqrt(s["z_max_m"] / s["z_min_m"]) - 1)
    if s.get("hfov_air_deg") is None:
        return None
    z = math.sqrt(s["z_min_m"] * s["z_max_m"])
    hf = hfov_water_deg(s["hfov_air_deg"], s.get("port", "flat"), s.get("refractive_index", 1.33))
    return dict(scale_mm_per_px=mm_per_px(z, hf, width_px),
                scale_source=f"assumed_standoff_{s['z_min_m']:g}-{s['z_max_m']:g}m",
                scale_rel_unc=math.sqrt(s["z_max_m"] / s["z_min_m"]) - 1)


MM_KEYS = ("length_main", "length_main_tipcorr", "length_total", *WIDTH_KEYS)
