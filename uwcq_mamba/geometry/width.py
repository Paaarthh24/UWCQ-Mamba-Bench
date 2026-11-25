"""Crack width at stations placed every 2 px along the skeleton."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .skeleton import branches_from_skeleton, edt_width, prune_spurs


@dataclass
class Station:
    branch: int
    rc: tuple[float, float]
    s: float
    width: float
    width_edt: float
    excluded: str | None
    below_res: bool
    tangent: tuple[float, float] | None = None


def measure_width(soft: np.ndarray, station_step: float = 2.0, spur_px: float = 6.0,
                  below_res: float = 2.0, window_widths: float = 3.0, min_window: float = 5.0,
                  thr: float = 0.5, smoothing: str = "savgol",
                  keep_longest: bool = False) -> tuple[list[Station], np.ndarray]:
    """Width stations on a binary mask; lengths in pixels."""
    soft = np.clip(soft.astype(np.float64), 0, 1)
    binary = soft >= thr
    if binary.sum() < 3:
        return [], np.zeros_like(binary)
    w_edt = edt_width(binary)
    skel = prune_spurs(binary, spur_px, width_px=w_edt, keep_longest=keep_longest)
    branches = branches_from_skeleton(skel, smoothing)
    if not branches:
        return [], skel
    _, (iy, ix) = ndimage.distance_transform_edt(~skel, return_indices=True)
    mass = np.zeros_like(soft)
    fg = soft > 0
    np.add.at(mass, (iy[fg], ix[fg]), soft[fg])
    stations: list[Station] = []
    for bi, br in enumerate(branches):
        rc, ds, s = br.rc, br.ds, br.s
        m = mass[rc[:, 0], rc[:, 1]]
        L = s[-1]
        if L <= 0:
            continue
        excl = []
        for end, deg in ((0, br.ends_degree[0]), (len(rc) - 1, br.ends_degree[1])):
            r, c = rc[end]
            if deg >= 3:
                excl.append(("junction", s[end], w_edt[r, c] / 2 + 2.0))
            elif deg == 1:
                excl.append(("tip", s[end], None))
        n_st = max(int(np.floor(L / station_step)), 1)
        for k in range(n_st + 1):
            sk_ = min(k * station_step, L)
            j = int(np.argmin(np.abs(s - sk_)))
            w_loc = max(w_edt[rc[j, 0], rc[j, 1]], 1.0)
            for _ in range(2):
                half = max(window_widths * w_loc, min_window) / 2
                sel = np.abs(s - sk_) <= half
                arc = ds[sel].sum()
                w_new = m[sel].sum() / arc if arc > 0 else np.nan
                if not np.isfinite(w_new):
                    break
                w_loc = w_new
            reason = None
            for kind, s_end, radius in excl:
                rad = radius if radius is not None else max(w_loc, 1.0)
                if abs(sk_ - s_end) < rad + (half if kind == "tip" else 0):
                    reason = kind
                    break
            tg = (rc[min(j + 3, len(rc) - 1)] - rc[max(j - 3, 0)]).astype(float)
            tg = tuple(tg / np.linalg.norm(tg)) if np.linalg.norm(tg) > 0 else None
            stations.append(Station(branch=bi, rc=(float(rc[j, 0]), float(rc[j, 1])), s=float(sk_),
                                    width=float(w_loc), width_edt=float(w_edt[rc[j, 0], rc[j, 1]]),
                                    excluded=reason, below_res=bool(w_loc < below_res), tangent=tg))
    return stations, skel


def aggregate(stations: list[Station], scale: float = 1.0, low_percentiles: bool = False) -> dict:
    """Width statistics over resolvable, non-excluded stations, scaled to mm by `scale`."""
    ok = [st for st in stations if st.excluded is None]
    res = [st.width * scale for st in ok if not st.below_res]
    out = {"n_stations": len(ok), "width_frac_below_res": (1 - len(res) / len(ok)) if ok else float("nan")}
    if res:
        a = np.asarray(res)
        out.update(width_p50=float(np.percentile(a, 50)), width_p90=float(np.percentile(a, 90)),
                   width_p95=float(np.percentile(a, 95)), width_mean=float(a.mean()),
                   width_max_diag=float(a.max()))
        if low_percentiles:
            out.update(width_p05=float(np.percentile(a, 5)), width_p10=float(np.percentile(a, 10)))
    else:
        out.update(width_p50=float("nan"), width_p90=float("nan"), width_p95=float("nan"),
                   width_mean=float("nan"), width_max_diag=float("nan"))
        if low_percentiles:
            out.update(width_p05=float("nan"), width_p10=float("nan"))
    return out
