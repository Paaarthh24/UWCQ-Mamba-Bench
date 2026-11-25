"""Crack length: longest tip-to-tip geodesic (main) and sum of kept branches (total)."""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import ndimage

from .skeleton import branches_from_skeleton, edt_width, prune_spurs, scaled_length


def length_metrics(binary: np.ndarray, spur_px: float = 6.0, smoothing: str = "savgol",
                   border_px: int = 2, image_border: np.ndarray | None = None, main_mode: str = "tips",
                   keep_longest: bool = False, mirror: tuple | None = None, axis_scale: tuple | None = None) -> dict:
    """Skeleton-graph lengths and PCA extents of a binary mask, in grid units."""
    binary = binary.astype(bool)
    out = dict(path_length_total=0.0, path_length_main=0.0, extent_long=0.0, extent_trans=0.0,
               anisotropy=float("nan"), pca_ambiguous=False, tortuosity=float("nan"),
               n_branches=0, n_junctions=0, n_endpoints=0, truncated=False)
    if mirror is not None:
        out["main_border_ends"] = 0
        out["main_free_end_radius"] = 0.0
    if binary.sum() < 3:
        return out
    if mirror is not None and any(mirror):
        P = int(2 * edt_width(binary).max()) + 10
        t, b, l, r = (P if side else 0 for side in mirror)
        big = np.pad(binary, ((t, b), (l, r)), mode="reflect")
        skel = prune_spurs(big, spur_px, width_px=edt_width(big), keep_longest=keep_longest)
        skel = skel[t:t + binary.shape[0], l:l + binary.shape[1]]
    else:
        skel = prune_spurs(binary, spur_px, width_px=edt_width(binary), keep_longest=keep_longest)
    branches = branches_from_skeleton(skel, smoothing)
    if not branches:
        return out
    G = nx.Graph()
    tips, junctions = set(), set()
    for br in branches:
        a, b = tuple(br.rc[0]), tuple(br.rc[-1])
        G.add_edge(a, b, weight=br.length, key=len(G.edges))
        for node, deg in ((a, br.ends_degree[0]), (b, br.ends_degree[1])):
            if deg == 1:
                tips.add(node)
            elif deg >= 3:
                junctions.add(node)
    out["path_length_total"] = float(sum(br.length for br in branches))
    out["n_branches"], out["n_junctions"], out["n_endpoints"] = len(branches), len(junctions), len(tips)
    if main_mode == "nodes":
        G = nx.Graph()
        for bi, br in enumerate(branches):
            a, b = tuple(br.rc[0]), tuple(br.rc[-1])
            if a != b and (not G.has_edge(a, b) or G[a][b]["weight"] > br.length):
                G.add_edge(a, b, weight=br.length, bi=bi)
    best, pair = 0.0, None
    tl = list(G.nodes) if main_mode == "nodes" else list(tips)
    for i, t in enumerate(tl):
        dist = nx.single_source_dijkstra_path_length(G, t, weight="weight")
        for u in tl[i + 1:]:
            if u in dist and dist[u] > best:
                best, pair = dist[u], (t, u)
    if pair is None:
        best = out["path_length_total"]
    out["path_length_main"] = float(best)
    if axis_scale is not None:
        mm = [scaled_length(br.rc, axis_scale[0], axis_scale[1], smoothing) for br in branches]
        out["path_length_total_mm"] = float(sum(mm))
        if pair is not None and main_mode == "nodes":
            path = nx.dijkstra_path(G, pair[0], pair[1], weight="weight")
            out["path_length_main_mm"] = float(sum(mm[G[u][v]["bi"]] for u, v in zip(path[:-1], path[1:])))
        else:
            out["path_length_main_mm"] = out["path_length_total_mm"]
    if pair is not None and mirror is not None:
        Hb, Wb = binary.shape
        on_border = lambda rc: bool(rc[0] < border_px or rc[1] < border_px or rc[0] >= Hb - border_px or rc[1] >= Wb - border_px
                                    or (image_border is not None and image_border[rc[0], rc[1]]))
        out["main_border_ends"] = int(on_border(pair[0])) + int(on_border(pair[1]))
        edt = ndimage.distance_transform_edt(np.pad(binary, 1))[1:-1, 1:-1]
        out["main_free_end_radius"] = float(sum(edt[rc] for rc in pair if not on_border(rc)))
    if pair is not None:
        chord = float(np.hypot(pair[0][0] - pair[1][0], pair[0][1] - pair[1][1]))
        out["tortuosity"] = best / chord if chord > 0 else float("nan")
    ys, xs = np.nonzero(binary)
    P = np.stack([xs, ys], 1).astype(float)
    P -= P.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(P.T))
    main, minor = evecs[:, 1], evecs[:, 0]
    pl, pt = P @ main, P @ minor
    out["extent_long"] = float(np.percentile(pl, 99) - np.percentile(pl, 1))
    out["extent_trans"] = float(np.percentile(pt, 99) - np.percentile(pt, 1))
    out["anisotropy"] = float(evals[0] / evals[1]) if evals[1] > 0 else float("nan")
    out["pca_ambiguous"] = bool(out["anisotropy"] > 0.5)
    H, W = binary.shape
    for (r, c) in tips:
        if r < border_px or c < border_px or r >= H - border_px or c >= W - border_px:
            out["truncated"] = True
        if image_border is not None and image_border[r, c]:
            out["truncated"] = True
    return out
