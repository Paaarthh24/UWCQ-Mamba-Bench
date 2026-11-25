"""Evaluate a trained UWCQ-Mamba checkpoint on a dataset's test split.

  python evaluate.py --data ~/Datasets/UW_CamCrack789 --split split_author_testsel.json --weights weights/uwcca.pth

Prints the segmentation metrics (mIoU, ODS, OIS, F1; MixerCSeg protocol), the mIoU in each Jerlov water type, and the
median relative error of the six crack measurements after one scale per quantity fitted on training images.
Results are written to <weights folder>/eval_<dataset>.json.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from uwcq_mamba.data import JERLOV, split_items
from uwcq_mamba.evaluation import (NAMES, author_metrics, calibration_subset, hist, measure, pooled_miou, predict_prob,
                                   read_image, reference, water_type)
from uwcq_mamba.model import build, build_inference

CFG = yaml.safe_load(open(Path(__file__).parent / "configs/measure.yaml"))


def _measure(args):
    packed, shape = args
    return measure(np.unpackbits(packed)[:shape[0] * shape[1]].reshape(shape).astype(bool), CFG)


def masks_of(model, items, keep_hist=False):
    """Run the model on every render -> packed masks (+ author histograms and pooled counts per render)."""
    out = []
    for i, (p, mp) in enumerate(items):
        prob = predict_prob(model, read_image(p))
        pred = prob >= 0.5
        rec = dict(id=f"{p.parent.name}/{p.name}", packed=np.packbits(pred), shape=pred.shape)
        if keep_hist:
            gt = cv2.imread(str(mp), 0) > 127
            tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
            rec.update(hist=hist(prob, gt), counts=(tp, fp, fn, gt.size), jerlov=water_type(p))
        out.append(rec)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="split_author_testsel.json")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--calib-per-source", type=int, default=1, help="training renders per source photo for the scales "
                    "(3 for UW-CrackMap, which has few source photos)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eager", action="store_true", help="run the unprepared model eagerly (same results, slower)")
    a = ap.parse_args()

    model = build(a.weights, a.device).eval() if a.eager else build_inference(a.weights, a.device)
    test = split_items(a.data, a.split, "test")
    print(f"test: {len(test)} renders", flush=True)
    R = masks_of(model, test, keep_hist=True)
    res = dict(dataset=Path(a.data).name, n_test=len(test), segmentation=author_metrics([r["hist"] for r in R]),
               miou_per_water_type={j: pooled_miou([r["counts"] for r in R if r["jerlov"] == j]) for j in JERLOV
                                    if any(r["jerlov"] == j for r in R)})
    print("segmentation:", {k: round(v, 4) for k, v in res["segmentation"].items()}, flush=True)

    ref = reference(a.data)
    cal = masks_of(model, calibration_subset(split_items(a.data, a.split, "train"), a.calib_per_source))
    with Pool(a.workers) as pool:
        q_test = np.array(pool.map(_measure, [(r["packed"], r["shape"]) for r in R], chunksize=8))
        q_cal = np.array(pool.map(_measure, [(r["packed"], r["shape"]) for r in cal], chunksize=8))
    g_test = np.array([ref.get(r["id"], [np.nan] * 6) for r in R])
    g_cal = np.array([ref.get(r["id"], [np.nan] * 6) for r in cal])
    scale = [float(np.nanmedian((g_cal[:, j] / q_cal[:, j])[(g_cal[:, j] > 0) & (q_cal[:, j] > 0)])) for j in range(6)]
    err = {}
    for j, nm in enumerate(NAMES):
        v = (g_test[:, j] > 0) & np.isfinite(q_test[:, j])
        err[nm] = float(100 * np.median(np.abs(q_test[v, j] * scale[j] / g_test[v, j] - 1)))
    err["average"] = float(np.mean(list(err.values())))
    res.update(measurement_scale=dict(zip(NAMES, scale)), measurement_error_pct=err)
    print("measurement error (%):", {k: round(v, 1) for k, v in err.items()}, flush=True)
    out = Path(a.weights).parent / f"eval_{Path(a.data).name}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
