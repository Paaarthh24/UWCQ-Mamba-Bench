"""Generate UWCQ-Bench: underwater renders of a public crack dataset (DeepCrack, CamCrack789, CrackMap or Crack500).

Every source photograph is rendered under the 10 Jerlov water types (I, IA, IB, II, III, 1C, 3C, 5C, 7C, 9C) at 3 water
depths (shallow 1-5 m, medium 5-15 m, deep 15-30 m), 30 renders per photograph. Image formation follows RSUIGM, with the
camera distance from Depth Anything V2 (smooth quadratic fit, crack pixels excluded) and patchy turbidity, caustics,
an ROV lamp, forward-scatter blur and a simulated camera (noise, white balance, JPEG). Masks are the original labels
binarised at > 127; no step moves, warps or crops a pixel, so all 30 renders of a photograph share its mask.

Input:  <root>/<dataset>/{train,val,test}_img and _lab (MixerCSeg layout; CamCrack789 pairs image-NN with target-NN).
Output: <out-root>/UW_<dataset>/{split}_img/<stem>_<jerlov>_<depth>.jpg and {split}_lab/<same>.png (0/255), the same
splits as the source, params/<split>/<stem>.json with every sampled parameter, manifest.csv and summary.json.
Deterministic: rng = default_rng([seed, i]), i = index in the sorted (split, image) list of the dataset.

Stages (each resumes where it stopped):
  python generate_benchmark.py depth  --root /path/to/datasets --dataset CrackMap        # GPU (Depth Anything V2)
  python generate_benchmark.py render --root /path/to/datasets --dataset CrackMap --workers 10
  python generate_benchmark.py index  --root /path/to/datasets --dataset CrackMap
The render stage stops cleanly when the disk has less than --min-free-gb left (resume later).
"""
import argparse, csv, json, os, shutil, sys, time
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uwcq_bench.depth_distance import quad_fit_coef, quad_eval, distance_from_disparity
from uwcq_bench.uw_synthesis_rsuigm import JERLOV, JERLOV_TYPES, synthesize_rsuigm
from uwcq_bench.uw_synthesis_v3 import DEPTH_CATEGORIES, sample_effects

ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["depth", "render", "index"])
ap.add_argument("--root", required=True, help="folder that holds the source datasets, e.g. /data/cracks")
ap.add_argument("--dataset", required=True, help="source dataset folder under --root, e.g. CrackMap")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--min-free-gb", type=float, default=5.0)
ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
ap.add_argument("--out-root", default=None, help="where UW_<dataset> is written (default: --root)")
ap.add_argument("--limit", type=int, default=0, help="first N sources only (smoke tests)")
args = ap.parse_args()
ROOT = args.root
SRC, OUT = os.path.join(ROOT, args.dataset), os.path.join(args.out_root or ROOT, "UW_" + args.dataset)
SPLITS = [s for s in ("train", "val", "test") if os.path.isdir(os.path.join(SRC, f"{s}_img"))]
lum = np.array([0.299, 0.587, 0.114])


def label_name(f):
    stem = os.path.splitext(f)[0]
    if args.dataset == "CamCrack789":                      # image-07.png <-> target-07.png
        return "target-" + stem.split("-")[-1] + ".png"
    return stem + ".png"


def sources():
    """Sorted list of (index, split, image file, label file)."""
    out = []
    for split in SPLITS:
        labs = set(os.listdir(os.path.join(SRC, f"{split}_lab")))
        for f in sorted(os.listdir(os.path.join(SRC, f"{split}_img"))):
            if os.path.splitext(f)[1].lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = label_name(f)
            assert m in labs, f"{split}: no label for {f}"
            out.append((split, f, m))
    out = [(i, *t) for i, t in enumerate(out)]
    return out[:args.limit] if args.limit else out


def load(split, f, m):
    img = Image.open(os.path.join(SRC, f"{split}_img", f)).convert("RGB")
    mask = (np.array(Image.open(os.path.join(SRC, f"{split}_lab", m)).convert("L")) > 127).astype(np.uint8) * 255
    return img, mask


def stage_depth():
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "depth_fits.json")
    fits = json.load(open(path)) if os.path.exists(path) else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(dev).eval()
    todo = [s for s in sources() if f"{s[1]}/{s[2]}" not in fits]
    t0 = time.time()
    for n, (i, split, f, m) in enumerate(todo):
        img, mask = load(split, f, m)
        with torch.no_grad():
            pred = model(**proc(images=img, return_tensors="pt").to(dev)).predicted_depth
            disp = torch.nn.functional.interpolate(pred[:, None], size=(img.height, img.width),
                                                   mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        disp = disp.astype(np.float32).astype(np.float64)       # fixed precision path (reproducible renders)
        fits[f"{split}/{f}"] = dict(coef=[float(c) for c in quad_fit_coef(disp, mask > 0)],
                                    h=img.height, w=img.width, model=args.model)
        if (n + 1) % 200 == 0 or n + 1 == len(todo):
            json.dump(fits, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
            print(f"[{args.dataset}] depth {n + 1}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)


def render_one(task):
    i, split, f, m = task
    stem = os.path.splitext(f)[0]
    pdir = os.path.join(OUT, "params", split)
    if os.path.exists(os.path.join(pdir, stem + ".json")):
        return i, 0
    if shutil.disk_usage(OUT).free < args.min_free_gb * 2**30:
        return i, -1
    fit_info = FITS[f"{split}/{f}"]
    img, mask = load(split, f, m)
    assert mask.shape == (img.height, img.width) == (fit_info["h"], fit_info["w"]), f
    fit = quad_eval(fit_info["coef"], img.height, img.width)
    x = np.asarray(img, dtype=np.float64) / 255.0
    ring = ndimage.binary_dilation(mask > 0, iterations=6) & ~(mask > 0)
    has_crack = bool((mask > 0).any() and ring.any())
    c_src = abs((x @ lum)[ring].mean() - (x @ lum)[mask > 0].mean()) if has_crack else 0.0
    mask_png = Image.fromarray(mask)
    rng = np.random.default_rng([args.seed, i])
    records = []
    for cat in DEPTH_CATEGORIES:
        for jt in JERLOV_TYPES:
            fx = sample_effects(cat, rng)
            tau = float(rng.uniform(0.3, 0.8))                          # optical depth at nearest point
            d_near = float(np.clip(tau / np.mean(JERLOV[jt]["c"]), 0.1, fx["d0_m"]))
            fx["d0_m"], fx["near_optical_depth"] = round(d_near, 3), round(tau, 3)
            dist, ratio = distance_from_disparity(fit, d_near)
            u8, info, jpeg = synthesize_rsuigm(x, jt, fx, rng, dist=dist, return_jpeg=True)
            name = f"{stem}_{jt}_{cat}"
            with open(os.path.join(OUT, f"{split}_img", name + ".jpg"), "wb") as fh:
                fh.write(jpeg)
            mask_png.save(os.path.join(OUT, f"{split}_lab", name + ".png"))
            y = u8 / 255.0
            c_syn = abs((y @ lum)[ring].mean() - (y @ lum)[mask > 0].mean()) if has_crack else 0.0
            fx = {k: v for k, v in fx.items() if k not in ("tilt", "angle")}
            records.append(dict(image=f"{split}_img/{name}.jpg", mask=f"{split}_lab/{name}.png", split=split,
                                source=f"{args.dataset}/{split}_img/{f}", source_mask=f"{args.dataset}/{split}_lab/{m}",
                                jerlov_type=jt, depth_category=cat, effects=fx,
                                distance_near_m=round(float(dist.min()), 3), distance_far_m=round(float(dist.max()), 3),
                                far_near_ratio=round(ratio, 3), **info,
                                crack_contrast_retained=round(float(c_syn / c_src), 3) if has_crack and c_src > 0 else None))
    os.makedirs(pdir, exist_ok=True)
    tmp = os.path.join(pdir, stem + ".json.tmp")
    json.dump(dict(source_index=i, seed=[args.seed, i], records=records), open(tmp, "w"))
    os.replace(tmp, os.path.join(pdir, stem + ".json"))         # written last: marks the image as done
    return i, len(records)


def _init(fits):
    global FITS
    FITS = fits


def stage_render():
    fits = json.load(open(os.path.join(OUT, "depth_fits.json")))
    todo = sources()
    missing = [s for s in todo if f"{s[1]}/{s[2]}" not in fits]
    assert not missing, f"{len(missing)} sources have no depth fit: run the depth stage first"
    for split in SPLITS:
        for sub in (f"{split}_img", f"{split}_lab", os.path.join("params", split)):
            os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    t0, done, low = time.time(), 0, 0
    with Pool(args.workers, initializer=_init, initargs=(fits,)) as pool:
        for n, (i, k) in enumerate(pool.imap_unordered(render_one, todo, chunksize=1)):
            if k < 0:
                low += 1; continue
            done += k
            if (n + 1) % 100 == 0 or n + 1 == len(todo):
                el = time.time() - t0
                print(f"[{args.dataset}] render {n + 1}/{len(todo)} sources, {done} new images, {el / 60:.1f} min, "
                      f"eta {(len(todo) - n - 1) * el / (n + 1) / 60:.0f} min", flush=True)
    if low:
        print(f"[{args.dataset}] STOPPED: {low} sources skipped, less than {args.min_free_gb} GB free on disk", flush=True)
        sys.exit(3)


def stage_index():
    srcs = sources()
    rows, missing = [], []
    with open(os.path.join(OUT, "params.jsonl"), "w") as pj:
        for i, split, f, m in srcs:
            p = os.path.join(OUT, "params", split, os.path.splitext(f)[0] + ".json")
            if not os.path.exists(p):
                missing.append(f"{split}/{f}"); continue
            for r in json.load(open(p))["records"]:
                pj.write(json.dumps(r) + "\n")
                rows.append((r["split"], r["image"], r["mask"], r["jerlov_type"], r["depth_category"],
                             r["source"], r["crack_contrast_retained"]))
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "image", "mask", "jerlov_type", "depth_category", "source", "crack_contrast_retained"])
        w.writerows(rows)
    nonexist = [r[1] for r in rows if not os.path.exists(os.path.join(OUT, r[1]))] + \
               [r[2] for r in rows if not os.path.exists(os.path.join(OUT, r[2]))]
    cc = {}
    for r in rows:
        if r[6] is not None:
            cc.setdefault((r[4], r[3]), []).append(r[6])
    summary = dict(dataset=args.dataset, recipe="UWCQ-Bench (RSUIGM x 10 Jerlov x 3 depths), labels > 127",
                   sources=len(srcs), sources_missing=missing, images=len(rows),
                   per_split={s: sum(r[0] == s for r in rows) for s in SPLITS},
                   sources_per_split={s: sum(t[1] == s for t in srcs) for s in SPLITS},
                   files_missing=nonexist[:20], n_files_missing=len(nonexist),
                   median_crack_contrast_retained={f"{d}/{t}": round(float(np.median(v)), 3) for (d, t), v in sorted(cc.items())})
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "median_crack_contrast_retained"}, indent=1))


if __name__ == "__main__":
    dict(depth=stage_depth, render=stage_render, index=stage_index)[args.stage]()
