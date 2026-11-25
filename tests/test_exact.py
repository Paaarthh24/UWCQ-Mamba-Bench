"""Bit-identity of the optimised inference (build_inference: prepare_inference + CUDA graph replay) against the release
inference (build(weights).eval() run eagerly), both through evaluation.predict_prob (fp16 autocast, batch 1, native size).
Per test render it compares the crack mask (prob >= 0.5) and the saved author-protocol map uint8(rint(255 prob / max prob)),
and also counts renders whose float32 probabilities differ in any bit. Exits non-zero on any mask or map mismatch.

  python tests/test_exact.py                       # all of UW-CrackMap and UW-DeepCrack, 1000 renders of the others
  python tests/test_exact.py --n 50                # quick: 50 evenly spaced renders per dataset
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uwcq_mamba.data import split_items  # noqa: E402
from uwcq_mamba.evaluation import predict_prob, read_image  # noqa: E402
from uwcq_mamba.model import build, build_inference  # noqa: E402

DATASETS = [("UW_CrackMap", "split_author_testsel.json", "uwcma", 0), ("UW_DeepCrack", "split_group_7_1_2.json", "uwdc", 0),
            ("UW_CamCrack789", "split_author_testsel.json", "uwcca", 1000), ("UW_Crack500", "split_author_testsel.json", "uwc5a", 1000)]


def saved_map(prob):
    m = prob.max()
    return np.clip(np.rint(255.0 * prob / m), 0, 255).astype(np.uint8) if m > 0 else np.zeros(prob.shape, np.uint8)


def even(items, n):
    return items if not n or n >= len(items) else [items[i] for i in np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets")
    ap.add_argument("--weights", default="weights/{}.pth",
                    help="checkpoint path with {} for the dataset tag (uwdc, uwcca, uwcma, uwc5a)")
    ap.add_argument("--n", type=int, default=None, help="renders per dataset (default: all / all / 1000 / 1000)")
    ap.add_argument("--datasets", default=",".join(d[0] for d in DATASETS))
    a = ap.parse_args()
    failed = False
    for name, split, tag, n in DATASETS:
        if name not in a.datasets.split(","):
            continue
        items = even(split_items(Path(a.data_root) / name, split, "test"), a.n if a.n is not None else n)
        ref = build(a.weights.format(tag), "cuda").eval()
        opt = build_inference(a.weights.format(tag), "cuda")
        bad_mask = bad_map = bad_bits = px = 0
        for p, _ in items:
            x = read_image(p)
            pa, pb = predict_prob(ref, x), predict_prob(opt, x)
            ma, mb = pa >= 0.5, pb >= 0.5
            bad_mask += int(not np.array_equal(ma, mb))
            bad_map += int(not np.array_equal(saved_map(pa), saved_map(pb)))
            bad_bits += int(not np.array_equal(pa.view(np.uint32), pb.view(np.uint32)))
            px += pa.size
        ok = bad_mask == 0 and bad_map == 0
        failed |= not ok
        print(f"{name}: {len(items)} renders, {px} px | mask mismatches {bad_mask} | saved-map mismatches {bad_map} | "
              f"renders with any prob bit different {bad_bits} | {'PASS' if ok else 'FAIL'}", flush=True)
        del ref, opt
        torch.cuda.empty_cache()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
