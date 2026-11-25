"""Inference latency of UWCQ-Mamba: batch 1, fp16 autocast, no_grad, median of 200 runs after 50 warm-up runs, with a CUDA
synchronize before and after each run; the variants take turns in blocks of 10 runs. Variants are cumulative:

  release   build(weights).eval(), as in the original release
  frozen    + prepare_inference() with its kernels off: node head dropped, parameters frozen, fp16 conv / linear weights
  routed    + SS2D8 inference path: route gathers (fp16, padded to whole scan chunks), fused route mean + out_norm + gate
  fused     + fused kernels for the other LN2d and the residual updates (prepare_inference() as shipped)
  graph     + CUDA graph replay per input shape (GraphRunner; build_inference() as shipped)

  python benchmark.py --weights best.pth [--sizes 512x512,544x384,480x640,256x256,360x640]
"""
import argparse
import time

import torch

from uwcq_mamba.model import LN2d, SS2D8, ConvNeXtBlock, GraphRunner, VSSBlock, build, prepare_inference


def variant(name, weights):
    m = build(weights, "cuda").eval()
    if name == "release":
        return m
    prepare_inference(m)
    level = ["frozen", "routed", "fused", "graph"].index(name)
    for s in m.modules():
        if isinstance(s, SS2D8):
            s.fast = level >= 1
        elif isinstance(s, (LN2d, VSSBlock, ConvNeXtBlock)):
            s.fast = level >= 2
    return GraphRunner(m) if name == "graph" else m


@torch.no_grad()
def latencies(models, h, w, warmup=50, runs=200, rounds=20):
    """Median ms per model. The models take turns in `rounds` blocks of runs / rounds calls, so that load from other GPU
    processes falls on all of them alike."""
    x = torch.rand(1, 3, h, w, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    ts = [[] for _ in models]
    with torch.autocast("cuda", dtype=torch.float16):
        for m in models:
            for _ in range(warmup):
                m(x)
        for _ in range(rounds):
            for m, t in zip(models, ts):
                for _ in range(runs // rounds):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    m(x)
                    torch.cuda.synchronize()
                    t.append(1000 * (time.perf_counter() - t0))
    return [sorted(t)[len(t) // 2] for t in ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sizes", default="512x512,544x384,480x640,256x256,360x640", help="HxW list")
    ap.add_argument("--variants", default="release,frozen,routed,fused,graph")
    a = ap.parse_args()
    sizes = [tuple(map(int, s.split("x"))) for s in a.sizes.split(",")]
    names = a.variants.split(",")
    models = [variant(v, a.weights) for v in names]
    res = [latencies(models, h, w) for h, w in sizes]
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}; median ms / FPS")
    print(f"{'variant':<10}" + "".join(f"{f'{h}x{w}':>17}" for h, w in sizes))
    for i, v in enumerate(names):
        print(f"{v:<10}" + "".join(f"{f'{r[i]:.2f} / {1000 / r[i]:.1f}':>17}" for r in res))
    print(f"{'speed-up':<10}" + "".join(f"{f'x{r[0] / r[-1]:.2f}':>17}" for r in res))


if __name__ == "__main__":
    main()
