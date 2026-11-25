"""Train UWCQ-Mamba on one underwater crack dataset.

  python train.py --data ~/Datasets/UW_CamCrack789 --split split_author_testsel.json --out runs/uwcca

Recipe (all paper runs): AdamW (lr 1e-3, weight decay 1e-4), 5 warm-up epochs then cosine decay, 50 epochs of 402
optimizer steps at an effective batch of 8 (batch 4 x 2 accumulation), 352 x 352 crops (256 on CrackMap), mixed precision,
gradient clipping at 5, EMA of the weights (decay 0.999). The checkpoint with the best selection IoU (scored every 5 epochs
on the split's "val" photos) is kept as best.pth.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from uwcq_mamba.data import CropDataset, split_items
from uwcq_mamba.evaluation import predict_mask, read_image
from uwcq_mamba.losses import TverskyDice, total_loss
from uwcq_mamba.model import UWCQMamba


class EMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.ema, self.decay, self.updates = copy.deepcopy(model).eval(), decay, 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach().to(v.dtype), alpha=1 - d)
            else:
                v.copy_(msd[k])


@torch.no_grad()
def selection_iou(model, items, max_items=1000):
    """Mean per-image IoU over the selection renders that contain a crack."""
    model.eval()
    r = []
    import cv2
    for p, mp in items[::max(1, -(-len(items) // max_items))]:
        gt = cv2.imread(str(mp), 0) >= 127
        if gt.any():
            pr = predict_mask(model, read_image(p))
            tp = (pr & gt).sum()
            r.append(tp / max(tp + (pr & ~gt).sum() + (~pr & gt).sum(), 1))
    model.train()
    return float(np.mean(r))


def worker_init(_):
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder with {train,val,test}_{img,lab} and the split file")
    ap.add_argument("--split", default="split_author_testsel.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--samples", type=int, default=3216, help="crops per epoch (402 steps x 8)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--crop", type=int, default=352)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--sel-items", type=int, default=1000, help="selection renders scored per check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = [(p, m) for p, m in split_items(a.data, a.split, "train")]
    sel = split_items(a.data, a.split, "val")
    ds = CropDataset(train, size=a.crop)
    dl = DataLoader(ds, batch_size=a.batch, sampler=RandomSampler(ds, replacement=True, num_samples=a.samples, generator=g),
                    num_workers=a.workers, worker_init_fn=worker_init, generator=g, pin_memory=dev == "cuda", drop_last=True,
                    persistent_workers=a.workers > 0)
    print(f"{len(train)} training renders, {len(sel)} selection renders", flush=True)

    model = UWCQMamba().to(dev).to(memory_format=torch.channels_last)
    amp = dev == "cuda"
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_ep = a.samples // (a.batch * a.accum)
    total, warm = a.epochs * steps_ep, 5 * steps_ep
    lr_f = lambda s: (s + 1) / warm if s < warm else max(1e-3, 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_f)
    scaler = torch.amp.GradScaler(dev, enabled=amp)
    ema = EMA(model)
    seg_loss = TverskyDice(0.5, 0.5)
    start, best, step = 0, -1.0, 0
    if a.resume and (out / "resume.pth").exists():
        ck = torch.load(out / "resume.pth", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.ema.load_state_dict(ck["ema"]); ema.updates = ck["ema_updates"]
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        if ck["scaler"]:
            scaler.load_state_dict(ck["scaler"])
        start, best, step = ck["epoch"], ck["best"], ck["step"]
        print(f"resumed at epoch {start}", flush=True)

    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    model.train()
    for ep in range(start, a.epochs):
        sums, n, micro, skipped = {}, 0, 0, 0
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            bn = [(t, t.detach().clone()) for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
                  for t in (m.running_mean, m.running_var)]
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                out_all = model.forward_all(b["image"].contiguous(memory_format=torch.channels_last))
            loss, terms = total_loss(model, out_all, b, seg_loss)
            if not torch.isfinite(loss):                  # skip the batch and undo its BatchNorm statistics update
                skipped += 1
                with torch.no_grad():
                    for t, v in bn:
                        t.copy_(v)
                continue
            scaler.scale(loss / a.accum).backward()
            micro += 1
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
            if micro < a.accum:
                continue
            micro = 0
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            ema.update(model)
            step += 1
        rec = dict(epoch=ep + 1, step=step, lr=opt.param_groups[0]["lr"], elapsed_s=round(time.time() - t0, 1), skipped=skipped,
                   **{k: round(v / max(n, 1), 5) for k, v in sums.items()})
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            rec["sel_iou"] = round(selection_iou(ema.ema, sel, a.sel_items), 4)
            if rec["sel_iou"] > best:
                best = rec["sel_iou"]
                torch.save(dict(ema=ema.ema.state_dict(), epoch=ep + 1, sel_iou=best), out / "best.pth")
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        torch.save(dict(model=model.state_dict(), ema=ema.ema.state_dict(), ema_updates=ema.updates, opt=opt.state_dict(),
                        sched=sched.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1, step=step, best=best), out / "resume.pth")
    torch.save(dict(ema=ema.ema.state_dict(), epoch=a.epochs), out / "last.pth")


if __name__ == "__main__":
    main()
