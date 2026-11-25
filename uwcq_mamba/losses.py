"""Training losses of UWCQ-Mamba: Tversky-Dice segmentation loss plus the auxiliary geometry losses (centreline, width,
orientation, signed distance) and the node width / centre loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import sample


class TverskyDice(nn.Module):
    """Tversky + soft Dice over the two classes; pixels labelled `ignore` (rotation fill, padding) are excluded."""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, ignore=255):
        super().__init__()
        self.alpha, self.beta, self.smooth, self.ignore = alpha, beta, smooth, ignore

    def forward(self, logits, y):
        valid = (y != self.ignore).unsqueeze(1)
        probs = torch.softmax(logits.float(), 1) * valid
        tgt = F.one_hot(torch.where(valid[:, 0], y, 0).long(), 2).permute(0, 3, 1, 2).float() * valid
        dims = (0, 2, 3)
        tp, fp, fn = (probs * tgt).sum(dims), (probs * (1 - tgt)).sum(dims), ((1 - probs) * tgt).sum(dims)
        tv = 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        dc = 1 - (2 * tp + self.smooth) / (probs.sum(dims) + tgt.sum(dims) + self.smooth)
        return tv.mean() + dc.mean()


def _valid(b):
    return (b["mask"] != 255).unsqueeze(1).float()


def sdf_loss(pred, b, scale=20.0):
    """L1 on the signed distance (shifted half a pixel so its zero lies on the mask boundary), weighted towards the edge."""
    t = b["sdf"].unsqueeze(1).float()
    t = t - 0.5 * torch.sign(t)
    w = _valid(b) * (1 + 4 * torch.exp(-t.abs() / 2))
    return (w * (pred.float() - t).abs()).sum() / (w.sum().clamp_min(1) * scale)


def centerline_loss(cl_logit, b, alpha=2.0, beta=4.0):
    """Focal loss towards a Gaussian centreline heat map (positives = skeleton pixels)."""
    v = _valid(b)
    gt = b["cl"].unsqueeze(1)
    pos = (b["skel"].unsqueeze(1) > 0).float() * v
    p = torch.sigmoid(cl_logit.float()).clamp(1e-4, 1 - 1e-4)
    lp = -((1 - p) ** alpha) * torch.log(p) * pos
    ln = -((1 - gt) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos) * v
    return (lp.sum() + ln.sum()) / pos.sum().clamp_min(1)


def width_loss(logw, b):
    g = (b["mask"] == 1).unsqueeze(1).float()
    return ((logw.float() - b["logw"].unsqueeze(1)).abs() * g).sum() / g.sum().clamp_min(1)


def ori_loss(ori, b):
    """1 - cos of the doubled-angle difference on crack pixels, weighted by the structure-tensor coherence."""
    w = (b["mask"] == 1).unsqueeze(1).float() * b["coh"].unsqueeze(1)
    v = ori.float() / ori.float().norm(dim=1, keepdim=True).clamp_min(1e-6)
    cos = (v * b["ori"]).sum(1, keepdim=True)
    return ((1 - cos) * w).sum() / w.sum().clamp_min(1)


@torch.no_grad()
def node_targets(skel, ori, sdf, max_nodes=256, jitter=1.5, w_max=24.0, smooth_amp=3.0):
    """Ground-truth skeleton nodes -> (batch index, centre (x, y), unit normal, width, centre offset). The walls are the
    first zero crossings of the signed distance along -n and +n; each crop's nodes are shifted along the normal by a
    smooth random field so the head learns to recentre."""
    dev = skel.device
    B, H, W = skel.shape
    bl, cl, nl = [], [], []
    for b in range(B):
        ys, xs = torch.nonzero(skel[b] > 0, as_tuple=True)
        if ys.numel() == 0:
            continue
        if ys.numel() > max_nodes:
            sel = torch.randperm(ys.numel(), device=dev)[:max_nodes]
            ys, xs = ys[sel], xs[sel]
        c2, s2 = ori[b, 0, ys, xs].float(), ori[b, 1, ys, xs].float()
        ok = (c2.abs() + s2.abs()) > 1e-3
        ys, xs, c2, s2 = ys[ok], xs[ok], c2[ok], s2[ok]
        th = 0.5 * torch.atan2(s2, c2)
        bl.append(torch.full_like(ys, b))
        cl.append(torch.stack([xs, ys], 1).float())
        nl.append(torch.stack([-torch.sin(th), torch.cos(th)], 1))
    if not bl:
        return None
    bidx, ctr, nrm = torch.cat(bl), torch.cat(cl), torch.cat(nl)
    t = sdf.float().unsqueeze(1)
    t = t - 0.5 * torch.sign(t)
    s = torch.arange(0, 20.0001, 0.25, device=dev)
    walls = []
    for sgn in (-1.0, 1.0):
        prof = sample(t, bidx, ctr[:, None, :] + sgn * s[None, :, None] * nrm[:, None, :])[:, 0]
        out = prof >= 0
        first = torch.where(out.any(1), out.float().argmax(1), torch.full_like(bidx, -1))
        i1 = first.clamp_min(1)
        i0 = i1 - 1
        p0, p1 = prof.gather(1, i0[:, None])[:, 0], prof.gather(1, i1[:, None])[:, 0]
        d = s[i0] + (-p0 / (p1 - p0).clamp_min(1e-6)).clamp(0, 1) * 0.25
        walls.append(torch.where((first > 0) & (prof[:, 0] < 0), d, torch.full_like(d, float("nan"))))
    l_gt, r_gt = walls
    w_gt = l_gt + r_gt
    keep = torch.isfinite(w_gt) & (w_gt <= w_max)
    if not keep.any():
        return None
    bidx, ctr, nrm, l_gt, r_gt, w_gt = bidx[keep], ctr[keep], nrm[keep], l_gt[keep], r_gt[keep], w_gt[keep]
    pix = ctr.round().long()
    lo = torch.rand(B, 1, 6, 6, device=dev) * 2 - 1
    on = (torch.rand(B, device=dev) < 0.5).float()[:, None, None, None]
    field = F.interpolate(lo * on, size=(H, W), mode="bicubic", align_corners=True).clamp(-1, 1)[:, 0]
    j = smooth_amp * field[bidx, pix[:, 1].clamp(0, H - 1), pix[:, 0].clamp(0, W - 1)] if jitter > 0 else torch.zeros_like(w_gt)
    return bidx, ctr + j[:, None] * nrm, nrm, w_gt, 0.5 * (r_gt - l_gt) - j


def node_loss(model, out, b):
    tg = node_targets(b["skel"], b["ori"], b["sdf"])
    if tg is None:
        z = out["logits"].float().sum() * 0.0
        return dict(node_w=z, node_d=z)
    bidx, ctr, nrm, w_gt, d_gt = tg
    f = out["feats"]
    w, d, _ = model.node(f["g2"], f["g2"], bidx, ctr, nrm, f["off"])
    return dict(node_w=((w - w_gt).abs() / (w_gt + 1)).mean(), node_d=((d - d_gt).abs() / 2).mean())


def total_loss(model, out, b, seg_loss):
    """Loss of one batch: 0.5 x Tversky-Dice + centreline + width + orientation + signed distance + node terms (all weight 1)."""
    terms = dict(seg=0.5 * seg_loss(out["logits"].float(), b["mask"]),
                 q_cl=centerline_loss(out["cl"], b), q_w=width_loss(out["logw"], b), q_ori=ori_loss(out["ori"], b),
                 sdf=sdf_loss(out["sdf"], b), **node_loss(model, out, b))
    return sum(terms.values()), terms
