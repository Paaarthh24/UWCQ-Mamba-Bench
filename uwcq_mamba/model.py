"""UWCQ-Mamba: lightweight underwater crack segmentation and quantification network (1.79 M parameters).

stem + stride-1 detail path -> VSS stages at H/4, H/8, H/16 (one selective scan shared over 8 raster routes)
-> fusion at H/2 -> 7 head channels (crack logits x2, signed distance, centreline, log-width, orientation x2)
-> full-resolution refinement and plain skip. The node head is used by the training loss only.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from . import kernels as RK
from .scan import selective_scan_cuda_oflex, selective_scan_fn

_ROUTE_IDX = {}


def input_norm(x):
    """Per-image normalisation: 0.45 + 0.085 (x - mean) / sd, sd of the 8x8-averaged deviation."""
    xf = x.float()
    d = xf - xf.mean((2, 3), keepdim=True)
    sd = F.avg_pool2d(d, 8).pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(0.01)
    return (0.45 + 0.085 * d / sd).to(x.dtype)


class LN2d(nn.Module):
    """LayerNorm over the channels of a (B, C, H, W) map."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w, self.b, self.eps = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c)), eps

    fast = False

    def forward(self, x):
        if _fast(self, x):
            return RK.ln2d(x, self.w, self.b, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) / torch.sqrt(s + self.eps) * self.w[:, None, None] + self.b[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, c, k=7, mlp=2):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)
        self.ln = LN2d(c)
        self.pw1, self.pw2 = nn.Conv2d(c, mlp * c, 1), nn.Conv2d(mlp * c, c, 1)
        self.gamma = nn.Parameter(torch.full((c,), 0.1))

    fast = False

    def forward(self, x):
        return _res(self, x, self.gamma, self.pw2(F.gelu(self.pw1(self.ln(self.dw(x))))))


class DWSepRes(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.BatchNorm2d(c),
                               nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())

    def forward(self, x):
        return x + self.f(x)


def _fast(m, *ts):
    """True when module m may use the Triton inference kernels on tensors ts."""
    return (m.fast and RK.available and not m.training and not torch.is_grad_enabled() and torch.is_autocast_enabled("cuda")
            and all(t.is_cuda and t.is_contiguous() and t.dtype in (torch.float16, torch.float32) for t in ts))


def _res(m, x, g, y):
    return RK.residual(x, g, y) if _fast(m, x, y) and x.shape == y.shape else x + g[:, None, None] * y


def _routes(x):
    """(B, C, H, W) -> 8 views whose row-major flattening gives the scans from each corner along rows and columns."""
    out = []
    for tr in (False, True):
        xt = x.transpose(2, 3) if tr else x
        for fh in (False, True):
            for fw in (False, True):
                v = xt.flip(2) if fh else xt
                out.append(v.flip(3) if fw else v)
    return out


def _unroute(ys):
    """Inverse of _routes for 8 maps -> their mean in the original orientation."""
    acc, i = 0, 0
    for tr in (False, True):
        for fh in (False, True):
            for fw in (False, True):
                v = ys[i]
                i += 1
                v = v.flip(3) if fw else v
                v = v.flip(2) if fh else v
                acc = acc + (v.transpose(2, 3) if tr else v)
    return acc / 8


def _route_index(H, W, device, dtype=torch.int64):
    """Gather indices (8, H*W) of the routes in the flattened map, and their inverses (cached per shape)."""
    key = (H, W, str(device), dtype)
    if key not in _ROUTE_IDX:
        g = torch.arange(H * W, device=device).view(1, 1, H, W)
        fwd = torch.stack([v.reshape(-1) for v in _routes(g)])
        _ROUTE_IDX[key] = fwd.to(dtype), fwd.argsort(1).to(dtype)
    return _ROUTE_IDX[key]


def _scan_len(L):
    """Sequence length padded to whole chunks of the CUDA scan kernel (chunk 128 ... 1024 up to L = 1024, then 2048), which
    takes its faster even-length path; the outputs at positions < L are unchanged (causal scan, same chunking)."""
    if selective_scan_cuda_oflex is None:
        return L
    for c in (128, 256, 512, 1024):
        if L <= c:
            return c
    return -(-L // 2048) * 2048


def _gather(t, idx):
    """t (B, C, L) any strides, idx (8, L) -> (8B, C, L) with route k of t[b] at row k * B + b."""
    B, C, L = t.shape
    return torch.gather(t[None].expand(8, B, C, L), 3, idx[:, None, None].expand(8, B, C, L)).view(8 * B, C, L)


ROUTE_GROUP = 1          # routes scanned together at inference (1, 2, 4 or 8): fewer = less memory, same outputs


class SS2D8(nn.Module):
    """One Mamba selective scan with shared parameters over the 8 corner-raster routes of a 2-D map."""

    def __init__(self, c, d_state=8, expand=2, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        di = expand * c
        self.di, self.n, self.r = di, d_state, max(4, math.ceil(c / 16))
        self.in_proj = nn.Conv2d(c, 2 * di, 1, bias=False)
        self.dw = nn.Conv2d(di, di, 3, padding=1, groups=di)
        self.x_proj = nn.Linear(di, self.r + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.r, di, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -self.r ** -0.5, self.r ** -0.5)
        dt = torch.exp(torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out_norm = LN2d(di)
        self.out_proj = nn.Conv2d(di, c, 1, bias=False)

    fast = False

    def forward(self, x):
        if self.fast and not self.training and not torch.is_grad_enabled():
            return self._forward_fast(x)
        B, _, H, W = x.shape
        xi, z = self.in_proj(x).chunk(2, 1)
        xi = F.silu(self.dw(xi))
        views = _routes(xi)
        shapes = [v.shape[2:] for v in views]
        rt = lambda t: torch.cat([v.reshape(B, t.shape[1], -1) for v in _routes(t)], 0)       # (8B, C, L)
        u = rt(xi).float()
        dt, Bm, Cm = torch.split(self.x_proj(xi.flatten(2).transpose(1, 2).float()), [self.r, self.n, self.n], -1)
        mp = lambda t: t.transpose(1, 2).reshape(B, -1, H, W)
        delta = rt(mp(F.linear(dt, self.dt_proj.weight)))
        Bm, Cm = rt(mp(Bm)).unsqueeze(1), rt(mp(Cm)).unsqueeze(1)
        y = selective_scan_fn(u.contiguous(), delta.float().contiguous(), -torch.exp(self.A_log.float()),
                              Bm.float().contiguous(), Cm.float().contiguous(), self.D.float(), self.dt_proj.bias.float())
        y = _unroute([yy.reshape(B, self.di, *s) for yy, s in zip(y.chunk(8, 0), shapes)]).to(x.dtype)
        return self.out_proj(self.out_norm(y) * F.silu(z))

    def _forward_fast(self, x):
        """Inference: the same values as forward. Each route stack is one gather instead of flip / cat copies. With Triton
        the fp16 stacks stay fp16 (the scan kernel converts them to float32 exactly, as .float() did) and are padded to
        whole scan chunks (_scan_len), and the route mean, out_norm and gate are fused kernels."""
        B, _, H, W = x.shape
        fwd, inv = _route_index(H, W, x.device)
        xi, z = self.in_proj(x).chunk(2, 1)
        xi = F.silu(self.dw(xi))
        xt = xi.flatten(2).transpose(1, 2)
        xt = xt if torch.is_autocast_enabled("cuda") and xt.dtype == torch.float16 else xt.float()      # autocast casts back
        dt, Bm, Cm = torch.split(self.x_proj(xt), [self.r, self.n, self.n], -1)
        dl = F.linear(dt, self.dt_proj.weight).transpose(1, 2)
        A = -torch.exp(self.A_log.float())
        if RK.available and x.is_cuda:
            f32, i32 = _route_index(H, W, x.device, torch.int32)
            src = (xi.flatten(2), dl, Bm.transpose(1, 2), Cm.transpose(1, 2))
            ty = torch.float16 if all(t.dtype == torch.float16 for t in src) else torch.float32    # the scan reads fp16 exactly
            G = ROUTE_GROUP
            out = torch.empty(B, self.di, H * W, device=x.device, dtype=x.dtype)
            acc = torch.empty(B, self.di, H * W, device=x.device, dtype=torch.float32) if G < 8 else out
            for k0 in range(0, 8, G):
                u, delta, Bm, Cm = (RK.route(t, f32[k0:k0 + G], ty, _scan_len(H * W)) for t in src)
                y = selective_scan_fn(u, delta, A, Bm.unsqueeze(1), Cm.unsqueeze(1), self.D.float(), self.dt_proj.bias.float())
                del u, delta, Bm, Cm
                RK.unroute_acc(y, i32[k0:k0 + G], acc, out, k0 == 0, k0 + G == 8)
                del y
            y = out.view(B, self.di, H, W)
            if torch.is_autocast_enabled("cuda") and y.dtype in (torch.float16, torch.float32):
                n = self.out_norm
                return self.out_proj(RK.ln2d(y, n.w, n.b, n.eps, gate=F.silu(z), out_dtype=torch.float16))
        else:
            u, delta = _gather(xi.flatten(2), fwd).float(), _gather(dl, fwd).float()
            Bm, Cm = _gather(Bm.transpose(1, 2), fwd).unsqueeze(1).float(), _gather(Cm.transpose(1, 2), fwd).unsqueeze(1).float()
            y = selective_scan_fn(u, delta, A, Bm, Cm, self.D.float(), self.dt_proj.bias.float()).view(8, B * self.di, -1)
            y = torch.gather(y, 2, inv[:, None].expand(y.shape)).view(8, B, self.di, H, W)
            acc = 0 + y[0]
            for k in range(1, 8):
                acc = acc + y[k]
            y = (acc / 8).to(x.dtype)
        return self.out_proj(self.out_norm(y) * F.silu(z))


class VSSBlock(nn.Module):
    def __init__(self, c, d_state=8, mlp=2):
        super().__init__()
        self.ln1, self.ssm = LN2d(c), SS2D8(c, d_state)
        self.ln2 = LN2d(c)
        self.ffn = nn.Sequential(nn.Conv2d(c, mlp * c, 1), nn.Conv2d(mlp * c, mlp * c, 3, padding=1, groups=mlp * c),
                                 nn.GELU(), nn.Conv2d(mlp * c, c, 1))
        self.g1, self.g2 = nn.Parameter(torch.full((c,), 0.1)), nn.Parameter(torch.full((c,), 0.1))

    fast = False

    def forward(self, x):
        x = _res(self, x, self.g1, self.ssm(self.ln1(x)))
        return _res(self, x, self.g2, self.ffn(self.ln2(x)))


def _down(i, o):
    return nn.Sequential(LN2d(i), nn.Conv2d(i, o, 3, 2, 1))


class FullResSkip(nn.Module):
    """Plain full-resolution skip: the H/2 fused map (1x1 to 16 ch) is upsampled bilinearly, fused with the stride-1 detail
    features (DW 3x3 + PW), and a zero-initialised 1x1 adds a correction to the 7 head channels."""

    def __init__(self, c1=16, c2=64, cq=16):
        super().__init__()
        self.q = nn.Conv2d(c2, cq, 1)
        c = cq + c1
        self.fuse1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.BatchNorm2d(c),
                                   nn.Conv2d(c, 32, 1, bias=False), nn.BatchNorm2d(32), nn.SiLU())
        self.head1 = nn.Conv2d(32, 7, 1)
        nn.init.zeros_(self.head1.weight)
        nn.init.zeros_(self.head1.bias)

    def _forward(self, g2, f1):
        q = F.interpolate(self.q(g2), size=f1.shape[2:], mode="bilinear", align_corners=False)
        if not torch.is_grad_enabled() and torch.is_autocast_enabled("cuda") and q.is_cuda:
            f1h = f1 if f1.dtype == torch.float16 else f1.to(q.dtype).half()   # fp16 -> fp32 -> fp16 is the identity
            c = torch.cat([q.half(), f1h], 1)                          # the conv casts the float32 concatenation to fp16
            del f1h
            del q
            t = self.fuse1[0](c)
            del c
            for layer in self.fuse1[1:]:
                t = layer(t)
            return self.head1(t)
        return self.head1(self.fuse1(torch.cat([q, f1.to(q.dtype)], 1)))

    def forward(self, g2, f1):
        if torch.is_grad_enabled():                       # recompute in backward: saves full-resolution activations
            return torch.utils.checkpoint.checkpoint(self._forward, g2, f1, use_reentrant=False)
        return self._forward(g2, f1)


def sample(feat, bidx, pts):
    """feat (B, C, H, W), bidx (N,), pts (N, P, 2) as (x, y) pixels -> (N, C, P), bilinear."""
    B, C, H, W = feat.shape
    g = torch.stack([2 * pts[..., 0] / max(W - 1, 1) - 1, 2 * pts[..., 1] / max(H - 1, 1) - 1], -1)
    out = feat.new_zeros(pts.shape[0], C, pts.shape[1])
    for b in range(B):
        m = bidx == b
        if m.any():
            s = F.grid_sample(feat[b:b + 1].float(), g[m][None].float(), mode="bilinear", align_corners=True)
            out[m] = s[0].permute(1, 0, 2).to(out.dtype)
    return out


class NodeHead(nn.Module):
    """Training-only head: at ground-truth skeleton nodes, predicts the crack width and the centre offset along the normal
    from a 17-point cross-section of the H/2 decoder map."""

    def __init__(self, c1=64, c2=64, k=17, span=12.0, hid=128):
        super().__init__()
        self.k, self.span = k, span
        self.prof = nn.Sequential(nn.Conv1d(c1, c1, 3, padding=1), nn.GELU(), nn.Conv1d(c1, c1, 3, padding=1), nn.GELU())
        self.prof_out = nn.Linear(c1 * k, 96)
        self.mlp = nn.Sequential(nn.Linear(96 + c2, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 3))
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1)
            self.mlp[-1].bias.copy_(torch.tensor([math.log(math.e ** 2 - 1), 0.0, 0.0]))

    def forward(self, f1, f2, bidx, ctr, nrm, off):
        nu = torch.linspace(-self.span, self.span, self.k, device=ctr.device)
        pts = ctr[:, None, :] + nu[None, :, None] * nrm[:, None, :]
        to2 = lambda t: (t + off.to(t.dtype) + 0.5) / 2 - 0.5          # cropped stride-1 px -> padded stride-2 px
        p = self.prof(sample(f1, bidx, to2(pts)).float())
        q = sample(f2, bidx, to2(ctr)[:, None, :])[..., 0].float()
        o = self.mlp(torch.cat([self.prof_out(p.flatten(1)), q], 1))
        return F.softplus(o[:, 0]), 4.0 * torch.tanh(o[:, 1]), o[:, 2]


class UWCQMamba(nn.Module):
    def __init__(self):
        super().__init__()
        c1, c2, c4, c8, c16 = 16, 24, 48, 96, 192
        self.stem2 = nn.Sequential(nn.Conv2d(3, c2, 3, 2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(),
                                   nn.Conv2d(c2, c2, 3, 1, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU())
        self.det1 = nn.Sequential(nn.Conv2d(3, c1, 3, 1, 1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(), DWSepRes(c1), DWSepRes(c1))
        self.s2 = nn.Sequential(ConvNeXtBlock(c2), ConvNeXtBlock(c2))
        self.d4, self.s4 = _down(c2, c4), nn.Sequential(*[VSSBlock(c4) for _ in range(2)])
        self.d8, self.s8 = _down(c4, c8), nn.Sequential(*[VSSBlock(c8) for _ in range(4)])
        self.d16, self.s16 = _down(c8, c16), nn.Sequential(*[VSSBlock(c16) for _ in range(2)])
        self.p4, self.p8, self.p16 = (nn.Conv2d(c, 48, 1) for c in (c4, c8, c16))
        self.fuse2 = nn.Sequential(nn.Conv2d(3 * 48 + c2, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.SiLU())
        self.heads = nn.Conv2d(64, 7, 1)          # crack logits (2) | sdf | centreline | log width | orientation (2)
        self.refine1 = nn.Sequential(nn.Conv2d(c1 + 7, c1 + 7, 3, padding=1, groups=c1 + 7), nn.Conv2d(c1 + 7, 16, 1), nn.SiLU(),
                                     nn.Conv2d(16, 7, 1))
        nn.init.zeros_(self.refine1[-1].weight)
        nn.init.zeros_(self.refine1[-1].bias)
        with torch.no_grad():
            self.heads.weight.mul_(0.1)
            self.heads.bias.zero_()
            self.heads.bias[2], self.heads.bias[3], self.heads.bias[4] = 2.0, -4.0, math.log(3.0)
        self.s1skip = FullResSkip(c1, 64)
        self.node = NodeHead(64, 64)

    def forward_all(self, x):
        """x (B, 3, H, W) RGB in [0, 1] -> dict of logits (B, 2, H, W) and the geometry fields (+ decoder map in training)."""
        H, W = x.shape[2:]
        x = input_norm(x)
        ph, pw = (-H) % 16, (-W) % 16
        top, left = ph // 2, pw // 2
        if ph or pw:
            x = F.pad(x, (left, pw - left, top, ph - top), mode="reflect")
        f1 = self.det1(x)
        f2 = self.s2(self.stem2(x))
        f4 = self.s4(self.d4(f2))
        f8 = self.s8(self.d8(f4))
        f16 = self.s16(self.d16(f8))
        up = lambda t, s: F.interpolate(t, size=s, mode="bilinear", align_corners=False)
        s2 = f2.shape[2:]
        lean = not torch.is_grad_enabled() and torch.is_autocast_enabled("cuda") and x.is_cuda
        h16 = (lambda t: t.half()) if lean else (lambda t: t)   # autocast casts the concatenation to fp16: same values
        a4 = h16(up(self.p4(f4), s2))
        del f4
        a8 = h16(up(self.p8(f8), s2))
        del f8
        a16 = h16(up(self.p16(f16), s2))
        del f16
        c = torch.cat([a4, a8, a16, h16(f2)], 1)
        del a4, a8, a16, f2
        g2 = self.fuse2(c)
        del c
        h = up(self.heads(g2), x.shape[2:])
        h = h + self.refine1(torch.cat([f1, h.to(f1.dtype)], 1))
        h = h + self.s1skip(g2, f1).to(h.dtype)
        h = h[..., top:top + H, left:left + W]
        crack = h[:, 1:2] - h[:, :1]
        out = dict(logits=torch.cat([torch.zeros_like(crack), crack], 1), sdf=h[:, 2:3], cl=h[:, 3:4], logw=h[:, 4:5], ori=h[:, 5:7])
        if self.training:
            out["feats"] = dict(g2=g2, off=torch.tensor([float(left), float(top)], device=g2.device))
        return out

    def forward(self, x):
        return self.forward_all(x)["logits"]


def build(weights=None, device="cpu"):
    """UWCQ-Mamba, optionally loaded from a checkpoint (its EMA weights)."""
    m = UWCQMamba()
    if weights:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        m.load_state_dict(ck.get("ema", ck))
    return m.to(device)


def prepare_inference(model):
    """In place, for inference only: drop the training-only node head, freeze the parameters, store the conv / linear
    weights that fp16 autocast casts anyway in fp16 (LN / BN, layer scales and scan parameters stay fp32), route the scans
    with precomputed gathers and switch LN2d and the residual updates to fused kernels (uwcq_mamba/kernels.py). Outputs
    under fp16 autocast are bit-identical to the unprepared model (tests/test_exact.py)."""
    model.eval()
    if hasattr(model, "node"):
        del model.node
    for p in model.parameters():
        p.requires_grad_(False)
    cuda = next(model.parameters()).is_cuda
    ssm = [m for m in model.modules() if isinstance(m, SS2D8)]
    for m in model.modules():
        if isinstance(m, (SS2D8, LN2d, VSSBlock, ConvNeXtBlock)):
            m.fast = True
    keep32 = {id(m.dt_proj.bias) for m in ssm}                  # read with .float() by the scan
    for m in model.modules():
        if cuda and isinstance(m, (nn.Conv2d, nn.Linear)):
            for p in (m.weight, m.bias):
                if p is not None and id(p) not in keep32:
                    p.data = p.data.half()
    return model


class GraphRunner(nn.Module):
    """model(x) replayed from a CUDA graph captured once per input shape under fp16 autocast and no_grad (the inference
    protocol); any other call runs the model eagerly. Keeps the graphs of the last `max_graphs` shapes captured."""

    def __init__(self, model, max_graphs=4):
        super().__init__()
        self.model, self.max_graphs, self.graphs = model, max_graphs, {}
        for p in model.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        if not (x.is_cuda and not torch.is_grad_enabled() and torch.is_autocast_enabled("cuda")
                and torch.get_autocast_dtype("cuda") == torch.float16 and not self.model.training):
            return self.model(x)
        key = (tuple(x.shape), x.dtype, x.device)
        if key not in self.graphs:
            if len(self.graphs) >= self.max_graphs:
                self.graphs.pop(next(iter(self.graphs)))
            self.graphs[key] = self._capture(x)
        g, xs, out = self.graphs[key]
        xs.copy_(x)
        g.replay()
        return out.clone()

    def _capture(self, x):
        xs = x.clone()
        s = torch.cuda.Stream(x.device)
        s.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self.model(xs)
        torch.cuda.current_stream(x.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.model(xs)
        return g, xs, out


def build_inference(weights, device="cuda", graphs=True):
    """build() + prepare_inference(), wrapped in a GraphRunner on CUDA."""
    m = prepare_inference(build(weights, device).eval())
    return GraphRunner(m) if graphs and torch.device(device).type == "cuda" else m
