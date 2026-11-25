"""Triton inference kernels, each bit-identical to the PyTorch ops it replaces: the route stacks of SS2D8 (pure copies),
the route mean (same fp32 sum order) and the elementwise parts of LN2d under fp16 autocast (same op order and rounding,
IEEE division / square root, no FMA contraction). `available` is False when Triton is not installed."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

available = triton is not None

if available:
    @triton.jit
    def _store(ptr, v, mask):
        """Store float32 v, rounded to nearest-even when ptr is float16 (Triton's own cast differs from PyTorch on ties)."""
        if ptr.dtype.element_ty == tl.float16:
            v = tl.inline_asm_elementwise("cvt.rn.f16.f32 $0, $1;", "=h,r", [v], dtype=tl.float16, is_pure=True, pack=1)
        tl.store(ptr, v, mask=mask)

    @triton.jit
    def _route_kernel(src, idx, out, B, C, L, Lp, sb, sc, sl, BC: tl.constexpr, BL: tl.constexpr):
        pl, pc, pkb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        k, b = pkb // B, pkb % B
        lo = pl * BL + tl.arange(0, BL)
        co = pc * BC + tl.arange(0, BC)
        ml, mc = lo < L, co < C
        p = tl.load(idx + k * L + lo, mask=ml, other=0)
        v = tl.load(src + b * sb + co[:, None] * sc + p[None, :] * sl, mask=mc[:, None] & ml[None, :], other=0.0)
        tl.store(out + (pkb * C + co[:, None]) * Lp + lo[None, :], v.to(out.dtype.element_ty), mask=mc[:, None] & (lo < Lp)[None, :])

    @triton.jit
    def _unroute_kernel(y, inv, out, R, L, Ly, BL: tl.constexpr):
        pl, r = tl.program_id(0), tl.program_id(1)
        lo = pl * BL + tl.arange(0, BL)
        ml = lo < L
        acc = tl.zeros([BL], dtype=tl.float32)
        for k in tl.static_range(8):
            p = tl.load(inv + k * L + lo, mask=ml, other=0)
            acc = acc + tl.load(y + (k * R + r) * Ly + p, mask=ml)
        _store(out + r * L + lo, acc * 0.125, ml)

    @triton.jit
    def _unroute_acc_kernel(y, inv, acc, out, R, L, Ly, G: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr,
                            BL: tl.constexpr):
        pl, r = tl.program_id(0), tl.program_id(1)
        lo = pl * BL + tl.arange(0, BL)
        ml = lo < L
        if FIRST:
            a = tl.zeros([BL], dtype=tl.float32)
        else:
            a = tl.load(acc + r * L + lo, mask=ml)
        for k in tl.static_range(G):
            p = tl.load(inv + k * L + lo, mask=ml, other=0)
            a = a + tl.load(y + (k * R + r) * Ly + p, mask=ml)
        if LAST:
            _store(out + r * L + lo, a * 0.125, ml)
        else:
            tl.store(acc + r * L + lo, a, mask=ml)

    @triton.jit
    def _ln_sq_kernel(x, u, d2, C, L, HALF: tl.constexpr, BL: tl.constexpr):
        pl, row = tl.program_id(0), tl.program_id(1)
        lo = pl * BL + tl.arange(0, BL)
        ml = lo < L
        d = tl.load(x + row * L + lo, mask=ml).to(tl.float32) - tl.load(u + (row // C) * L + lo, mask=ml).to(tl.float32)
        if HALF:
            d = tl.inline_asm_elementwise("cvt.rn.f16.f32 $0, $1;", "=h,r", [d], dtype=tl.float16, is_pure=True, pack=1).to(tl.float32)
        tl.store(d2 + row * L + lo, d * d, mask=ml)

    @triton.jit
    def _ln_out_kernel(x, u, s, w, b, g, out, C, L, eps, HALF: tl.constexpr, GATE: tl.constexpr, BL: tl.constexpr):
        pl, row = tl.program_id(0), tl.program_id(1)
        lo = pl * BL + tl.arange(0, BL)
        ml = lo < L
        c, i = row % C, (row // C) * L + lo
        d = tl.load(x + row * L + lo, mask=ml).to(tl.float32) - tl.load(u + i, mask=ml).to(tl.float32)
        if HALF:
            d = tl.inline_asm_elementwise("cvt.rn.f16.f32 $0, $1;", "=h,r", [d], dtype=tl.float16, is_pure=True, pack=1).to(tl.float32)
        o = tl.div_rn(d, tl.sqrt_rn(tl.load(s + i, mask=ml) + eps)) * tl.load(w + c) + tl.load(b + c)
        if GATE:
            o = o * tl.load(g + row * L + lo, mask=ml).to(tl.float32)
        _store(out + row * L + lo, o, ml)

    @triton.jit
    def _res_kernel(x, g, y, out, C, L, BL: tl.constexpr):
        pl, row = tl.program_id(0), tl.program_id(1)
        lo = pl * BL + tl.arange(0, BL)
        ml = lo < L
        o = tl.load(x + row * L + lo, mask=ml).to(tl.float32) + tl.load(g + row % C) * tl.load(y + row * L + lo, mask=ml).to(tl.float32)
        tl.store(out + row * L + lo, o, mask=ml)


def residual(x, g, y):
    """x + g[:, None, None] * y for contiguous (B, C, H, W) x, y (float16 / float32) and float32 g -> float32."""
    B, C, H, W = x.shape
    L, BL = H * W, 512
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    _res_kernel[(triton.cdiv(L, BL), B * C)](x, g, y, out, C, L, BL=BL, enable_fp_fusion=False)
    return out


def ln2d(x, w, b, eps, gate=None, out_dtype=torch.float32):
    """LN2d.forward of a contiguous (B, C, H, W) map under fp16 autocast (optionally times `gate`, cast to `out_dtype`)."""
    B, C, H, W = x.shape
    L, BL = H * W, 512
    u = x.mean(1, keepdim=True)
    d2 = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(L, BL), B * C)
    half = x.dtype == torch.float16
    _ln_sq_kernel[grid](x, u, d2, C, L, HALF=half, BL=BL, enable_fp_fusion=False)
    s = d2.mean(1, keepdim=True)
    out = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    g = x if gate is None else gate.contiguous()
    _ln_out_kernel[grid](x, u, s, w, b, g, out, C, L, eps, HALF=half, GATE=gate is not None, BL=BL, enable_fp_fusion=False)
    return out


def route(t, idx, dtype=torch.float32, length=None):
    """t (B, C, L) any strides, idx (8, L) int32 -> (8B, C, length) in `dtype` (exact: float32, or t's own dtype), route
    k of t[b] at row k * B + b, zero after L."""
    B, C, L = t.shape
    Lp, K = length or L, idx.shape[0]
    out = torch.empty(K * B, C, Lp, device=t.device, dtype=dtype)
    BC, BL = 16, 128
    _route_kernel[(triton.cdiv(Lp, BL), triton.cdiv(C, BC), K * B)](t, idx, out, B, C, L, Lp, *t.stride(), BC=BC, BL=BL)
    return out


def unroute_mean(y, inv, dtype):
    """y (8B, C, >= L) float32 in route order, inv (8, L) int32 -> (B, C, L) in `dtype`: the mean of the 8 routes mapped
    back, summed in route order starting from 0 like _unroute."""
    R, L = y.shape[0] // 8 * y.shape[1], inv.shape[1]
    out = torch.empty(y.shape[0] // 8, y.shape[1], L, device=y.device, dtype=dtype)
    BL = 256
    _unroute_kernel[(triton.cdiv(L, BL), R)](y.contiguous(), inv, out, R, L, y.shape[2], BL=BL)
    return out


def unroute_acc(y, inv, acc, out, first, last):
    """One group of routes of unroute_mean: y (G B, C, >= L) float32 for routes inv (G, L); adds them in order to the
    float32 running sum `acc` (B, C, L) (started from 0 when `first`), and on the `last` group writes acc / 8 to `out`."""
    G, L = inv.shape
    R = y.shape[0] // G * y.shape[1]
    _unroute_acc_kernel[(triton.cdiv(L, 256), R)](y.contiguous(), inv, acc, out, R, L, y.shape[2], G=G, FIRST=first,
                                                  LAST=last, BL=256)
