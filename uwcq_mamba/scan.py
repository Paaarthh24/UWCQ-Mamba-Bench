"""Selective scan (Mamba-1). Uses the compiled VMamba kernel `selective_scan_cuda_oflex` when it is installed and the
inputs are on the GPU; otherwise a chunked PyTorch implementation with the same numerics (slower)."""
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

try:
    import selective_scan_cuda_oflex
except ImportError:
    selective_scan_cuda_oflex = None


def _scan_chunk(x0, delta, A, B, u, C):
    dA = torch.exp(torch.einsum("bkcl,kcn->bkcln", delta, A))
    dBu = torch.einsum("bkcl,bknl,bkcl->bkcln", delta, B, u)
    x, ys = x0, []
    for i in range(delta.shape[-1]):
        x = dA[:, :, :, i] * x + dBu[:, :, :, i]
        ys.append(torch.einsum("bkcn,bkn->bkc", x, C[:, :, :, i]))
    return x, torch.stack(ys, 3)


def selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus=True, chunk=256):
    """u, delta (Bt, K*C, L); A (K*C, N); B, C (Bt, K, N, L); D, delta_bias (K*C,). Returns float32 (Bt, K*C, L)."""
    bt, K, N, L = B.shape
    kc = u.shape[1]
    c = kc // K
    delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    dr, ur, Ar = delta.view(bt, K, c, L), u.view(bt, K, c, L), A.view(K, c, N)
    ckpt = torch.is_grad_enabled() and any(t.requires_grad for t in (dr, ur, B, C, Ar))
    x = A.new_zeros((bt, K, c, N))
    ys = []
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        args = (x, dr[..., s:e], Ar, B[..., s:e], ur[..., s:e], C[..., s:e])
        x, y = torch.utils.checkpoint.checkpoint(_scan_chunk, *args, use_reentrant=False) if ckpt else _scan_chunk(*args)
        ys.append(y)
    return torch.cat(ys, 3).reshape(bt, kc, L) + u * D.unsqueeze(-1)


class _SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, u, delta, A, B, C, D, delta_bias, delta_softplus):
        ctx.delta_softplus = delta_softplus
        out, x, *_ = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, True)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        grads = selective_scan_cuda_oflex.bwd(u, delta, A, B, C, D, delta_bias, dout.contiguous(), x, ctx.delta_softplus, 1)
        return (*grads[:7], None)


def selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus=True):
    if selective_scan_cuda_oflex is not None and u.is_cuda:
        return _SelectiveScanCuda.apply(u, delta, A, B, C, D, delta_bias, delta_softplus)
    return selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus)
