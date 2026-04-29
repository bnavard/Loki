"""
Spatiotemporal attention modules for Marionette's 3D UNet: basic transformer
blocks, plus spatial / temporal / 3D self-attention. The 3D and spatial
self-attention modes accept an optional `ref_kv` feature map (captured from
the frozen reference UNet at the same layer) which is concatenated onto the
gen K/V pool so every gen query attends to both gen and ref tokens — the
ReferenceNet / AnimateAnyone identity-injection pattern.
"""
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm_base.ldm.modules.diffusionmodules.util import GroupNorm32, LayerNorm32

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
FIX_LEGACY_FAIL = os.environ.get("FIX_LEGACY_FAIL", False)
if FIX_LEGACY_FAIL:
    print("Fixing legacy failed k and v layers")

_USE_FP16_ATTENTION = os.environ.get("USE_FP16_ATTENTION", False)
_USE_FLASH = os.environ.get("USE_FLASH_ATTENTION", False)
if _USE_FLASH:
    from flash_attn import flash_attn_func


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return GroupNorm32(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def legacy_attention(q, k, v, scale):
    """Plain Q·Kᵀ → softmax → ·V attention. No mask path — the talking-head
    pipeline never passes one (every caller of `AttentionModule.forward` in
    this codebase leaves `mask=None`), and the previous version of this
    function referenced an undefined `h` inside a `if exists(mask):` branch
    that was effectively dead. Delete-rather-than-fix because there's no
    realistic future use for masked self-attention here; reintroduce the
    parameter (with `h` threaded in explicitly) only if a real consumer
    appears."""
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    sim = sim.softmax(dim=-1)
    return einsum('b i j, b j d -> b i d', sim, v)


class AttentionModule(nn.Module):
    """Self-attention in one of three modes — `spatial`, `temporal`, or `3d` —
    over the gen-token tensor. `spatial` and `3d` modes additionally accept
    an optional `ref_kv` feature map that is concatenated onto the K/V pool
    so each gen query attends to both gen and ref tokens (the ReferenceNet
    K/V-injection pattern). Temporal mode collapses a different axis and
    does not consume `ref_kv`."""

    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.,
                 mode="spatial", num_timesteps=0):
        super().__init__()
        if mode not in ("spatial", "temporal", "3d"):
            raise ValueError(f"Unrecognized attention mode: {mode}")
        if mode in ("temporal", "3d"):
            assert num_timesteps > 0

        inner_dim = dim_head * heads
        self.mode = mode
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.num_timesteps = num_timesteps

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_v_fixed = False

        is_zero_module = mode == "temporal"
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim) if is_zero_module else zero_module(nn.Linear(inner_dim, query_dim)),
            nn.Dropout(dropout)
        )

    def forward(self, x, ref_kv=None):
        """`ref_kv` is a pre-norm feature map `(B, N_ref, D)` captured from a
        frozen reference UNet at this layer. When present, its K/V contribution
        is concatenated onto the gen K/V pool so every gen query attends to
        both gen tokens and ref tokens. Ignored for temporal mode (it collapses
        a different axis)."""
        h = self.heads
        t = self.num_timesteps
        b = x.shape[0]

        if FIX_LEGACY_FAIL and not self.k_v_fixed:
            with torch.no_grad():
                self.to_v.weight.data = self.to_k.weight.data
            self.k_v_fixed = True

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        inject_ref = ref_kv is not None and self.mode in ("spatial", "3d")
        if inject_ref:
            k_ref = self.to_k(ref_kv)
            v_ref = self.to_v(ref_kv)

        if _USE_FLASH or XFORMERS_IS_AVAILBLE:
            if self.mode == "3d":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> b (n t) h d', h=h, t=t), (q, k, v))
                if inject_ref:
                    k_ref = rearrange(k_ref, 'b n (h d) -> b n h d', h=h)
                    v_ref = rearrange(v_ref, 'b n (h d) -> b n h d', h=h)
                    k = torch.cat([k, k_ref], dim=1)
                    v = torch.cat([v, v_ref], dim=1)
            elif self.mode == "temporal":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> (b n) t h d', h=h, t=t), (q, k, v))
            else:  # spatial
                q, k, v = map(lambda yt: rearrange(yt, 'b n (h d) -> b n h d', h=h), (q, k, v))
                if inject_ref:
                    # Gen q/k/v are at layout (B*T, N, h, d); ref is (B, N_ref, h, d).
                    # Broadcast ref across T so each gen slot sees the same ref tokens.
                    k_ref = rearrange(k_ref, 'b n (h d) -> b n h d', h=h)
                    v_ref = rearrange(v_ref, 'b n (h d) -> b n h d', h=h)
                    k_ref = repeat(k_ref, 'b n h d -> (b t) n h d', t=t)
                    v_ref = repeat(v_ref, 'b n h d -> (b t) n h d', t=t)
                    k = torch.cat([k, k_ref], dim=1)
                    v = torch.cat([v, v_ref], dim=1)

            if _USE_FLASH:
                dtype_before = q.dtype
                out = flash_attn_func(q.half(), k.half(), v.half()).type(dtype_before)
            else:
                if _USE_FP16_ATTENTION:
                    dtype_before = q.dtype
                    out = xformers.ops.memory_efficient_attention(q.half(), k.half(), v.half(), attn_bias=None, op=None).type(dtype_before)
                else:
                    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

            if self.mode == "3d":
                out = rearrange(out, 'b (n t) h d -> (b t) n (h d)', b=b // t, h=h, t=t)
            elif self.mode == "temporal":
                out = rearrange(out, '(b n) t h d -> (b t) n (h d)', b=b // t, h=h, t=t)
            else:  # spatial
                out = rearrange(out, 'b n h d -> b n (h d)', h=h)
        else:
            if self.mode == "3d":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> (b h) (n t) d', h=h, t=t), (q, k, v))
                if inject_ref:
                    k_ref = rearrange(k_ref, 'b n (h d) -> (b h) n d', h=h)
                    v_ref = rearrange(v_ref, 'b n (h d) -> (b h) n d', h=h)
                    k = torch.cat([k, k_ref], dim=1)
                    v = torch.cat([v, v_ref], dim=1)
            elif self.mode == "temporal":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> (b h n) t d', h=h, t=t), (q, k, v))
            else:  # spatial
                q, k, v = map(lambda yt: rearrange(yt, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
                if inject_ref:
                    k_ref = rearrange(k_ref, 'b n (h d) -> (b h) n d', h=h)
                    v_ref = rearrange(v_ref, 'b n (h d) -> (b h) n d', h=h)
                    k_ref = repeat(k_ref, '(b h) n d -> (b t h) n d', h=h, t=t)
                    v_ref = repeat(v_ref, '(b h) n d -> (b t h) n d', h=h, t=t)
                    k = torch.cat([k, k_ref], dim=1)
                    v = torch.cat([v, v_ref], dim=1)

            if _USE_FP16_ATTENTION:
                dtype_before = q.dtype
                out = legacy_attention(q.half(), k.half(), v.half(), self.scale).type(dtype_before)
            else:
                out = legacy_attention(q, k, v, self.scale)

            if self.mode == "3d":
                out = rearrange(out, '(b h) (n t) d -> (b t) n (h d)', b=b // t, h=h, t=t)
            elif self.mode == "temporal":
                out = rearrange(out, '(b h n) t d -> (b t) n (h d)', b=b // t, h=h, t=t)
            else:  # spatial
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """One transformer block in the gen UNet: spatial-or-3D self-attention
    (with optional ref-K/V injection), then optional temporal self-attention,
    then a feed-forward. No cross-attention path — the model is conditioned
    spatially via `ConditioningEncoder` (additive feature map) and via the
    ref-attention K/V injection."""

    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True,
                 temporal_connection_type="none", num_timesteps=0):
        super().__init__()
        self.temporal_connection_type = temporal_connection_type
        if temporal_connection_type != "none":
            assert num_timesteps > 0

        self.attn1 = AttentionModule(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            mode="spatial" if temporal_connection_type != "3d" else "3d",
            num_timesteps=num_timesteps,
        )
        self.norm1 = LayerNorm32(dim)

        if temporal_connection_type == "temporal":
            self.attn_t = AttentionModule(
                query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                mode="temporal", num_timesteps=num_timesteps,
            )
            self.norm_t = LayerNorm32(dim)

        self.norm3 = LayerNorm32(dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

    def forward(self, x, ref_kv=None):
        x = self.attn1(self.norm1(x), ref_kv=ref_kv) + x

        if self.temporal_connection_type == "temporal":
            x = self.attn_t(self.norm_t(x)) + x

        x = self.ff(self.norm3(x)) + x
        return x


class SpatioTemporalTransformer(nn.Module):
    """Transformer block for image-like data with optional temporal attention.

    Owned by `MarionetteUNet`. Each instance lives at one resolution in the
    UNet hierarchy; its `BasicTransformerBlock` does spatial / 3D self-attn
    (+ ref K/V injection) and optional temporal self-attn. The forward
    signature accepts a `context` keyword positionally for compatibility
    with the LDM base's `TimestepEmbedSequential` dispatcher, but the value
    is ignored.
    """

    def __init__(self, in_channels, n_heads, d_head, dropout=0.,
                 temporal_connection_type="none", num_timesteps=0):
        super().__init__()

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, dropout=dropout,
                temporal_connection_type=temporal_connection_type,
                num_timesteps=num_timesteps,
            )
        ])
        self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))

    def forward(self, x, context=None):
        """`ref_kv` is read from `self._ref_kv_feature` if set by the owning
        UNet before the forward pass. This attribute-based plumbing avoids
        widening the signature, which would require patching
        `TimestepEmbedSequential`'s dispatcher in the LDM base.

        `context` is accepted positionally so the LDM dispatcher can pass it
        without a signature mismatch; it is unused.
        """
        del context
        ref_kv = getattr(self, "_ref_kv_feature", None)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, ref_kv=ref_kv)
        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x + x_in
