"""
Spatiotemporal attention modules for Marionette's 3D UNet: basic transformer
blocks, spatial/temporal/3D self-attention, and cross-attention for audio
context conditioning.
"""
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from controlnet.ldm.modules.diffusionmodules.util import checkpoint, GroupNorm32, LayerNorm32

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


def legacy_attention(q, k, v, scale, mask=None):
    if _ATTN_PRECISION == "fp32":
        with torch.autocast(enabled=False, device_type='cuda'):
            q, k = q.float(), k.float()
            sim = einsum('b i d, b j d -> b i j', q, k) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    sim = sim.softmax(dim=-1)
    return einsum('b i j, b j d -> b i d', sim, v)


class AttentionModule(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.,
                 mode="spatial", context_dim=None, num_timesteps=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.mode = mode
        if mode == "context":
            assert context_dim is not None
            kv_dim = context_dim
        elif mode in ("spatial", "temporal", "3d"):
            kv_dim = query_dim
            if mode in ("temporal", "3d"):
                assert num_timesteps > 0
        else:
            raise ValueError(f"Unrecognized attention mode: {mode}")

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.num_timesteps = num_timesteps

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=False)
        self.k_v_fixed = False

        is_zero_module = mode == "temporal"
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim) if is_zero_module else zero_module(nn.Linear(inner_dim, query_dim)),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        t = self.num_timesteps
        b = x.shape[0]

        q = self.to_q(x)
        if self.mode == "context":
            assert context is not None
            k = self.to_k(context)
            v = self.to_v(context)
        else:
            if FIX_LEGACY_FAIL and not self.k_v_fixed:
                with torch.no_grad():
                    self.to_v.weight.data = self.to_k.weight.data
                self.k_v_fixed = True
            k = self.to_k(x)
            v = self.to_v(x)

        if _USE_FLASH or XFORMERS_IS_AVAILBLE:
            if self.mode == "3d":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> b (n t) h d', h=h, t=t), (q, k, v))
            elif self.mode == "temporal":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> (b n) t h d', h=h, t=t), (q, k, v))
            elif self.mode in ("context", "spatial"):
                q, k, v = map(lambda yt: rearrange(yt, 'b n (h d) -> b n h d', h=h), (q, k, v))

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
            elif self.mode in ("context", "spatial"):
                out = rearrange(out, 'b n h d -> b n (h d)', h=h)
        else:
            if self.mode == "3d":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> (b h) (n t) d', h=h, t=t), (q, k, v))
            elif self.mode == "temporal":
                q, k, v = map(lambda yt: rearrange(yt, '(b t) n (h d) -> (b h n) t d', h=h, t=t), (q, k, v))
            elif self.mode in ("context", "spatial"):
                q, k, v = map(lambda yt: rearrange(yt, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            if _USE_FP16_ATTENTION:
                dtype_before = q.dtype
                out = legacy_attention(q.half(), k.half(), v.half(), self.scale, mask=mask).type(dtype_before)
            else:
                out = legacy_attention(q, k, v, self.scale, mask=mask)

            if self.mode == "3d":
                out = rearrange(out, '(b h) (n t) d -> (b t) n (h d)', b=b // t, h=h, t=t)
            elif self.mode == "temporal":
                out = rearrange(out, '(b h n) t d -> (b t) n (h d)', b=b // t, h=h, t=t)
            elif self.mode in ("context", "spatial"):
                out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., use_context=True,
                 context_dim=None, gated_ff=True,
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

        self.use_context = use_context
        if use_context:
            self.attn2 = AttentionModule(
                query_dim=dim, context_dim=context_dim, heads=n_heads,
                dim_head=d_head, dropout=dropout, mode="context",
            )
            self.norm2 = LayerNorm32(dim)

        if temporal_connection_type == "temporal":
            self.attn_t = AttentionModule(
                query_dim=dim, context_dim=None, heads=n_heads,
                dim_head=d_head, dropout=dropout, num_timesteps=num_timesteps,
            )
            self.norm_t = LayerNorm32(dim)

        self.norm3 = LayerNorm32(dim)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=None) + x

        if self.use_context:
            assert context is not None, "Audio context must be provided when use_context=True"
            x = self.attn2(self.norm2(x), context=context) + x

        if self.temporal_connection_type == "temporal":
            x = self.attn_t(self.norm_t(x), context=None) + x

        x = self.ff(self.norm3(x)) + x
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Transformer block for image-like data with optional temporal attention and
    optional cross-attention (used for audio conditioning in the talking-head model).
    """

    def __init__(self, in_channels, n_heads, d_head, dropout=0.,
                 use_context=True, context_dim=None,
                 temporal_connection_type="none", num_timesteps=0):
        super().__init__()

        if use_context:
            assert context_dim is not None

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, dropout=dropout,
                use_context=use_context, context_dim=context_dim,
                temporal_connection_type=temporal_connection_type,
                num_timesteps=num_timesteps,
            )
        ])
        self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.has_context = True

    def forward(self, x, context=None):
        assert not isinstance(context, list)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return x + x_in
