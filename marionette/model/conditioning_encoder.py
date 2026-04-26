"""
Conditioning encoder — a lightweight SD 2.1-style conv stack that takes the
full-resolution spatial conditioning tensor (expression map / deformation /
driving video at 512×512) and downsamples it to the UNet's latent resolution
(64×64) with `model_channels` output channels, ready to be added to the UNet's
first feature map.

Replaces the old `cond_linear` path (a per-pixel channel projection at latent
resolution). The old path discarded all spatial structure above 64×64; this
encoder preserves and learns from the high-frequency signal in the conditioning
map (lip-corner edges, brow creases, mouth interior boundaries, etc.).

Architecture (matches Stable Diffusion 2.1's UNet block style):

    Input (B*T, C_cond, 512, 512)
    │
    ├─ stem: Conv3×3   (C_cond → base_ch=64)        → 64 × 512 × 512
    ├─ ResBlock(64   → 128), DownsampleConv(stride 2) → 128 × 256 × 256
    ├─ ResBlock(128  → 256), DownsampleConv(stride 2) → 256 × 128 × 128
    ├─ ResBlock(256  → 320), DownsampleConv(stride 2) → 320 × 64  × 64
    ├─ ResBlock(320  → 320)                           → 320 × 64  × 64
    ├─ GroupNorm + SiLU
    └─ zero-init Conv3×3(320 → 320)                   → 320 × 64  × 64

Every normalisation is `GroupNorm(32, ch)`, every activation is SiLU — same as
the SD UNet so the emitted feature map is compatible with the UNet's input
expectations. The final Conv3×3 is zero-initialised so the encoder contributes
**zero** at step 0 and the pretrained UNet starts from its known-good state.
The encoder's other weights gradually learn to add signal, mirroring the
behaviour of the old `cond_linear` path under `zero_module`.

The module is trained jointly with the UNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_init(conv: nn.Conv2d) -> nn.Conv2d:
    """Zero out conv weights and bias so the module returns zeros at init."""
    with torch.no_grad():
        conv.weight.zero_()
        if conv.bias is not None:
            conv.bias.zero_()
    return conv


class _ResBlock(nn.Module):
    """SD-style residual block: (GN -> SiLU -> Conv3×3) ×2 + skip.

    Channel-change is absorbed into the residual via a 1×1 skip projection
    (SD does this too — `openaimodel.ResBlock.skip_connection`).
    """

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _Downsample(nn.Module):
    """Strided Conv3×3 downsample — matches SD's `Downsample(use_conv=True)`."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditioningEncoder(nn.Module):
    """Downsample full-resolution spatial conditioning to `model_channels`
    feature maps at the UNet's latent resolution.

    Args:
        in_channels      : number of channels in the conditioning tensor
                           (matches `SpatialConditioning.n_conditioning_channels`;
                           typically 46, 4, or 1).
        model_channels   : UNet's first feature map channel count (320 for SD 2.1).
        input_resolution : spatial size of the incoming conditioning tensor
                           (default 512).
        output_resolution: target resolution — must match the UNet's first
                           feature map spatial size (default 64 for SD 2.1 at
                           512px images, i.e. 512 / 8 via the VAE).
        stage_channels   : channel progression through the downsample stages.
                           Default `(64, 128, 256, model_channels)` — 4 entries
                           define the channel counts at the four resolutions
                           512 → 256 → 128 → 64.
        num_groups       : GroupNorm group count (32, matches SD).

    Invariants: `input_resolution / 2**n == output_resolution` where `n` is
    the number of downsample stages (len(stage_channels) - 1). Defaults give
    `512 / 2**3 == 64`. Raises at init if the math doesn't work out.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int = 320,
        input_resolution: int = 512,
        output_resolution: int = 64,
        stage_channels: tuple[int, ...] | None = None,
        num_groups: int = 32,
    ):
        super().__init__()

        if stage_channels is None:
            stage_channels = (64, 128, 256, model_channels)
        if stage_channels[-1] != model_channels:
            raise ValueError(
                f"stage_channels[-1] ({stage_channels[-1]}) must equal "
                f"model_channels ({model_channels}) — the encoder's output "
                f"channel count must match the UNet's first feature map."
            )

        n_downsamples = len(stage_channels) - 1
        ratio = input_resolution / output_resolution
        expected_ratio = 2 ** n_downsamples
        if ratio != expected_ratio:
            raise ValueError(
                f"input_resolution ({input_resolution}) / output_resolution "
                f"({output_resolution}) = {ratio}, but stage_channels has "
                f"{n_downsamples} downsample stages (expected ratio "
                f"{expected_ratio}). Adjust stage_channels or the resolutions."
            )

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        # Stem: lift C_cond to the first stage's channel count while staying at full res.
        self.stem = nn.Conv2d(in_channels, stage_channels[0], kernel_size=3, padding=1)

        # Downsample stages: ResBlock(ch_i → ch_{i+1}) + Downsample at 2× stride.
        self.stages = nn.ModuleList()
        for i in range(n_downsamples):
            self.stages.append(nn.ModuleList([
                _ResBlock(stage_channels[i], stage_channels[i + 1], num_groups),
                _Downsample(stage_channels[i + 1]),
            ]))

        # Refinement ResBlock at the output resolution.
        self.refine = _ResBlock(
            stage_channels[-1], stage_channels[-1], num_groups,
        )

        # Final norm + zero-initialised projection → zero output at init.
        self.out_norm = nn.GroupNorm(num_groups, stage_channels[-1])
        self.out_conv = _zero_init(
            nn.Conv2d(stage_channels[-1], model_channels, kernel_size=3, padding=1)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cond: (N, C_cond, H_in, W_in) — spatial conditioning at full res.
        Returns:
            (N, model_channels, H_out, W_out) — zero-initialised at step 0.
        """
        if cond.shape[-2:] != (self.input_resolution, self.input_resolution):
            raise ValueError(
                f"ConditioningEncoder expected spatial shape "
                f"({self.input_resolution}, {self.input_resolution}), got "
                f"{tuple(cond.shape[-2:])}."
            )

        h = self.stem(cond)
        for resblock, down in self.stages:
            h = resblock(h)
            h = down(h)
        h = self.refine(h)
        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        return h
