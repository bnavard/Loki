"""
Head pose encoder using 6DRepNet for the Marionette video diffusion model.

Extracts per-frame head pose (yaw, pitch, roll) from cropped face images using
a frozen 6DRepNet backbone, then projects the 3-angle vector into an embedding
of the same dimensionality as the UNet's timestep embedding. This embedding is
added to the timestep embedding so that head pose modulates every ResBlock and
transformer block in the UNet via the existing conditioning pathway.

The 6DRepNet backbone is frozen (no gradients). Only the learned projection
MLP is trained. This way the pose signal is always accurate (pretrained) and
only the mapping to the UNet's embedding space is learned.
"""

import math
import torch
import torch.nn as nn
from torchvision import transforms


class PoseEncoder(nn.Module):
    """
    Frozen 6DRepNet backbone → learned projection → pose embedding.

    Args:
        embed_dim: output embedding dimension. Must match the UNet's timestep
                   embedding dimension (model_channels * 4, typically 1280).
        freeze_backbone: if True (default), the 6DRepNet backbone is frozen.
    """

    def __init__(self, embed_dim: int = 1280, freeze_backbone: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self._backbone = None
        self._frozen = freeze_backbone

        # Preprocessing: 6DRepNet expects ImageNet-normalized 224x224 inputs
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Sinusoidal embedding for each of the 3 angles (like timestep embedding)
        # 3 angles × 128 frequencies × 2 (sin + cos) = 768 dims
        n_freqs = 128
        self.register_buffer(
            "angle_freqs",
            torch.exp(torch.linspace(0, math.log(1000.0), n_freqs)),
        )

        # Learned projection: sinusoidal(3 angles) → embed_dim
        sin_dim = 3 * n_freqs * 2  # 768
        self.projection = nn.Sequential(
            nn.Linear(sin_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _ensure_backbone(self, device):
        """Lazy-load the 6DRepNet backbone on first forward call."""
        if self._backbone is not None:
            return
        from sixdrepnet import SixDRepNet
        wrapper = SixDRepNet()
        self._backbone = wrapper.model.to(device).eval()
        if self._frozen:
            self._backbone.requires_grad_(False)

    @staticmethod
    def _rot_mat_to_euler(rot_mat: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3, 3) rotation matrices to (B, 3) Euler angles in degrees."""
        sy = torch.sqrt(rot_mat[:, 0, 0] ** 2 + rot_mat[:, 1, 0] ** 2)
        pitch = torch.atan2(-rot_mat[:, 2, 0], sy)
        yaw = torch.atan2(rot_mat[:, 1, 0], rot_mat[:, 0, 0])
        roll = torch.atan2(rot_mat[:, 2, 1], rot_mat[:, 2, 2])
        return torch.stack([pitch, yaw, roll], dim=-1) * (180.0 / math.pi)

    def _sinusoidal_embed(self, angles: torch.Tensor) -> torch.Tensor:
        """Embed (B, 3) angles into (B, 768) via sinusoidal encoding."""
        # angles: (B, 3) in degrees
        # Expand: (B, 3, 1) * (n_freqs,) → (B, 3, n_freqs)
        scaled = angles.unsqueeze(-1) * self.angle_freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(1)

    def forward(self, face_frames: torch.Tensor) -> torch.Tensor:
        """
        Extract head pose from face frames and produce a pose embedding.

        Args:
            face_frames: (B, T, 3, H, W) face crops in [0, 1] range.
                         These are the driver's natural video frames, already
                         cropped to the face region.

        Returns:
            (B*T, embed_dim) pose embedding, ready to be added to the UNet's
            timestep embedding.
        """
        device = face_frames.device
        self._ensure_backbone(device)

        B, T, C, H, W = face_frames.shape
        flat = face_frames.reshape(B * T, C, H, W)

        # Preprocess for 6DRepNet: resize to 224, ImageNet normalize
        preprocessed = self.preprocess(flat)

        # 6DRepNet backbone → rotation matrix
        with torch.no_grad():
            rot_mat = self._backbone(preprocessed)  # (B*T, 3, 3)

        # Rotation matrix → Euler angles (degrees)
        euler = self._rot_mat_to_euler(rot_mat)  # (B*T, 3)

        # Sinusoidal embedding → learned projection
        sin_emb = self._sinusoidal_embed(euler)  # (B*T, 768)
        return self.projection(sin_emb)  # (B*T, embed_dim)
