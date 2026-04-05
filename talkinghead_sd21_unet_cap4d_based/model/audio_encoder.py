"""
Audio encoder: raw 16kHz waveform → per-frame cross-attention tokens.

A frozen wav2vec2-base-960h backbone encodes audio windows into token sequences.
A learned linear projection maps from backbone dim (768) to the UNet's
context_dim (1024, matching SD 2.1's pretrained cross-attention weights).

Each video frame gets audio from ±context_frames neighbors (~200ms window).
The backbone is frozen during training to preserve pretrained speech features;
only the projection layer trains.

Input / output shapes:
  AudioEncoder.forward(waveform):
    waveform : (B, T, window_samples)    — raw 16 kHz audio
    returns  : (B, T, num_tokens, context_dim)

For the dummy test, pass random float tensors of shape (B, T, window_samples);
the model handles arbitrary window lengths.
"""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        context_dim: int = 768,        # must match UNet context_dim
        freeze_backbone: bool = True,
        use_pretrained: bool = True,   # set False in unit tests to skip download
    ):
        """
        Args:
            model_name:      HuggingFace model identifier for wav2vec2 or HuBERT.
            context_dim:     Output feature dimension (projected to match UNet).
            freeze_backbone: If True, backbone weights are frozen during training.
            use_pretrained:  If False, initialise backbone with random weights
                             (useful for shape-only unit tests without internet).
        """
        super().__init__()

        self.context_dim = context_dim

        # Load backbone
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Config
            if use_pretrained:
                self.backbone = Wav2Vec2Model.from_pretrained(model_name)
            else:
                config = Wav2Vec2Config()
                self.backbone = Wav2Vec2Model(config)
            backbone_dim = self.backbone.config.hidden_size  # 768 for base models
        except ImportError:
            raise ImportError(
                "transformers package is required for AudioEncoder. "
                "Install with: pip install transformers"
            )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # disable masking (mask_time_length > short sequences)

        # Project backbone output to UNet context_dim
        self.proj = nn.Linear(backbone_dim, context_dim) if backbone_dim != context_dim else nn.Identity()

    def train(self, mode=True):
        """Override to keep frozen backbone in eval mode (avoids wav2vec2 masking)."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def _encode_backbone(self, waveform_flat: torch.Tensor) -> torch.Tensor:
        """
        Run backbone on a flat (BT, window_samples) tensor.
        Returns (BT, num_tokens, backbone_dim).
        """
        out = self.backbone(waveform_flat).last_hidden_state  # (BT, num_tokens, dim)
        return out

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T, window_samples)  — raw 16 kHz mono audio
        Returns:
            (B, T, num_tokens, context_dim)
        """
        B, T, L = waveform.shape
        waveform_flat = waveform.reshape(B * T, L)

        if self.freeze_backbone or not self.backbone.training:
            with torch.no_grad():
                feat = self._encode_backbone(waveform_flat)
        else:
            feat = self._encode_backbone_train(waveform_flat)

        feat = self.proj(feat)                          # (B*T, num_tokens, context_dim)
        feat = feat.reshape(B, T, feat.shape[1], self.context_dim)
        return feat

    def _encode_backbone_train(self, waveform_flat):
        """Backbone forward for when it is NOT frozen (allows gradients)."""
        return self.backbone(waveform_flat).last_hidden_state
