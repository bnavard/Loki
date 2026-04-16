"""
SyncNet v2 architecture (Chung & Zisserman, 2016).

Two-stream network that embeds 5-frame lip video windows and their
corresponding MFCC audio windows into a shared 1024-dim space. Used to
compute LSE-D (distance) and LSE-C (confidence) lip-sync metrics.

Pretrained weights: syncnet_v2.model from
  https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model

The checkpoint is a full torch.save(model) dump from the original repo.
load_syncnet_v2() extracts the state_dict and loads it into this clean
reimplementation.
"""

import torch
import torch.nn as nn


class SyncNetV2(nn.Module):
    """SyncNet v2: audio (MFCC) + video (3D-conv) → 1024-dim embeddings each."""

    def __init__(self):
        super().__init__()

        # --- Audio stream ---
        # Input: (B, 1, 13, 20)  — 13 MFCC coefficients × 20 time steps (5 video frames)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=0),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.audio_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
        )

        # --- Video / lip stream ---
        # Input: (B, 3, 5, 224, 224)  — 5 consecutive RGB frames
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),

            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512), nn.ReLU(inplace=True),
        )
        self.video_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
        )

    def forward_audio(self, mfcc: torch.Tensor) -> torch.Tensor:
        """(B, 1, 13, 20) → (B, 1024)"""
        x = self.audio_encoder(mfcc)
        return self.audio_fc(x.view(x.size(0), -1))

    def forward_video(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, 3, 5, 224, 224) → (B, 1024)"""
        x = self.video_encoder(frames)
        return self.video_fc(x.view(x.size(0), -1))


def load_syncnet_v2(checkpoint_path: str, device: str = "cuda") -> SyncNetV2:
    """Load pretrained syncnet_v2.model weights into a clean SyncNetV2 instance.

    The checkpoint is a full torch.save(model) from the original repo. We
    extract its state_dict and remap keys to match our naming.
    """
    model = SyncNetV2()

    # The original checkpoint is torch.save(model_instance), so torch.load
    # returns the model object directly. Extract its state_dict.
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict) and "state_dict" in loaded:
        src_sd = loaded["state_dict"]
    elif isinstance(loaded, nn.Module):
        src_sd = loaded.state_dict()
    elif isinstance(loaded, dict):
        src_sd = loaded
    else:
        src_sd = loaded.state_dict()

    # Map original keys (netcnnaud.*, netcnnlip.*, netfcaud.*, netfclip.*)
    # to our keys (audio_encoder.*, video_encoder.*, audio_fc.*, video_fc.*).
    key_map = {
        "netcnnaud.": "audio_encoder.",
        "netfcaud.":  "audio_fc.",
        "netcnnlip.": "video_encoder.",
        "netfclip.":  "video_fc.",
    }
    remapped = {}
    for k, v in src_sd.items():
        new_k = k
        for old_prefix, new_prefix in key_map.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix):]
                break
        remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"[syncnet] Warning: missing keys: {missing}")
    if unexpected:
        print(f"[syncnet] Warning: unexpected keys: {unexpected}")

    model.eval().to(device)
    return model
