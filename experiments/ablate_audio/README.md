# Ablate Audio Cross-Attention

Does wav2vec2 cross-attention materially improve lip sync and expressiveness,
or is the spatial FLAME signal alone sufficient?

## Setup

Two configs, identical except for the audio conditioning path:

1. **With audio** — any existing config in `marionette/configs/` (all currently
   ship with `audio_encoder_config` populated and the UNet's
   `use_audio_context: true` (default)).
2. **Without audio** — a config that sets `audio_encoder_config: null` and
   `unet_config.params.use_audio_context: false`. These flags must be set
   together; `THDiffusion` validates their consistency at construction time
   and raises a `ValueError` if they mismatch.

A template `no_audio.yaml` is staged here as a starting point — copy one of the
existing configs, null out the audio encoder, and set `use_audio_context: false`.

## What to measure

- Lip-sync (SyncNet offset/confidence against the source audio).
- Motion smoothness and temporal jitter.
- FID / SSIM against held-out val clips.