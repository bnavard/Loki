# Ablate Audio Cross-Attention

Does wav2vec2 cross-attention materially improve lip sync and expressiveness,
or is the spatial FLAME signal alone sufficient?

## How to compose each variant

| Variant     | Overlays                              |
|-------------|---------------------------------------|
| With audio  | *(base default, no overlays)*         |
| No audio    | `overlays/audio/off.yaml`             |

The `audio/off` overlay sets both `audio_encoder_config: null` and
`unet_config.params.use_audio_context: false`. `THDiffusion` validates that
these two flags agree at construction time and raises a clear error on
mismatch, so you can't accidentally train a half-ablated model.

## What to measure

- Lip-sync (SyncNet offset/confidence against the source audio).
- Motion smoothness and temporal jitter.
- FID / SSIM against held-out val clips.