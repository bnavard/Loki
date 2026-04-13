# Archived caching scripts

These scripts supported the earlier Wan-based text-to-expression-field approach,
which was abandoned in favor of the Marigold-style SD3.5 generator in
`marigold_training/`. They are kept for reference but are not part of any
current training pipeline.

- `cache_vae_latents.py` — Encodes expression fields via the Wan2.2 VAE.
  The current approach uses SD3.5's VAE.
- `cache_text_embeddings.py` — Encodes captions via the Wan2.2 UMT5-XXL text
  encoder. The current Marigold setup uses null-text conditioning (no text
  embeddings needed).