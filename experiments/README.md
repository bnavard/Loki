# Experiments

Everything Marionette does outside the core training/inference loop —
the canonical full-stack training, single-variable conditioning ablations,
identity-paired evaluation against a checkpoint, side-by-side comparison
against published SOTA baselines, and the FLAME-native metric harness that
chews through any of those output trees.

## Contents

### [`marionette_baseline/`](marionette_baseline/) — canonical training run
Point-of-truth training against [`marionette/configs/base.yaml`](../marionette/configs/base.yaml)
unchanged. Same-identity self-supervised video diffusion: 45-channel FLAME
spatial conditioning, SD 2.1 generation UNet with 3D spatiotemporal
attention, and a frozen SD 2.1 reference UNet whose per-layer
self-attention features are injected as K/V tokens into the gen UNet.

### [`condition_ablation/`](condition_ablation/) — conditioning-pathway ablations
Single-variable ablations on the FLAME spatial-conditioning pathway: the
`no_posenc` arm drops the 42-channel positional encoding; the `no_deform`
arm drops the 3-channel deformation map. Each arm trains a fresh 30k-step
run from the same SD 2.1 init + seed as the baseline; the only thing that
differs per arm is which channels of `spatial_cond` the model sees.

### [`marionette_eval/`](marionette_eval/) — same + cross identity evaluation on HDTF
Loads any baseline or ablation checkpoint and generates side-by-side
panel/mp4 outputs on the curated HDTF benchmark. Same-identity
reconstruction plus a deranged-pair cross-identity sweep. Output layout
matches every SOTA wrapper so a single glob compares Marionette and every
baseline 1-to-1 on the same physical identity (or pair).

### [`sota_comparison/`](sota_comparison/) — SOTA baseline wrappers
Uniform CLI wrappers around five published talking-head baselines:
[SadTalker](sota_comparison/sadtalker/), [AniTalker](sota_comparison/anitalker/),
[EchoMimic](sota_comparison/echomimic/), [HunyuanPortrait](sota_comparison/hunyuan_portrait/),
[X-Portrait](sota_comparison/xportrait/). Every wrapper consumes the same
curated HDTF manifest and emits `samples/<sample_id>/panel.mp4` so a
single glob compares Marionette and every baseline 1-to-1 on the same
identity (or identity pair). One-shot `setup_env.sh` per baseline.

### [`evaluation_metrics/`](evaluation_metrics/) — FLAME-native quality metrics
Three metrics on every run dir, routed by protocol:
- `head_rot_dist` — geodesic angular distance between pred and target
  head rotations, computed FLAME-natively from `rot · neck_rot` (degrees,
  lower is better).
- `expression_l1` — pose-disentangled L1 of the rasterised FLAME
  deformation map between pred and target fits (lower is better).
- `id_cosine` — ArcFace identity cosine vs. the ref-clip prior
  (cross-identity only; higher is better).

Operates on any `outputs/**/samples/<sample_id>/panel.mp4` tree —
Marionette's eval and every SOTA baseline's output share the layout.

## Conventions

- Base training / inference configs live in `marionette/configs/`.
  Experiment-specific configs live inside each experiment subfolder.
- Outputs land at repo root under `outputs/<experiment>/…`.
- `<sample_id>` naming is UID-based (`id_0457` for same-identity,
  `id_0457_id_0009` for cross-identity) and consistent across every
  `sota_comparison/<baseline>/` **and** `marionette_eval/` — all of them
  enumerate from the frozen curated manifest under
  `experiments/sota_comparison/manifests/hdtf.json` via the shared
  `experiments.sota_comparison.dataset.pairing.build_samples`. So a single
  glob `outputs/**/samples/<sample_id>/panel.mp4` compares Marionette and
  every SOTA baseline 1-to-1 on the same physical identity (or pair).
