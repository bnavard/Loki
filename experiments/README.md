# Experiments

Everything Marionette does outside the core training/inference loop —
the canonical full-stack training, single-variable conditioning ablations,
identity-paired evaluation against a checkpoint, side-by-side comparison
against published SOTA baselines, and the numerical metric harness that
chews through any of those output trees.

## Contents

### [`marionette_baseline/`](marionette_baseline/) — canonical training run
Point-of-truth training against [`marionette/configs/base.yaml`](../marionette/configs/base.yaml)
unchanged. The audio-on, full-FLAME reference every ablation arm is
compared against.

### [`condition_ablation/`](condition_ablation/) — conditioning-pathway ablations
Single-variable ablations on the conditioning inputs feeding the gen UNet:
audio on/off, FLAME-vs-natural-video, pos_enc-vs-deform. Each arm trains
a fresh 30k-step run from the same SD 2.1 init + seed as the baseline; the
only thing that differs per arm is which conditioning channels the model
sees. See the subfolder README for the full matrix.

### [`marionette_eval/`](marionette_eval/) — cross + same identity evaluation
Loads any baseline or ablation checkpoint and generates side-by-side
panel/mp4 outputs on the validation set. Same-identity reconstruction plus
a deranged-pair cross-identity sweep across all usable YouTube identities.

### [`sota_comparison/`](sota_comparison/) — SOTA baseline wrappers
Uniform CLI wrappers around five published talking-head baselines:
[SadTalker](sota_comparison/sadtalker/), [AniTalker](sota_comparison/anitalker/),
[EchoMimic](sota_comparison/echomimic/), [HunyuanPortrait](sota_comparison/hunyuan_portrait/),
[X-Portrait](sota_comparison/xportrait/). Every wrapper consumes the same
curated benchmark manifest and emits `samples/<sample_id>/panel.mp4` so a
single glob compares Marionette and every baseline 1-to-1 on the same
identity (or identity pair). One-shot `setup_env.sh` per baseline.

### [`evaluation_metrics/`](evaluation_metrics/) — lip-sync evaluation
SyncNet-based audio-visual sync metrics (LSE-D, LSE-C, AV offset) for
judging generated videos against their driver audio. Operates on any
`outputs/**/samples/<sample_id>/panel.mp4` tree — Marionette's eval and
every SOTA baseline's output share the layout.

## Conventions

- Base training / inference configs live in `marionette/configs/`.
  Experiment-specific configs live inside each experiment subfolder.
- Outputs land at repo root under `outputs/<experiment>/…`.
- `<sample_id>` naming is UID-based (`id_0457` for same-identity,
  `id_0457_id_0009` for cross-identity) and consistent across every
  `sota_comparison/<baseline>/` **and** `marionette_eval/` — all of them
  enumerate from the frozen curated manifest under
  `experiments/sota_comparison/manifests/` via the shared
  `experiments.sota_comparison.dataset.pairing.build_samples`. So a single
  glob `outputs/**/samples/<sample_id>/panel.mp4` compares Marionette and
  every SOTA baseline 1-to-1 on the same physical identity (or pair).
