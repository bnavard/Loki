# marionette_v2 · Reorganization Plan

Living document for the modularity / single-responsibility sweep.
Keep this file updated as items land.

---

## Naming (DONE — 2026-04-20)

Strip the v1-era `TH` (TalkingHead) prefix from classes & files; use names
tied to v2's actual role (warp-conditioned retargeting diffusion).

| Old | New | File |
|---|---|---|
| `THDiffusion` | `MarionetteDiffusion` | `marionette/model/diffusion.py` |
| `THUnetModel` | `MarionetteUNet` | `marionette/model/unet.py` |
| `THSampler` | `SlidingWindowSampler` | `marionette/model/sampler.py` |
| `THConditioning` | `SpatialConditioning` | `marionette/conditioning/conditioning.py` |

Config `target:` paths, imports, docstrings, and README references all
updated in the same commit.

---

## Target layout (ideal; subset marked "DO" is the practical work)

```
marionette/
├── data/
│   ├── dataset.py             # TalkingHeadDataset orchestrator                (EXISTS)
│   ├── indexing.py            # build_window_index                            [DO — phase C]
│   ├── flame_ops.py           # compute_flame_for_frame, build_flame_item     [DO — phase C]
│   ├── video_io.py            # load_frame, FrameReader                       [DO — phase C]
│   ├── image_ops.py           # crop_image, rescale_image                     [DO — phase C]
│   ├── verts.py               # verts_to_pytorch3d, get_bbox_from_verts       [DO — phase C]
│   ├── types.py               # SampleDict, HintDict TypedDicts               [DO — phase D]
│   ├── warp.py                # warp_reference_to_target (pure)                (EXISTS)
│   └── corruptions/
│       ├── __init__.py        # WarpCorruption dispatcher                     [DO — phase F]
│       ├── regions.py         # eye / mouth vertex-id loading                 [DO — phase F]
│       ├── primitives.py      # eye_drop, mouth_drop, cutouts, glasses        [DO — phase F]
│       └── photometric.py     # jitter                                        [DO — phase F]
├── conditioning/
│   ├── conditioning.py        # SpatialConditioning facade                     (EXISTS)
│   ├── positional.py          # PositionalEncoding                            [optional split]
│   ├── rasterize.py           # fused_property_rasterize                      [optional split]
│   ├── warp_sampling.py       # sample_warp                                   [optional split]
│   └── mesh2img.py            # PropRenderer, VertexShader                     (EXISTS)
├── model/
│   ├── diffusion.py           # MarionetteDiffusion                            (EXISTS, renamed)
│   ├── unet.py                # MarionetteUNet                                 (EXISTS, renamed)
│   ├── sampler.py             # SlidingWindowSampler                           (EXISTS, renamed)
│   ├── attention.py                                                            (EXISTS)
│   ├── encoders/
│   │   └── warp.py            # WarpEncoder                                   [optional]
│   └── schedule.py            # shift_schedule, enforce_zero_terminal_snr     [optional split from utils.py]
├── training/
│   ├── cfg.py                 # cfg_mix (pure)                                [DO — phase B]
│   ├── loss.py                # ref_masked_eps_mse (pure)                     [DO — phase B]
│   ├── trainer.py             # run_training                                  [optional]
│   └── lightning_wrapper.py   # LightningWrapper                              [optional]
├── inference/
│   ├── retargeting.py         # build_retargeted_verts                        [DO — phase A]
│   ├── conditioning_batch.py  # build_cond_batch_for_inference                [DO — phase A]
│   └── generate.py            # generate_cross_identity (thin orchestrator)   [DO — phase A]
├── callbacks/
│   ├── sanity_check.py                                                         (EXISTS)
│   ├── identity_viz.py                                                         (EXISTS)
│   └── viz_panel.py           # shared _colorize_3ch, _resize_nn, _build_panel [DO — minor]
├── flame/                     # unchanged                                      (EXISTS)
├── configs/                   # unchanged                                      (EXISTS)
└── cli/                       # optional — not doing now
    ├── train.py
    └── generate.py
```

---

## Phases (DO set only) · ordered by leverage

Order: **A → E → B → F → D → C** (previously agreed).

### Phase A · inference/ extraction (HIGH leverage, LOW risk)
**Status**: DONE (2026-04-20)

Goal: shrink `marionette/generate.py` from ~340 → ~80 lines by extracting
pure-function helpers into `marionette/inference/`.

Extractions (all pure, type-annotated):
1. `build_retargeted_verts(ref_fit, driver_fit, ref_crop_box, n_frames, flame_skinner)` → `(verts_np, offsets_np)`
2. `build_ref_verts_ndc(ref_fit, ref_crop_box, flame_skinner)` → `np.ndarray`
3. `encode_ref_image_to_latent(model, ref_img_norm, device)` → `torch.Tensor`  `(1, 1, 4, h, w)`
4. `build_inference_cond_batch(verts_np, offsets_np, ref_img_t, ref_verts_ndc, latent_res, n_frames, z_input, device)` → `dict`
5. `encode_warp_features(model.warp_encoder, warped_ref_t)` → `torch.Tensor`  `(1, T, D, h, w)`
6. `round_to_sampler_window(n_frames, V, R)` → `int`

Rewrite `generate_cross_identity` as a thin orchestrator calling these in sequence.

Drop the old standalone helpers inside `generate.py`; CLI `main()` keeps its
argparse and stays in `generate.py`.

### Phase E · unet.py cleanup (MEDIUM leverage, LOW risk)
**Status**: DONE (2026-04-20)

Inside `MarionetteUNet`:
- `_apply_ref_passthrough(x, z_input, ref_mask) → (x_swapped, x_input, ref_mask_inv)` (pure-ish method)
- `_inject_spatial_conditioning(pos_enc, warp_features) → pos_embedding`
- `forward` becomes: swap-in refs → flatten T → compute pos_embedding → base UNet call → swap-out refs.

### Phase B · training/cfg.py + training/loss.py (MEDIUM leverage, LOW risk)
**Status**: DONE (2026-04-20)

- `cfg_mix(c_cond, c_uncond, p, device, batch_size) → control_dict`: bernoulli draw per sample, zeros non-`None` keys.
- `ref_masked_eps_mse(eps_pred, eps_target, ref_mask) → (loss_simple_mean, ref_mask_1d)`: uniform pixel-MSE masked to non-reference slots.

`MarionetteDiffusion.get_input` and `p_losses` become thin callers.

### Phase F · corruptions/ package split (LOW leverage, LOW risk)
**Status**: DONE (2026-04-20)

- `corruptions/regions.py` — `load_region_vert_ids(pkl_path) → (left_eye, right_eye, mouth)`, `verts_ndc_to_px`, `polygon_mask`.
- `corruptions/primitives.py` — 5 per-corruption functions (eye_drop, mouth_drop, cutouts, glasses, jitter) — already pure-ish, just move.
- `corruptions/__init__.py` — `WarpCorruption` class remains the dispatcher; mounted as `from marionette.data.corruptions import WarpCorruption`.

### Phase D · TypedDict contracts (LOW-MEDIUM leverage, no runtime effect)
**Status**: DONE (2026-04-20)

`marionette/data/types.py` defining `SampleDict`, `HintDict`, and
(optionally) `ControlDict`. Type-annotate `TalkingHeadDataset.__getitem__`,
`SpatialConditioning.forward` (conditionally — LDM inheritance may limit this),
and `generate_cross_identity`.

### Phase C · data/ helper split (POLISH, LOW risk)
**Status**: DONE (2026-04-20)

Split `marionette/data/utils.py` (93 lines, mixed concerns) into:
- `video_io.py` — `load_frame`, `FrameReader`
- `image_ops.py` — `crop_image`, `rescale_image`
- `verts.py` — `verts_to_pytorch3d`, `get_bbox_from_verts`, `get_square_bbox`

Update all importers.

---

## Non-goals (explicitly skipped from the ideal layout)

- `cli/` split — scripts already work; not worth the import-hop.
- Full `training/` and `inference/` as top-level subpackages — only the
  pure-function extractions inside matter. Directory-level split adds no
  correctness win.
- `__init__.py` re-exports — add if/when external callers appear.

---

## Guidelines

- **Single responsibility**: one function = one task. If you need "and" to
  describe what it does, split.
- **Pure where possible**: a function that only reads its args and returns
  values is testable in isolation. Orchestrators sequence pure helpers.
- **No lazy attribute registration in `nn.Module`**: every buffer/param
  registered in `__init__`. (DDP-synced state must be symmetric across
  ranks from wrap time.)
- **Comments explain WHY not WHAT**: identifier names carry the what.
  Comments flag hidden constraints, invariants, or workarounds.
- **No comments about removed code**: if the old variant is gone, don't
  leave a breadcrumb; let `git log` carry that.
