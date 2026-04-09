"""
Dummy pipeline tests for the Talking-Head Diffusion Model.

All tests use random tensors — NO real checkpoints, FLAME assets, or audio files
are required. The tests verify that tensor shapes flow correctly through each
module and that no assertion / shape error fires.

Run:
    cd <repo_root>
    python talkinghead/tests/test_pipeline.py
  or:
    python -m pytest talkinghead/tests/test_pipeline.py -v   (if pytest installed)

Test coverage:
  1. THConditioning  — channel count property + conditional/unconditional shapes
  2. AudioEncoder    — waveform → context tokens (no pretrained weights downloaded)
  3. THUnetModel     — full UNet forward pass with spatial + audio conditioning
  4. THDiffusion     — training step (get_input + p_losses) with mock VAE
  5. THSampler       — inference loop (2 DDIM steps, fake model)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
# Use CUDA — required for xformers attention (CPU not supported)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

B  = 2      # batch size
T  = 4      # video frames (1 ref + 3 gen), small for speed
H  = 64     # latent spatial resolution
R_IMG = 128 # full-res image size — test VAE uses ch_mult=[1,2] → 2x downsampling → 128/2=64=H

C_LATENT    = 4    # VAE latent channels
C_COND      = 46   # conditioning channels (42 pos + 3 expr + 1 ref_mask)
CONTEXT_DIM = 768  # audio cross-attention dimension
MODEL_CH    = 64   # reduced channels for fast tests (config uses 320)
N_AUDIO_TOK = 5    # audio tokens per frame (after backbone)
# 8000 samples = 0.5s at 16kHz → ~25 tokens after wav2vec2 stride-320
# must be > mask_time_length (10) to avoid masking assertion in training mode
AUDIO_WIN   = 8000


# ---------------------------------------------------------------------------
# Helper: fake control dict as expected by THUnetModel.forward / THSampler
#
# Shape contract (no inner T dim at per-frame level for the sampler):
#   pos_enc       : (B, T, H, H, C_cond)   — for UNet direct call
#   z_input       : (B, T, C, H, H)
#   ref_mask      : (B, T, 1, H, H)
#   audio_context : (B, T, S, D)
# ---------------------------------------------------------------------------
def make_control(B, T, H, C_cond, C_latent, context_dim, n_audio_tok, device=DEVICE):
    ref_mask = torch.zeros(B, T, 1, H, H, device=device)
    ref_mask[:, 0] = 1.0  # first frame is the reference

    z_input = torch.randn(B, T, C_latent, H, H, device=device)
    z_input[:, 1:] = 0.0  # only reference slot has a GT latent

    return {
        "pos_enc":       torch.randn(B, T, H, H, C_cond, device=device),
        "z_input":       z_input,
        "ref_mask":      ref_mask,
        "audio_context": torch.randn(B, T, n_audio_tok, context_dim, device=device),
    }


# Helper: per-frame conditioning for the sampler
# Shape: (N_frames, H, H, C) — NO inner T dim, sampler adds it via 2-D indexing
def make_sampler_cond(n, H, C_cond, C_latent, context_dim, n_audio_tok, device=DEVICE):
    ref_mask = torch.zeros(n, 1, H, H, device=device)
    return {
        "pos_enc":       torch.randn(n, H, H, C_cond, device=device),
        "z_input":       torch.randn(n, C_latent, H, H, device=device),
        "ref_mask":      ref_mask,
        "audio_context": torch.randn(n, n_audio_tok, context_dim, device=device),
    }


# ===========================================================================
# Test 1: THConditioning
# ===========================================================================
class TestTHConditioning:
    def test_channel_count(self):
        """n_conditioning_channels must be 46 with default settings."""
        from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
        cond = THConditioning(
            image_size=H, positional_channels=42,
            use_ray_directions=False, use_expr_deformation=True, use_crop_mask=False,
        )
        assert cond.n_conditioning_channels == C_COND, (
            f"Expected {C_COND}, got {cond.n_conditioning_channels}"
        )

    def test_channel_count_with_ray(self):
        """With ray directions enabled: 42 + 3 + 3 + 1 = 49 channels."""
        from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
        cond = THConditioning(
            image_size=H, positional_channels=42,
            use_ray_directions=True, use_expr_deformation=True, use_crop_mask=False,
        )
        assert cond.n_conditioning_channels == 49

    def test_unconditional_output_shape(self):
        """Unconditional path returns all-zero pos_enc of correct shape."""
        from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
        cond = THConditioning(image_size=H, positional_channels=42,
                              use_ray_directions=False, use_expr_deformation=True)
        V_dim = 100
        batch = {
            "verts_2d":       torch.randn(B, T, V_dim, 2),
            "offsets_3d":     torch.randn(B, T, V_dim, 3),
            "reference_mask": torch.zeros(B, T, 1, H, H),
        }
        out = cond(batch, unconditional=True)

        assert out["pos_enc"].shape  == (B, T, H, H, C_COND), f"Got {out['pos_enc'].shape}"
        assert out["ref_mask"].shape == (B, T, 1, H, H)
        assert out["z_input"] is None
        assert out["pos_enc"].abs().sum().item() == 0.0, "Unconditional pos_enc must be zeros"

    def test_conditional_output_shape_mocked_renderer(self):
        """
        Conditional path with mocked PropRenderer (no FLAME assets needed).
        pytorch3d may be broken in this env, so we patch the renderer directly.
        """
        from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
        from unittest.mock import MagicMock, patch

        V_dim = 100
        cond = THConditioning(
            image_size=H, positional_channels=42,
            use_ray_directions=False, use_expr_deformation=True, super_resolution=1,
        )

        # Mock the renderer — avoid pytorch3d entirely
        fake_pose_map = torch.randn(B * T, H, H, 6)   # 3 pose + 3 expr channels
        fake_mask     = torch.ones(B * T, H, H, 1).bool()

        mock_renderer = MagicMock()
        mock_renderer.render.return_value = (fake_pose_map, fake_mask)
        cond._renderer = mock_renderer   # inject directly, bypassing lazy-init

        batch = {
            "verts_2d":       torch.randn(B, T, V_dim, 2),
            "offsets_3d":     torch.randn(B, T, V_dim, 3),
            "reference_mask": torch.zeros(B, T, 1, H, H),
            "z":              torch.randn(B, T, C_LATENT, H, H),
        }
        out = cond(batch, unconditional=False)

        assert out["pos_enc"].shape  == (B, T, H, H, C_COND), f"Got {out['pos_enc'].shape}"
        assert out["z_input"].shape  == (B, T, C_LATENT, H, H)
        assert out["ref_mask"].shape == (B, T, 1, H, H)


# ===========================================================================
# Test 2: AudioEncoder
# ===========================================================================
class TestAudioEncoder:
    def test_output_shape(self):
        """
        Random-weight AudioEncoder: (B, T, AUDIO_WIN) → (B, T, num_tokens, context_dim).
        Uses eval() mode to disable wav2vec2's time masking (which requires
        sequence_length > mask_time_length).
        """
        from talkinghead_sd21_unet_cap4d_based.model.audio_encoder import AudioEncoder

        enc = AudioEncoder(context_dim=CONTEXT_DIM, use_pretrained=False)
        enc.eval()  # disable masking in backbone

        waveform = torch.randn(B, T, AUDIO_WIN)
        with torch.no_grad():
            out = enc(waveform)

        assert out.ndim == 4, f"Expected 4-D (B,T,S,D), got {out.shape}"
        assert out.shape[0] == B
        assert out.shape[1] == T
        assert out.shape[3] == CONTEXT_DIM
        print(f"  AudioEncoder output: {tuple(out.shape)}")


# ===========================================================================
# Test 3: THUnetModel
# ===========================================================================
class TestTHUnetModel:
    def _build_unet(self):
        from talkinghead_sd21_unet_cap4d_based.model.th_unet import THUnetModel
        return THUnetModel(
            image_size=H, in_channels=C_LATENT, out_channels=C_LATENT,
            model_channels=MODEL_CH, condition_channels=C_COND,
            context_dim=CONTEXT_DIM, time_steps=T, temporal_mode="3d",
            attention_resolutions=[4, 2], num_res_blocks=1,
            channel_mult=[1, 2], num_head_channels=32,
            use_spatial_transformer=True, use_linear_in_transformer=True,
            transformer_depth=1, legacy=False,
        ).to(DEVICE)

    def test_forward_shape(self):
        """UNet output shape must be (B, T, C_latent, H, H)."""
        unet    = self._build_unet()
        control = make_control(B, T, H, C_COND, C_LATENT, CONTEXT_DIM, N_AUDIO_TOK)

        x         = torch.randn(B, T, C_LATENT, H, H, device=DEVICE)
        timesteps = torch.randint(0, 1000, (B, T), device=DEVICE)

        with torch.no_grad():
            out = unet(x, timesteps, context=None, control=control)

        assert out.shape == (B, T, C_LATENT, H, H), f"UNet output shape: {out.shape}"

    def test_reference_frame_passthrough(self):
        """
        For reference frame slots (ref_mask=1), output must equal (x - z_input),
        i.e. the GT noise residual is passed through unchanged.
        """
        unet    = self._build_unet()
        control = make_control(B, T, H, C_COND, C_LATENT, CONTEXT_DIM, N_AUDIO_TOK)

        x         = torch.randn(B, T, C_LATENT, H, H, device=DEVICE)
        timesteps = torch.randint(0, 1000, (B, T), device=DEVICE)

        with torch.no_grad():
            out = unet(x, timesteps, context=None, control=control)

        expected_ref = x[:, 0] - control["z_input"][:, 0]
        assert torch.allclose(out[:, 0], expected_ref, atol=1e-5), (
            "Reference frame output does not equal (x - z_input)"
        )


# ===========================================================================
# Test 4: THDiffusion (training step with lightweight sub-modules)
# ===========================================================================
class TestTHDiffusion:
    def _build_model(self):
        from talkinghead_sd21_unet_cap4d_based.model.th_unet import THUnetModel
        from talkinghead_sd21_unet_cap4d_based.conditioning.th_conditioning import THConditioning
        from talkinghead_sd21_unet_cap4d_based.model.audio_encoder import AudioEncoder
        from talkinghead_sd21_unet_cap4d_based.model.th_diffusion import THDiffusion
        from omegaconf import OmegaConf

        vae_cfg = OmegaConf.create({
            "embed_dim": C_LATENT,
            "ddconfig": {
                "double_z": True, "z_channels": C_LATENT,
                "resolution": 32, "in_channels": 3, "out_ch": 3,
                "ch": 32, "ch_mult": [1, 2], "num_res_blocks": 1,
                "attn_resolutions": [], "dropout": 0.0,
            },
            "lossconfig": {"target": "torch.nn.Identity"},
        })

        model = THDiffusion(
            control_key="hint", only_mid_control=False, n_frames=T,
            audio_key="audio",
            first_stage_key="jpg", cond_stage_key="hint",
            conditioning_key="crossattn", image_size=H,
            channels=C_LATENT, scale_factor=0.18215, use_ema=False,
            timesteps=100, linear_start=0.00085, linear_end=0.0120,
            parameterization="eps", l_simple_weight=1.0,
            original_elbo_weight=0.0, v_posterior=0.0,
            learn_logvar=False, logvar_init=0.0, num_timesteps_cond=1,
            unet_config={
                "target": "talkinghead.model.th_unet.THUnetModel",
                "params": {
                    "image_size": H, "in_channels": C_LATENT, "out_channels": C_LATENT,
                    "model_channels": MODEL_CH, "condition_channels": C_COND,
                    "context_dim": CONTEXT_DIM, "time_steps": T, "temporal_mode": "3d",
                    "attention_resolutions": [4, 2], "num_res_blocks": 1,
                    "channel_mult": [1, 2], "num_head_channels": 32,
                    "use_spatial_transformer": True, "use_linear_in_transformer": True,
                    "transformer_depth": 1, "legacy": False,
                },
            },
            first_stage_config={
                "target": "controlnet.ldm.models.autoencoder.AutoencoderKL",
                "params": OmegaConf.to_container(vae_cfg),
            },
            cond_stage_config={
                "target": "talkinghead.conditioning.th_conditioning.THConditioning",
                "params": {
                    "image_size": H, "positional_channels": 42,
                    "use_ray_directions": False, "use_expr_deformation": True,
                    "super_resolution": 1,
                },
            },
        ).to(DEVICE)

        model.audio_encoder = AudioEncoder(
            context_dim=CONTEXT_DIM, use_pretrained=False
        ).to(DEVICE)
        model.audio_encoder.eval()

        return model

    def _make_batch(self):
        V_verts  = 100
        ref_mask = torch.zeros(B, T, 1, H, H, device=DEVICE)
        ref_mask[:, 0] = 1.0
        return {
            "jpg":   torch.randn(B, T, R_IMG, R_IMG, 3, device=DEVICE),
            "audio": torch.randn(B, T, AUDIO_WIN, device=DEVICE),
            "hint": {
                "verts_2d":       torch.randn(B, T, V_verts, 2, device=DEVICE),
                "offsets_3d":     torch.randn(B, T, V_verts, 3, device=DEVICE),
                "reference_mask": ref_mask,
            },
        }

    def test_get_input_shapes(self):
        from unittest.mock import MagicMock

        model = self._build_model()
        batch = self._make_batch()

        # Inject mock renderer to avoid pytorch3d
        fake_pose_map = torch.randn(B * T, H, H, 6, device=DEVICE)
        fake_mask     = torch.ones(B * T, H, H, 1, dtype=torch.bool, device=DEVICE)
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = (fake_pose_map, fake_mask)
        model.cond_stage_model._renderer = mock_renderer

        z, cond = model.get_input(batch, "jpg")

        assert z.shape == (B, T, C_LATENT, H, H), f"z shape: {z.shape}"
        ctrl = cond["c_concat"][0]
        assert ctrl["pos_enc"].shape       == (B, T, H, H, C_COND)
        assert ctrl["audio_context"].shape == (B, T, ctrl["audio_context"].shape[2], CONTEXT_DIM)
        assert ctrl["z_input"].shape       == (B, T, C_LATENT, H, H)
        assert ctrl["ref_mask"].shape      == (B, T, 1, H, H)

    def test_training_loss(self):
        """Full training forward should return a positive scalar loss."""
        from unittest.mock import MagicMock

        model = self._build_model()
        batch = self._make_batch()

        fake_pose_map = torch.randn(B * T, H, H, 6, device=DEVICE)
        fake_mask     = torch.ones(B * T, H, H, 1, dtype=torch.bool, device=DEVICE)
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = (fake_pose_map, fake_mask)
        model.cond_stage_model._renderer = mock_renderer

        z, cond = model.get_input(batch, "jpg")
        loss, loss_dict = model(z, cond)

        assert loss.ndim == 0,            f"Loss should be scalar, got {loss.shape}"
        assert loss.item() > 0.0,         "Loss should be positive"
        assert "train/loss" in loss_dict
        print(f"  Training loss: {loss.item():.4f}")


# ===========================================================================
# Test 5: THSampler
# ===========================================================================
class TestTHSampler:
    def _make_fake_model(self):
        """Minimal mock satisfying THSampler's model interface."""
        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                ts    = 100
                betas = np.linspace(0.0001, 0.02, ts, dtype=np.float32)
                alphas     = 1. - betas
                alphas_cp  = np.cumprod(alphas)
                self.num_timesteps       = ts
                self.alphas_cumprod      = torch.tensor(alphas_cp, device=DEVICE)
                self.betas               = torch.tensor(betas, device=DEVICE)
                self.alphas_cumprod_prev = torch.tensor(np.append(1., alphas_cp[:-1]), device=DEVICE)

                # Required by register_buffer call in sampler
                def _reg(name, val):
                    setattr(self, name, val)
                self.register_buffer = _reg

            def apply_model(self, x, t, cond):
                return torch.randn_like(x)

            @property
            def device(self):
                return DEVICE

        return FakeModel()

    def test_sample_output_shape(self):
        from talkinghead_sd21_unet_cap4d_based.model.th_sampler import THSampler

        R     = 1
        V     = 4   # UNet window: 1 ref + 3 gen
        n_gen = 6   # total frames to generate (divisible by G=V-R=3)

        model   = self._make_fake_model()
        sampler = THSampler(model)

        # Sampler indexes dim-0 of each cond dict.
        # After 2-D indexing (ref_batches[[b]] shape (1,R)), result is (1, R, ...).
        # So per-frame cond must NOT have an inner T dim.
        ref_c   = make_sampler_cond(R,     H, C_COND, C_LATENT, CONTEXT_DIM, N_AUDIO_TOK)
        ref_u   = make_sampler_cond(R,     H, C_COND, C_LATENT, CONTEXT_DIM, N_AUDIO_TOK)
        gen_c   = make_sampler_cond(n_gen, H, C_COND, C_LATENT, CONTEXT_DIM, N_AUDIO_TOK)
        gen_u   = make_sampler_cond(n_gen, H, C_COND, C_LATENT, CONTEXT_DIM, N_AUDIO_TOK)

        latents = sampler.sample(
            S=2,
            ref_cond=ref_c, ref_uncond=ref_u,
            gen_cond=gen_c, gen_uncond=gen_u,
            latent_shape=(C_LATENT, H, H),
            V=V, R=R, cfg_scale=2.0, verbose=False,
        )

        assert latents.shape == (n_gen, C_LATENT, H, H), (
            f"Sampler output shape mismatch: {latents.shape}"
        )
        print(f"  Sampler output: {tuple(latents.shape)} ✓")


# ===========================================================================
# Runner
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"Talking-Head Pipeline Shape Tests  (device: {DEVICE})")
    print("=" * 60)

    tests = [
        ("THConditioning.channel_count",        TestTHConditioning().test_channel_count),
        ("THConditioning.channel_count_ray",     TestTHConditioning().test_channel_count_with_ray),
        ("THConditioning.unconditional_shape",   TestTHConditioning().test_unconditional_output_shape),
        ("THConditioning.conditional_shape",     TestTHConditioning().test_conditional_output_shape_mocked_renderer),
        ("AudioEncoder.output_shape",            TestAudioEncoder().test_output_shape),
        ("THUnetModel.forward_shape",            TestTHUnetModel().test_forward_shape),
        ("THUnetModel.ref_passthrough",          TestTHUnetModel().test_reference_frame_passthrough),
        ("THDiffusion.get_input_shapes",         TestTHDiffusion().test_get_input_shapes),
        ("THDiffusion.training_loss",            TestTHDiffusion().test_training_loss),
        ("THSampler.sample_output_shape",        TestTHSampler().test_sample_output_shape),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}")
            print(f"         {e}")
            traceback.print_exc()
            failed += 1

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
