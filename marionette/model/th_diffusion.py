"""
Latent video diffusion model for talking-head generation.

Wraps the SD 2.1 UNet in a training loop with:
  - Image → VAE latent encoding
  - FLAME conditioning via THConditioning (spatial addition)
  - Audio encoding via wav2vec2 (cross-attention context)
  - Expression-weighted loss (amplify gradients on active face regions)
  - Classifier-free guidance dropout
  - Loss masking to non-reference frames

The expr_weight_map is always available for loss weighting, even when the
UNet's spatial conditioning is ablated (drop_expression_map=True). The
expr_weight_alpha parameter controls amplification strength (0 = uniform).
"""

import einops
import torch
import numpy as np
from functools import partial
from einops import rearrange, repeat
from torchvision.utils import make_grid

from ldm_base.ldm.models.diffusion.ddpm import LatentDiffusion
from ldm_base.ldm.util import exists, default, instantiate_from_config
from ldm_base.ldm.modules.diffusionmodules.util import make_beta_schedule
from ldm_base.ldm.models.diffusion.ddim import DDIMSampler

from marionette.model.utils import shift_schedule, enforce_zero_terminal_snr


class THDiffusion(LatentDiffusion):
    """
    Talking-Head Latent Diffusion Model.

    Inherits the full DDPM/LDM training infrastructure from LatentDiffusion
    (VAE first-stage, noise schedule, DDIM sampler, logging).

    Key additions over MMLDM:
      - audio_encoder: processes raw waveform windows → per-frame context tokens
      - audio_context is concatenated into the control dict alongside pos_enc/z_input
      - CFG dropout zeros both spatial conditioning AND audio context

    Expected batch keys:
        "jpg"    : (B, T, H, W, 3)    — image frames in [-1, 1]
        "audio"  : (B, T, W_audio)    — raw 16 kHz waveform window per frame
        "hint"   : dict               — spatial conditioning keys (verts_2d, etc.)

    The first frame (index 0) is always the reference frame; reference_mask should
    be 1 at index 0 and 0 elsewhere.
    """

    def __init__(
        self,
        control_key: str,
        only_mid_control: bool,
        n_frames: int,
        audio_key: str = "audio",
        audio_encoder_config: dict = None,   # instantiate_from_config spec
        *args,
        cfg_probability: float = 0.1,
        shift_schedule_flag: bool = False,
        sqrt_shift: bool = False,
        zero_snr_shift: bool = True,
        minus_one_shift: bool = True,
        negative_shift: bool = False,
        **kwargs,
    ):
        self.n_frames          = n_frames
        self.shift_schedule    = shift_schedule_flag
        self.sqrt_shift        = sqrt_shift
        self.minus_one_shift   = minus_one_shift
        self.control_key       = control_key
        self.only_mid_control  = only_mid_control
        self.cfg_probability   = cfg_probability
        self.negative_shift    = negative_shift
        self.zero_snr_shift    = zero_snr_shift
        self.audio_key         = audio_key
        self.expr_weight_alpha = kwargs.pop("expr_weight_alpha", 0.0)

        super().__init__(*args, **kwargs)

        # Audio encoder (optional — if None, model runs without audio conditioning)
        self.audio_encoder = None
        if audio_encoder_config is not None:
            self.audio_encoder = instantiate_from_config(audio_encoder_config)

        # Sanity: audio encoder presence must match UNet cross-attention flag
        unet_wants_audio = getattr(self.model.diffusion_model, "use_context", True)
        if unet_wants_audio and self.audio_encoder is None:
            raise ValueError(
                "THDiffusion: UNet has use_audio_context=True but no audio_encoder "
                "was configured. Either provide audio_encoder_config or set "
                "unet_config.params.use_audio_context: false."
            )
        if not unet_wants_audio and self.audio_encoder is not None:
            raise ValueError(
                "THDiffusion: UNet has use_audio_context=False but an audio_encoder "
                "was configured. Either set unet_config.params.use_audio_context: true "
                "or remove audio_encoder_config."
            )

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------
    def get_input(self, batch, k, bs=None, force_conditional=False, *args, **kwargs):
        with torch.no_grad():
            # ---- encode images to latent space ----
            x = batch[k]
            if x.ndim == 3:
                x = x[..., None]
            x = rearrange(x, 'b t h w c -> b t c h w').to(memory_format=torch.contiguous_format)
            if bs is not None:
                x = x[:bs]
            b_, t_ = x.shape[:2]
            x_flat = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            encoder_posterior = self.encode_first_stage(x_flat)
            z_flat = self.get_first_stage_encoding(encoder_posterior).detach()
            z = einops.rearrange(z_flat, '(b t) c h w -> b t c h w', b=b_)

            # ---- store GT latents in control dict for UNet reference masking ----
            batch[self.control_key]['z'] = z.detach()

            # ---- encode audio to per-frame context tokens ----
            audio_context = None
            if self.audio_encoder is not None and self.audio_key in batch:
                audio = batch[self.audio_key]            # (B, T, window_samples)
                if bs is not None:
                    audio = audio[:bs]
                audio_context = self.audio_encoder(audio)  # (B, T, S, D)

            # ---- unconditional conditioning (zeros) ----
            c_uncond = self.get_unconditional_conditioning(batch[self.control_key])
            # Zero out audio context for unconditional branch
            c_uncond["audio_context"] = (
                torch.zeros_like(audio_context) if audio_context is not None else None
            )

            loss_mask = batch.get("mask", None)

        # ---- conditional conditioning ----
        c_cond = self.get_learned_conditioning(batch[self.control_key])
        c_cond["audio_context"] = audio_context


        # ---- stochastic CFG mixing ----
        # expr_weight_map is excluded from CFG mixing — it's only used for loss
        # weighting and has a different channel count than pos_enc in ablation mode.
        _no_cfg_keys = {"expr_weight_map"}
        if not force_conditional:
            is_uncond = torch.rand(b_, device=x.device) < self.cfg_probability
            is_cond   = torch.logical_not(is_uncond)
            control = {}
            for key in c_cond:
                if c_cond[key] is None:
                    control[key] = None
                    continue
                if key in _no_cfg_keys:
                    control[key] = c_cond[key]
                    continue
                control[key] = (
                    einops.einsum(c_uncond[key], is_uncond, 'b ..., b -> b ...') +
                    einops.einsum(c_cond[key],   is_cond,   'b ..., b -> b ...')
                )
        else:
            control = c_cond

        if bs is not None:
            for key in control:
                if control[key] is not None:
                    control[key] = control[key][:bs]

        return z, dict(c_concat=[control], c_uncond=[c_uncond], mask=loss_mask)

    # ------------------------------------------------------------------
    # Decode (adds time dimension handling)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False):
        b_, t_ = z.shape[:2]
        z = einops.rearrange(z, 'b t c h w -> (b t) c h w')
        z = super().decode_first_stage(z, predict_cids)
        return einops.rearrange(z, '(b t) c h w -> b t c h w', b=b_)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, x.shape[:2], device=self.device).long()
        assert c is not None
        assert not self.shorten_cond_schedule
        return self.p_losses(x, c, t, *args, **kwargs)

    # ------------------------------------------------------------------
    # Apply diffusion model (UNet call)
    # ------------------------------------------------------------------
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        assert len(cond['c_concat']) == 1
        control = cond['c_concat'][0]

        # Pass audio context as `context` for cross-attention
        # audio_context shape: (B, T, S, D) — UNet reshapes to (B*T, S, D) internally
        audio_context = control.get("audio_context", None)

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=audio_context,   # will be handled inside THUnetModel.forward
            control=control,
            only_mid_control=self.only_mid_control,
        )
        return eps

    # ------------------------------------------------------------------
    # Loss computation (identical to MMLDM — masked to non-reference frames)
    # ------------------------------------------------------------------
    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        b_, t_ = t.shape[:2]
        t_flat         = einops.rearrange(t, 'b t -> (b t)')
        noise_flat     = einops.rearrange(noise,   'b t c h w -> (b t) c h w')
        x_start_flat   = einops.rearrange(x_start, 'b t c h w -> (b t) c h w')
        x_noisy_flat   = self.q_sample(x_start=x_start_flat, t=t_flat, noise=noise_flat)
        x_noisy        = einops.rearrange(x_noisy_flat, '(b t) c h w -> b t c h w', b=b_)

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        assert self.parameterization == 'eps'
        target = noise

        # Per-pixel loss: (B, T, C, H, W)
        loss_pixel = self.get_loss(model_output, target, mean=False)

        # ----- Spatial weight map: expression + foreground -----
        T_frames = x_start.shape[1]
        spatial_weight = torch.ones(b_, T_frames, 1, x_start.shape[3], x_start.shape[4],
                                    device=x_start.device)

        control = cond['c_concat'][0]

        # Expression-weighted: amplify loss in high-deformation regions.
        # Uses expr_weight_map (always 46 channels) rather than pos_enc,
        # so it works even when expression maps are ablated from UNet conditioning.
        if self.expr_weight_alpha > 0.0 and 'expr_weight_map' in control:
            expr_map_full = control['expr_weight_map']  # (B, T, H, W, 46) always
            if expr_map_full.shape[-1] > 45:
                expr_deform = expr_map_full[..., 42:45]  # (B, T, H, W, 3)
                expr_mag = expr_deform.norm(dim=-1)  # (B, T, H, W)
                # Normalize per sample to [0, 1]
                flat = expr_mag.reshape(b_, T_frames, -1)
                e_min = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
                e_max = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
                expr_norm = (expr_mag - e_min) / (e_max - e_min + 1e-8)
                expr_weight = 1.0 + self.expr_weight_alpha * expr_norm  # (B, T, H, W)
                spatial_weight = spatial_weight * expr_weight.unsqueeze(2)

        # Apply spatial weight to per-pixel loss
        loss_weighted = loss_pixel * spatial_weight
        weight_sum = spatial_weight.sum(dim=[2, 3, 4]).clamp(min=1.0)
        loss_simple = loss_weighted.sum(dim=[2, 3, 4]) / weight_sum  # (B, T)

        # Mask: only penalise generated (non-reference) frames
        ref_mask_full = control['ref_mask']  # (B, T, 1, H, W)
        ref_mask = torch.logical_not(ref_mask_full[:, :, 0, 0, 0]).float()  # (B, T)
        loss_simple_mean = (loss_simple * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1).clamp(min=1)
        loss_dict[f'{prefix}/loss_simple'] = loss_simple_mean.mean()

        logvar_t = self.logvar[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        loss = (loss * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1).clamp(min=1)
        if self.learn_logvar:
            loss_dict[f'{prefix}/loss_gamma'] = loss.mean()
            loss_dict['logvar'] = self.logvar.data.mean()

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False)
        loss_vlb = (loss_vlb * spatial_weight).sum(dim=[2, 3, 4]) / weight_sum
        loss_vlb = (self.lvlb_weights[t] * loss_vlb * ref_mask).sum(dim=-1) / ref_mask.sum(dim=-1).clamp(min=1)
        loss_vlb = loss_vlb.mean()
        loss_dict[f'{prefix}/loss_vlb'] = loss_vlb
        loss += self.original_elbo_weight * loss_vlb
        loss_dict[f'{prefix}/loss'] = loss

        return loss, loss_dict

    # ------------------------------------------------------------------
    # Conditioning helpers
    # ------------------------------------------------------------------
    def get_learned_conditioning(self, c):
        return self.cond_stage_model(c, unconditional=False)

    @torch.no_grad()
    def get_unconditional_conditioning(self, c):
        return self.cond_stage_model(c, unconditional=True)

    # ------------------------------------------------------------------
    # Noise schedule (identical to MMLDM)
    # ------------------------------------------------------------------
    def register_schedule(self, given_betas=None, beta_schedule="linear",
                          timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps,
                                       linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)

        if self.zero_snr_shift:
            print("Enforcing zero terminal SNR in noise schedule.")
            betas = enforce_zero_terminal_snr(betas)

        betas[betas > 0.99] = 0.99
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        if self.shift_schedule:
            n_gen = self.n_frames - 1 if self.minus_one_shift else self.n_frames
            shift_ratio = (64 ** 2) / (self.image_size ** 2 * n_gen)
            if self.negative_shift:
                shift_ratio = 1. / shift_ratio
            if self.sqrt_shift:
                shift_ratio = np.sqrt(shift_ratio)
            new_alpha_cumprod, new_betas = shift_schedule(alphas_cumprod, shift_ratio=shift_ratio)
            print(f"Shifted noise schedule by factor {shift_ratio:.4f}.")
            alphas       = 1. - new_betas
            betas        = new_betas
            alphas_cumprod = new_alpha_cumprod

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start  = linear_start
        self.linear_end    = linear_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas',                     to_torch(betas),                    persistent=False)
        self.register_buffer('alphas_cumprod',            to_torch(alphas_cumprod),            persistent=False)
        self.register_buffer('alphas_cumprod_prev',       to_torch(alphas_cumprod_prev),       persistent=False)
        self.register_buffer('sqrt_alphas_cumprod',       to_torch(np.sqrt(alphas_cumprod)),   persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)), persistent=False)
        self.register_buffer('log_one_minus_alphas_cumprod',  to_torch(np.log(1. - alphas_cumprod)),  persistent=False)
        self.register_buffer('sqrt_recip_alphas_cumprod',     to_torch(np.sqrt(1. / alphas_cumprod)), persistent=False)
        self.register_buffer('sqrt_recipm1_alphas_cumprod',   to_torch(np.sqrt(1. / alphas_cumprod - 1)), persistent=False)

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance',             to_torch(posterior_variance),  persistent=False)
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))), persistent=False)
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)), persistent=False)
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)), persistent=False)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())
        if self.audio_encoder is not None:
            # Only add unfrozen audio encoder parameters
            params += [p for p in self.audio_encoder.parameters() if p.requires_grad]
        if self.cond_stage_trainable:
            params += [p for p in self.cond_stage_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr)

    # ------------------------------------------------------------------
    # VRAM management helper (mirrors MMLDM)
    # ------------------------------------------------------------------
    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model  = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model  = self.cond_stage_model.cuda()

    # ------------------------------------------------------------------
    # Sample helper (used by log_images)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        shape = (self.n_frames, 4, self.image_size, self.image_size)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates
