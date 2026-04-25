"""
Latent video diffusion model for talking-head generation.

Wraps the SD 2.1 UNet in a training loop with:
  - Image → VAE latent encoding (frozen VAE).
  - FLAME spatial conditioning via SpatialConditioning (45ch: pos_enc + deform).
  - Audio encoding via wav2vec2 (cross-attention context, optional).
  - ReferenceNet-style identity injection: a frozen SD 2.1 UNet processes the
    VAE-encoded reference frame once per sample; per-layer self-attention
    inputs are cached and concatenated to the gen UNet's self-attention K/V
    pool (see `marionette.model.ref_unet.RefFeatureExtractor`).
  - Classifier-free guidance dropout (null token = zero-filled cond dict).

Loss is uniform across all gen slots; the ref slot lives in a separate UNet
and is not part of the gen UNet's output — so there is no per-slot loss
mask.

Batch contract:
  "target_video" : (B, T+1, H, W, 3)  frames, in [-1, 1]. Slot 0 is a random
                                       reference frame from the clip; slots
                                       1..T are the contiguous generation
                                       target window.
  "audio"        : (B, T, W_audio)     raw 16 kHz per-frame windows for the
                                       T gen slots (no audio for the ref).
  "hint"         : dict                FLAME conditioning for the T gen slots
                                       (driver_verts, driver_deform).
"""

import einops
import torch
import numpy as np
from functools import partial
from einops import rearrange

from ldm_base.ldm.models.diffusion.ddpm import LatentDiffusion
from ldm_base.ldm.util import exists, default, instantiate_from_config
from ldm_base.ldm.modules.diffusionmodules.util import make_beta_schedule

from marionette.model.utils import shift_schedule, enforce_zero_terminal_snr
from marionette.model.ref_unet import RefFeatureExtractor


class MarionetteDiffusion(LatentDiffusion):
    """Talking-head latent video diffusion with ReferenceNet identity injection."""

    def __init__(
        self,
        control_key: str,
        only_mid_control: bool,
        n_frames: int,
        audio_key: str = "audio",
        audio_encoder_config: dict = None,
        ref_unet_config: dict = None,
        *args,
        cfg_probability: float = 0.1,
        shift_schedule_flag: bool = False,
        sqrt_shift: bool = False,
        zero_snr_shift: bool = True,
        minus_one_shift: bool = True,
        negative_shift: bool = False,
        **kwargs,
    ):
        self.n_frames         = n_frames
        self.shift_schedule   = shift_schedule_flag
        self.sqrt_shift       = sqrt_shift
        self.minus_one_shift  = minus_one_shift
        self.control_key      = control_key
        self.only_mid_control = only_mid_control
        self.cfg_probability  = cfg_probability
        self.negative_shift   = negative_shift
        self.zero_snr_shift   = zero_snr_shift
        self.audio_key        = audio_key

        super().__init__(*args, **kwargs)

        self.audio_encoder = None
        if audio_encoder_config is not None:
            self.audio_encoder = instantiate_from_config(audio_encoder_config)

        unet_wants_audio = getattr(self.model.diffusion_model, "use_context", True)
        if unet_wants_audio and self.audio_encoder is None:
            raise ValueError(
                "MarionetteDiffusion: UNet has use_audio_context=True but no "
                "audio_encoder was configured."
            )
        if not unet_wants_audio and self.audio_encoder is not None:
            raise ValueError(
                "MarionetteDiffusion: UNet has use_audio_context=False but an "
                "audio_encoder was configured."
            )

        if ref_unet_config is None:
            raise ValueError(
                "MarionetteDiffusion: ref_unet_config is required. Provide a "
                "`unet_config` dict for the frozen SD 2.1 reference UNet."
            )
        self.ref_extractor = RefFeatureExtractor(ref_unet_config)

    def get_input(self, batch, k, bs=None, force_conditional=False, *args, **kwargs):
        """Encode target_video, split slot 0 (ref) from slots 1..T (gen),
        and build the CFG-mixed control dict.

        Returns `(gen_z, cond)` where `gen_z` is the T-slot clean latent that
        Lightning will noise during training. `ref_z` (clean ref latent) rides
        in the control dict so `apply_model` can feed the ref UNet.
        """
        with torch.no_grad():
            x = batch[k]
            if x.ndim == 3:
                x = x[..., None]
            x = rearrange(x, 'b t h w c -> b t c h w').to(memory_format=torch.contiguous_format)
            if bs is not None:
                x = x[:bs]
            b_, t_plus_1 = x.shape[:2]

            x_flat = einops.rearrange(x, 'b t c h w -> (b t) c h w')
            encoder_posterior = self.encode_first_stage(x_flat)
            z_flat = self.get_first_stage_encoding(encoder_posterior).detach()
            z = einops.rearrange(z_flat, '(b t) c h w -> b t c h w', b=b_)

            ref_z = z[:, 0]        # (B, 4, h, w)   — feeds ref UNet
            gen_z = z[:, 1:]       # (B, T, 4, h, w) — denoising target

            audio_context = None
            if self.audio_encoder is not None and self.audio_key in batch:
                audio = batch[self.audio_key]
                if bs is not None:
                    audio = audio[:bs]
                audio_context = self.audio_encoder(audio)

            loss_mask = batch.get("mask", None)

        c_cond = self.cond_stage_model(batch[self.control_key])
        c_cond["audio_context"] = audio_context
        c_cond["ref_z"]         = ref_z

        # Null token: zero every tensor in c_cond. CFG dropout replaces a
        # sample's conditioning with this null with probability cfg_probability.
        c_uncond = {
            key: (torch.zeros_like(v) if torch.is_tensor(v) else v)
            for key, v in c_cond.items()
        }

        if not force_conditional:
            is_uncond = torch.rand(b_, device=x.device) < self.cfg_probability
            is_cond   = torch.logical_not(is_uncond)
            control = {}
            for key in c_cond:
                if c_cond[key] is None:
                    control[key] = None
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

        return gen_z, dict(c_concat=[control], c_uncond=[c_uncond], mask=loss_mask)

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False):
        b_, _ = z.shape[:2]
        z = einops.rearrange(z, 'b t c h w -> (b t) c h w')
        z = super().decode_first_stage(z, predict_cids)
        return einops.rearrange(z, '(b t) c h w -> b t c h w', b=b_)

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, x.shape[:2], device=self.device).long()
        assert c is not None
        assert not self.shorten_cond_schedule
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        assert len(cond['c_concat']) == 1
        control = cond['c_concat'][0]

        ref_z = control["ref_z"]
        ref_features = self.ref_extractor(ref_z)

        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            control=control,
            ref_features=ref_features,
            only_mid_control=self.only_mid_control,
        )
        return eps

    def p_losses(self, x_start, cond, t, noise=None):
        """Uniform ε-prediction loss over all gen slots. No ref-slot masking —
        the ref lives in the separate RefFeatureExtractor, not in x_start."""
        noise = default(noise, lambda: torch.randn_like(x_start))

        b_, t_ = t.shape[:2]
        t_flat       = einops.rearrange(t,       'b t -> (b t)')
        noise_flat   = einops.rearrange(noise,   'b t c h w -> (b t) c h w')
        x_start_flat = einops.rearrange(x_start, 'b t c h w -> (b t) c h w')
        x_noisy_flat = self.q_sample(x_start=x_start_flat, t=t_flat, noise=noise_flat)
        x_noisy      = einops.rearrange(x_noisy_flat, '(b t) c h w -> b t c h w', b=b_)

        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        assert self.parameterization == 'eps'
        target = noise

        loss_pixel  = self.get_loss(model_output, target, mean=False)
        loss_simple = loss_pixel.mean(dim=[2, 3, 4])                   # (B, T)

        loss_simple_mean = loss_simple.mean()
        loss_dict[f'{prefix}/loss_simple'] = loss_simple_mean

        logvar_t = self.logvar[t]
        loss = (loss_simple / torch.exp(logvar_t) + logvar_t).mean(dim=-1)
        if self.learn_logvar:
            loss_dict[f'{prefix}/loss_gamma'] = loss.mean()
            loss_dict['logvar'] = self.logvar.data.mean()

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=[2, 3, 4])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict[f'{prefix}/loss_vlb'] = loss_vlb
        loss += self.original_elbo_weight * loss_vlb
        loss_dict[f'{prefix}/loss'] = loss

        return loss, loss_dict

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
            alphas         = 1. - new_betas
            betas          = new_betas
            alphas_cumprod = new_alpha_cumprod

        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start  = linear_start
        self.linear_end    = linear_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas',                             to_torch(betas),                         persistent=False)
        self.register_buffer('alphas_cumprod',                    to_torch(alphas_cumprod),                persistent=False)
        self.register_buffer('alphas_cumprod_prev',               to_torch(alphas_cumprod_prev),           persistent=False)
        self.register_buffer('sqrt_alphas_cumprod',               to_torch(np.sqrt(alphas_cumprod)),       persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod',     to_torch(np.sqrt(1. - alphas_cumprod)),  persistent=False)
        self.register_buffer('log_one_minus_alphas_cumprod',      to_torch(np.log(1. - alphas_cumprod)),   persistent=False)
        self.register_buffer('sqrt_recip_alphas_cumprod',         to_torch(np.sqrt(1. / alphas_cumprod)),  persistent=False)
        self.register_buffer('sqrt_recipm1_alphas_cumprod',       to_torch(np.sqrt(1. / alphas_cumprod - 1)), persistent=False)

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance',             to_torch(posterior_variance), persistent=False)
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

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.parameters())
        if self.audio_encoder is not None:
            params += [p for p in self.audio_encoder.parameters() if p.requires_grad]
        if self.cond_stage_trainable:
            params += [p for p in self.cond_stage_model.parameters() if p.requires_grad]
        # ref_extractor is frozen — all its params have requires_grad=False,
        # so they're excluded automatically.
        return torch.optim.AdamW(params, lr=lr)

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model  = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model  = self.cond_stage_model.cuda()

    @torch.no_grad()
    def sample_video(
        self,
        control: dict,
        control_uncond: dict,
        n_frames: int,
        latent_shape: tuple,
        n_ddim_steps: int = 50,
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        """DDIM-sample a T-slot video latent with classifier-free guidance.

        `ref_features` are extracted once (ref UNet is frozen and the ref
        latent is fixed across sampling steps) and reused at every DDIM step.
        The ref never occupies a slot in the output — it's injected via
        self-attention K/V inside `MarionetteUNet`.
        """
        from ldm_base.ldm.modules.diffusionmodules.util import (
            make_ddim_timesteps, make_ddim_sampling_parameters,
        )

        device = next(self.parameters()).device
        unet   = self.model.diffusion_model

        ref_features        = self.ref_extractor(control["ref_z"])
        ref_features_uncond = [torch.zeros_like(f) for f in ref_features]

        ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=n_ddim_steps,
            num_ddpm_timesteps=self.num_timesteps,
            verbose=False,
        )
        _, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod.detach().cpu(),
            ddim_timesteps=ddim_timesteps,
            eta=0.0,
            verbose=False,
        )
        # `make_ddim_sampling_parameters` returns numpy arrays (except `sigmas`,
        # which it casts to a torch tensor). Normalize all three to tensors.
        ddim_alphas                = torch.as_tensor(ddim_alphas,      device=device, dtype=torch.float32)
        ddim_alphas_prev           = torch.as_tensor(ddim_alphas_prev, device=device, dtype=torch.float32)
        ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - ddim_alphas)

        x = torch.randn(1, n_frames, *latent_shape, device=device)

        for i in reversed(range(n_ddim_steps)):
            t_long = torch.full(
                (1, n_frames), int(ddim_timesteps[i]),
                device=device, dtype=torch.long,
            )

            eps_cond = unet(
                x=x, timesteps=t_long,
                control=control, ref_features=ref_features,
                only_mid_control=self.only_mid_control,
            )
            eps_uncond = unet(
                x=x, timesteps=t_long,
                control=control_uncond, ref_features=ref_features_uncond,
                only_mid_control=self.only_mid_control,
            )
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            a_t       = ddim_alphas[i]
            a_prev    = ddim_alphas_prev[i]
            sqrt_1mat = ddim_sqrt_one_minus_alphas[i]

            pred_x0 = (x - sqrt_1mat * eps) / torch.sqrt(a_t)
            dir_xt  = torch.sqrt(1.0 - a_prev) * eps
            x       = torch.sqrt(a_prev) * pred_x0 + dir_xt

        return x.squeeze(0)   # (T, C, h, w)
