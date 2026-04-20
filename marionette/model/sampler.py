"""
Stochastic I/O sampler for Marionette talking-head generation.

Key features:
  1. R defaults to 1 (single reference frame for talking head). Can still be
     increased if multiple reference images are provided.
  2. audio_context is threaded through ref_cond / gen_cond dicts and
     concatenated along the time axis alongside the spatial conditioning keys —
     the UNet receives a combined (B, V, S, D) tensor that it reshapes internally.

Usage:
    sampler = SlidingWindowSampler(model)
    latents = sampler.sample(
        S=50,
        ref_cond=ref_cond,    # dict with spatial_cond, z_input, ref_mask, audio_context
        ref_uncond=ref_uncond,
        gen_cond=gen_cond,    # dict with spatial_cond, z_input(zeros), ref_mask, audio_context
        gen_uncond=gen_uncond,
        latent_shape=(4, 64, 64),
        V=16,   # total slots per forward pass (1 ref + 15 gen)
        R=1,    # reference frames (usually 1 for talking head)
        cfg_scale=2.0,
    )
"""

from typing import Tuple, Dict
import torch
import numpy as np
from tqdm import tqdm

from ldm_base.ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
)


class SlidingWindowSampler:
    """
    Stochastic I/O sampler for talking-head generation.

    Core algorithm is identical to CAP4D's StochasticIOSampler:
      - Concatenate R reference latents with (V-R) generated latents along T.
      - Run the UNet with CFG.
      - Extract only the generated portion [R:] for the DDIM update.

    For talking head, R=1 (one identity reference frame).
    Audio context is handled transparently — if present in the cond dicts it is
    concatenated along the time axis just like any other key.
    """

    def __init__(self, model, **kwargs):
        super().__init__()

        if isinstance(model, dict):
            self.device_model_map = model
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device_model_map = {device: model}

        for key in self.device_model_map:
            self.main_model = self.device_model_map[key]
            self.ddpm_num_timesteps = self.main_model.num_timesteps
            break

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.main_model.alphas_cumprod
        to_torch = lambda x: x.clone().detach().to(torch.float32)

        self.register_buffer('betas',                to_torch(self.main_model.betas))
        self.register_buffer('alphas_cumprod',       to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',  to_torch(self.main_model.alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod',  to_torch(np.sqrt(alphas_cumprod.detach().cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.detach().cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',  to_torch(np.log(1. - alphas_cumprod.detach().cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',     to_torch(np.sqrt(1. / alphas_cumprod.detach().cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',   to_torch(np.sqrt(1. / alphas_cumprod.detach().cpu() - 1)))

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.detach().cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer('ddim_sigmas',      ddim_sigmas)
        self.register_buffer('ddim_alphas',      ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_orig = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_orig)

    @torch.no_grad()
    def sample(
        self,
        S: int,
        ref_cond:   Dict[str, torch.Tensor],
        ref_uncond: Dict[str, torch.Tensor],
        gen_cond:   Dict[str, torch.Tensor],
        gen_uncond: Dict[str, torch.Tensor],
        latent_shape: Tuple[int, int, int],   # (C, H, W)
        V: int   = 16,                        # total UNet context window size
        R: int   = 1,                         # number of reference frames (default 1)
        cfg_scale: float = 2.0,
        eta: float = 0.,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Generate video frames autoregressively with stochastic I/O conditioning.

        Args:
            S            : DDIM denoising steps.
            ref_cond     : conditioning dict for reference frames (contains GT latents).
            ref_uncond   : unconditional conditioning for reference frames (zeros).
            gen_cond     : conditioning dict for frames to generate.
            gen_uncond   : unconditional conditioning for generated frames.
            latent_shape : (C, H, W) shape of a single latent frame.
            V            : UNet context window (R refs + V-R generated per pass).
            R            : number of reference frames (1 for single-image driving).
            cfg_scale    : classifier-free guidance strength.
            eta          : DDIM stochasticity (0 = deterministic).

        Returns:
            Tensor of shape (n_gen, C, H, W) — all generated latents.
        """
        mem_device = next(iter(gen_cond.items()))[1].device
        n_devices  = len(self.device_model_map)

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)

        n_gen     = next(iter(gen_cond.items()))[1].shape[0]
        n_all_ref = next(iter(ref_cond.items()))[1].shape[0]
        R = min(n_all_ref, R)
        G = V - R  # generated slots per forward pass

        assert n_gen % G == 0, (
            f"n_gen ({n_gen}) must be divisible by G=V-R ({G})"
        )
        n_its = n_gen // G

        all_x_T = torch.randn((n_gen, *latent_shape), device=mem_device)
        all_e_t = torch.zeros_like(all_x_T)

        timesteps  = self.ddim_timesteps
        time_range = np.flip(timesteps)

        print(f"SlidingWindowSampler: {timesteps.shape[0]} steps | R={R} ref | {n_gen} frames to generate")

        def dict_index(d, indices, device=None):
            return {k: (v[indices].to(device) if device else v[indices])
                    for k, v in d.items() if v is not None}

        # Build iteration batches (supports multi-GPU)
        batch_indices = []
        for l in range(int(np.ceil(n_its / n_devices))):
            device_batch = []
            for dev_id in range(min(n_devices, n_its)):
                idx = l * n_devices + dev_id
                if idx < n_its:
                    device_batch.append([idx])
            batch_indices.append(device_batch)

        for i, step in enumerate(tqdm(time_range, desc='SlidingWindowSampler', total=len(time_range))):
            index = len(time_range) - i - 1
            ts    = torch.full((1, V), step, device=mem_device, dtype=torch.long)
            all_e_t.zero_()

            # Sample which reference frames to condition on this step
            if R == 1:
                ref_batches = np.zeros((n_its, R), dtype=np.int64)
            else:
                ref_batches = np.stack([
                    np.random.permutation(np.arange(n_all_ref))[:R]
                    for _ in range(n_its)
                ], axis=0)

            gen_batches = np.reshape(np.random.permutation(np.arange(n_gen)), (n_its, -1))

            x_in_list, t_in_list, c_in_list, e_t_list = [], [], [], []

            for dev_batches in batch_indices:
                for dev_id, dev_batch in enumerate(dev_batches):
                    dev_key    = list(self.device_model_map)[dev_id]
                    dev_device = self.device_model_map[dev_key].device

                    curr_ref_cond   = dict_index(ref_cond,   ref_batches[dev_batch], dev_device)
                    curr_ref_uncond = dict_index(ref_uncond, ref_batches[dev_batch], dev_device)
                    curr_gen_cond   = dict_index(gen_cond,   gen_batches[dev_batch], dev_device)
                    curr_gen_uncond = dict_index(gen_uncond, gen_batches[dev_batch], dev_device)

                    curr_x_T = all_x_T[gen_batches[dev_batch]].to(dev_device)

                    # Concatenate ref and gen along time axis for each key
                    curr_cond   = {k: torch.cat([curr_ref_cond[k],   curr_gen_cond[k]],   dim=1)
                                   for k in curr_ref_cond if k in curr_gen_cond}
                    curr_uncond = {k: torch.cat([curr_ref_uncond[k], curr_gen_uncond[k]], dim=1)
                                   for k in curr_ref_uncond if k in curr_gen_uncond}

                    # Stack uncond + cond for a single CFG forward pass.
                    c_in = {k: torch.cat([curr_uncond[k], curr_cond[k]], dim=0)
                            for k in curr_cond}

                    t_in = torch.cat([ts] * 2, dim=0).to(dev_device)
                    c_in = dict(c_concat=[c_in])

                    # Input: [R ref latents | (V-R) noisy generated latents]
                    x_in = torch.cat([curr_cond["z_input"][:, :R], curr_x_T], dim=1)
                    x_in = torch.cat([x_in] * 2, dim=0).to(dev_device)

                    x_in_list.append(x_in)
                    t_in_list.append(t_in)
                    c_in_list.append(c_in)

                for dev_id, dev_batch in enumerate(dev_batches):
                    dev_key = list(self.device_model_map)[dev_id]
                    model_uncond, model_t = self.device_model_map[dev_key].apply_model(
                        x_in_list[dev_id], t_in_list[dev_id], c_in_list[dev_id],
                    ).chunk(2)
                    model_output = model_uncond + cfg_scale * (model_t - model_uncond)

                    # Extract only the generated portion
                    e_t = model_output[:, R:]
                    e_t_list.append(e_t)

                for dev_id, dev_batch in enumerate(dev_batches):
                    all_e_t[gen_batches[dev_batch]] += e_t_list[dev_id].to(mem_device)

            # DDIM update step
            alpha_t             = self.ddim_alphas.float()[index]
            sqrt_one_minus_at   = self.ddim_sqrt_one_minus_alphas[index]
            sigma_t             = self.ddim_sigmas[index]
            alpha_prev_t        = torch.tensor(self.ddim_alphas_prev).float()[index]

            alpha_prev_t      = alpha_prev_t.double()
            sqrt_one_minus_at = sqrt_one_minus_at.double()
            alpha_t           = alpha_t.double()

            e_t_factor = (
                -alpha_prev_t.sqrt() * sqrt_one_minus_at / alpha_t.sqrt()
                + (1. - alpha_prev_t - sigma_t ** 2).sqrt()
            )
            x_t_factor = alpha_prev_t.sqrt() / alpha_t.sqrt()

            all_x_T = all_x_T * x_t_factor.float() + all_e_t * e_t_factor.float()

        return all_x_T
