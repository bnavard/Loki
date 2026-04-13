import os

import numpy as np
import torch
import torchvision
from pathlib import Path
from PIL import Image
from pytorch_lightning.callbacks import Callback
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import einops
import torch.nn.functional as F
import gc


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    # @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, gpu_id=0):
        root = os.path.join(save_dir, "image_log", split, f"e-{current_epoch:06d}")

        rows = []

        for k in images:
            if len(images[k].shape) == 4:
                # Single images
                # TODO: Implement
                continue
            if len(images[k].shape) == 5:
                # We have videos
                b, t = images[k].shape[:2]
                imgs = einops.rearrange(images[k], 'b t c h w -> (b t) c h w')
                grid = torchvision.utils.make_grid(imgs, nrow=b * t)

            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            grid = grid.permute(1, 2, 0).numpy()
            grid = (grid * 255).astype(np.uint8)

            rows.append(grid)
        
        filename = "gs-{:06}_b-{:06}_{:02d}_i.jpg".format(global_step, batch_idx, gpu_id)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(np.concatenate(rows, axis=0)).save(path)

    def log_cond(self, pl_module, batch):
        cond_model = pl_module.cond_stage_model
        cond_key = pl_module.control_key

        c_cond = cond_model(batch[cond_key], conditioned=True)
        enc_vis = cond_model.get_vis(c_cond["pos_enc"])

        for key in enc_vis:
            vis = enc_vis[key]
            b_ = vis.shape[0]
            vis = einops.rearrange(vis, 'b t h w c -> (b t) c h w')
            vis = F.interpolate(vis, scale_factor=8., mode="nearest")
            enc_vis[key] = einops.rearrange(vis, '(b t) c h w -> b t c h w', b=b_)
        
        return enc_vis

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            # draw vertices!
            with torch.no_grad():
                cond_vis = self.log_cond(pl_module, batch)

                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        if not images[k].shape[-1] == 3:
                            images[k] = torch.clamp(images[k], -1., 1.)
                
            for key in cond_vis:
                images[key] = cond_vis[key].detach().cpu().clamp(-1., 1.)

            self.log_local(
                pl_module.logger.save_dir, 
                split, 
                images,
                pl_module.global_step, 
                pl_module.current_epoch, 
                batch_idx,
                gpu_id=pl_module.global_rank,
            )
            
            gc.collect()

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
