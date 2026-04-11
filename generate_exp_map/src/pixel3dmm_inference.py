import traceback
from tqdm import tqdm
import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from time import time

from pixel3dmm.utils.uv import uv_pred_to_mesh
from pixel3dmm.lightning.p3dmm_system import system as p3dmm_system
from pixel3dmm import env_paths


def pad_to_3_channels(img):
    if img.shape[-1] == 3:
        return img
    elif img.shape[-1] == 1:
        return np.concatenate([img, np.zeros_like(img[..., :1]), np.zeros_like(img[..., :1])], axis=-1)
    elif img.shape[-1] == 2:
        return np.concatenate([img, np.zeros_like(img[..., :1])], axis=-1)
    else:
        raise ValueError('too many dimensions in prediction type!')


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w


def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d_x = gaussian_fn(kernlen, std=std * 5)
    gkern1d_y = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d_y, gkern1d_x)
    return gkern2d


valid_verts = np.load(f'{env_paths.VALID_VERTICES_WIDE_REGION}')


def main(cfg):
    """
    Process images one at a time (matching original network_inference.py behavior)
    Model is loaded once and reused for all images
    """
    
    if cfg.model.prediction_type == 'flame_params':
        cfg.data.mirror_aug = False

    # Process one image at a time (like original network_inference.py)
    batch_size = 1  

    checkpoints = {
        'uv_map': f"{env_paths.CKPT_UV_PRED}",
        'normals': f"{env_paths.CKPT_N_PRED}",
    }

    model_checkpoint = checkpoints[cfg.model.prediction_type]

    prediction_types = cfg.model.prediction_type.split(',')

    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=11, bias=False, padding='same')
    g_weights = gkern(11, 2)
    g_weights /= torch.sum(g_weights)
    conv.weight = torch.nn.Parameter(g_weights.unsqueeze(0).unsqueeze(0))

    OUT_NAMES = str(cfg.video_name).split(',')

    print(f'<<<<<<<< STARTING PIXEL3DMM INFERENCE for {cfg.video_name} in {prediction_types} MODE >>>>>>>>')

    for OUT_NAME in OUT_NAMES:
        folder = f'{env_paths.PREPROCESSED_DATA}/{OUT_NAME}/'
        IMAGE_FOLDER = f'{folder}/cropped'
        SEGEMNTATION_FOLDER = f'{folder}/seg_og/'

        out_folders = {}
        out_folders_wGT = {}
        out_folders_viz = {}

        for prediction_type in prediction_types:
            out_folders[prediction_type] = f'{env_paths.PREPROCESSED_DATA}/{OUT_NAME}/p3dmm/{prediction_type}/'
            out_folders_wGT[prediction_type] = f'{env_paths.PREPROCESSED_DATA}/{OUT_NAME}/p3dmm_wGT/{prediction_type}/'
            os.makedirs(out_folders[prediction_type], exist_ok=True)
            os.makedirs(out_folders_wGT[prediction_type], exist_ok=True)
            out_folders_viz[prediction_type] = f'{env_paths.PREPROCESSED_DATA}/{OUT_NAME}/p3dmm_extraViz/{prediction_type}/'
            os.makedirs(out_folders_viz[prediction_type], exist_ok=True)

        image_names = os.listdir(f'{IMAGE_FOLDER}')
        image_names.sort()

        if os.path.exists(out_folders[prediction_type]):
            if len(os.listdir(out_folders[prediction_type])) == len(image_names):
                return

        # Load model once per video (same as original)
        print("Loading model checkpoint...")
        model = p3dmm_system.load_from_checkpoint(model_checkpoint, strict=False)
        model = model.cuda()
        print("Model loaded!")

        # Process images one at a time (like original)
        for i in tqdm(range(len(image_names))):
            try:
                # Load and prepare image
                img = np.array(Image.open(f'{IMAGE_FOLDER}/{image_names[i]}').resize((512, 512))) / 255
                img = torch.from_numpy(img)[None, None].float().cuda()  # 1,1,512,512,3
                
                img_seg = np.array(Image.open(f'{SEGEMNTATION_FOLDER}/{image_names[i][:-4]}.png').resize((512, 512), Image.NEAREST))
                if len(img_seg.shape) == 3:
                    img_seg = img_seg[..., 0]
                
                mask = ((img_seg == 2) | ((img_seg > 3) & (img_seg < 14))) & ~(img_seg == 11)
                mask = torch.from_numpy(mask).long().cuda()[None, None]  # 1, 1, 512, 512
                
                batch = {
                    'tar_msk': mask,
                    'tar_rgb': img,
                }
                batch_mirrored = {
                    'tar_rgb': torch.flip(batch['tar_rgb'], dims=[3]).cuda(),
                    'tar_msk': torch.flip(batch['tar_msk'], dims=[3]).cuda(),
                }

                # Run inference
                with torch.no_grad():
                    output, conf = model.net(batch)
                    output_mirrored, conf = model.net(batch_mirrored)

                    if 'uv_map' in output:
                        fliped_uv_pred = torch.flip(output_mirrored['uv_map'], dims=[4])
                        fliped_uv_pred[:, :, 0, :, :] *= -1
                        fliped_uv_pred[:, :, 0, :, :] += 2*0.0075
                        output['uv_map'] = (output['uv_map'] + fliped_uv_pred)/2
                    if 'normals' in output:
                        fliped_uv_pred = torch.flip(output_mirrored['normals'], dims=[4])
                        fliped_uv_pred[:, :, 0, :, :] *= -1
                        output['normals'] = (output['normals'] + fliped_uv_pred)/2

                # Save results
                for prediction_type in prediction_types:
                    i_batch = 0
                    i_view = 0
                    gt_rgb = batch['tar_rgb']

                    # normalize to [0,1] range
                    if prediction_type == 'uv_map':
                        tmp_output = torch.clamp((output[prediction_type][i_batch, i_view] + 1) / 2, 0, 1)
                    elif prediction_type in ['normals', 'normals_can']:
                        tmp_output = output[prediction_type][i_batch, i_view]
                        tmp_output = tmp_output / torch.norm(tmp_output, dim=0).unsqueeze(0)
                        tmp_output = torch.clamp((tmp_output + 1) / 2, 0, 1)
                        tmp_output = torch.stack(
                            [tmp_output[0, ...], 1 - tmp_output[2, ...], 1 - tmp_output[1, ...]],
                            dim=0)

                    Image.fromarray(
                        pad_to_3_channels(
                            tmp_output.permute(1, 2, 0).detach().cpu().float().numpy() * 255).astype(
                            np.uint8)).save(
                        f'{out_folders[prediction_type]}/{image_names[i][:-4]}.png')

            except Exception as exx:
                traceback.print_exc()
                pass


if __name__ == '__main__':
    base_conf = OmegaConf.load(f'{env_paths.CODE_BASE}/configs/base.yaml')
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)
    main(cfg)
