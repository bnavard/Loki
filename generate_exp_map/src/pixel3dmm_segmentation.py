import os
import sys
import traceback

from math import ceil

import PIL.Image
import torch
import distinctipy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import facer
import tyro

from pixel3dmm import env_paths

colors = distinctipy.get_colors(22, rng=0)


def viz_results(img, seq_classes, n_classes, suppress_plot = False):

    seg_img = np.zeros([img.shape[-2], img.shape[-1], 3])
    bad_indices = [
        0,  # background,
        1,  # neck
        3,  # cloth
        4,  # ear_r (images-space r)
        5,  # ear_l
        14,  # hair,
        16,  # ??
        17,  # earring_r
        18,  # ?
    ]
    bad_indices = []

    for i in range(n_classes):
        if i not in bad_indices:
            seg_img[seq_classes[0, :, :] == i] = np.array(colors[i])*255

    if not suppress_plot:
        plt.imshow(seg_img.astype(np.uint8))
        plt.show()
    return Image.fromarray(seg_img.astype(np.uint8))


# CRITICAL FIX: Don't load models at module level!
# Move model loading into function after CUDA_VISIBLE_DEVICES is set
_face_detector = None
_face_parser = None


def get_models(device):
    """Lazy load models only when needed, respecting CUDA_VISIBLE_DEVICES"""
    global _face_detector, _face_parser
    
    if _face_detector is None:
        print(f"Loading models on device: {device}")
        _face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        _face_parser = facer.face_parser('farl/celebm/448', device=device)
    
    return _face_detector, _face_parser


def main(video_name: str, batch_size: int = 4):  # Increased default batch size
    """
    OPTIMIZATIONS:
    1. Model loading moved inside function (after CUDA_VISIBLE_DEVICES set)
    2. Increased batch size for better GPU utilization
    3. Batched image loading
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models AFTER CUDA_VISIBLE_DEVICES is already set
    face_detector, face_parser = get_models(device)

    out = f'{env_paths.PREPROCESSED_DATA}/{video_name}'
    out_seg = f'{out}/seg_og/'
    out_seg_annot = f'{out}/seg_non_crop_annotations/'
    os.makedirs(out_seg, exist_ok=True)
    os.makedirs(out_seg_annot, exist_ok=True)
    folder = f'{out}/cropped/'

    frames = [f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]
    frames.sort()

    if len(os.listdir(out_seg)) == len(frames):
        print(f'<<<<<<<< ALREADY COMPLETED SEGMENTATION FOR {video_name}, SKIPPING >>>>>>>>')
        return

    # Process in larger batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        image_stack = []
        frame_stack = []

        # Load batch of images
        for file in batch_frames:
            if os.path.exists(f'{out_seg}/{file[:-4]}.png'):
                continue
                
            img = Image.open(f'{folder}/{file}')
            image = facer.hwc2bchw(torch.from_numpy(np.array(img)[..., :3])).to(device=device)
            image_stack.append(image)
            frame_stack.append(file[:-4])

        if not image_stack:
            continue

        # Process batch
        image_batch = torch.cat(image_stack, dim=0)
        
        try:
            with torch.inference_mode():
                faces = face_detector(image_batch)
                torch.cuda.empty_cache()
                faces = face_parser(image_batch, faces, bbox_scale_factor=1.25)
                torch.cuda.empty_cache()

            seg_logits = faces['seg']['logits']
            back_ground = torch.all(seg_logits == 0, dim=1, keepdim=True).detach().squeeze(1).cpu().numpy()
            seg_probs = seg_logits.softmax(dim=1)
            seg_classes = seg_probs.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
            seg_classes[back_ground] = seg_probs.shape[1] + 1

            # Save results
            for _iidx in range(seg_probs.shape[0]):
                if _iidx >= len(frame_stack):
                    continue
                frame = frame_stack[_iidx]
                iidx = faces['image_ids'][_iidx].item()
                try:
                    I_color = viz_results(image_batch[iidx:iidx+1], 
                                        seq_classes=seg_classes[_iidx:_iidx+1], 
                                        n_classes=seg_probs.shape[1] + 1, 
                                        suppress_plot=True)
                    I_color.save(f'{out_seg_annot}/color_{frame}.png')
                except Exception:
                    pass
                I = Image.fromarray(seg_classes[_iidx])
                I.save(f'{out_seg}/{frame}.png')
                
            torch.cuda.empty_cache()
            
        except Exception as exx:
            traceback.print_exc()
            continue


if __name__ == '__main__':
    tyro.cli(main)
