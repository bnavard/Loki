import os
import tyro
import sys

from pixel3dmm import env_paths

# Add paths for direct imports
sys.path.insert(0, f'{env_paths.CODE_BASE}/scripts/')
sys.path.insert(0, f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA/')

# Direct imports instead of os.system calls
from run_cropping import main as run_cropping_main
from demo import main as mica_demo_main, get_cfg_defaults

# Import optimized facer segmentation
from pixel3dmm_segmentation import main as run_facer_segmentation_main


def main(video_or_images_path: str):
    """
    Optimized preprocessing - uses direct imports instead of os.system
    No subprocess overhead, models loaded once per worker
    """
    # Resolve to absolute path immediately to avoid issues with relative paths
    video_or_images_path = os.path.abspath(video_or_images_path)

    if os.path.isdir(video_or_images_path):
        vid_name = video_or_images_path.split('/')[-1]
    else:
        vid_name = video_or_images_path.split('/')[-1][:-4]
    
    # Step 1: Run cropping (direct import - no subprocess)
    try:
        run_cropping_main(video_or_images_path=video_or_images_path)
    except Exception as e:
        raise RuntimeError(f"run_cropping.py failed for {video_or_images_path}") from e
    
    # Step 2: Run MICA demo (direct import - no subprocess)
    try:
        # Create args object for MICA
        class Args:
            def __init__(self):
                self.video_name = vid_name
                self.i = f'{env_paths.PREPROCESSED_DATA}/{vid_name}/cropped/'
                self.a = f'{env_paths.PREPROCESSED_DATA}/{vid_name}/arcface/'
                self.o = f'{env_paths.PREPROCESSED_DATA}/{vid_name}/mica/'
                self.m = f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA/data/pretrained/mica.tar'
        
        args = Args()
        
        # Check if already completed
        if os.path.exists(f'{env_paths.PREPROCESSED_DATA}/{vid_name}/mica/'):
            if len(os.listdir(f'{env_paths.PREPROCESSED_DATA}/{vid_name}/mica/')) >= 10:
                print(f'<<<<<<<< ALREADY COMPLETE MICA PREDICTION FOR {vid_name}, SKIPPING >>>>>>>>')
            else:
                cfg = get_cfg_defaults()
                mica_demo_main(cfg, args)
        else:
            cfg = get_cfg_defaults()
            mica_demo_main(cfg, args)
    except Exception as e:
        raise RuntimeError(f"MICA/demo.py failed while running {video_or_images_path}") from e

    # Step 3: Run facer segmentation (optimized version with lazy loading)
    try:
        run_facer_segmentation_main(video_name=vid_name, batch_size=4)
    except Exception as e:
        raise RuntimeError(f"run_facer_segmentation.py failed while running {video_or_images_path}") from e


if __name__ == '__main__':
    tyro.cli(main)
