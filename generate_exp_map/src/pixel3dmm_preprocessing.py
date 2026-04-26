import os
import sys

from pixel3dmm import env_paths

# Add pixel3dmm scripts to path (for run_cropping, etc.)
# Note: PIPNet's FaceBoxesV2 is NOT added to sys.path to avoid
# conflicting with MICA's utils. Instead, faceboxes_detector.py
# uses relative imports (patched during setup).
sys.path.insert(0, f'{env_paths.CODE_BASE}/scripts/')

# Import cropping (uses PIPNet) and facer segmentation at module level
from run_cropping import main as run_cropping_main
from pixel3dmm_segmentation import main as run_facer_segmentation_main


def main(video_or_images_path: str):
    """
    Optimized preprocessing - uses direct imports instead of os.system.
    """
    video_or_images_path = os.path.abspath(video_or_images_path)

    if os.path.isdir(video_or_images_path):
        vid_name = video_or_images_path.split('/')[-1]
    else:
        vid_name = video_or_images_path.split('/')[-1][:-4]

    # Step 1: Run cropping (uses PIPNet via run_cropping)
    try:
        run_cropping_main(video_or_images_path=video_or_images_path)
    except Exception as e:
        raise RuntimeError(f"run_cropping.py failed for {video_or_images_path}") from e

    # Step 2: Run MICA demo
    # Import lazily to avoid utils namespace conflict with PIPNet's FaceBoxesV2/utils
    try:
        mica_path = f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA/'
        if mica_path not in sys.path:
            sys.path.insert(0, mica_path)
        from demo import main as mica_demo_main, get_cfg_defaults

        class Args:
            def __init__(self):
                self.video_name = vid_name
                self.i = f'{env_paths.PREPROCESSED_DATA}/{vid_name}/cropped/'
                self.a = f'{env_paths.PREPROCESSED_DATA}/{vid_name}/arcface/'
                self.o = f'{env_paths.PREPROCESSED_DATA}/{vid_name}/mica/'
                self.m = f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA/data/pretrained/mica.tar'

        args = Args()

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
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"MICA/demo.py failed while running {video_or_images_path}") from e

    # Step 3: Run facer segmentation
    try:
        run_facer_segmentation_main(video_name=vid_name, batch_size=4)
    except Exception as e:
        raise RuntimeError(f"run_facer_segmentation.py failed while running {video_or_images_path}") from e


if __name__ == '__main__':
    import tyro
    tyro.cli(main)
