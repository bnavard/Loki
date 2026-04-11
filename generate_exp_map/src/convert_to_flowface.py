"""
Convert pixel3dmm tracking output to FlowFace format (fit.npz).

Takes the per-frame checkpoint files and meshes from pixel3dmm tracking,
re-fits FLAME parameters via optimization, adds gaze tracking (L2CS)
and background matting (RobustVideoMatting), and exports fit.npz.

This is Phase 2 of the expression map generation pipeline.
Phase 1 (process_video.py) produces the tracking checkpoints.
Phase 2 (this script) converts them to the fit.npz format used by
all downstream pipelines.

Usage:
    cd <repo_root>

    python generate_exp_map/scripts/convert_to_flowface.py \
        --video_path data/talkvid/talkvid/CLIP_ID.mp4 \
        --tracking_path data/flame_tracking/tracking/CLIP_ID_nV1_noPho_uv2000.0_n1000.0 \
        --preprocess_path data/flame_tracking/preprocessing/CLIP_ID \
        --output_path data/flowface/CLIP_ID
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import trimesh

# Use our repo's FLAME code (no external flowface/cap4d dependency)
from talkinghead_sd21_unet_cap4d_based.flame.flame import (
    CAP4DFlameSkinner, OPENCV2PYTORCH3D, transform_vertices,
)

# Dependencies co-located in generate_exp_map/src/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from l2cs_eye_tracker import L2CSTracker, compute_eyeball_rotation
from robust_video_matting.model.model import MattingNetwork


# ============================================================================
# Constants
# ============================================================================

ORBIT_PERIOD = 8
ORBIT_AMPLITUDE_YAW = 55
ORBIT_AMPLITUDE_PITCH = 20


# ============================================================================
# FrameReader (inlined from cap4d/datasets/utils.py)
# ============================================================================

class FrameReader:
    def __init__(self, video_path):
        self.frame_list = sorted(list(Path(video_path).glob("*.*")))

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        return cv2.imread(str(self.frame_list[index]))[..., [2, 1, 0]]


def pivot_camera_intrinsic(extrinsics, target, angles, distance_factor=1.):
    """Rotate a camera around a target point."""
    extrinsics = np.linalg.inv(extrinsics)
    R_c2w = extrinsics[:3, :3]
    t_c2w = extrinsics[:3, 3]
    v = (t_c2w - target) * distance_factor
    R_delta = R.from_euler("YX", angles, degrees=True).as_matrix()
    new_R_c2w = R_c2w @ R_delta
    new_v = R_c2w @ R_delta @ np.linalg.inv(R_c2w) @ v
    new_t_c2w = target + new_v
    new_extrinsics = np.eye(4)
    new_extrinsics[:3, :3] = new_R_c2w
    new_extrinsics[:3, 3] = new_t_c2w
    return np.linalg.inv(new_extrinsics)


# ============================================================================
# FLAME Fitting Model
# ============================================================================

class FlameFittingModel(nn.Module):
    def __init__(self, flame, n_timesteps, vertex_weights, use_jaw_rotation=False):
        super().__init__()
        self.use_jaw_rotation = use_jaw_rotation
        self.n_timesteps = n_timesteps
        self.flame = flame

        self.shape = nn.Parameter(torch.zeros(flame.n_shape_params))
        self.expr = nn.Parameter(torch.zeros(n_timesteps, flame.n_expr_params))
        self.rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.tra = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.eye_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.neck_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        if use_jaw_rotation:
            self.jaw_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
            self.register_buffer("jaw_std", torch.deg2rad(torch.tensor([45, 5, 0.01])), persistent=False)
        else:
            self.jaw_rot = None

        self.register_buffer("vertex_weights", vertex_weights / vertex_weights.sum(), persistent=False)
        self.register_buffer("opencv2pytorch", OPENCV2PYTORCH3D, persistent=False)

    def forward(self):
        flame_sequence = {
            "shape": self.shape, "expr": self.expr, "rot": self.rot,
            "tra": self.tra, "eye_rot": self.eye_rot,
            "jaw_rot": self.jaw_rot, "neck_rot": self.neck_rot,
        }
        verts_3d, _ = self.flame(flame_sequence)
        verts_3d_cv = transform_vertices(self.opencv2pytorch[None], verts_3d)
        return {"verts_3d": verts_3d_cv}

    def fit(self, verts_3d, init_lr=1e-2, n_steps=6000,
            w_shape_reg=1e-2, w_expr_reg=1e-2, verbose=True, pos_warm_up_steps=500):
        opt = torch.optim.Adam(lr=init_lr, params=self.parameters(), betas=(0.96, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.5)

        pbar = tqdm(range(n_steps)) if verbose else range(n_steps)
        for i in pbar:
            if i < pos_warm_up_steps:
                self.expr.data *= 0.
                self.shape.data *= 0.
                self.eye_rot.data *= 0.

            opt.zero_grad(set_to_none=True)
            output_dict = self.forward()

            l_vert = (output_dict["verts_3d"] - verts_3d) / 0.01
            l_vert = l_vert.norm(dim=-1)
            l_vert_max = l_vert.max()
            l_vert = l_vert ** 2
            l_vert = (l_vert * self.vertex_weights[None]).sum(dim=-1).mean()

            l_shape = (self.shape ** 2).sum(dim=-1).mean()
            expr_params = self.expr
            if self.use_jaw_rotation:
                expr_params = torch.cat([expr_params, self.jaw_rot / self.jaw_std[None]], dim=-1)
            l_expr = (expr_params ** 2).sum(dim=-1).mean()

            loss = l_vert + l_shape * w_shape_reg + l_expr * w_expr_reg
            loss.backward()
            opt.step()

            if i > pos_warm_up_steps:
                scheduler.step(loss.item())
            if opt.param_groups[0]["lr"] < 1e-5:
                break
            if i % 10 == 0 and verbose:
                pbar.set_description(
                    f"lr: {opt.param_groups[0]['lr']:.1e}, loss: {loss.item():.3f}, "
                    f"vert: {l_vert.item():.3f}, max: {l_vert_max.item():.3f}"
                )

        return l_vert, output_dict["verts_3d"]

    def export_results(self):
        fit_3d = {
            "shape": self.shape.data.detach().cpu().numpy(),
            "expr": self.expr.data.detach().cpu().numpy(),
            "rot": self.rot.data.detach().cpu().numpy(),
            "tra": self.tra.data.detach().cpu().numpy(),
            "eye_rot": self.eye_rot.data.detach().cpu().numpy(),
            "neck_rot": self.neck_rot.data.detach().cpu().numpy(),
        }
        if self.jaw_rot is not None:
            fit_3d["jaw_rot"] = self.jaw_rot.data.detach().cpu().numpy()
        return fit_3d


# ============================================================================
# FLAME Fitting
# ============================================================================

def fit_flame(verts_3d, gaze_directions, cam_rt, use_jaw_rotation=False,
              n_shape_params=150, n_expr_params=65, device="cpu",
              n_steps=6000, w_shape_reg=1e-2, w_expr_reg=1e-2,
              smooth_eye_rotations=False):
    verts_3d = torch.tensor(verts_3d).float().to(device)
    vert_weights = torch.tensor(np.load("data/assets/flame/flowface_vertex_weights.npy"))

    flame_path = "data/assets/flame/flame2023_no_jaw.pkl"
    if use_jaw_rotation:
        flame_path = "data/assets/flame/flame2023.pkl"

    flame = CAP4DFlameSkinner(
        flame_path, n_shape_params=n_shape_params, n_expr_params=n_expr_params,
        blink_blendshape_path="data/assets/flame/blink_blendshape.npy",
    ).to(device)

    fitter = FlameFittingModel(
        flame, verts_3d.shape[0], vertex_weights=vert_weights,
        use_jaw_rotation=use_jaw_rotation,
    ).to(device)
    _, pred_verts_3d = fitter.fit(
        verts_3d, n_steps=n_steps,
        w_shape_reg=w_shape_reg, w_expr_reg=w_expr_reg,
    )

    # Fix eye rotations from gaze tracking
    for frame_id in range(verts_3d.shape[0]):
        if gaze_directions[frame_id] is None:
            eye_rot = np.zeros(3)
        else:
            yaw, pitch = gaze_directions[frame_id][0]
            eye_rot = compute_eyeball_rotation(
                yaw.cpu().numpy(), pitch.cpu().numpy(),
                cam_rt[:3, :3], cam_rt[:3, 3],
                fitter.rot.data[frame_id].detach().cpu().numpy(),
                fitter.tra.data[frame_id].detach().cpu().numpy(),
            )
        fitter.eye_rot.data[frame_id] = torch.from_numpy(eye_rot).float().to(device)

    # Clamp eye rotation magnitude
    clamp_factor = fitter.eye_rot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    fitter.eye_rot.data = fitter.eye_rot.data / clamp_factor * clamp_factor.clamp(max=1.)

    if smooth_eye_rotations:
        fitter.eye_rot.data = torch.from_numpy(
            gaussian_filter1d(fitter.eye_rot.data.cpu().numpy(), sigma=2, axis=0)
        ).float()

    pred_verts_3d = fitter.forward()["verts_3d"]

    return (
        fitter.export_results(),
        pred_verts_3d.detach().cpu().numpy(),
        flame.template_faces.cpu().numpy(),
    )


# ============================================================================
# Camera calibration conversion
# ============================================================================

def convert_calibration(tracking_resolution, crop_box, k):
    x0, y0, x1, y1 = crop_box
    crop_w, crop_h = x1 - x0, y1 - y0
    H_track, W_track = tracking_resolution
    scale_x, scale_y = crop_w / W_track, crop_h / H_track
    k[0, :] *= scale_x
    k[1, :] *= scale_y
    k[0, 2] += x0
    k[1, 2] += y0
    return k


def auto_downsample_ratio(h, w):
    return min(512 / max(h, w), 1)


# ============================================================================
# Main conversion
# ============================================================================

def main(args):
    tracking_path = Path(args.tracking_path)
    preprocess_path = Path(args.preprocess_path)

    pixel_frames = sorted(list((tracking_path / "checkpoint").glob("*.*")))
    assert len(pixel_frames) > 0, "No pixel3dmm tracking results found."

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)

    video_path = Path(args.video_path)
    assert video_path.exists(), f"Input video not found: {video_path}"

    if video_path.is_dir():
        frame_reader = FrameReader(video_path)
        shutil.copytree(video_path, output_path / "images" / "cam0", dirs_exist_ok=True)
        is_video = False
    else:
        frame_reader = VideoReader(str(video_path))
        shutil.copy(video_path, output_path / "images" / f"cam0{video_path.suffix}")
        is_video = True

    # --- Gaze tracking + background matting ---
    print("Running gaze tracking (L2CS) and background matting (RVM)...")
    l2cs_tracker = L2CSTracker(
        device=args.device,
        gaze_weight_path="data/weights/l2cs/L2CSNet_gaze360.pkl",
    )

    matting_model = MattingNetwork()
    matting_model.load_state_dict(torch.load("data/weights/rvm/rvm_mobilenetv3.pth"))
    matting_model.eval().to(args.device)

    output_bg_path = output_path / "bg" / "cam0"
    output_bg_path.mkdir(exist_ok=True, parents=True)

    gaze_directions = []
    rec = [None] * 4

    for frame_id, frame_img in enumerate(tqdm(frame_reader, desc="Gaze+BG")):
        if not isinstance(frame_img, np.ndarray):
            frame_img = frame_img.asnumpy()

        frame_img = torch.from_numpy(frame_img)[None] / 255.

        if args.enable_gaze_tracking:
            with torch.no_grad():
                gaze = l2cs_tracker.process(frame_img)
        else:
            gaze = None
        gaze_directions.append(gaze)

        downsample_ratio = auto_downsample_ratio(*frame_img.shape[2:])
        src = frame_img.permute(0, 3, 1, 2).to(args.device)
        with torch.no_grad():
            fgr, pha, *rec = matting_model(src, *rec, downsample_ratio)
        if not is_video:
            rec = [None] * 4

        cv2.imwrite(
            str(output_bg_path / f"{frame_id:04d}.png"),
            (pha[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
        )

    # --- Load tracking results ---
    print("Loading pixel3dmm tracking results...")
    crop_box = np.load(preprocess_path / "crop_ymin_ymax_xmin_xmax.npy")
    crop_box = crop_box[[2, 0, 3, 1]]

    all_vertices = []
    for frame_id, frame_path in enumerate(tqdm(pixel_frames, desc="Loading meshes")):
        frame_name = frame_path.stem
        mesh = trimesh.load(tracking_path / "mesh" / f"{frame_name}.ply")
        vertices = mesh.vertices

        frame_info = torch.load(frame_path, weights_only=False)
        vertices = frame_info["flame"]["R_rotation_matrix"][0] @ vertices.T + frame_info["flame"]["t"].T
        vertices = vertices.T

        if frame_id == 0:
            rt = np.eye(4)
            rt[:3, :3] = frame_info["camera"]["R_base_0"][0]
            rt[:3, 3] = frame_info["camera"]["t_base_0"][0]
            rt = OPENCV2PYTORCH3D.inverse().numpy() @ rt
            k = np.eye(3)
            size = 256
            k[:2, 2] = (frame_info["camera"]["pp"][0] + 1.) * (size / 2 + 0.5)
            k[0, 0] = frame_info["camera"]["fl"][0, 0] * size
            k[1, 1] = frame_info["camera"]["fl"][0, 0] * size

            tracking_resolution = frame_info["img_size"]
            orig_resolution = frame_reader[frame_id].shape[:2] if not isinstance(
                frame_reader[frame_id], np.ndarray) else frame_reader[frame_id].asnumpy().shape[:2] if hasattr(
                frame_reader[frame_id], 'asnumpy') else frame_reader[frame_id].shape[:2]

            k_converted = convert_calibration(tracking_resolution, crop_box, k)

        all_vertices.append(vertices)

    # --- FLAME fitting ---
    print("Fitting FLAME parameters...")
    verts_3d = np.stack(all_vertices, axis=0)

    fit, pred_verts_3d, template_faces = fit_flame(
        verts_3d, gaze_directions,
        OPENCV2PYTORCH3D.numpy() @ rt,
        use_jaw_rotation=False,
        n_shape_params=150, n_expr_params=65,
        device=args.device,
        n_steps=8000,
        w_shape_reg=1e-4, w_expr_reg=1e-4,
        smooth_eye_rotations=is_video,
    )

    # Save converted meshes
    converted_mesh_dir = tracking_path / "flowface_mesh"
    converted_mesh_dir.mkdir(exist_ok=True)
    for frame_id, frame_path in enumerate(pixel_frames):
        trimesh.Trimesh(
            pred_verts_3d[frame_id], faces=template_faces
        ).export(converted_mesh_dir / f"{frame_path.stem}.ply")

    # --- Export fit.npz ---
    out_flame = {
        "fx": k_converted[0, 0][None, None].astype(np.float32),
        "fy": k_converted[1, 1][None, None].astype(np.float32),
        "cx": k_converted[0, 2][None, None].astype(np.float32),
        "cy": k_converted[1, 2][None, None].astype(np.float32),
        "extr": rt[None].astype(np.float32),
        "resolutions": np.array([orig_resolution]),
        "camera_order": ["cam0"],
        "rot": fit["rot"], "tra": fit["tra"],
        "shape": fit["shape"], "expr": fit["expr"],
        "neck_rot": fit["neck_rot"], "eye_rot": fit["eye_rot"],
    }

    np.savez(output_path / "fit.npz", **out_flame)
    print(f"Saved fit.npz to {output_path / 'fit.npz'}")

    # Reference images
    n_frames = len(pixel_frames)
    np.random.seed(123)
    selected_ids = sorted(np.random.permutation(np.arange(n_frames))[:args.max_n_ref])
    reference_info = [["cam0", int(sid)] for sid in selected_ids]
    with open(output_path / "reference_images.json", "w") as f:
        json.dump(reference_info, f, indent=4)

    # Camera trajectories (for video input)
    if is_video:
        fps = frame_reader.get_avg_fps()
        n_orbit_frames = n_frames

        trajectory = {
            "extr": out_flame["extr"].repeat(n_orbit_frames, axis=0),
            "fx": out_flame["fx"].repeat(n_orbit_frames, axis=0),
            "fy": out_flame["fy"].repeat(n_orbit_frames, axis=0),
            "cx": out_flame["cx"].repeat(n_orbit_frames, axis=0),
            "cy": out_flame["cy"].repeat(n_orbit_frames, axis=0),
            "resolution": orig_resolution,
            "fps": fps,
        }
        np.savez(output_path / "cam_static.npz", **trajectory)

        t = np.arange(n_orbit_frames) / fps / ORBIT_PERIOD
        yaw_angles = np.cos(t * 2 * np.pi) * ORBIT_AMPLITUDE_YAW
        pitch_angles = np.sin(t * 2 * np.pi) * ORBIT_AMPLITUDE_PITCH
        for i in range(n_orbit_frames):
            target = out_flame["tra"][0].copy()
            target[1:] = -target[1:]
            trajectory["extr"][i] = pivot_camera_intrinsic(
                trajectory["extr"][i], target, [yaw_angles[i], pitch_angles[i]]
            )
        np.savez(output_path / "cam_orbit.npz", **trajectory)

    print(f"Conversion complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pixel3dmm tracking to FlowFace format")
    parser.add_argument("--video_path", required=True, help="Input video or frame directory")
    parser.add_argument("--tracking_path", required=True, help="pixel3dmm tracking output directory")
    parser.add_argument("--preprocess_path", required=True, help="pixel3dmm preprocessing output")
    parser.add_argument("--output_path", required=True, help="Output FlowFace directory (fit.npz)")
    parser.add_argument("--max_n_ref", type=int, default=100)
    parser.add_argument("--enable_gaze_tracking", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
