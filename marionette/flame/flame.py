"""
FLAME 3DMM skinning and projection.

Contains the base FlameSkinner (originally from flowface/cap4d), the extended
CAP4DFlameSkinner with mouth vertices and expression offsets, and all supporting
utilities (Rodrigues, vertex projection, pkl loading, coordinate transforms).

Everything is self-contained — no external flowface dependency.
"""

import pickle
from typing import Any, Dict, List, MutableMapping

import einops
import numpy as np
import torch
import torch.nn.functional as F

from marionette.flame.mouth import FlameMouth

FLAME_PKL_PATH = "data/assets/flame/flame2023_no_jaw.pkl"
JAW_REGRESSOR_PATH = "data/assets/flame/jaw_regressor.npy"
BLINK_BLENDSHAPE_PATH = "data/assets/flame/blink_blendshape.npy"

FLAME_N_SHAPE = 300
FLAME_N_EXPR = 100
FLAME_N_VERTS = 5023

# OpenCV ↔ PyTorch3D coordinate convention: flip Y and Z axes.
OPENCV2PYTORCH3D = torch.eye(4)
OPENCV2PYTORCH3D[1, 1] = -1
OPENCV2PYTORCH3D[2, 2] = -1

def _convert_array(
    arr_dict: MutableMapping[str, Any],
    key: str,
    new_dtype: Any = None,
    squeeze: bool = True,
):
    arr = arr_dict[key]
    if callable(getattr(arr, "todense", None)):  # scipy sparse matrix
        arr = arr.todense()
    if new_dtype is None:
        if isinstance(arr.dtype, torch.dtype):
            if arr.dtype.is_floating_point:
                new_dtype = np.float32
        elif np.issubdtype(arr.dtype, np.floating):
            new_dtype = np.float32
        else:
            new_dtype = np.int64
    np_arr = np.array(arr, dtype=new_dtype)
    if squeeze:
        np_arr = np_arr.squeeze()
    arr_dict[key] = np_arr


def _load_model_pkl(path_model_pkl: str) -> MutableMapping[str, Any]:
    """Load a FLAME .pkl file and convert all entries to numpy arrays."""
    model_dict: MutableMapping[str, Any] = pickle.load(
        open(path_model_pkl, "rb"), encoding="latin1"
    )
    keys_to_delete: List[str] = []
    for key, value in model_dict.items():
        if not hasattr(value, "shape"):
            keys_to_delete.append(key)
        elif key == "f":
            _convert_array(model_dict, "f", new_dtype=np.int32)
        else:
            _convert_array(model_dict, key)
    for key in keys_to_delete:
        del model_dict[key]
    model_dict["kintree_table"][0, 0] = -1  # fix 2^32 - 1 sentinel
    return model_dict


# =============================================================================
# Geometric utilities
# =============================================================================

def _dot(x: torch.Tensor, y: torch.Tensor, dim=-1, keepdim=False) -> torch.Tensor:
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


def _safe_length(x: torch.Tensor, dim=-1, keepdim=False, eps=1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(_dot(x, x, dim=dim, keepdim=keepdim), min=eps))


def batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Axis-angle (B, 3) → rotation matrices (B, 3, 3) via Rodrigues' formula."""
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = _safe_length(rot_vecs, keepdim=True, eps=epsilon)
    rot_dir = rot_vecs / angle

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.view(batch_size, 3, 3)

    ident = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    return ident + sin * K + (1 - cos) * torch.bmm(K, K)


def transform_vertices(transform: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Apply [B, 4, 4] transform to [B, N, 3] vertices."""
    transformed = transform[:, :3, :3] @ vertices.permute(0, 2, 1)
    transformed = transformed + transform[:, :3, [3]]
    return transformed.permute(0, 2, 1)


def project_vertices(verts_3d: torch.Tensor, cam_parameters: Dict) -> torch.Tensor:
    """
    Project 3D vertices onto 2D screen coordinates.

    Args:
        verts_3d:       (N_t, V, 3)
        cam_parameters: dict with fx, fy, cx, cy (each N_c), extr (N_c, 4, 4)

    Returns:
        verts_2d: (N_c, N_t, V, 3)
    """
    extr = cam_parameters["extr"]
    verts_3d_cam = einops.einsum(extr[:, :3, :3], verts_3d, "N_c i j, N_t V j -> N_c N_t V i")
    verts_3d_cam = verts_3d_cam + extr[:, None, None, :3, 3]

    fx = cam_parameters["fx"][:, None]
    fy = cam_parameters["fy"][:, None]
    cx = cam_parameters["cx"][:, None]
    cy = cam_parameters["cy"][:, None]

    verts_2d = torch.stack([
        verts_3d_cam[..., 0] / verts_3d_cam[..., 2] * fx + cx,
        verts_3d_cam[..., 1] / verts_3d_cam[..., 2] * fy + cy,
        verts_3d_cam[..., 2] / verts_3d_cam[..., 2].mean(dim=-1)[..., None] * (fx + fy) / 2,
    ], dim=-1)

    return verts_2d


# =============================================================================
# FlameSkinner (base class, originally from flowface)
# =============================================================================

class FlameSkinner(torch.nn.Module):
    """
    Generates skinned FLAME meshes from shape, expression and pose parameters.
    No trainable parameters.
    """

    def __init__(
        self,
        flame_pkl_path: str,
        n_shape_params: int = FLAME_N_SHAPE,
        n_expr_params: int = FLAME_N_EXPR,
        blink_blendshape_path: str = None,
    ):
        super().__init__()

        assert n_shape_params <= FLAME_N_SHAPE
        assert n_expr_params <= FLAME_N_EXPR

        flame_dict = _load_model_pkl(flame_pkl_path)

        shape_eigenvecs = torch.tensor(flame_dict["shapedirs"][..., :n_shape_params])
        expr_eigenvecs = torch.tensor(
            flame_dict["shapedirs"][..., FLAME_N_SHAPE:FLAME_N_SHAPE + n_expr_params]
        )

        if blink_blendshape_path is not None:
            blink_blendshape = torch.tensor(np.load(blink_blendshape_path))
            expr_eigenvecs[:, :, -1] = blink_blendshape

        template_vertices = torch.tensor(flame_dict["v_template"])
        template_faces = torch.tensor(flame_dict["f"]).long()
        pose_eigenvecs = torch.tensor(flame_dict["posedirs"])
        pose_eigenvecs = einops.rearrange(pose_eigenvecs, "v xyz j -> j (v xyz)")
        joint_regressor = torch.tensor(flame_dict["J_regressor"])
        joint_parents = torch.tensor(flame_dict["kintree_table"][0]).long()
        skinning_weights = torch.tensor(flame_dict["weights"])

        self.n_shape_params = n_shape_params
        self.n_expr_params = n_expr_params

        self.register_buffer("template_vertices", template_vertices, persistent=False)
        self.register_buffer("template_faces", template_faces, persistent=False)
        self.register_buffer("shape_eigenvecs", shape_eigenvecs, persistent=False)
        self.register_buffer("expr_eigenvecs", expr_eigenvecs, persistent=False)
        self.register_buffer("pose_eigenvecs", pose_eigenvecs, persistent=False)
        self.register_buffer("joint_regressor", joint_regressor, persistent=False)
        self.register_buffer("joint_parents", joint_parents, persistent=False)
        self.register_buffer("skinning_weights", skinning_weights, persistent=False)

        self.cached_shape_eigenvecs = None
        self.cached_expr_eigenvecs = None
        self.cached_j_regressor = None
        self.cached_lbs_weights = None
        self.cached_pose_dirs = None
        self.cached_template_vertices = None

    def _get_template_vertices(self, vert_mask=None):
        if vert_mask is not None:
            if self.cached_template_vertices is None:
                self.cached_template_vertices = self.template_vertices[None, vert_mask]
            return self.cached_template_vertices
        return self.template_vertices[None]

    def _get_shape_offsets(self, shape_params, vert_mask=None):
        assert shape_params.shape[1] == self.n_shape_params
        if vert_mask is not None:
            if self.cached_shape_eigenvecs is None:
                self.cached_shape_eigenvecs = self.shape_eigenvecs[vert_mask]
            shape_eigenvecs = self.cached_shape_eigenvecs
        else:
            shape_eigenvecs = self.shape_eigenvecs
        return einops.einsum(shape_params, shape_eigenvecs, "b betas, V xyz betas -> b V xyz")

    def _get_expr_offsets(self, expr_params, vert_mask=None):
        assert expr_params.shape[1] == self.n_expr_params
        if vert_mask is not None:
            if self.cached_expr_eigenvecs is None:
                self.cached_expr_eigenvecs = self.expr_eigenvecs[vert_mask]
            expr_eigenvecs = self.cached_expr_eigenvecs
        else:
            expr_eigenvecs = self.expr_eigenvecs
        return einops.einsum(expr_params, expr_eigenvecs, "b betas, V xyz betas -> b V xyz")

    def _apply_joint_rotation(
        self, vertices, rotations,
        return_joints=False, return_transforms=False, vert_mask=None,
    ):
        j_regressor = self.joint_regressor
        lbs_weights = self.skinning_weights
        pose_dirs = einops.rearrange(
            self.pose_eigenvecs, "(J i j) (V xyz) -> J i j V xyz", i=3, j=3, xyz=3
        )

        if vert_mask is not None:
            if self.cached_j_regressor is None:
                self.cached_j_regressor = j_regressor[:, vert_mask]
                self.cached_lbs_weights = lbs_weights[vert_mask]
                self.cached_pose_dirs = pose_dirs[:, :, :, vert_mask]
            j_regressor = self.cached_j_regressor
            lbs_weights = self.cached_lbs_weights
            pose_dirs = self.cached_pose_dirs

        identity = torch.eye(3, dtype=torch.float32, device=vertices.device)
        pose_offset_params = rotations[:, [0, 2, 3, 4]] - identity
        pose_offset_params = pose_offset_params - identity
        pose_offsets = einops.einsum(pose_offset_params, pose_dirs, "B J i j, J i j V xyz -> B V xyz")

        assert rotations.shape[1] == j_regressor.shape[0]

        joints = einops.einsum(vertices, j_regressor, "b V xyz, J V -> b J xyz")
        v_posed = vertices + pose_offsets

        transforms = F.pad(rotations, [0, 1, 0, 1, 0, 0, 0, 0])
        transforms[..., -1, -1] = 1.
        transforms[..., :3, -1] = joints - (rotations @ joints[..., None])[..., 0]
        weighted_transforms = einops.einsum(lbs_weights, transforms, "V J, b J i j -> b V i j")
        v_posed_homo = F.pad(v_posed, [0, 1], value=1)
        v_rotated = einops.einsum(weighted_transforms, v_posed_homo, "b V i j, b V j -> b V i")

        output = [v_rotated[..., :3]]
        if return_joints:
            output.append(joints)
        if return_transforms:
            output.append(weighted_transforms)
        return output

    def forward(self, flame_sequence, vert_mask=None, return_gaze=False):
        shape_offsets = self._get_shape_offsets(flame_sequence["shape"][None], vert_mask)
        shape_verts = self._get_template_vertices(vert_mask) + shape_offsets

        expr_offsets = self._get_expr_offsets(flame_sequence["expr"], vert_mask)
        verts = shape_verts + expr_offsets

        rotations = torch.eye(3, device=verts.device)[None, None].repeat(verts.shape[0], 5, 1, 1)
        if "neck_rot" in flame_sequence and flame_sequence["neck_rot"] is not None:
            rotations[:, 1, ...] = batch_rodrigues(flame_sequence["neck_rot"])
        if "jaw_rot" in flame_sequence and flame_sequence["jaw_rot"] is not None:
            rotations[:, 2, ...] = batch_rodrigues(flame_sequence["jaw_rot"])
        if "eye_rot" in flame_sequence and flame_sequence["eye_rot"] is not None:
            eye_rot = batch_rodrigues(flame_sequence["eye_rot"])
            rotations[:, 3, ...] = eye_rot
            rotations[:, 4, ...] = eye_rot

        verts = self._apply_joint_rotation(verts, rotations=rotations, vert_mask=vert_mask)[0]

        base_rot = batch_rodrigues(flame_sequence["rot"])
        base_tra = flame_sequence["tra"][..., None]
        verts = (base_rot @ verts.permute(0, 2, 1) + base_tra).permute(0, 2, 1)

        if return_gaze:
            forward_vec = torch.tensor([0., 0., 1.], device=rotations.device)
            gaze_dirs = base_rot[:, None] @ rotations[:, 3:5] @ forward_vec[None, None, :, None]
            return verts, gaze_dirs[..., 0]
        else:
            return verts


# =============================================================================
# CAP4DFlameSkinner (extended with mouth vertices and expression offsets)
# =============================================================================

class CAP4DFlameSkinner(FlameSkinner):
    def __init__(
        self,
        flame_pkl_path: str = FLAME_PKL_PATH,
        n_shape_params: int = FLAME_N_SHAPE,
        n_expr_params: int = FLAME_N_EXPR,
        blink_blendshape_path: str = BLINK_BLENDSHAPE_PATH,
        add_mouth: bool = False,
        add_lower_jaw: bool = False,
        jaw_regressor_path: str = JAW_REGRESSOR_PATH,
    ):
        super().__init__(flame_pkl_path, n_shape_params, n_expr_params, blink_blendshape_path)

        self.add_mouth = add_mouth
        if add_mouth:
            self.mouth = FlameMouth()

        self.add_lower_jaw = add_lower_jaw
        if add_lower_jaw:
            self.lower_jaw = FlameMouth()
            jaw_regressor = torch.tensor(np.load(jaw_regressor_path))
            self.register_buffer("jaw_regressor", jaw_regressor)

    def forward(
        self,
        flame_sequence: Dict,
        return_offsets: bool = True,
        return_transforms: bool = False,
    ) -> torch.Tensor:
        shape_offsets = self._get_shape_offsets(flame_sequence["shape"][None], None)
        shape_verts = self._get_template_vertices(None) + shape_offsets

        expr_offsets = self._get_expr_offsets(flame_sequence["expr"], None)
        verts = shape_verts + expr_offsets

        rotations = torch.eye(3, device=verts.device)[None, None].repeat(verts.shape[0], 5, 1, 1)
        if "neck_rot" in flame_sequence and flame_sequence["neck_rot"] is not None:
            rotations[:, 0, ...] = batch_rodrigues(flame_sequence["neck_rot"])
        if "jaw_rot" in flame_sequence and flame_sequence["jaw_rot"] is not None:
            rotations[:, 2, ...] = batch_rodrigues(flame_sequence["jaw_rot"])
        if "eye_rot" in flame_sequence and flame_sequence["eye_rot"] is not None:
            eye_rot = batch_rodrigues(flame_sequence["eye_rot"])
            rotations[:, 3, ...] = eye_rot
            rotations[:, 4, ...] = eye_rot

        verts, v_transforms = self._apply_joint_rotation(verts, rotations=rotations, vert_mask=None, return_transforms=True)

        offsets = verts - shape_verts
        if self.add_mouth:
            mouth_verts = self.mouth(shape_verts, self.joint_regressor)
            mouth_verts = mouth_verts.repeat(verts.shape[0], 1, 1)
            verts = torch.cat([verts, mouth_verts], dim=1)
            offsets = torch.cat([offsets, torch.zeros_like(mouth_verts)], dim=1)
            v_transforms = torch.cat([v_transforms, torch.zeros(mouth_verts.shape[0], mouth_verts.shape[1], 4, 4, device=v_transforms.device)], dim=1)
        if self.add_lower_jaw:
            jaw_rot = einops.einsum(flame_sequence["expr"], self.jaw_regressor, 'b exp, exp r -> b r')
            neutral_jaw_verts = self.lower_jaw(shape_verts, self.joint_regressor, batch_rodrigues(jaw_rot * 0.))
            jaw_verts = self.lower_jaw(shape_verts, self.joint_regressor, batch_rodrigues(jaw_rot))
            verts = torch.cat([verts, jaw_verts], dim=1)
            offsets = torch.cat([offsets, jaw_verts - neutral_jaw_verts], dim=1)
            jaw_transforms = torch.zeros(jaw_verts.shape[0], 4, 4, device=v_transforms.device)
            jaw_transforms[:, :3, :3] = batch_rodrigues(jaw_rot)
            jaw_transforms[..., -1, -1] = 1.
            jaw_transforms = jaw_transforms[:, None].repeat(1, jaw_verts.shape[1], 1, 1)
            v_transforms = torch.cat([v_transforms, jaw_transforms], dim=1)

        base_rot = batch_rodrigues(flame_sequence["rot"])
        base_tra = flame_sequence["tra"][..., None]
        verts = (base_rot @ verts.permute(0, 2, 1) + base_tra).permute(0, 2, 1)

        output = [verts]
        if return_offsets:
            output.append(offsets)
        if return_transforms:
            base_transform = torch.cat([base_rot, base_tra], dim=2)
            base_transform = torch.cat([base_transform, torch.zeros_like(base_transform[:, :1, ...])], dim=1)
            base_transform[..., -1, -1] = 1.
            v_transforms = einops.einsum(base_transform, v_transforms, 'b i j, b N j k -> b N i k')
            output.append(v_transforms)

        return output

def compute_flame(
    flame: CAP4DFlameSkinner,
    fit_3d: Dict[str, np.ndarray],
):
    flame_sequence = {
        "shape": torch.tensor(fit_3d["shape"]).float(),
        "expr": torch.tensor(fit_3d["expr"]).float(),
        "rot": torch.tensor(fit_3d["rot"]).float(),
        "tra": torch.tensor(fit_3d["tra"]).float(),
        "eye_rot": torch.tensor(fit_3d["eye_rot"]).float(),
        "jaw_rot": None,
        "neck_rot": None,
    }
    if "neck_rot" in fit_3d:
        flame_sequence["neck_rot"] = torch.tensor(fit_3d["neck_rot"]).float()
    if "jaw_rot" in fit_3d:
        flame_sequence["jaw_rot"] = torch.tensor(fit_3d["jaw_rot"]).float()

    verts_3d, offsets_3d = flame(flame_sequence, return_offsets=True)

    fx, fy, cx, cy = [torch.tensor(fit_3d[key]).float() for key in ["fx", "fy", "cx", "cy"]]
    extr = torch.tensor(fit_3d["extr"]).float()
    cam_parameters = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "extr": extr}

    verts_3d_cv = transform_vertices(OPENCV2PYTORCH3D[None].to(verts_3d.device), verts_3d)
    verts_2d = project_vertices(verts_3d_cv, cam_parameters)

    return {
        "verts_3d": verts_3d.cpu().numpy(),
        "verts_3d_cv": verts_3d_cv.cpu().numpy(),
        "verts_2d": verts_2d.cpu().numpy(),
        "offsets_3d": offsets_3d.cpu().numpy(),
    }
