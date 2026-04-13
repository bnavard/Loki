"""
Mesh-to-image rasterization: rasterizes FLAME mesh vertex properties (positions
and per-vertex offsets) into 2D feature maps used as spatial conditioning.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import (
    BlendParams,
    PerspectiveCameras,
    hard_rgb_blend,
    rasterize_meshes,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures.meshes import Meshes
from pytorch3d.io import load_obj


def create_camera_objects(
    K: torch.Tensor, RT: torch.Tensor, resolution: Tuple[int, int]
) -> PerspectiveCameras:
    R = RT[:, :3, :3]
    tvec = RT[:, :3, 3]

    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=-1)
    principal_point = K[:, :2, 2]

    H, W = resolution
    img_size = torch.tensor([[W, H]] * len(K), dtype=torch.int, device=K.device)

    scale = img_size.min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)

    c0 = img_size / 2.0

    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    R_pytorch3d = R.clone().permute(0, 2, 1)
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=img_size,
        device=K.device,
    )


class VertexShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _get_mesh_ndc(self, meshes, cameras):
        eps = None
        verts_world = meshes.verts_padded()
        verts_view = cameras.get_world_to_view_transform().transform_points(verts_world, eps=eps)
        projection_trafo = cameras.get_projection_transform().compose(cameras.get_ndc_camera_transform())
        verts_ndc = projection_trafo.transform_points(verts_view, eps=eps)
        verts_ndc[..., 2] = verts_view[..., 2]
        return meshes.update_padded(new_verts_padded=verts_ndc)

    def _get_fragments(self, cameras, meshes_ndc, img_shape, blur_sigma):
        znear = None
        if cameras is not None:
            znear = cameras.get_znear()
            if isinstance(znear, torch.Tensor):
                znear = znear.min().detach().item()
        z_clip = None if znear is None else znear / 2

        fragments = rasterize_meshes(
            meshes_ndc,
            image_size=img_shape,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blur_sigma,
            faces_per_pixel=4 if blur_sigma > 0.0 else 1,
            bin_size=None,
            max_faces_per_bin=None,
            clip_barycentric_coords=True,
            perspective_correct=cameras is not None,
            cull_backfaces=True,
            z_clip_value=z_clip,
            cull_to_frustum=False,
        )
        return Fragments(
            pix_to_face=fragments[0],
            zbuf=fragments[1],
            bary_coords=fragments[2],
            dists=fragments[3],
        )

    def _rasterize_property(self, property, fragments):
        prop_packed = torch.cat([property[i] for i in range(property.shape[0])], dim=0)
        return interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, prop_packed)

    def _rasterize_vertices(self, vertices, fragments):
        rasterized_properties = {}
        for key, prop in vertices.items():
            if key == "positions":
                continue
            rasterized_properties[key] = self._rasterize_property(prop, fragments)
        return rasterized_properties

    def forward(self, vertices, faces, intrinsics, extrinsics, img_shape, blur_sigma, return_meshes_and_cameras=False):
        meshes = Meshes(verts=vertices["positions"], faces=faces.to(vertices["positions"].device))
        cameras = None
        if intrinsics is not None:
            cameras = create_camera_objects(intrinsics, extrinsics, img_shape)
            meshes = self._get_mesh_ndc(meshes, cameras)
        fragments = self._get_fragments(cameras, meshes, img_shape, blur_sigma)
        pixels = self._rasterize_vertices(vertices, fragments)

        if return_meshes_and_cameras:
            return pixels, fragments, meshes, cameras
        return pixels, fragments


class PropRenderer(nn.Module):
    """Rasterizes per-vertex properties given ndc meshes."""

    def __init__(
        self,
        template_path="./data/assets/flame/cap4d_flame_template.obj",
        head_vert_path="./data/assets/flame/head_vertices.txt",
        n_mouth_verts=200,
        prop_type="verts",
    ) -> None:
        super().__init__()

        self.v_shader = VertexShader()

        verts, faces, aux = load_obj(template_path)

        self.register_buffer("faces", faces.verts_idx)
        self.register_buffer("faces_uvs", faces.textures_idx)

        vert_mask = torch.zeros(verts.shape[0]).bool()
        head_verts = torch.tensor(np.genfromtxt(head_vert_path)).long()
        vert_mask[head_verts] = 1
        vert_mask[-n_mouth_verts:] = 1

        face_mask = vert_mask[self.faces].max(dim=-1)[0]
        self.register_buffer("face_mask", face_mask)

        if prop_type == "verts":
            self.register_buffer("props", verts)
            self.props = self.props - self.props.mean(dim=-2, keepdim=True)
            self.props = self.props / self.props.max()
        elif prop_type == "uvs":
            self.register_buffer("props", aux.verts_uvs)
            self.props = self.props * 2. - 1.
            self.props[..., 1] = -self.props[..., 1]

    def render(self, vertices, img_shape, prop=None):
        b = vertices.shape[0]
        props_unpacked = self.props[self.faces][None].repeat(b, 1, 1, 1)

        verts = {
            "positions": vertices,
            "prop": props_unpacked,
        }

        if prop is not None:
            add_prop = prop[:, self.faces]
            verts["add_prop"] = add_prop

        pixels, fragments = self.v_shader(
            verts,
            self.faces[None].repeat(vertices.shape[0], 1, 1),
            None,
            None,
            img_shape,
            0.
        )

        img = pixels["prop"][..., 0, :]

        if prop is not None:
            img = torch.cat([img, pixels["add_prop"][..., 0, :]], dim=-1)

        render_mask = fragments.pix_to_face != -1
        face_mask = self.face_mask.repeat(b)
        face_masked = face_mask[torch.clamp(fragments.pix_to_face, 0)]
        render_mask = torch.logical_and(render_mask, face_masked)

        return img, render_mask
