"""
Copied directly from cap4d/flame/mouth.py — no changes needed.
"""
import torch
import torch.nn as nn
import numpy as np
import einops


def generate_uv_sphere(r=1.0, latitude_steps=30, longitude_steps=30):
    # Creat half a sphere!
    # Generate latitude and longitude
    latitudes = torch.linspace(-np.pi / 2, np.pi / 2, latitude_steps)[:latitude_steps // 2]
    longitudes = torch.linspace(0, 2 * np.pi, longitude_steps)

    verts = []
    # Create a meshgrid for latitudes and longitudes
    for lat in latitudes:
        for lon in longitudes:
            verts.append([
                r * torch.cos(lat) * torch.cos(lon),
                r * torch.cos(lat) * torch.sin(lon),
                r * torch.sin(lat),
            ])

    verts = torch.tensor(verts)

    # Generate triangle indices
    indices = []
    for i in range(latitude_steps // 2 - 1):
        for j in range(longitude_steps):
            # Current vertex
            lat_1_lon_1 = i * longitude_steps + j
            lat_1_lon_2 = i * longitude_steps + (j + 1) % longitude_steps
            # Next row's vertices
            lat_2_lon_1 = (i + 1) * longitude_steps + j
            lat_2_lon_2 = (i + 1) * longitude_steps + (j + 1) % longitude_steps

            if i < latitude_steps - 2:
                indices.append([lat_1_lon_1, lat_2_lon_2, lat_2_lon_1])

            if i > 0:
                indices.append([lat_1_lon_1, lat_1_lon_2, lat_2_lon_2])

    # Convert indices to a tensor
    face_indices = torch.tensor(indices, dtype=torch.long)

    return verts, face_indices


class FlameMouth(nn.Module):
    def __init__(
        self,
        long_steps=20,
        lat_steps=20,
        lip_v_index=3533,
        lip_offset=0.005,
    ):
        super().__init__()

        v_sphere, f_sphere = generate_uv_sphere(
            r=1.,
            latitude_steps=lat_steps,
            longitude_steps=long_steps,
        )
        v_sphere[:, 1] = -v_sphere[:, 1]  # flip axis
        v_sphere[:, 2] = -v_sphere[:, 2]  # flip axis to align in right direction

        self.lip_v_index = lip_v_index
        self.register_buffer("vertices", v_sphere)
        self.register_buffer("faces", f_sphere)

        self.lip_offset = lip_offset

    def forward(
        self,
        neutral_verts,
        joint_regressor,
        jaw_rotation=None,
    ):
        jaw_joint = einops.einsum(neutral_verts, joint_regressor[2], "b V xyz, V -> b xyz") # (B, 3)

        lip_vert = neutral_verts[:, self.lip_v_index]

        offset = lip_vert - jaw_joint

        distance = offset.norm(dim=-1, keepdim=True)
        direction = offset / distance
        y = torch.zeros_like(direction, device=direction.device)
        y[:, 1] = 1
        new_x = torch.cross(y, direction, dim=-1)
        new_x = new_x / new_x.norm(dim=-1, keepdim=True)
        new_y = torch.cross(direction, new_x, dim=-1)
        new_y = new_y / new_y.norm(dim=-1, keepdim=True)
        new_z = direction

        rot_mat = torch.stack([new_x, new_y, new_z], dim=-1)
        v_sphere = self.vertices[None] * distance[..., None] * 0.25

        v_sphere = (rot_mat @ v_sphere.permute(0, 2, 1)).permute(0, 2, 1)
        center = jaw_joint + offset * 0.75 - self.lip_offset * direction
        v_sphere = v_sphere + center

        if jaw_rotation is not None:
            v_offset = jaw_rotation @ (v_sphere - jaw_joint).permute(0, 2, 1)
            v_sphere = jaw_joint + v_offset.permute(0, 2, 1)

        return v_sphere
