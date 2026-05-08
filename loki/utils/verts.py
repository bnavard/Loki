"""FLAME vertex projection + face-bbox utilities (pytorch3d NDC)."""
import numpy as np


CROP_MARGIN = 0.2


def verts_to_pytorch3d(verts_2d, crop_box):
    """Map pixel-space verts into pytorch3d NDC [-1, 1] relative to crop_box.

    pytorch3d NDC convention: +x = left, +y = up. Both axes are sign-flipped
    here; downstream consumers (the FLAME mesh rasterizer in
    `loki.conditioning.mesh2img.PropRenderer`) expect this convention.
    """
    verts_2d[..., 0] = -((verts_2d[..., 0] - crop_box[..., 0]) / (crop_box[..., 2] - crop_box[..., 0]) * 2. - 1.)
    verts_2d[..., 1] = -((verts_2d[..., 1] - crop_box[..., 1]) / (crop_box[..., 3] - crop_box[..., 1]) * 2. - 1.)
    return verts_2d


def get_square_bbox(bbox, border_margin=0.1, mode="max"):
    """Inflate a (x0, y0, x1, y1) bbox to a square with a margin."""
    bbox = bbox.astype(int)
    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]
    cx, cy = (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2
    if mode == "max":
        dim = int(max(bbox_h, bbox_w) // 2.0 * (1.0 + border_margin))
    elif mode == "min":
        dim = int(min(bbox_h, bbox_w) // 2.0 * (1.0 + border_margin))
    return (cx - dim, cy - dim, cx + dim, cy + dim)


def get_bbox_from_verts(verts_2d, vert_mask):
    """Axis-aligned square bbox around the verts selected by vert_mask (+ CROP_MARGIN)."""
    head_verts = verts_2d[vert_mask]
    head_bbox = [
        head_verts[..., 0].min(), head_verts[..., 1].min(),
        head_verts[..., 0].max(), head_verts[..., 1].max(),
    ]
    return np.array(get_square_bbox(np.array(head_bbox), border_margin=CROP_MARGIN))
