"""
Image / video / FLAME utilities used by the TalkingHeadDataset and inference code.
"""
from pathlib import Path

import numpy as np
import cv2
from decord import VideoReader


CROP_MARGIN = 0.2


def crop_image(img, crop_box, bg_value=0):
    img_h = img.shape[0]
    img_w = img.shape[1]
    crop_h = crop_box[3] - crop_box[1]
    crop_w = crop_box[2] - crop_box[0]
    x_start = max(0, -crop_box[0])
    x_end = max(0, crop_box[2] - img_w)
    y_start = max(0, -crop_box[1])
    y_end = max(0, crop_box[3] - img_h)
    cropped_img = np.ones((crop_h, crop_w, *img.shape[2:]), dtype=img.dtype) * bg_value
    cropped_img[y_start: crop_h - y_end, x_start: crop_w - x_end, ...] = img[
        crop_box[1] + y_start: crop_box[3] - y_end,
        crop_box[0] + x_start: crop_box[2] - x_end,
        ...,
    ]
    return cropped_img


def rescale_image(img, target_resolution):
    interpolation_mode = cv2.INTER_LINEAR
    if target_resolution < img.shape[0]:
        interpolation_mode = cv2.INTER_AREA
    return cv2.resize(img, (target_resolution, target_resolution), interpolation=interpolation_mode)


def verts_to_pytorch3d(verts_2d, crop_box):
    verts_2d[..., 0] = -((verts_2d[..., 0] - crop_box[..., 0]) / (crop_box[..., 2] - crop_box[..., 0]) * 2. - 1.)
    verts_2d[..., 1] = -((verts_2d[..., 1] - crop_box[..., 1]) / (crop_box[..., 3] - crop_box[..., 1]) * 2. - 1.)
    return verts_2d


def get_square_bbox(bbox, border_margin=0.1, mode="max"):
    bbox = bbox.astype(int)
    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]
    b_center = ((bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2)
    if mode == "max":
        dim = int(max(bbox_h, bbox_w) // 2.0 * (1.0 + border_margin))
    elif mode == "min":
        dim = int(min(bbox_h, bbox_w) // 2.0 * (1.0 + border_margin))
    return (
        b_center[0] - dim,
        b_center[1] - dim,
        b_center[0] + dim,
        b_center[1] + dim,
    )


def get_bbox_from_verts(verts_2d, vert_mask):
    head_verts = verts_2d[vert_mask]
    head_bbox = [head_verts[..., 0].min(), head_verts[..., 1].min(), head_verts[..., 0].max(), head_verts[..., 1].max()]
    crop_box = get_square_bbox(np.array(head_bbox), border_margin=CROP_MARGIN)
    return np.array(crop_box)


class FrameReader:
    def __init__(self, video_path):
        self.frame_list = sorted(list(Path(video_path).glob("*.*")))

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        img = cv2.imread(str(self.frame_list[index]))[..., [2, 1, 0]]
        return img


def load_frame(video_path, frame_id):
    if Path(video_path).is_dir():
        video_reader = FrameReader(video_path)
    else:
        video_reader = VideoReader(str(video_path))

    if frame_id >= len(video_reader):
        frame_id = len(video_reader) - 1

    frame_img = video_reader[frame_id]
    if not isinstance(frame_img, np.ndarray):
        frame_img = frame_img.asnumpy()
    return frame_img
