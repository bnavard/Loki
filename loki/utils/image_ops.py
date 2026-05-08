"""Padded crop + isotropic resize — used for face-centered image loading."""
import numpy as np
import cv2


def crop_image(img, crop_box, bg_value=0):
    """Crop img to crop_box, padding out-of-bounds regions with bg_value."""
    img_h, img_w = img.shape[:2]
    crop_h = crop_box[3] - crop_box[1]
    crop_w = crop_box[2] - crop_box[0]
    x_start = max(0, -crop_box[0])
    x_end   = max(0, crop_box[2] - img_w)
    y_start = max(0, -crop_box[1])
    y_end   = max(0, crop_box[3] - img_h)
    out = np.ones((crop_h, crop_w, *img.shape[2:]), dtype=img.dtype) * bg_value
    out[y_start: crop_h - y_end, x_start: crop_w - x_end, ...] = img[
        crop_box[1] + y_start: crop_box[3] - y_end,
        crop_box[0] + x_start: crop_box[2] - x_end,
        ...,
    ]
    return out


def rescale_image(img, target_resolution):
    """Square-resize img to (target_resolution, target_resolution)."""
    interp = cv2.INTER_AREA if target_resolution < img.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(img, (target_resolution, target_resolution), interpolation=interp)
