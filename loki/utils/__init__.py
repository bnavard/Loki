"""Public surface for loki's utility modules. Import from here
(`from loki.utils import ...`) rather than reaching into the
submodules directly — the internal layout can change, the public names
should not."""
from loki.utils.video_io import FrameReader, load_frame
from loki.utils.image_ops import crop_image, rescale_image
from loki.utils.verts import (
    CROP_MARGIN,
    verts_to_pytorch3d,
    get_square_bbox,
    get_bbox_from_verts,
)
from loki.utils.viz import (
    VisualizationCallback,
    slice_cond_rgb,
    add_label,
    save_labeled_grid,
    save_video,
    make_grid_tensor,
)
from loki.utils.log_tee import install_log_tee

__all__ = [
    # video I/O
    "FrameReader", "load_frame",
    # image ops
    "crop_image", "rescale_image",
    # verts
    "CROP_MARGIN", "verts_to_pytorch3d", "get_square_bbox", "get_bbox_from_verts",
    # viz
    "VisualizationCallback", "slice_cond_rgb", "add_label",
    "save_labeled_grid", "save_video", "make_grid_tensor",
    # log tee
    "install_log_tee",
]
