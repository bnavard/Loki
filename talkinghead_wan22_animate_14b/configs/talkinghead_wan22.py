# Combined config: Wan2.2-Animate-14B + S2V audio injection for talking-head generation.
#
# Merges:
#   - wan_animate_14B: base DiT + I2V + FaceAdapter (expression conditioning)
#   - wan_s2v_14B:     audio injection via AdaIN at selected DiT layers
#
# The text conditioning (T5) is kept active with a fixed prompt to preserve
# pretrained weight distribution. Audio is injected via the S2V mechanism,
# and face/expression via the Animate FaceAdapter.

from easydict import EasyDict
from talkinghead_wan22_animate_14b.wan.configs.shared_config import wan_shared_cfg

talkinghead_wan22 = EasyDict(__name__='Config: Talking-Head Wan2.2 Animate+S2V 14B')
talkinghead_wan22.update(wan_shared_cfg)

# ---- T5 text encoder ----
talkinghead_wan22.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
talkinghead_wan22.t5_tokenizer = 'google/umt5-xxl'

# ---- CLIP (for reference image encoding) ----
talkinghead_wan22.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
talkinghead_wan22.clip_tokenizer = 'xlm-roberta-large'
talkinghead_wan22.lora_checkpoint = 'relighting_lora.ckpt'

# ---- VAE (Wan2.1 3D causal VAE, stride 4x8x8) ----
talkinghead_wan22.vae_checkpoint = 'Wan2.1_VAE.pth'
talkinghead_wan22.vae_stride = (4, 8, 8)

# ---- wav2vec (for S2V audio injection) ----
talkinghead_wan22.wav2vec = "wav2vec2-large-xlsr-53-english"

# ---- DiT transformer (Animate base) ----
talkinghead_wan22.patch_size = (1, 2, 2)
talkinghead_wan22.dim = 5120
talkinghead_wan22.ffn_dim = 13824
talkinghead_wan22.freq_dim = 256
talkinghead_wan22.num_heads = 40
talkinghead_wan22.num_layers = 40
talkinghead_wan22.window_size = (-1, -1)
talkinghead_wan22.qk_norm = True
talkinghead_wan22.cross_attn_norm = True
talkinghead_wan22.eps = 1e-6

# ---- Animate: face motion encoder + FaceAdapter ----
talkinghead_wan22.use_face_encoder = True
talkinghead_wan22.motion_encoder_dim = 512

# ---- S2V: audio injection config ----
talkinghead_wan22.audio = EasyDict()
talkinghead_wan22.audio.enable = True
talkinghead_wan22.audio.audio_dim = 1024
talkinghead_wan22.audio.enable_adain = True
talkinghead_wan22.audio.adain_mode = "attn_norm"
talkinghead_wan22.audio.audio_inject_layers = [
    0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39
]
talkinghead_wan22.audio.enable_framepack = True
talkinghead_wan22.audio.framepack_drop_mode = 'padd'

# ---- Inference defaults ----
talkinghead_wan22.sample_shift = 5.0
talkinghead_wan22.sample_steps = 20
talkinghead_wan22.sample_guide_scale = 1.0
talkinghead_wan22.frame_num = 77
talkinghead_wan22.sample_fps = 30
talkinghead_wan22.prompt = 'a person talking'
talkinghead_wan22.sample_neg_prompt = (
    "blurry, worst quality, blurry details, subtitle, ugly, "
    "deformed, extra fingers, poorly drawn hands, poorly drawn face"
)
