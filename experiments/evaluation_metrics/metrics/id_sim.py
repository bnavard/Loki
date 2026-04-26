"""ArcFace identity-preservation metric — cross-identity protocol.

For each generated video, score = mean cosine similarity between the
**generated frames' ArcFace embeddings** and a fixed identity-prior built
from the **reference clip**. Higher is better; bounded in [-1, 1].

Why this metric belongs to cross-identity only: in same-identity
reconstruction the generation is paired against the *same* clip, so PSNR/
SSIM/LPIPS already capture pixel-level identity match. In cross-identity
the prediction is made to look like the ref identity while *moving* like
the driver, and there is no GT video to pair against; ArcFace cosine on
the per-frame face embedding is the standard measure for "did the face
keep the ref's identity?" (X-Portrait, HunyuanPortrait both report it).

Identity prior choice: averaging ArcFace embeddings over **all frames of
the ref clip**, L2-normalizing the mean. More robust than picking a single
ref frame (which depends on `ref_frame_idx`, a value the runner samples
internally and doesn't dump per-sample) and matches the
ArcFace-of-an-identity convention used in face-recognition benchmarks.

Detector backbone: InsightFace's `buffalo_l` — pip-installable, weights
auto-download into `~/.insightface/`. Both the face detector (RetinaFace)
and the embedding net (ArcFace, R100) come from the same pack so we get
consistent crops and embeddings.
"""
from __future__ import annotations

import numpy as np
import torch


class IDSimilarity:
    """Stateful ArcFace wrapper. One InsightFace `FaceAnalysis` per process —
    the underlying ONNXRuntime sessions are not safe to share across
    multiprocessing workers; construct one per worker if parallelizing.

    Frames are passed in as `(T, 3, H, W)` float32 in `[0, 1]` (the IO
    convention) and converted to BGR uint8 internally because InsightFace
    expects OpenCV-style frames.
    """

    def __init__(
        self,
        device:    str = "cuda",
        det_size:  tuple[int, int] = (640, 640),
        # Cosine fallback when the detector misses on a frame: skip the
        # frame and renormalize over hits, mirroring LMD's `detect_rate`.
    ) -> None:
        from insightface.app import FaceAnalysis
        # Provider list controls execution; CUDA first with CPU fallback so
        # the metric still runs on host-only boxes (slowly).
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda") else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    # ---------------- internals ----------------

    def _frames_to_bgr_uint8(self, frames: torch.Tensor) -> list[np.ndarray]:
        """`(T, 3, H, W)` float32 in `[0, 1]` (RGB) → list of `(H, W, 3)`
        uint8 BGR ndarrays (InsightFace's OpenCV expectation)."""
        arr = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return [a[..., ::-1].copy() for a in arr]   # RGB → BGR

    def _embed_frames(self, frames: torch.Tensor) -> np.ndarray:
        """Returns `(N_hits, 512)` L2-normalized embeddings for the largest
        face per frame. Frames where no face was detected are dropped — the
        caller decides what to do with the resulting hit count."""
        out: list[np.ndarray] = []
        for img_bgr in self._frames_to_bgr_uint8(frames):
            faces = self.app.get(img_bgr)
            if not faces:
                continue
            # Largest detection, by bbox area.
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            emb = face.normed_embedding   # already L2-normed by InsightFace
            out.append(emb.astype(np.float32))
        return np.stack(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)

    # ---------------- public API ----------------

    def build_identity_prior(self, ref_frames: torch.Tensor) -> np.ndarray:
        """Average ArcFace embedding across all `ref_frames` of the reference
        clip, L2-normalized. Returns shape `(512,)` or `None` if no frame
        produced a detection (caller must handle)."""
        embs = self._embed_frames(ref_frames)
        if embs.shape[0] == 0:
            return None  # type: ignore[return-value]
        mean = embs.mean(axis=0)
        return mean / (np.linalg.norm(mean) + 1e-8)

    def score(
        self,
        gen_frames: torch.Tensor,
        identity_prior: np.ndarray,
    ) -> tuple[float, float]:
        """Mean cosine similarity between every detected face in
        `gen_frames` and `identity_prior`.

        Args:
            gen_frames: `(T, 3, H, W)` in `[0, 1]`.
            identity_prior: `(512,)` L2-normalized vector from
                `build_identity_prior`.

        Returns:
            `(mean_cosine, detect_rate)` — `detect_rate` is hits / T.
            `mean_cosine` is `nan` if the detector missed on every frame.
        """
        embs = self._embed_frames(gen_frames)
        T = gen_frames.shape[0]
        if embs.shape[0] == 0:
            return float("nan"), 0.0
        # InsightFace returns L2-normed embeddings, and identity_prior is
        # also normalized — dot product == cosine.
        cos = (embs @ identity_prior).astype(np.float32)
        return float(cos.mean()), float(embs.shape[0] / T)
