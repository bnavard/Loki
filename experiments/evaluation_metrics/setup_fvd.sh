#!/bin/bash
# =============================================================================
# One-time setup for the FVD path.
#
# Installs `cdfvd`, applies the upstream-URL patch to its VideoMAEv2
# loader, and pre-stages the SSv2-finetuned giant checkpoint (~1.9 GB)
# into the env's site-packages so the first FVD run doesn't block on a
# 2 GB download.
#
# Idempotent — re-runs are a no-op once everything is already in place.
#
# Also installs the runtime deps for the per-sample pixel-aligned
# metrics (PSNR / SSIM via torchmetrics, LPIPS).
#
# Usage (from repo root):
#
#     bash experiments/evaluation_metrics/setup_fvd.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=src/_activate.sh
source "${HERE}/src/_activate.sh"

echo "============================================================"
echo "[setup_fvd] activating env from src/_activate.sh: $(which python)"
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: pip-install the runtime deps.
# -----------------------------------------------------------------------------
# Notes:
# * The PyPI name is `cd-fvd` (with a hyphen); the import is `cdfvd` (no
#   hyphen). Don't try `pip install cdfvd` — that name doesn't exist.
# * `av<14` is a load-bearing pin: cd-fvd's video-folder loader imports
#   `av.AVError`, which PyAV 14 renamed to `av.error.FFmpegError`. Lift
#   this once cd-fvd catches up.
# * `lpips==0.1.4` for reproducibility — its AlexNet weights are baked
#   into the package and any future repackaging would change reported
#   numbers.
# * `torchmetrics>=1.3` — the SSIM API stabilized at 1.3.
echo ""
echo "===== Step 1: pip install cd-fvd torchmetrics lpips av<14 ====="
"${PYTHON}" -m pip install --quiet --upgrade pip
"${PYTHON}" -m pip install --quiet \
    "cd-fvd" \
    "torchmetrics>=1.3" \
    "lpips==0.1.4" \
    "av<14"
echo "[setup_fvd] pip install OK."

# -----------------------------------------------------------------------------
# Step 2: patch cdfvd's VideoMAE-v2 loader.
# -----------------------------------------------------------------------------
# Upstream cdfvd hardcodes a download URL (pjlab-gvm-data on Aliyun OSS)
# that's been taken down — `requests.get` saves the 404 HTML body as a
# .pth, `torch.load` then dies with `invalid load key, '<'`. Our patch
# at `patches/cdfvd_videomaev2_utils.py` redirects to the HuggingFace
# mirror at `OpenGVLab/VideoMAE2/mae-g/` (ungated, identical filename).
# Copy the patch over the installed file so the fix lives in code, not
# just on-disk state.
echo ""
echo "===== Step 2: patch cdfvd VideoMAEv2 loader ====="
CDFVD_VMAE_DIR="$("${PYTHON}" -c 'import cdfvd, os; print(os.path.join(os.path.dirname(cdfvd.__file__), "third_party", "VideoMAEv2"))')"
PATCH_SRC="${HERE}/patches/cdfvd_videomaev2_utils.py"
PATCH_DST="${CDFVD_VMAE_DIR}/utils.py"
if [[ ! -f "${PATCH_SRC}" ]]; then
    echo "ERROR: patch source missing at ${PATCH_SRC}"
    exit 1
fi
if cmp -s "${PATCH_SRC}" "${PATCH_DST}"; then
    echo "[setup_fvd] SKIP: ${PATCH_DST} already matches the patched version."
else
    cp "${PATCH_SRC}" "${PATCH_DST}"
    echo "[setup_fvd] patched ${PATCH_DST}"
fi

# -----------------------------------------------------------------------------
# Step 3: pre-download VideoMAE-v2 SSv2-finetuned giant (~1.9 GB).
# -----------------------------------------------------------------------------
# Backbone for the cdfvd VideoMAE-v2 FVD path. Pre-staging here means
# the first FVD run doesn't block on a 2 GB download. Skipped if already
# present.
echo ""
echo "===== Step 3: download VideoMAE-v2 SSv2-ft checkpoint ====="
VMAE_PATH="${CDFVD_VMAE_DIR}/vit_g_hybrid_pt_1200e_ssv2_ft.pth"
VMAE_URL="https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth"
if [[ -f "${VMAE_PATH}" ]] && [[ "$(stat -c '%s' "${VMAE_PATH}")" -gt 1000000000 ]]; then
    echo "[setup_fvd] SKIP: ${VMAE_PATH} already present ($(du -h "${VMAE_PATH}" | cut -f1))."
else
    rm -f "${VMAE_PATH}"   # clear any stub / partial / HTML-disguised-as-pth
    wget --progress=bar:force -O "${VMAE_PATH}" "${VMAE_URL}"
    echo "[setup_fvd] downloaded ${VMAE_PATH}"
fi

echo ""
echo "============================================================"
echo "[setup_fvd] done."
echo ""
echo "Next steps (from repo root):"
echo "  PYTHONPATH=. python experiments/evaluation_metrics/compute_fvd.py"
echo "============================================================"