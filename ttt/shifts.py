"""
Distribution shifts for Phase 2: fixed corruptions and deterministic AugMix views.

Two corruption families are evaluated on the frozen 8k eval subset, gaussian
noise and gaussian blur, at severities 1/3/5. MEMO adapts on the marginal
entropy of AugMix views of the shifted image, so views must be reproducible:
a cached view has to match what a deployment would compute, and a rerun has to
match the cache. Seeds therefore come from a checksum of the sample id, since
Python's hash() is salted per process, and AugMix runs inside a forked RNG so
building a cache never perturbs sampling elsewhere in the same process.
"""

import zlib
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageFilter

# Severity constants follow ImageNet-C (Hendrycks & Dietterich, 2019).
GAUSSIAN_NOISE_STD = {1: 0.08, 2: 0.12, 3: 0.18, 4: 0.26, 5: 0.38}
GAUSSIAN_BLUR_SIGMA = {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 6.0}
SEVERITIES = (1, 2, 3, 4, 5)


def _gaussian_noise(image: Image.Image, severity: int, rng) -> Image.Image:
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = x + rng.normal(loc=0.0, scale=GAUSSIAN_NOISE_STD[severity], size=x.shape)
    return Image.fromarray((np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8))


def _gaussian_blur(image: Image.Image, severity: int, rng) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=GAUSSIAN_BLUR_SIGMA[severity]))


CORRUPTIONS: Dict[str, Any] = {
    "gaussian_noise": _gaussian_noise,
    "gaussian_blur": _gaussian_blur,
}


def seed_for(sample_id: str, view: Any, salt: Any = 0) -> int:
    """Stable 32-bit seed for one (sample, view).

    crc32 of the joined fields: identical in every process and session, unlike
    hash(), which is salted per process and would make a cached view
    irreproducible on the next run.
    """
    return zlib.crc32(f"{sample_id}:{view}:{salt}".encode()) & 0xFFFFFFFF


def corrupt(image: Image.Image, kind: str, severity: int, sample_id: str = "") -> Image.Image:
    """Apply one corruption at one severity.

    Args:
        image: Source image.
        kind: Key of CORRUPTIONS.
        severity: 1-5, the ImageNet-C scale.
        sample_id: Ties the noise draw to the sample, so a rerun reproduces it.

    Returns:
        The corrupted image, RGB.

    Raises:
        ValueError: On an unknown corruption or severity.
    """
    if kind not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption '{kind}'. Valid: {', '.join(sorted(CORRUPTIONS))}")
    if severity not in SEVERITIES:
        raise ValueError(f"Severity {severity} outside 1-5.")
    rng = np.random.default_rng(seed_for(sample_id, "corrupt", severity))
    return CORRUPTIONS[kind](image.convert("RGB"), severity, rng)


def augmix_views(
    image: Image.Image,
    n_views: int = 4,
    sample_id: str = "",
    severity: int = 3,
    mixture_width: int = 3,
) -> List[Image.Image]:
    """n_views AugMix views of one image, deterministic in (sample_id, view).

    MEMO's marginal entropy is H(mean(p)) over these views. The original image
    is not among them: it is encoded separately and carries the prediction.

    Args:
        image: Source image, already corrupted for a shift condition.
        n_views: Number of views (4 in the main configuration).
        sample_id: Seeds the views, so the cache is reproducible.
        severity: AugMix severity.
        mixture_width: Number of augmentation chains mixed per view.

    Returns:
        List of n_views images.
    """
    from torchvision.transforms import AugMix

    augmix = AugMix(severity=severity, mixture_width=mixture_width)
    source = image.convert("RGB")
    views = []
    for view in range(n_views):
        # Forked RNG: AugMix draws from torch's global generator, and a cache
        # build must not shift the stream any other code is drawing from.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_for(sample_id, view))
            views.append(augmix(source))
    return views
